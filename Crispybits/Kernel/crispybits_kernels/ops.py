
from __future__ import annotations
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from . import _C as _CUDA
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    _CUDA = None
    _IMPORT_ERROR = e

_C = _CUDA
if os.environ.get("CRISPYBITS_BACKEND", "").lower() == "emulator":
    from . import emulator as _C  # noqa: F811

DECODE_THRESHOLD = 4    
ACT = {"none": 0, None: 0, "relu": 1, "silu": 2}
KINDS = ("gemv", "qkv", "gate_up")


def set_backend(name: str):
   
    global _C
    if name == "cuda":
        _C = _CUDA
    elif name == "emulator":
        from . import emulator
        _C = emulator
    else:
        raise ValueError(name)
    tile_n.cache_clear()


def backend_name() -> str:
    return "emulator" if (_C is not None and getattr(_C, "IS_EMULATOR", False)) else "cuda"


@dataclass
class PackedLinearWeight:
    packed: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int
    group_size: int
    in_features: int
    out_features: int
    bias: Optional[torch.Tensor] = None

    rope_perm_head_dim: int = 0

    def cuda(self, device=None, dtype=torch.float16):
        device = device or "cuda"
        return replace(
            self,
            packed=self.packed.to(device=device, dtype=torch.int32).contiguous(),
            scales=self.scales.to(device=device, dtype=dtype).contiguous(),
            zeros=self.zeros.to(device=device, dtype=dtype).contiguous(),
            bias=None if self.bias is None else self.bias.to(device=device, dtype=dtype).contiguous(),
        )


def _check_ext():
    if _C is None:
        raise RuntimeError(f"crispybits CUDA extension is not built: {_IMPORT_ERROR}")


def _empty(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, device=x.device, dtype=x.dtype)


def _bias_tensor(w: PackedLinearWeight, x: torch.Tensor):
    return w.bias if w.bias is not None else _empty(x)


def _ws(workspace, x):
    return workspace if workspace is not None else torch.empty(0, device=x.device, dtype=torch.int32)


def new_workspace(device) -> torch.Tensor:

    return torch.zeros(2, dtype=torch.int32, device=device)


def rope_row_order(out_features: int, head_dim: int, device=None) -> torch.Tensor:
    """Stored row p <- original row order[p]: per head, (0, h/2, 1, h/2+1, ...)."""
    if head_dim % 2 or out_features % head_dim:
        raise ValueError("out_features must be a multiple of an even head_dim")
    h = head_dim // 2
    within = torch.stack([torch.arange(h), torch.arange(h) + h], 1).reshape(-1)
    heads = torch.arange(out_features // head_dim)[:, None] * head_dim
    return (heads + within[None]).reshape(-1).to(device)


def rope_inverse_order(out_features: int, head_dim: int, device=None) -> torch.Tensor:
    order = rope_row_order(out_features, head_dim)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel())
    return inv.to(device)


def permute_for_rope(w: PackedLinearWeight, head_dim: int) -> PackedLinearWeight:
 
    if w.rope_perm_head_dim:
        if w.rope_perm_head_dim != head_dim:
            raise ValueError("weight already permuted with a different head_dim")
        return w
    o = rope_row_order(w.out_features, head_dim, w.packed.device)
    return replace(w, packed=w.packed[o].contiguous(), scales=w.scales[o].contiguous(),
                   zeros=w.zeros[o].contiguous(),
                   bias=None if w.bias is None else w.bias[o].contiguous(),
                   rope_perm_head_dim=head_dim)


def _unpermute_cols(y: torch.Tensor, w: PackedLinearWeight) -> torch.Tensor:
    """Kernel-order outputs of a permuted weight -> original column order."""
    if not w.rope_perm_head_dim:
        return y
    inv = rope_inverse_order(w.out_features, w.rope_perm_head_dim, y.device)
    return y.index_select(-1, inv)



@lru_cache(maxsize=None)
def tile_n() -> int:
    
    _check_ext()
    return int(_C.tile_n())


@lru_cache(maxsize=None)
def _sm_count_for(device_index: int) -> int:
    if backend_name() == "emulator":
        return int(_C.sm_count())
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def virtual_rows(kind: str, *out_features: int) -> int:
    
    if kind == "gemv":
        return out_features[0]
    if kind == "qkv":
        return sum(out_features)
    if kind == "gate_up":
        return 2 * out_features[0]
    raise ValueError(kind)


def candidate_split_k(batch: int, n: int, bn: int, sms: int, rho: int, max_split: Optional[int] = None) -> int:
    """P_K = ceil(rho*S / (B*ceil(N/B_N))) if C_out < C_target, else 1."""
    rho = max(1, int(rho))
    c_out = batch * ceil(n / bn)
    c_target = rho * sms
    if c_out >= c_target:
        return 1
    pk = max(1, ceil(c_target / c_out))
    return pk if max_split is None else min(max_split, pk)


def _resolve(split_k: Optional[int], rho: int, batch: int, v_rows: int, k: int, device) -> int:
    """split_k=None means: apply the underfill rule at call time for this batch size."""
    nchunks = ceil(k / 32)
    if split_k is not None:
        return max(1, min(int(split_k), nchunks))
    idx = device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else 0)
    return max(1, min(candidate_split_k(batch, v_rows, tile_n(), _sm_count_for(idx), rho), nchunks))



def packed_linear(x: torch.Tensor, w: PackedLinearWeight, split_k: Optional[int] = None, rho: int = 1,
                  act: Optional[str] = None, decode_threshold: int = DECODE_THRESHOLD,
                  workspace: Optional[torch.Tensor] = None, use_tensor_cores: bool = True):
   
    _check_ext()
    if w.rope_perm_head_dim:
        raise ValueError("this weight is stored RoPE-permuted; use fused_qkv (or _unpermute_cols)")
    return _packed_linear_raw(x, w, split_k, rho, act, decode_threshold, workspace, use_tensor_cores)


def _packed_linear_raw(x, w, split_k=None, rho=1, act=None, decode_threshold=DECODE_THRESHOLD,
                       workspace=None, use_tensor_cores=True):
    original = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    bias = _bias_tensor(w, x2)
    a = ACT[act]
    if x2.shape[0] <= decode_threshold:
        pk = _resolve(split_k, rho, x2.shape[0], w.out_features, w.in_features, x2.device)
        y = _C.packed_gemv(x2, w.packed, w.scales, w.zeros, bias, w.bits, w.in_features, w.group_size,
                           pk, rho, a, _ws(workspace, x2))
    else:
        y = _C.packed_gemm(x2, w.packed, w.scales, w.zeros, bias, w.bits, w.in_features, w.group_size,
                           a, use_tensor_cores)
    return y.reshape(*original, w.out_features)


def fused_gate_up(x: torch.Tensor, gate: PackedLinearWeight, up: PackedLinearWeight,
                  split_k: Optional[int] = None, rho: int = 1, decode_threshold: int = DECODE_THRESHOLD,
                  workspace: Optional[torch.Tensor] = None):
    
    _check_ext()
    if gate.bits != up.bits:
        raise ValueError("fused Gate/Up requires the same precision for Gate and Up within a block")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    if x2.shape[0] <= decode_threshold:
        pk = _resolve(split_k, rho, x2.shape[0], virtual_rows("gate_up", gate.out_features),
                      gate.in_features, x2.device)
        y = _C.fused_gate_up(x2, gate.packed, gate.scales, gate.zeros, up.packed, up.scales, up.zeros,
                             gate.bits, gate.in_features, gate.group_size, pk, rho, _ws(workspace, x2))
    else:
        g = _C.packed_gemm(x2, gate.packed, gate.scales, gate.zeros, _empty(x2), gate.bits, gate.in_features,
                           gate.group_size, ACT["silu"], True)
        u = _C.packed_gemm(x2, up.packed, up.scales, up.zeros, _empty(x2), up.bits, up.in_features,
                           up.group_size, 0, True)
        y = g * u
    return y.reshape(*x.shape[:-1], gate.out_features)


def apply_rope_reference(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int) -> torch.Tensor:
    """HF rotate_half RoPE on a flat [M, H*head_dim] tensor with cos/sin [M, head_dim]."""
    M, W = t.shape
    th = t.float().view(M, W // head_dim, head_dim)
    c, s = cos.float()[:, None, :], sin.float()[:, None, :]
    h = head_dim // 2
    rot = torch.cat([-th[..., h:], th[..., :h]], dim=-1)
    return (th * c + rot * s).view(M, W).to(t.dtype)


def fused_qkv(x: torch.Tensor, q: PackedLinearWeight, k: PackedLinearWeight, v: PackedLinearWeight,
              split_k: Optional[int] = None, rho: int = 1,
              rope: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
              decode_threshold: int = DECODE_THRESHOLD, workspace: Optional[torch.Tensor] = None):
    
    _check_ext()
    if not (q.bits == k.bits == v.bits):
        raise ValueError("Q/K/V must use the block precision")
    if q.rope_perm_head_dim != k.rope_perm_head_dim or v.rope_perm_head_dim:
        raise ValueError("Q and K must be permuted with the same head_dim; V must not be permuted")
    perm = q.rope_perm_head_dim
    pre = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M = x2.shape[0]
    if rope is not None:
        cos, sin, head_dim = rope
        head_dim = int(head_dim)
        if perm != head_dim:
            raise ValueError("fused RoPE needs Q/K permuted with permute_for_rope(w, head_dim) "
                             "(install_bitmap(fuse_rope=True) does this)")
        cos2 = cos.expand(*pre, head_dim).reshape(M, head_dim)
        sin2 = sin.expand(*pre, head_dim).reshape(M, head_dim)
    if M <= decode_threshold:
        v_rows = virtual_rows("qkv", q.out_features, k.out_features, v.out_features)
        pk = _resolve(split_k, rho, M, v_rows, q.in_features, x2.device)
        if rope is None:
            rc, rs, hd = _empty(x2), _empty(x2), 0
        else:
            rc, rs, hd = cos2.float().contiguous(), sin2.float().contiguous(), head_dim
        oq, ok, ov = _C.fused_qkv(x2, q.packed, q.scales, q.zeros, _bias_tensor(q, x2),
                                  k.packed, k.scales, k.zeros, _bias_tensor(k, x2),
                                  v.packed, v.scales, v.zeros, _bias_tensor(v, x2),
                                  rc, rs, hd, perm, q.bits, q.in_features, q.group_size, pk, rho,
                                  _ws(workspace, x2))
    else:
        oq, ok, ov = (_unpermute_cols(_packed_linear_raw(x2, w, decode_threshold=0), w) for w in (q, k, v))
        if rope is not None:
            oq = apply_rope_reference(oq, cos2, sin2, head_dim)
            ok = apply_rope_reference(ok, cos2, sin2, head_dim)
    return oq.reshape(*pre, q.out_features), ok.reshape(*pre, k.out_features), ov.reshape(*pre, v.out_features)


def sm_count():
    _check_ext(); return int(_C.sm_count())


def max_rho(bits: int, kind: str = "gemv"):
    _check_ext(); return int(_C.max_rho(bits, kind))



class PackedLinear(nn.Module):
    

    def __init__(self, w: PackedLinearWeight, split_k: Optional[int] = None, rho: int = 1, act: Optional[str] = None):
        super().__init__()
        self.register_buffer("packed", w.packed)
        self.register_buffer("scales", w.scales)
        self.register_buffer("zeros", w.zeros)
        self.register_buffer("bias", w.bias)
        self.register_buffer("workspace", new_workspace(w.packed.device), persistent=False)
        self.bits = int(w.bits)
        self.group_size = int(w.group_size)
        self.in_features = int(w.in_features)
        self.out_features = int(w.out_features)
        self.rope_perm_head_dim = int(w.rope_perm_head_dim)
        self.split_k = None if split_k is None else int(split_k)
        self.rho = int(rho)
        self.act = act

    def weight_spec(self) -> PackedLinearWeight:
        return PackedLinearWeight(self.packed, self.scales, self.zeros, self.bits, self.group_size,
                                  self.in_features, self.out_features, self.bias, self.rope_perm_head_dim)

    def permute_for_rope_(self, head_dim: int):
        """In place: store rows pair-interleaved for the fused RoPE epilogue."""
        w = permute_for_rope(self.weight_spec(), head_dim)
        self.packed, self.scales, self.zeros, self.bias = w.packed, w.scales, w.zeros, w.bias
        self.rope_perm_head_dim = head_dim
        return self

    def forward(self, x):
        return packed_linear(x, self.weight_spec(), self.split_k, self.rho, self.act, workspace=self.workspace)

    def extra_repr(self):
        r = f", rope_perm={self.rope_perm_head_dim}" if self.rope_perm_head_dim else ""
        return (f"in={self.in_features}, out={self.out_features}, W{self.bits}, g={self.group_size}, "
                f"split_k={self.split_k}, rho={self.rho}, act={self.act}{r}")
