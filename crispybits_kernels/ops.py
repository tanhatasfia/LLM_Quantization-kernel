from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    from . import _C
except Exception as e:  # pragma: no cover
    _C = None
    _IMPORT_ERROR = e


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

    def cuda(self, device=None, dtype=torch.float16):
        device = device or "cuda"
        return PackedLinearWeight(
            self.packed.to(device=device, dtype=torch.int32, non_blocking=True),
            self.scales.to(device=device, dtype=dtype, non_blocking=True),
            self.zeros.to(device=device, dtype=dtype, non_blocking=True),
            self.bits,self.group_size,self.in_features,self.out_features,
            None if self.bias is None else self.bias.to(device=device,dtype=dtype,non_blocking=True),
        )


def _check_ext():
    if _C is None:
        raise RuntimeError(f"crispybits CUDA extension is not built: {_IMPORT_ERROR}")


def _bias_tensor(w: PackedLinearWeight, x: torch.Tensor):
    return w.bias if w.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)


def packed_linear(x: torch.Tensor, w: PackedLinearWeight, split_k: int = 1, rho: int = 1, decode_threshold: int = 4):
    """Direct packed low-bit linear. No full FP16 weight matrix is materialized."""
    _check_ext()
    original = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    bias = _bias_tensor(w, x2)
    if x2.shape[0] <= decode_threshold:
        y = _C.packed_gemv(x2,w.packed,w.scales,w.zeros,bias,w.bits,w.in_features,w.group_size,split_k,rho)
    else:
        y = _C.packed_gemm(x2,w.packed,w.scales,w.zeros,bias,w.bits,w.in_features,w.group_size)
    return y.reshape(*original, w.out_features)


class PackedLinear(nn.Module):
    def __init__(self, w: PackedLinearWeight, split_k: int = 1, rho: int = 1):
        super().__init__()
        self.register_buffer("packed", w.packed)
        self.register_buffer("scales", w.scales)
        self.register_buffer("zeros", w.zeros)
        if w.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", w.bias)
        self.bits = int(w.bits)
        self.group_size = int(w.group_size)
        self.in_features = int(w.in_features)
        self.out_features = int(w.out_features)
        self.split_k = int(split_k)
        self.rho = int(rho)

    def _weight(self):
        return PackedLinearWeight(self.packed,self.scales,self.zeros,self.bits,self.group_size,self.in_features,self.out_features,self.bias)

    def forward(self, x):
        return packed_linear(x,self._weight(),self.split_k,self.rho)


def fused_gate_up(x: torch.Tensor, gate: PackedLinearWeight, up: PackedLinearWeight, split_k: int=1, rho: int=1):
    _check_ext()
    x2=x.reshape(-1,x.shape[-1]).contiguous()
    if gate.bits != up.bits:
        raise ValueError("reference fused kernel requires the same precision for Gate and Up within a block")
    y=_C.fused_gate_up(x2,gate.packed,gate.scales,gate.zeros,up.packed,up.scales,up.zeros,gate.bits,gate.in_features,gate.group_size,split_k,rho)
    return y.reshape(*x.shape[:-1],gate.out_features)


def fused_qkv(x: torch.Tensor,q: PackedLinearWeight,k: PackedLinearWeight,v: PackedLinearWeight,split_k:int=1,rho:int=1):
    _check_ext();x2=x.reshape(-1,x.shape[-1]).contiguous()
    if not (q.bits==k.bits==v.bits): raise ValueError("Q/K/V must use the block precision")
    empty=lambda w: _bias_tensor(w,x2)
    oq,ok,ov=_C.fused_qkv(x2,q.packed,q.scales,q.zeros,empty(q),k.packed,k.scales,k.zeros,empty(k),v.packed,v.scales,v.zeros,empty(v),q.bits,q.in_features,q.group_size,split_k,rho)
    pre=x.shape[:-1]
    return oq.reshape(*pre,q.out_features),ok.reshape(*pre,k.out_features),ov.reshape(*pre,v.out_features)


def sm_count():
    _check_ext();return int(_C.sm_count())

def max_rho(bits:int):
    _check_ext();return int(_C.max_rho(bits))
