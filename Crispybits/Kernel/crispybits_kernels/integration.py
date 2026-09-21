
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .io import from_packed_dict
from .ops import PackedLinear, fused_gate_up, fused_qkv, new_workspace

_PREROTATED = "_crispybits_prerotated"


def _blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    raise ValueError("Supported integration targets are Hugging Face LLaMA-family and OPT causal LMs")


def _get(root: nn.Module, name: str):
    for p in name.split('.'):
        root = getattr(root, p, None)
        if root is None:
            return None
    return root


def _set_by_dotted_name(root: nn.Module, name: str, module: nn.Module):
    parts = name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


class FusedQKV(nn.Module):
   

    def __init__(self, q: PackedLinear, k: PackedLinear, v: PackedLinear, split_k=None, rho: int = 1,
                 rope_head_dim: int = 0):
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.split_k, self.rho = split_k, int(rho)
        self.rope_head_dim = int(rope_head_dim)
        self.register_buffer("workspace", new_workspace(q.packed.device), persistent=False)
        self._cache = None        # (x, [q, k, v], remaining idx, rotated)

    def compute(self, x, rope=None):
        return fused_qkv(x, self.q.weight_spec(), self.k.weight_spec(), self.v.weight_spec(), self.split_k,
                         self.rho, rope=rope, workspace=self.workspace)

    def prime(self, x, cos, sin):
        """Compute rotated Q/K and V for `x` ahead of the proxies (fused RoPE path)."""
        outs = self.compute(x, rope=(cos, sin, self.rope_head_dim))
        self._cache = (x, outs, {0, 1, 2}, True)

    def get(self, x, idx: int):
        c = self._cache
        if c is None or c[0] is not x or idx not in c[2]:
            if c is not None and c[3]:
                self._cache = None
                raise RuntimeError("CrispyBits fused RoPE: q/k/v_proj received a different tensor than the one "
                                   "primed by the attention wrapper; install with fuse_rope=False for this model")
            c = (x, self.compute(x), {0, 1, 2}, False)
        out = c[1][idx]
        c[2].discard(idx)
        self._cache = None if not c[2] else c
        return out


class _QKVProxy(nn.Module):
    def __init__(self, owner: FusedQKV, idx: int):
        super().__init__()
        object.__setattr__(self, "_owner", owner)   # not registered: avoid duplicate params
        self.idx = idx
        src = (owner.q, owner.k, owner.v)[idx]
        self.in_features, self.out_features = src.in_features, src.out_features

    def forward(self, x):
        return self._owner.get(x, self.idx)

    def extra_repr(self):
        return f"fused {'qkv'[self.idx]}_proj (see crispybits_qkv)"


class FusedLlamaMLP(nn.Module):
    def __init__(self, gate: PackedLinear, up: PackedLinear, down: PackedLinear, split_k=None, rho: int = 1):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down
        self.split_k, self.rho = split_k, int(rho)
        self.register_buffer("workspace", new_workspace(gate.packed.device), persistent=False)

    def forward(self, x):
        h = fused_gate_up(x, self.gate_proj.weight_spec(), self.up_proj.weight_spec(), self.split_k, self.rho,
                          workspace=self.workspace)
        return self.down_proj(h)


def _is_act(mod, names):
    return mod is not None and type(mod).__name__ in names



def _patch_rotary(module_name: str) -> bool:
 
    mod = sys.modules.get(module_name)
    orig = getattr(mod, "apply_rotary_pos_emb", None)
    if orig is None:
        return False
    if getattr(orig, "_crispybits_patched", False):
        return True

    def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        if getattr(cos, _PREROTATED, False):
            return q, k
        return orig(q, k, cos, sin, *args, **kwargs)

    apply_rotary_pos_emb._crispybits_patched = True
    apply_rotary_pos_emb.__wrapped__ = orig
    mod.apply_rotary_pos_emb = apply_rotary_pos_emb
    return True


def _mark(t: torch.Tensor) -> torch.Tensor:
    a = t.view_as(t)           
    setattr(a, _PREROTATED, True)
    return a


def _wrap_attention_forward(attn: nn.Module, owner: FusedQKV):
    orig_forward = attn.forward

    def forward(*args, **kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        pe = kwargs.get("position_embeddings")
        pos_in_args = pe is None and len(args) > 1 and isinstance(args[1], (tuple, list))
        if pos_in_args:
            pe = args[1]
        if hs is None or pe is None:
            return orig_forward(*args, **kwargs)       
        cos, sin = pe
        owner.prime(hs, cos, sin)
        marked = (_mark(cos), _mark(sin))
        if pos_in_args:
            args = (args[0], marked, *args[2:])
        else:
            kwargs["position_embeddings"] = marked
        try:
            return orig_forward(*args, **kwargs)
        finally:
            owner._cache = None

    attn.forward = forward
    attn._crispybits_orig_forward = orig_forward


def _head_dim(attn):
    hd = getattr(attn, "head_dim", None)
    if hd is None and hasattr(attn, "config"):
        c = attn.config
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return int(hd) if hd else 0


def _fuse_block(block: nn.Module, tuning: Dict, i: int, fuse_rope: bool):

    attn = getattr(block, "self_attn", None)
    if attn is not None and all(isinstance(getattr(attn, n, None), PackedLinear) for n in ("q_proj", "k_proj", "v_proj")):
        cfg = tuning.get(f"block.{i}.self_attn.qkv", {})
        rope_hd = 0
        is_llama_like = hasattr(block, "mlp") and not hasattr(block, "fc1")
        if fuse_rope and is_llama_like:
            hd = _head_dim(attn)
            if (hd and hd % 2 == 0 and attn.q_proj.out_features % hd == 0 and attn.k_proj.out_features % hd == 0
                    and _patch_rotary(type(attn).__module__)):
                attn.q_proj.permute_for_rope_(hd)
                attn.k_proj.permute_for_rope_(hd)
                rope_hd = hd
        owner = FusedQKV(attn.q_proj, attn.k_proj, attn.v_proj, cfg.get("split_k"), cfg.get("rho", 1), rope_hd)
        attn.crispybits_qkv = owner
        attn.q_proj, attn.k_proj, attn.v_proj = (_QKVProxy(owner, j) for j in range(3))
        if rope_hd:
            _wrap_attention_forward(attn, owner)
 
    mlp = getattr(block, "mlp", None)
    if (mlp is not None and all(isinstance(getattr(mlp, n, None), PackedLinear) for n in ("gate_proj", "up_proj", "down_proj"))
            and _is_act(getattr(mlp, "act_fn", None), ("SiLU", "SiLUActivation"))):
        cfg = tuning.get(f"block.{i}.mlp.gate_up", {})
        block.mlp = FusedLlamaMLP(mlp.gate_proj, mlp.up_proj, mlp.down_proj, cfg.get("split_k"), cfg.get("rho", 1))
  
    if isinstance(getattr(block, "fc1", None), PackedLinear) and _is_act(getattr(block, "activation_fn", None), ("ReLU",)):
        block.fc1.act = "relu"
        block.activation_fn = nn.Identity()


def install_bitmap(model: nn.Module, packed_dir: str, bitmap, device="cuda", dtype=torch.float16,
                   tuning: Optional[Dict] = None, fuse: bool = True, fuse_rope: bool = True):
   
    blocks = _blocks(model)
    if len(blocks) != len(bitmap):
        raise ValueError("bitmap length does not match model")
    tuning = tuning or {}
    for i, (block, bits) in enumerate(zip(blocks, bitmap)):
        data = torch.load(Path(packed_dir) / f"block_{i:03d}_w{int(bits)}.pt", map_location="cpu", weights_only=False)
        for name, d in data.items():
            if not isinstance(_get(block, name), nn.Linear):
                raise ValueError(f"block {i}: '{name}' is not an nn.Linear in this model")
            w = from_packed_dict(d, device=device, dtype=dtype)
            cfg = tuning.get(f"block.{i}.{name}", {})
            _set_by_dotted_name(block, name, PackedLinear(w, split_k=cfg.get("split_k"), rho=int(cfg.get("rho", 1))))
        if fuse:
            _fuse_block(block, tuning, i, fuse_rope)
    return model


def load_tuning(path: Optional[str]):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def save_tuning(tuning: Dict, path: str):
    with open(path, "w") as f:
        json.dump(tuning, f, indent=2, sort_keys=True)
