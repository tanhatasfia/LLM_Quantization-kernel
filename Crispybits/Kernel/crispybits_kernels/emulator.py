
from __future__ import annotations
import math
import torch

from .reference import unpack_codes

IS_EMULATOR = True
TILE_N = 32                      
SM_COUNT = 108                  
MAX_RHO = 4
_ACT = {0: lambda t: t, 1: torch.relu, 2: torch.nn.functional.silu}


def sm_count():
    return SM_COUNT


def tile_n():
    return TILE_N


def max_rho(bits, kind="gemv"):
    if kind not in ("gemv", "qkv", "gate_up"):
        raise RuntimeError("kind must be 'gemv', 'qkv' or 'gate_up'")
    return MAX_RHO


def _check(x, packed, scales, zeros, bits, K, gs, name):
    assert x.dim() == 2 and x.shape[1] == K and x.is_contiguous(), f"{name}: bad x"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), f"{name}: x dtype"
    assert packed.dtype == torch.int32 and packed.is_contiguous(), f"{name}: packed must be contiguous int32"
    assert bits in (2, 3, 4), f"{name}: bits"
    assert gs > 0 and gs % 32 == 0, f"{name}: group_size must be a positive multiple of 32"
    assert packed.shape[1] == math.ceil(K / 32) * bits, f"{name}: packed row must be ceil(K/32)*bits words"
    assert scales.shape == zeros.shape and scales.shape[0] == packed.shape[0], f"{name}: metadata shape"
    assert scales.shape[1] == math.ceil(K / gs), f"{name}: groups"
    assert scales.dtype == x.dtype and zeros.dtype == x.dtype, f"{name}: metadata dtype must match x"


def _check_ws(ws):
    if ws is not None and ws.numel() >= 2:
        assert ws.dtype == torch.int32
      
        assert ws[:2].tolist() == [0, 0], "workspace not reset (was it shared across streams?)"


def _partials(x, rows, scales, zeros, bits, K, gs, split_k):
    
    B = x.shape[0]
    C = math.ceil(K / 32)
    split_k = max(1, min(int(split_k), C))
    q = unpack_codes(rows.cpu(), bits, C * 32).to(x.device).float().view(-1, C, 32)
    xp = torch.zeros(B, C * 32, dtype=torch.float32, device=x.device)
    xp[:, :K] = x.float()
    xc = xp.view(B, C, 32)
    dot = torch.einsum("bcj,vcj->bvc", xc, q)
    xsum = xc.sum(-1)                                     
    g = (torch.arange(C, device=x.device) * 32) // gs
    s = scales.float()[:, g]                           
    z = zeros.float()[:, g]
    contrib = s[None] * (dot - z[None] * xsum[:, None, :])  
    parts = []
    for sp in range(split_k):
        c0, c1 = (C * sp) // split_k, (C * (sp + 1)) // split_k
        parts.append(contrib[..., c0:c1].sum(-1))
    return torch.stack(parts)


def _bias(b, n, like):
    if b is None or b.numel() == 0:
        return torch.zeros(n, device=like.device)
    assert b.numel() == n, "bias length must equal out_features"
    return b.float()


def packed_gemv(x, packed, scales, zeros, bias, bits, K, gs, split_k, rho, act, ws):
    _check(x, packed, scales, zeros, bits, K, gs, "packed_gemv"); _check_ws(ws)
    acc = _partials(x, packed, scales, zeros, bits, K, gs, split_k).sum(0)
    return _ACT[int(act)](acc + _bias(bias, packed.shape[0], x)).to(x.dtype)


def fused_gate_up(x, wg, sg, zg, wu, su, zu, bits, K, gs, split_k, rho, ws):
    _check(x, wg, sg, zg, bits, K, gs, "gate"); _check(x, wu, su, zu, bits, K, gs, "up"); _check_ws(ws)
    assert wg.shape == wu.shape
    N = wg.shape[0]
    # virtual rows: gate n -> 2n, up n -> 2n+1
    rows = torch.stack([wg, wu], 1).reshape(2 * N, -1)
    s = torch.stack([sg, su], 1).reshape(2 * N, -1)
    z = torch.stack([zg, zu], 1).reshape(2 * N, -1)
    acc = _partials(x, rows, s, z, bits, K, gs, split_k).sum(0)
    mine, partner = acc[:, 0::2], acc[:, 1::2]           
    return (torch.nn.functional.silu(mine) * partner).to(x.dtype)


def _unperm_index(n, hd, device):
    r = torch.arange(n, device=device)
    if hd <= 0:
        return r
    head, p = r // hd, r % hd
    return head * hd + torch.where(p % 2 == 1, p // 2 + hd // 2, p // 2)


def fused_qkv(x, wq, sq, zq, bq, wk, sk, zk, bk, wv, sv, zv, bv, rope_cos, rope_sin, head_dim, perm_hd,
              bits, K, gs, split_k, rho, ws):
    for w, s, z, n in ((wq, sq, zq, "q"), (wk, sk, zk, "k"), (wv, sv, zv, "v")):
        _check(x, w, s, z, bits, K, gs, n)
    _check_ws(ws)
    rope = rope_cos is not None and rope_cos.numel() > 0
    if perm_hd > 0:
        assert perm_hd % 2 == 0 and wq.shape[0] % perm_hd == 0 and wk.shape[0] % perm_hd == 0
    if rope:
        assert perm_hd > 0 and perm_hd == head_dim, "fused RoPE needs pair-interleaved Q/K"
        assert rope_cos.numel() == x.shape[0] * head_dim
    rows = torch.cat([wq, wk, wv]); s = torch.cat([sq, sk, sv]); z = torch.cat([zq, zk, zv])
    acc = _partials(x, rows, s, z, bits, K, gs, split_k).sum(0)
    outs, off = [], 0
    for i, (w, b) in enumerate(((wq, bq), (wk, bk), (wv, bv))):
        n = w.shape[0]
        y = acc[:, off:off + n] + _bias(b, n, x)[None]
        off += n
        hd = perm_hd if i < 2 else 0
        orig = _unperm_index(n, hd, x.device)
        if rope and i < 2:
            partner = y[:, torch.arange(n, device=x.device) ^ 1]   
            d = orig % head_dim
            c, sn = rope_cos.float()[:, d], rope_sin.float()[:, d]
            y = torch.where(d[None] < head_dim // 2, y * c - partner * sn, y * c + partner * sn)
        out = torch.empty_like(y)
        out[:, orig] = y
        outs.append(out.to(x.dtype))
    return outs


def packed_gemm(x, packed, scales, zeros, bias, bits, K, gs, act, use_tensor_cores):
    _check(x, packed, scales, zeros, bits, K, gs, "packed_gemm")
    C = math.ceil(K / 32)
    q = unpack_codes(packed.cpu(), bits, K).to(x.device).float()
    g = torch.arange(K, device=x.device) // gs
    w = (q - zeros.float()[:, g]) * scales.float()[:, g]
    if use_tensor_cores and x.dtype in (torch.float16, torch.bfloat16):
        w = w.to(x.dtype).float()                       
    y = x.float() @ w.T + _bias(bias, packed.shape[0], x)[None]
    return _ACT[int(act)](y).to(x.dtype)
