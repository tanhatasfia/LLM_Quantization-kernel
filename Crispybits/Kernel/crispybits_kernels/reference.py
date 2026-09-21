
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


def quantize_affine(w: torch.Tensor, bits: int, group_size: int = 128):
    
    assert bits in (2, 3, 4)
    N, K = w.shape
    G = math.ceil(K / group_size)
    Kp = G * group_size
    wf = torch.zeros(N, Kp, dtype=torch.float32, device=w.device)
    wf[:, :K] = w.float()
    if Kp != K:  
        wf[:, K:] = wf[:, K - 1:K]
    g = wf.view(N, G, group_size)
    qmax = (1 << bits) - 1
    mn, mx = g.amin(-1), g.amax(-1)
    scale = ((mx - mn) / qmax).clamp_min(1e-8)
    zero = torch.round(-mn / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(g / scale[..., None]) + zero[..., None], 0, qmax)
    q = q.view(N, Kp)[:, :K].to(torch.int32)
    return q, scale, zero


def dequantize(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    K = q.shape[1]
    s = scales.float().repeat_interleave(group_size, dim=1)[:, :K]
    z = zeros.float().repeat_interleave(group_size, dim=1)[:, :K]
    return (q.float() - z) * s


def pack_codes(q: torch.Tensor, bits: int) -> torch.Tensor:

    assert bits in (2, 3, 4)
    N, K = q.shape
    Kp = math.ceil(K / 32) * 32
    qq = torch.zeros(N, Kp, dtype=torch.int64, device=q.device)
    qq[:, :K] = q.to(torch.int64) & ((1 << bits) - 1)
    qq = qq.view(N, Kp // 32, 32)                           
    words = torch.zeros(N, Kp // 32, bits + 1, dtype=torch.int64, device=q.device)
    for j in range(32):
        pos = j * bits
        wi, off = pos // 32, pos % 32
        v = qq[:, :, j] << off
        words[:, :, wi] |= v & 0xFFFFFFFF
        words[:, :, wi + 1] |= v >> 32                     
    assert torch.all(words[:, :, bits] == 0)
    w = words[:, :, :bits].reshape(N, Kp // 32 * bits)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)          
    return w.to(torch.int32).contiguous()


def unpack_codes(packed: torch.Tensor, bits: int, K: int) -> torch.Tensor:
   
    w = packed.to(torch.int64) & 0xFFFFFFFF
    k = torch.arange(K, device=packed.device)
    pos = k * bits
    wi, off = pos // 32, pos % 32
    lo = w[:, wi] >> off
    nxt = torch.clamp(wi + 1, max=w.shape[1] - 1)
    hi = torch.where(off + bits > 32, w[:, nxt] << (32 - off), torch.zeros_like(lo))
    return ((lo | hi) & ((1 << bits) - 1)).to(torch.int32)


def pack_linear_dict(weight: torch.Tensor, bits: int, group_size: int = 128,
                     bias: Optional[torch.Tensor] = None, dtype=torch.float16) -> Dict:
   
    q, s, z = quantize_affine(weight, bits, group_size)
    return dict(packed=pack_codes(q, bits).cpu(), scales=s.to(dtype).cpu(), zeros=z.to(dtype).cpu(),
                bits=bits, group_size=group_size, in_features=weight.shape[1], out_features=weight.shape[0],
                bias=None if bias is None else bias.detach().to(dtype).cpu())


def dequantized_weight(d: Dict) -> torch.Tensor:

    q = unpack_codes(d["packed"], d["bits"], d["in_features"])
    return dequantize(q, d["scales"], d["zeros"], d["group_size"])


def write_packed_dir(model: torch.nn.Module, out_dir: str, bits_options: Iterable[int] = (2, 3, 4),
                     group_size: int = 128, dtype=torch.float16) -> None:

    from .integration import _blocks
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for i, block in enumerate(_blocks(model)):
        linears = {n: m for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)}
        for b in bits_options:
            data = {n: pack_linear_dict(m.weight.detach(), b, group_size, m.bias, dtype) for n, m in linears.items()}
            torch.save(data, out / f"block_{i:03d}_w{b}.pt")
