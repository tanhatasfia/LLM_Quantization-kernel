from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class QuantizedTensor:
    q: torch.Tensor          
    scales: torch.Tensor    
    zeros: torch.Tensor     
    bits: int
    group_size: int
    original_shape: Tuple[int, int]


def _validate(weight: torch.Tensor, bits: int, group_size: int) -> None:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D weight [out,in], got {tuple(weight.shape)}")
    if bits not in (2, 3, 4):
        raise ValueError("CrispyBits uses candidate precisions {2,3,4} bits")
    if group_size <= 0:
        raise ValueError("group_size must be positive")


def affine_quantize_weight(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
    eps: float = 1e-8,
) -> QuantizedTensor:
   
    _validate(weight, bits, group_size)
    device = weight.device
    dtype = weight.dtype
    out_features, in_features = weight.shape
    n_groups = (in_features + group_size - 1) // group_size
    qmax = (1 << bits) - 1

    q = torch.empty((out_features, in_features), dtype=torch.uint8, device=device)
    scales = torch.empty((out_features, n_groups), dtype=dtype, device=device)
    zeros = torch.empty((out_features, n_groups), dtype=dtype, device=device)

    for g in range(n_groups):
        s = g * group_size
        e = min(in_features, s + group_size)
        w = weight[:, s:e].float()
        w_min = w.amin(dim=1)
        w_max = w.amax(dim=1)
        scale = (w_max - w_min) / float(qmax)
        deg = scale.abs() < eps
      
        const_abs = w_max.abs()
        const_scale = torch.where(const_abs < eps, torch.ones_like(const_abs), const_abs / float(qmax))
        scale = torch.where(deg, const_scale, scale)
        zero = torch.round(-w_min / scale).clamp_(0, qmax)
        zero = torch.where(deg & (w_max >= 0), torch.zeros_like(zero), zero)
        zero = torch.where(deg & (w_max < 0), torch.full_like(zero, float(qmax)), zero)
        qg = torch.round(w / scale[:, None] + zero[:, None]).clamp_(0, qmax)

        q[:, s:e] = qg.to(torch.uint8)
        scales[:, g] = scale.to(dtype)
        zeros[:, g] = zero.to(dtype)

    return QuantizedTensor(
        q=q,
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group_size,
        original_shape=(out_features, in_features),
    )


def dequantize_weight(qt: QuantizedTensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    q = qt.q
    out_features, in_features = qt.original_shape
    dtype = dtype or qt.scales.dtype
    w = torch.empty((out_features, in_features), device=q.device, dtype=dtype)
    n_groups = qt.scales.shape[1]
    for g in range(n_groups):
        s = g * qt.group_size
        e = min(in_features, s + qt.group_size)
        qg = q[:, s:e].to(torch.float32)
        scale = qt.scales[:, g].float()[:, None]
        zero = qt.zeros[:, g].float()[:, None]
        w[:, s:e] = ((qg - zero) * scale).to(dtype)
    return w


def fake_quantize_weight(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    return dequantize_weight(affine_quantize_weight(weight, bits, group_size), dtype=weight.dtype)
