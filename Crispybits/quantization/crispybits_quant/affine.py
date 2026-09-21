from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class QuantizedTensor:
    q: torch.Tensor          # uint8/int32 integer codes, same logical shape as weight
    scales: torch.Tensor     # [out_features, n_groups]
    zeros: torch.Tensor      # [out_features, n_groups]
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
    """Group-wise asymmetric affine weight-only quantization.

    For each output row and each contiguous K-dimension group:
        min   = min(min(w), 0),  max = max(max(w), 0)
        scale = (max - min) / (2^b - 1)
        zero  = round(-min / scale)
        q     = clamp(round(w / scale) + zero, 0, 2^b-1)
        w_hat = (q - zero) * scale

    The paper states group size 128 and W2/W3/W4 candidates. Scales and
    zero-points are stored in floating point by the packed runtime.
    """
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
        # Extend the range to include 0 (standard asymmetric min-max, as in
        # GPTQ/AWQ RTN). Without this, a group that is entirely positive or
        # entirely negative gets its zero-point clamped to 0 or qmax and most
        # codes saturate. For groups that already span zero this is a no-op.
        w_min = w.amin(dim=1).clamp(max=0.0)
        w_max = w.amax(dim=1).clamp(min=0.0)
        scale = (w_max - w_min) / float(qmax)
        # Only an all-zero group has zero range now; any scale works there.
        scale = torch.where(scale < eps, torch.ones_like(scale), scale)
        # Because w_min <= 0 <= w_max, this already lies in [0, qmax];
        # the clamp only guards against floating-point round-off.
        zero = torch.round(-w_min / scale).clamp_(0, qmax)
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
