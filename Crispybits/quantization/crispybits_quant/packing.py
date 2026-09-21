from __future__ import annotations

from dataclasses import dataclass
import torch

from .affine import QuantizedTensor


@dataclass
class PackedWeight:
    packed: torch.Tensor     # uint32 [out, words_per_row]
    scales: torch.Tensor     # fp16/bf16/fp32 [out, groups]
    zeros: torch.Tensor      # fp16/bf16/fp32 [out, groups]
    bits: int
    group_size: int
    out_features: int
    in_features: int

    @property
    def words_per_row(self) -> int:
        return self.packed.shape[1]


def words_for_k(in_features: int, bits: int) -> int:
    return (in_features * bits + 31) // 32


def pack_codes(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack row-major integer codes into a continuous 32-bit bitstream.

    W2: 16 values / 32-bit word.
    W4:  8 values / 32-bit word.
    W3: 32 values / 3 words, with cross-word values handled explicitly.
    """
    if q.ndim != 2:
        raise ValueError("q must have shape [out_features, in_features]")
    if bits not in (2, 3, 4):
        raise ValueError("bits must be 2, 3, or 4")
    q_cpu = q.detach().to("cpu", torch.int64)
    out_features, in_features = q_cpu.shape
    words = words_for_k(in_features, bits)
    packed = torch.zeros((out_features, words), dtype=torch.int64)
    mask = (1 << bits) - 1

    # CPU reference packer. Packing is offline, so clarity is preferred.
    for r in range(out_features):
        for k in range(in_features):
            v = int(q_cpu[r, k].item()) & mask
            bitpos = k * bits
            wi = bitpos >> 5
            off = bitpos & 31
            packed[r, wi] |= v << off
            spill = off + bits - 32
            if spill > 0:
                packed[r, wi + 1] |= v >> (bits - spill)

    # torch has no uint32 arithmetic on all devices; int64 storage here is
    # converted to int32 with identical 32-bit bit-patterns for CUDA.
    packed = (packed & 0xFFFFFFFF).to(torch.int64)
    signed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed).to(torch.int32)
    return signed


def unpack_codes(packed: torch.Tensor, in_features: int, bits: int) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError("packed must have shape [out_features, words]")
    p = packed.detach().to("cpu", torch.int64) & 0xFFFFFFFF
    out_features = p.shape[0]
    q = torch.empty((out_features, in_features), dtype=torch.uint8)
    mask = (1 << bits) - 1
    for r in range(out_features):
        for k in range(in_features):
            bitpos = k * bits
            wi = bitpos >> 5
            off = bitpos & 31
            if off + bits <= 32:
                v = (int(p[r, wi].item()) >> off) & mask
            else:
                lo_bits = 32 - off
                hi_bits = bits - lo_bits
                lo = (int(p[r, wi].item()) >> off) & ((1 << lo_bits) - 1)
                hi = int(p[r, wi + 1].item()) & ((1 << hi_bits) - 1)
                v = lo | (hi << lo_bits)
            q[r, k] = v
    return q


def pack_quantized(qt: QuantizedTensor, metadata_dtype: torch.dtype = torch.float16) -> PackedWeight:
    packed = pack_codes(qt.q, qt.bits).to(qt.q.device)
    return PackedWeight(
        packed=packed,
        scales=qt.scales.to(metadata_dtype),
        zeros=qt.zeros.to(metadata_dtype),
        bits=qt.bits,
        group_size=qt.group_size,
        out_features=qt.original_shape[0],
        in_features=qt.original_shape[1],
    )


def unpack_dequantize(pw: PackedWeight, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    q = unpack_codes(pw.packed, pw.in_features, pw.bits).to(pw.scales.device)
    out = torch.empty((pw.out_features, pw.in_features), device=q.device, dtype=dtype)
    ng = pw.scales.shape[1]
    for g in range(ng):
        s = g * pw.group_size
        e = min(pw.in_features, s + pw.group_size)
        out[:, s:e] = (
            (q[:, s:e].float() - pw.zeros[:, g].float()[:, None])
            * pw.scales[:, g].float()[:, None]
        ).to(dtype)
    return out
