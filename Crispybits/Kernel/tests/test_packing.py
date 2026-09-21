
import pytest
import torch

from crispybits_kernels.reference import pack_codes, unpack_codes, quantize_affine, dequantize


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("K", [32, 128, 200, 4096, 5120])
def test_pack_roundtrip(bits, K):
    g = torch.Generator().manual_seed(bits * 1000 + K)
    q = torch.randint(0, 1 << bits, (7, K), generator=g, dtype=torch.int32)
    p = pack_codes(q, bits)
    assert p.dtype == torch.int32
    assert p.shape[1] * 32 >= K * bits
    assert torch.equal(unpack_codes(p, bits, K), q)


@pytest.mark.parametrize("bits,per_word", [(2, 16), (4, 8)])
def test_codes_per_word(bits, per_word):
    q = torch.arange(per_word, dtype=torch.int32).remainder(1 << bits)[None]
    q = torch.cat([q, torch.zeros(1, 32 - per_word, dtype=torch.int32)], 1)
    p = pack_codes(q, bits)
    w0 = int(p[0, 0]) & 0xFFFFFFFF
    assert [(w0 >> (bits * i)) & ((1 << bits) - 1) for i in range(per_word)] == q[0, :per_word].tolist()


def test_w3_three_words_per_32_codes():
    q = torch.full((1, 32), 7, dtype=torch.int32)
    p = pack_codes(q, 3)
    assert p.shape == (1, 3)
    assert all((int(v) & 0xFFFFFFFF) == 0xFFFFFFFF for v in p[0])
    # code 10 straddles words 0 and 1 (bits 30..32)
    q = torch.zeros(1, 32, dtype=torch.int32); q[0, 10] = 0b101
    p = pack_codes(q, 3)
    assert (int(p[0, 0]) & 0xFFFFFFFF) >> 30 == 0b01 and (int(p[0, 1]) & 1) == 0b1


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_quantize_dequantize_error_bounded(bits):
    torch.manual_seed(0)
    w = torch.randn(16, 384)
    q, s, z = quantize_affine(w, bits, 128)
    wd = dequantize(q, s, z, 128)
    step = s.repeat_interleave(128, 1)
    assert torch.all((wd - w).abs() <= step * 0.5 + 1e-5 + step * 1.0)  


def test_external_packer_matches_if_available():
  
    packing = pytest.importorskip("crispybits_quant.packing")
    affine = pytest.importorskip("crispybits_quant.affine")
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    for bits in (2, 3, 4):
        qt = affine.affine_quantize_weight(w, bits, 128)
        pw = packing.pack_quantized(qt)
        q = unpack_codes(pw.packed.cpu(), bits, 512)
        wd = dequantize(q, pw.scales.cpu(), pw.zeros.cpu(), 128)
        assert (wd - w).abs().max() < 2 * pw.scales.float().max(), f"layout mismatch for W{bits}"
