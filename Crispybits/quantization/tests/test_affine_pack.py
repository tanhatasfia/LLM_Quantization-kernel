import torch
from crispybits_quant.affine import affine_quantize_weight, dequantize_weight
from crispybits_quant.packing import pack_codes, unpack_codes


def test_pack_roundtrip():
    torch.manual_seed(0)
    for bits in (2,3,4):
        q = torch.randint(0, 1 << bits, (7, 259), dtype=torch.uint8)
        p = pack_codes(q, bits)
        q2 = unpack_codes(p, q.shape[1], bits)
        assert torch.equal(q, q2)


def test_quant_shapes():
    w = torch.randn(5, 257, dtype=torch.float32)
    qt = affine_quantize_weight(w, 3, 128)
    wh = dequantize_weight(qt)
    assert wh.shape == w.shape
    assert qt.scales.shape == (5, 3)
