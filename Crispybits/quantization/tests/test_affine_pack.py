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


def test_groups_not_spanning_zero():
    # Regression: all-positive / all-negative / constant groups must not saturate.
    torch.manual_seed(0)
    for bits in (2, 3, 4):
        qmax = (1 << bits) - 1
        for w in (torch.rand(3, 128) + 1.0,          # all positive
                  -(torch.rand(3, 128) + 1.0),       # all negative
                  torch.full((3, 128), 0.37),        # constant positive
                  torch.full((3, 128), -0.37),       # constant negative
                  torch.zeros(3, 128)):              # all zero
            qt = affine_quantize_weight(w, bits, 128)
            wh = dequantize_weight(qt)
            # range now includes 0, so error <= half a step of (max(w,0)-min(w,0))
            step = (w.clamp(min=0).amax(1) - w.clamp(max=0).amin(1)) / qmax
            assert torch.all((wh - w).abs().amax(1) <= step / 2 + 1e-5)
            assert torch.all((qt.zeros >= 0) & (qt.zeros <= qmax))


def test_zero_spanning_groups_unchanged():
    # For groups that already span zero, results equal plain min-max affine.
    torch.manual_seed(1)
    w = torch.randn(4, 256) * 0.02
    w[:, 0], w[:, 128] = -0.05, 0.05   # guarantee each group spans zero
    w[:, 1], w[:, 129] = 0.05, -0.05
    for bits in (2, 3, 4):
        qmax = (1 << bits) - 1
        qt = affine_quantize_weight(w, bits, 128)
        for g in range(2):
            wg = w[:, g*128:(g+1)*128]
            s = (wg.amax(1) - wg.amin(1)) / qmax
            z = torch.round(-wg.amin(1) / s).clamp(0, qmax)
            q = torch.round(wg / s[:, None] + z[:, None]).clamp(0, qmax)
            assert torch.equal(qt.q[:, g*128:(g+1)*128].long(), q.long())
