
import struct
import pytest
import torch

from crispybits_kernels.reference import pack_codes
from crispybits_kernels.ops import rope_row_order

M32 = 0xFFFFFFFF


def cb_code(w, j, bits):                     
    pos = j * bits; wi = pos >> 5; off = pos & 31
    v = (w[wi] >> off) & M32
    if off + bits > 32:
        v |= (w[wi + 1] << (32 - off)) & M32
    return v & ((1 << bits) - 1)


def cb_code_to_float(q):                     
    f = struct.unpack("<f", struct.pack("<I", (q | 0x4B000000) & M32))[0]
    return f - 8388608.0


def cb_unperm(r, hd):                       
    if hd <= 0:
        return r
    head, p = divmod(r, hd)
    return head * hd + ((p >> 1) + hd // 2 if p & 1 else (p >> 1))


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_chunk_unpack_matches_layout(bits):
    g = torch.Generator().manual_seed(bits)
    q = torch.randint(0, 1 << bits, (64, 32 * 5), generator=g, dtype=torch.int32)
    p = pack_codes(q, bits)
    for r in range(q.shape[0]):
        row = [int(v) & M32 for v in p[r]]
        for c in range(5):                      
            w = row[c * bits:(c + 1) * bits]
            assert [cb_code(w, j, bits) for j in range(32)] == q[r, c * 32:(c + 1) * 32].tolist()


def test_magic_float_conversion_exact():
    assert all(cb_code_to_float(q) == float(q) for q in range(16))


@pytest.mark.parametrize("hd", [64, 128])
def test_unperm_inverts_host_permutation(hd):
    n = 4 * hd
    order = rope_row_order(n, hd)             
    assert [cb_unperm(p, hd) for p in range(n)] == order.tolist()

    for p in range(0, n, 2):
        a, b = cb_unperm(p, hd), cb_unperm(p + 1, hd)
        assert b - a == hd // 2 and a // hd == b // hd


def test_staging_layout_is_a_conflict_free_transpose():
    TC, THREADS = 64, 128
    XP = TC + 1
    seen = {}
    for e in range(TC * 32):                   
        c, j = e >> 5, e & 31
        seen[j * XP + c] = (c, j)
    assert len(seen) == TC * 32                 
    for base in range(0, TC * 32, 32):         
        assert len({((e & 31) * XP + (e >> 5)) % 32 for e in range(base, base + 32)}) == 32
    for cc in range(0, TC, 32):                
        for j in range(32):
            assert len({(j * XP + cc + l) % 32 for l in range(32)}) == 32


def test_pair_epilogue_rows_stay_in_one_warp():
    RPW, WARPS = 8, 4
    BN = RPW * WARPS
    for tile in range(3):
        for warp in range(WARPS):
            v0 = tile * BN + warp * RPW
            for lane in range(RPW):           
                assert (v0 + lane) ^ 1 == v0 + (lane ^ 1)


def test_split_boundaries_cover_all_chunks():
    for C in (1, 7, 128, 161):
        for P in range(1, min(C, 40) + 1):
            b = [(C * s) // P for s in range(P + 1)]
            assert b[0] == 0 and b[-1] == C and all(b[i] < b[i + 1] for i in range(P))
