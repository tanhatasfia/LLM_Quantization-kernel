
import pytest
import torch

from crispybits_kernels import (fused_gate_up, fused_qkv, max_rho, packed_linear, apply_rope_reference,
                                candidate_split_k, permute_for_rope, new_workspace, sm_count, tile_n)
from crispybits_kernels.io import from_packed_dict
from crispybits_kernels.reference import pack_linear_dict, dequantized_weight

TOL = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 1e-4}


def make(n, k, bits, dtype, device, bias=False, seed=0, gs=128):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n, k, generator=g) * 0.05
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    d = pack_linear_dict(W, bits, gs, b, dtype)
    Wd = dequantized_weight(d).to(device)
    bd = None if b is None else d["bias"].float().to(device)
    return from_packed_dict(d, device, dtype), Wd, bd


def ref_linear(x, Wd, bd, act=None):
    y = x.float() @ Wd.T
    if bd is not None:
        y = y + bd
    if act == "relu":
        y = torch.relu(y)
    elif act == "silu":
        y = torch.nn.functional.silu(y)
    return y


def close(y, ref, dtype):
    err = (y.float() - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    assert err / scale < TOL[dtype], f"rel err {err / scale:.2e}"


SHAPES = [(128, 256), (300, 200), (4096, 4096), (1024, 5120), (5120, 1024), (33, 4128)]


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("split_k", [1, 2, 7, None])
def test_gemv(env, bits, n, k, B, split_k):
    if env == "cpu" and n * k > 2_000_000 and split_k not in (1, None):
        pytest.skip("large shape: covered on GPU")
    w, Wd, _ = make(n, k, bits, torch.float16, env)
    x = torch.randn(B, k, device=env, dtype=torch.float16)
    for rho in sorted({0, 1, max(1, max_rho(bits))}):         
        close(packed_linear(x, w, split_k, rho), ref_linear(x, Wd, None), torch.float16)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("act", [None, "relu", "silu"])
@pytest.mark.parametrize("split_k", [1, 4])
def test_gemv_bias_act_dtypes(env, dtype, act, split_k):
    w, Wd, bd = make(1000, 768, 3, dtype, env, bias=True)
    x = torch.randn(2, 768, device=env, dtype=dtype)
    close(packed_linear(x, w, split_k, 2, act), ref_linear(x, Wd, bd, act), dtype)


@pytest.mark.parametrize("gs", [32, 64, 256])
def test_group_sizes(env, gs):
    w, Wd, _ = make(200, 640, 3, torch.float16, env, gs=gs)
    x = torch.randn(1, 640, device=env, dtype=torch.float16)
    close(packed_linear(x, w, 3, 1), ref_linear(x, Wd, None), torch.float16)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("M", [5, 33, 130])
@pytest.mark.parametrize("act", [None, "relu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_prefill_gemm(env, bits, M, act, dtype):
    w, Wd, bd = make(260, 400, bits, dtype, env, bias=True)
    x = torch.randn(M, 400, device=env, dtype=dtype)
    tol_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16   
    close(packed_linear(x, w, act=act), ref_linear(x, Wd, bd, act), tol_dtype)
    close(packed_linear(x, w, act=act, use_tensor_cores=False), ref_linear(x, Wd, bd, act), tol_dtype)


def test_workspace_reuse_self_resets(env):
  
    w, Wd, _ = make(5120, 1024, 3, torch.float16, env)
    x = torch.randn(1, 1024, device=env, dtype=torch.float16)
    ws = new_workspace(env)
    for pk, rho in [(1, 1), (4, 2), (1, 0), (7, 1), (1, 3)] * 3:
        y = packed_linear(x, w, pk, rho, workspace=ws)
        assert torch.isfinite(y).all()
        close(y, ref_linear(x, Wd, None), torch.float16)
    if env == "cuda":
        torch.cuda.synchronize()
    assert ws.tolist() == [0, 0]


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("split_k", [1, 3])
def test_fused_gate_up(env, bits, B, split_k):
    g, Gd, _ = make(1376, 512, bits, torch.float16, env, seed=1)
    u, Ud, _ = make(1376, 512, bits, torch.float16, env, seed=2)
    x = torch.randn(B, 512, device=env, dtype=torch.float16)
    ref = torch.nn.functional.silu(x.float() @ Gd.T) * (x.float() @ Ud.T)
    close(fused_gate_up(x, g, u, split_k, 2), ref, torch.float16)


def test_fused_gate_up_prefill_fallback(env):
    g, Gd, _ = make(384, 256, 4, torch.float16, env, seed=1)
    u, Ud, _ = make(384, 256, 4, torch.float16, env, seed=2)
    x = torch.randn(9, 256, device=env, dtype=torch.float16)
    ref = torch.nn.functional.silu(x.float() @ Gd.T) * (x.float() @ Ud.T)
    close(fused_gate_up(x, g, u), ref, torch.float16)


def _rope_tables(M, head_dim, device):
    pos = torch.randint(0, 2048, (M,)).float()
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = pos[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos().to(device), emb.sin().to(device)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("B", [1, 3, 6])                  
@pytest.mark.parametrize("split_k", [1, 5])
@pytest.mark.parametrize("rope,head_dim", [(False, 0), (False, 128), (True, 128), (True, 64)])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_qkv_gqa(env, bits, B, split_k, rope, head_dim, bias):
    K, nq, nkv, hd = 1024, 8, 2, (head_dim or 128)
    q, Qd, qb = make(nq * hd, K, bits, torch.float16, env, bias, 1)
    k, Kd, kb = make(nkv * hd, K, bits, torch.float16, env, bias, 2)
    v, Vd, vb = make(nkv * hd, K, bits, torch.float16, env, bias, 3)
    if head_dim:                                        
        q, k = permute_for_rope(q, head_dim), permute_for_rope(k, head_dim)
    x = torch.randn(B, K, device=env, dtype=torch.float16)
    rq, rk, rv = ref_linear(x, Qd, qb), ref_linear(x, Kd, kb), ref_linear(x, Vd, vb)
    r = None
    if rope:
        cos, sin = _rope_tables(B, head_dim, env)
        rq = apply_rope_reference(rq, cos, sin, head_dim)
        rk = apply_rope_reference(rk, cos, sin, head_dim)
        r = (cos, sin, head_dim)
    oq, ok, ov = fused_qkv(x, q, k, v, split_k, 2, rope=r)
    close(oq, rq, torch.float16); close(ok, rk, torch.float16); close(ov, rv, torch.float16)


def test_rope_requires_permuted_weights(env):
    q, _, _ = make(256, 256, 3, torch.float16, env)
    x = torch.randn(1, 256, device=env, dtype=torch.float16)
    cos, sin = _rope_tables(1, 128, env)
    with pytest.raises(ValueError):
        fused_qkv(x, q, q, q, rope=(cos, sin, 128))
    with pytest.raises(ValueError):
        packed_linear(x, permute_for_rope(q, 128))


def test_policy_rule(env):
    S, bn = sm_count(), tile_n()
    assert candidate_split_k(1, bn * 4, bn, S, 1) == max(1, -(-S // 4))
    assert candidate_split_k(64, 4096, bn, S, 1) == 1         
    assert candidate_split_k(1, 10 ** 6, bn, S, 1) == 1


def test_non_default_stream(env):
    if env != "cuda":
        pytest.skip("CUDA streams")
    w, Wd, _ = make(512, 512, 3, torch.float16, env)
    x = torch.randn(1, 512, device=env, dtype=torch.float16)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        y = packed_linear(x, w, 3, 1)
    s.synchronize()
    close(y, ref_linear(x, Wd, None), torch.float16)


def test_cuda_graph_capture(env):
    if env != "cuda":
        pytest.skip("CUDA graphs")
    w, Wd, _ = make(4096, 4096, 3, torch.float16, env)
    x = torch.randn(1, 4096, device=env, dtype=torch.float16)
    ws = new_workspace(env)
    packed_linear(x, w, 1, 1, workspace=ws)              
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = packed_linear(x, w, 1, 1, workspace=ws)
    for _ in range(3):
        x.copy_(torch.randn_like(x)); g.replay(); torch.cuda.synchronize()
        close(y, ref_linear(x, Wd, None), torch.float16)
