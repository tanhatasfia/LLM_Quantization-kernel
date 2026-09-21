

import copy
import pytest
import torch

transformers = pytest.importorskip("transformers")

from crispybits_kernels import install_bitmap, tune_model, FusedQKV, FusedLlamaMLP
from crispybits_kernels.integration import _blocks
from crispybits_kernels.reference import write_packed_dir, pack_linear_dict, dequantized_weight


def _dequantize_model_(model, bitmap):
    for block, bits in zip(_blocks(model), bitmap):
        for _, m in block.named_modules():
            if isinstance(m, torch.nn.Linear):
                d = pack_linear_dict(m.weight.detach().float().cpu(), bits, 128, None, torch.float16)
                m.weight.data.copy_(dequantized_weight(d).to(m.weight))


def _sharpen_attention_(model, factor=8.0):

    with torch.no_grad():
        for blk in _blocks(model):
            blk.self_attn.q_proj.weight.mul_(factor)
            blk.self_attn.k_proj.weight.mul_(factor)
    return model


def _tiny(kind):
    if kind == "llama":
        cfg = transformers.LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                                       num_attention_heads=4, num_key_value_heads=2, vocab_size=512)
        return _sharpen_attention_(transformers.LlamaForCausalLM(cfg))
    cfg = transformers.OPTConfig(hidden_size=256, ffn_dim=512, num_hidden_layers=3, num_attention_heads=4,
                                 vocab_size=512, word_embed_proj_dim=256)
    return transformers.OPTForCausalLM(cfg)


@pytest.mark.parametrize("kind,fuse,fuse_rope", [("llama", False, False), ("llama", True, False),
                                                 ("llama", True, True), ("opt", False, False), ("opt", True, True)])
def test_model_matches_dequantized(env, tmp_path, kind, fuse, fuse_rope):
    torch.manual_seed(0)
    dtype = torch.float16 if env == "cuda" else torch.float32
    base = _tiny(kind).to(dtype).to(env).eval()
    bitmap = [4, 2, 3]
    write_packed_dir(base, tmp_path, bits_options=(2, 3, 4), dtype=dtype)
    ref = copy.deepcopy(base); _dequantize_model_(ref, bitmap)
    model = install_bitmap(copy.deepcopy(base), tmp_path, bitmap, device=env, dtype=dtype,
                           fuse=fuse, fuse_rope=fuse_rope)
    if fuse and kind == "llama":
        owners = [m for m in model.modules() if isinstance(m, FusedQKV)]
        assert owners and any(isinstance(m, FusedLlamaMLP) for m in model.modules())
        assert all(bool(o.rope_head_dim) == fuse_rope for o in owners)
    if fuse:
        tune_model(model, iters=2, warmup=1, dtype=dtype)
    ids = torch.randint(0, 512, (1, 12), device=env)
    with torch.no_grad():
 
        a = ref(ids, use_cache=True); b = model(ids, use_cache=True)
        la, lb = a.logits[:, -1], b.logits[:, -1]
        rel = (la.float() - lb.float()).abs().max() / la.float().abs().max()
        assert rel < 2e-2, f"prefill logits diverge: {rel:.3e}"
        pa, pb = a.past_key_values, b.past_key_values
        for _ in range(4):
            nxt = la.argmax(-1, keepdim=True)
            a = ref(nxt, past_key_values=pa, use_cache=True); b = model(nxt, past_key_values=pb, use_cache=True)
            la, lb, pa, pb = a.logits[:, -1], b.logits[:, -1], a.past_key_values, b.past_key_values
            rel = (la.float() - lb.float()).abs().max() / la.float().abs().max()
            assert rel < 2e-2, f"decode logits diverge: {rel:.3e}"


def test_fused_rope_path_is_used_and_checked(env, tmp_path, monkeypatch):

    import sys
    from crispybits_kernels import integration
    torch.manual_seed(0)
    dtype = torch.float16 if env == "cuda" else torch.float32
    base = _tiny("llama").to(dtype).to(env).eval()
    write_packed_dir(base, tmp_path, bits_options=(3,), dtype=dtype)
    ref = copy.deepcopy(base); _dequantize_model_(ref, [3, 3, 3])
    model = install_bitmap(copy.deepcopy(base), tmp_path, [3, 3, 3], device=env, dtype=dtype, fuse_rope=True)
    mod = sys.modules[type(_blocks(model)[0].self_attn).__module__]
    assert getattr(mod.apply_rotary_pos_emb, "_crispybits_patched", False)

    primes = []
    orig_prime = FusedQKV.prime
    monkeypatch.setattr(FusedQKV, "prime", lambda self, *a: (primes.append(1), orig_prime(self, *a))[1])
    ids = torch.randint(0, 512, (1, 16), device=env)

    def block0_update(m):   
        h = m(ids, output_hidden_states=True).hidden_states
        return (h[1] - h[0]).float()

    with torch.no_grad():
        la = block0_update(ref); lb = block0_update(model)
    assert len(primes) == 3                                 
    assert (la - lb).abs().max() / la.abs().max() < 2e-2
    for blk in _blocks(model):
        assert blk.self_attn.crispybits_qkv._cache is None

    monkeypatch.setattr(integration, "_mark", lambda t: t)  
    with torch.no_grad():
        lb2 = block0_update(model)
    err2 = (la - lb2).abs().max() / la.abs().max()
    assert err2 > 0.2, f"double rotation went undetected ({err2:.3f})"


def test_proxy_mismatch_raises(env):
    from crispybits_kernels.ops import PackedLinear
    from crispybits_kernels.io import from_packed_dict
    from crispybits_kernels.integration import _QKVProxy
    dtype = torch.float16 if env == "cuda" else torch.float32
    ws = [PackedLinear(from_packed_dict(pack_linear_dict(torch.randn(128, 128) * .05, 3, 128, None, dtype), env, dtype))
          for _ in range(3)]
    ws[0].permute_for_rope_(64); ws[1].permute_for_rope_(64)
    owner = FusedQKV(*ws, rope_head_dim=64)
    x = torch.randn(1, 128, device=env, dtype=dtype)
    cos, sin = torch.ones(1, 64, device=env), torch.zeros(1, 64, device=env)
    owner.prime(x, cos, sin)
    with pytest.raises(RuntimeError):
        _QKVProxy(owner, 0)(x.clone())                    
