#!/usr/bin/env python3

import torch
from _common import base_parser, setup, make_weight
from crispybits_kernels.autotune import tune_gemv, _time
from crispybits_kernels.ops import packed_linear, new_workspace
from crispybits_kernels.reference import dequantized_weight, pack_linear_dict

p = base_parser(__doc__)
p.add_argument("--m", type=int, default=1)
p.add_argument("--n", type=int, required=True, help="output features")
p.add_argument("--k", type=int, required=True, help="input features")
p.add_argument("--bits", type=int, choices=[2, 3, 4], required=True)
p.add_argument("--exhaustive", action="store_true")
a = p.parse_args()
dev = setup(a)

torch.manual_seed(0)
w, Wfp, _ = make_weight(a.n, a.k, a.bits, dev)
x = torch.randn(a.m, a.k, device=dev, dtype=torch.float16)
from crispybits_kernels.reference import unpack_codes, dequantize
ref = x.float() @ dequantize(unpack_codes(w.packed.cpu(), a.bits, a.k), w.scales.cpu(), w.zeros.cpu(), 128).to(dev).T
err = (packed_linear(x, w, 1, 1).float() - ref).abs().max() / ref.abs().max()
assert err < 3e-3, f"packed kernel incorrect (rel err {err:.2e}); not timing"

rep = tune_gemv(x, w, warmup=a.warmup // 2, iters=max(20, a.iters // 4), exhaustive=a.exhaustive)
for r in sorted(rep.results, key=lambda r: (r.rho, r.split_k)):
    tag = "rule" if r.policy else "    "
    print(f"  {tag} rho={r.rho:2d} P_K={r.split_k:2d}  {r.ms*1e3:8.2f} us   (C_out={r.c_out}, C_target={r.c_target})")
print("selected", rep.best)
if a.exhaustive:
    print("oracle  ", rep.oracle, f"  policy regret {rep.regret:+.1%}")
ws = new_workspace(dev)
low = _time(lambda: packed_linear(x, w, rep.best.split_k, rep.best.rho, workspace=ws), a.warmup, a.iters)
fp = _time(lambda: torch.nn.functional.linear(x, Wfp), a.warmup, a.iters)
gb = w.packed.numel() * 4 + w.scales.numel() * 2 * 2
print(f"packed W{a.bits}={low*1e3:.2f} us ({gb/low/1e6:.0f} GB/s effective)  fp16={fp*1e3:.2f} us "
      f"({a.n*a.k*2/fp/1e6:.0f} GB/s)  speedup={fp/low:.3f}x")
