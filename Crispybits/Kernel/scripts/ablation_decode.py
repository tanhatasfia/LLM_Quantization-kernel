#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from _common import MODELS, base_parser, setup, make_weight
from crispybits_kernels.autotune import _time, tune_gemv, tune_qkv, tune_gate_up
from crispybits_kernels.ops import fused_gate_up, fused_qkv, packed_linear, max_rho, new_workspace

p = base_parser(__doc__)
p.add_argument("--model", choices=sorted(MODELS), default="llama2-7b")
p.add_argument("--bits", type=int, choices=[2, 3, 4], default=3)
a = p.parse_args()
dev = setup(a)
H, I, nh, nkv, fam = MODELS[a.model]
hd = H // nh
opt = fam == "opt"
mk = lambda n, k, s, bias=False: make_weight(n, k, a.bits, dev, bias=bias, seed=s)
W = {"q": mk(nh * hd, H, 1, opt), "k": mk(nkv * hd, H, 2, opt), "v": mk(nkv * hd, H, 3, opt), "o": mk(H, H, 4, opt)}
if opt:
    W.update(fc1=mk(I, H, 5, True), fc2=mk(H, I, 6, True))
else:
    W.update(gate=mk(I, H, 5), up=mk(I, H, 6), down=mk(H, I, 7))
pw = {k: v[0] for k, v in W.items()}
xh = torch.randn(1, H, device=dev, dtype=torch.float16)
xi = torch.randn(1, I, device=dev, dtype=torch.float16)
ws = {k: new_workspace(dev) for k in ("q", "k", "v", "o", "gate", "up", "down", "fc1", "fc2", "qkv", "gu")}
cfg = {}  


def run_unfused(sk_rho):
    for n in ("q", "k", "v"):
        packed_linear(xh, pw[n], *sk_rho(n), workspace=ws[n])
    packed_linear(xh, pw["o"], *sk_rho("o"), workspace=ws["o"])
    if opt:
        torch.relu(packed_linear(xh, pw["fc1"], *sk_rho("fc1"), workspace=ws["fc1"]))
        packed_linear(xi, pw["fc2"], *sk_rho("fc2"), workspace=ws["fc2"])
    else:
        g = packed_linear(xh, pw["gate"], *sk_rho("gate"), workspace=ws["gate"])
        u = packed_linear(xh, pw["up"], *sk_rho("up"), workspace=ws["up"])
        F.silu(g) * u
        packed_linear(xi, pw["down"], *sk_rho("down"), workspace=ws["down"])


def run_fused(sk_rho):
    fused_qkv(xh, pw["q"], pw["k"], pw["v"], *sk_rho("qkv"), workspace=ws["qkv"])
    packed_linear(xh, pw["o"], *sk_rho("o"), workspace=ws["o"])
    if opt:
        packed_linear(xh, pw["fc1"], *sk_rho("fc1"), act="relu", workspace=ws["fc1"])
        packed_linear(xi, pw["fc2"], *sk_rho("fc2"), workspace=ws["fc2"])
    else:
        fused_gate_up(xh, pw["gate"], pw["up"], *sk_rho("gu"), workspace=ws["gu"])
        packed_linear(xi, pw["down"], *sk_rho("down"), workspace=ws["down"])


def fp16_block():
    for n in ("q", "k", "v", "o"):
        F.linear(xh, W[n][1], W[n][2])
    if opt:
        torch.relu(F.linear(xh, W["fc1"][1], W["fc1"][2])); F.linear(xi, W["fc2"][1], W["fc2"][2])
    else:
        F.silu(F.linear(xh, W["gate"][1])) * F.linear(xh, W["up"][1]); F.linear(xi, W["down"][1])


fused_mods = ["qkv", "o"] + (["fc1", "fc2"] if opt else ["gu", "down"])
t = lambda fn: _time(fn, a.warmup, a.iters)
rows = []
rows.append(("1 base packed (non-persistent, P_K=1)", t(lambda: run_unfused(lambda n: (1, 0)))))
rows.append(("2 + projection fusion", t(lambda: run_fused(lambda n: (1, 0)))))
rows.append(("3 + persistent CTA (rho=1)", t(lambda: run_fused(lambda n: (1, 1)))))


def tune(name, exhaustive=False, max_split=None):
    it = dict(warmup=5, iters=max(10, a.iters // 5), exhaustive=exhaustive, max_split=max_split)
    if name == "qkv":
        return tune_qkv(xh, pw["q"], pw["k"], pw["v"], **it)
    if name == "gu":
        return tune_gate_up(xh, pw["gate"], pw["up"], **it)
    x = xi if name in ("down", "fc2") else xh
    return tune_gemv(x, pw[name], act="relu" if name == "fc1" else None, **it)


kind = {"qkv": "qkv", "gu": "gate_up"}
best_rho = {}
for m in fused_mods:                                    
    best_rho[m] = tune(m, max_split=1).best.rho
rows.append(("4 + occupancy-aware rho", t(lambda: run_fused(lambda n: (1, best_rho[n])))))
for pk in (2, 4, 8):
    rows.append((f"5 fixed Split-K P_K={pk}", t(lambda: run_fused(lambda n: (pk, best_rho[n])))))
reports = {m: tune(m, exhaustive=True) for m in fused_mods}
pol = {m: (r.best.split_k, r.best.rho) for m, r in reports.items()}
orc = {m: (r.oracle.split_k, r.oracle.rho) for m, r in reports.items()}
rows.append(("6 SM-utilization-aware K-parallelism (policy)", t(lambda: run_fused(lambda n: pol[n]))))
rows.append(("7 oracle (exhaustive grid)", t(lambda: run_fused(lambda n: orc[n]))))
fp = t(fp16_block)

base = rows[0][1]
print(f"\n{a.model} W{a.bits}, batch 1, one block's projections")
for name, ms in rows:
    print(f"  {name:48s} {ms*1e3:8.1f} us   {base/ms:5.2f}x vs base   {fp/ms:5.2f}x vs fp16")
print(f"  {'FP16 cuBLAS':48s} {fp*1e3:8.1f} us")
print("\nper-module (P_K, rho): policy -> oracle, regret")
for m, r in reports.items():
    print(f"  {m:5s} {pol[m]} -> {orc[m]}   {r.regret:+.1%}")
