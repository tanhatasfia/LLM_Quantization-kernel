#!/usr/bin/env python3

import torch
from _common import base_parser, setup, make_weight
from crispybits_kernels.autotune import _time
from crispybits_kernels.ops import packed_linear

p = base_parser(__doc__)
p.add_argument("--m", type=int, default=1024, help="prompt tokens")
p.add_argument("--shapes", default="4096x4096,11008x4096,4096x11008", help="NxK list")
p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
a = p.parse_args()
dev = setup(a)
for shape in a.shapes.split(","):
    n, k = map(int, shape.split("x"))
    x = torch.randn(a.m, k, device=dev, dtype=torch.float16)
    for bits in a.bits:
        w, Wfp, _ = make_weight(n, k, bits, dev)
        ref = packed_linear(x, w, use_tensor_cores=False).float()
        y = packed_linear(x, w).float()
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        assert err < 5e-3, f"tensor-core path disagrees with scalar path ({err:.2e})"
        tc = _time(lambda: packed_linear(x, w), a.warmup, a.iters)
        sc = _time(lambda: packed_linear(x, w, use_tensor_cores=False), 3, max(3, a.iters // 20))
        fp = _time(lambda: torch.nn.functional.linear(x, Wfp), a.warmup, a.iters)
        tf = 2 * a.m * n * k / 1e9
        print(f"M={a.m} N={n} K={k} W{bits}: tensor-core {tc*1e3:9.1f} us ({tf/tc:6.1f} TFLOP/s) | "
              f"scalar {sc*1e3:10.1f} us | fp16 cuBLAS {fp*1e3:9.1f} us ({tf/fp:6.1f} TFLOP/s) | "
              f"vs fp16 {fp/tc:.2f}x")
