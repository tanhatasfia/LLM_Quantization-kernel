#!/usr/bin/env python3
import argparse, sys, os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../crispybits_quantization")))
from crispybits_quant.affine import affine_quantize_weight
from crispybits_quant.packing import pack_quantized
from crispybits_kernels.ops import PackedLinearWeight, packed_linear
from crispybits_kernels.autotune import tune_gemv

p=argparse.ArgumentParser();p.add_argument("--m",type=int,default=1);p.add_argument("--n",type=int,required=True);p.add_argument("--k",type=int,required=True);p.add_argument("--bits",type=int,choices=[2,3,4],required=True);p.add_argument("--iters",type=int,default=100);a=p.parse_args()
torch.manual_seed(0);x=torch.randn(a.m,a.k,device="cuda",dtype=torch.float16);W=torch.randn(a.n,a.k,device="cuda",dtype=torch.float16)
qt=affine_quantize_weight(W,a.bits,128);pw=pack_quantized(qt);w=PackedLinearWeight(pw.packed,pw.scales,pw.zeros,a.bits,128,a.k,a.n,None).cuda(dtype=torch.float16)
best,allr=tune_gemv(x,w,iters=max(20,a.iters//2));print("autotune",best)
for _ in range(10):packed_linear(x,w,best.split_k,best.rho)
torch.cuda.synchronize();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
for _ in range(a.iters): packed_linear(x,w,best.split_k,best.rho)
e.record();torch.cuda.synchronize();low=s.elapsed_time(e)/a.iters
s.record()
for _ in range(a.iters): torch.nn.functional.linear(x,W)
e.record();torch.cuda.synchronize();fp=s.elapsed_time(e)/a.iters
print(f"packed={low:.4f} ms fp16={fp:.4f} ms speedup={fp/low:.3f}x")
