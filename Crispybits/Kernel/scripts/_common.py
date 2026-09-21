"""Shared helpers for the benchmark scripts."""
import argparse
import torch

from crispybits_kernels import ops
from crispybits_kernels.io import from_packed_dict
from crispybits_kernels.reference import pack_linear_dict

MODELS = {
    "llama2-7b": (4096, 11008, 32, 32, "llama"),
    "llama2-13b": (5120, 13824, 40, 40, "llama"),
    "llama3-8b": (4096, 14336, 32, 8, "llama"),
    "opt-1.3b": (2048, 8192, 32, 32, "opt"),
    "opt-2.7b": (2560, 10240, 32, 32, "opt"),
    "tiny": (256, 512, 4, 2, "llama"),          
}


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--backend", choices=["cuda", "emulator"], default="cuda",
                   help="emulator = CPU smoke test of the script logic; timings are meaningless")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=25)
    return p


def setup(args):
    ops.set_backend(args.backend)
    device = "cuda" if args.backend == "cuda" else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"GPU: {torch.cuda.get_device_name()}  SMs={ops.sm_count()}  B_N={ops.tile_n()}")
    return device


def make_weight(n, k, bits, device, dtype=torch.float16, bias=False, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(n, k, generator=g) * 0.02
    b = torch.randn(n, generator=g) * 0.02 if bias else None
    d = pack_linear_dict(W.to(device), bits, 128, b, dtype)
    return from_packed_dict(d, device, dtype), W.to(device=device, dtype=dtype), (None if b is None else b.to(device, dtype))
