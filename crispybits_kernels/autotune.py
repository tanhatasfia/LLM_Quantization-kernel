from __future__ import annotations
from dataclasses import dataclass
from math import ceil
import torch

from .ops import PackedLinearWeight, packed_linear, sm_count, max_rho


@dataclass
class TuneResult:
    rho: int
    split_k: int
    ms: float
    c_out: int
    c_target: int


def candidate_split_k(batch: int, n: int, bn: int, sms: int, rho: int, max_split: int = 16) -> int:
    c_out = batch * ceil(n / bn)
    c_target = rho * sms
    if c_out >= c_target:
        return 1
    return min(max_split, max(1, ceil(c_target / c_out)))


def tune_gemv(x: torch.Tensor, w: PackedLinearWeight, warmup: int=10, iters: int=50, bn: int=128, max_split: int=16):
   
    S=sm_count(); occ=max(1,max_rho(w.bits)); B=x.reshape(-1,x.shape[-1]).shape[0];N=w.out_features
    choices=[]
    for rho in range(1,occ+1):
        pk=candidate_split_k(B,N,bn,S,rho,max_split)
        # Avoid duplicate (rho,pk) timing only if desired; we keep rho because
        # persistent block count changes even with equal split.
        for _ in range(warmup): packed_linear(x,w,pk,rho)
        torch.cuda.synchronize()
        st=torch.cuda.Event(True);en=torch.cuda.Event(True);st.record()
        for _ in range(iters): packed_linear(x,w,pk,rho)
        en.record();torch.cuda.synchronize();ms=st.elapsed_time(en)/iters
        choices.append(TuneResult(rho,pk,ms,B*((N+bn-1)//bn),rho*S))
    return min(choices,key=lambda z:z.ms),choices
