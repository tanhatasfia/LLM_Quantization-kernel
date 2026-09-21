
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from math import ceil
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .ops import (PackedLinear, PackedLinearWeight, candidate_split_k, fused_gate_up, fused_qkv, max_rho,
                  new_workspace, packed_linear, sm_count, tile_n, virtual_rows)

EXHAUSTIVE_PK = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


@dataclass
class TuneResult:
    rho: int
    split_k: int
    ms: float
    c_out: int
    c_target: int
    policy: bool = True          


@dataclass
class TuneReport:
    best: TuneResult            
    oracle: TuneResult           
    results: List[TuneResult]

    @property
    def regret(self) -> float:
        return self.best.ms / self.oracle.ms - 1.0


def _time(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / iters
    t0 = time.perf_counter()                      
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def _tune(run: Callable[[int, int], object], B: int, V: int, K: int, bits: int, kind: str,
          warmup: int, iters: int, max_split: Optional[int], exhaustive: bool = False) -> TuneReport:
    S, bn = sm_count(), tile_n()
    occ = max(1, max_rho(bits, kind))
    nchunks = ceil(K / 32)
    tiles = ceil(V / bn)
    c_out = B * tiles
    results: List[TuneResult] = []
    seen = set()

    def measure(rho, pk, policy):
        pk = max(1, min(pk, nchunks))
        
        key = (pk, min(B * tiles * pk, rho * S))
        if key in seen:
            for r in results:                    
                if (r.split_k, min(B * tiles * r.split_k, r.rho * S)) == key and policy:
                    r.policy = True
            return
        seen.add(key)
        ms = _time(lambda: run(pk, rho), warmup, iters)
        results.append(TuneResult(rho, pk, ms, c_out, rho * S, policy))

    for rho in range(1, occ + 1):
        measure(rho, candidate_split_k(B, V, bn, S, rho, max_split), True)
    if exhaustive:
        for rho in range(1, occ + 1):
            for pk in EXHAUSTIVE_PK:
                if max_split is None or pk <= max_split:
                    measure(rho, pk, False)
    best = min((r for r in results if r.policy), key=lambda r: r.ms)
    oracle = min(results, key=lambda r: r.ms)
    return TuneReport(best, oracle, results)


def tune_gemv(x: torch.Tensor, w: PackedLinearWeight, warmup: int = 10, iters: int = 50,
              max_split: Optional[int] = None, act: Optional[str] = None, exhaustive: bool = False) -> TuneReport:
    B = x.reshape(-1, x.shape[-1]).shape[0]
    ws = new_workspace(x.device)
    return _tune(lambda pk, rho: packed_linear(x, w, pk, rho, act, workspace=ws), B,
                 virtual_rows("gemv", w.out_features), w.in_features, w.bits, "gemv",
                 warmup, iters, max_split, exhaustive)


def tune_qkv(x, q: PackedLinearWeight, k: PackedLinearWeight, v: PackedLinearWeight,
             warmup: int = 10, iters: int = 50, max_split: Optional[int] = None,
             exhaustive: bool = False) -> TuneReport:
    B = x.reshape(-1, x.shape[-1]).shape[0]
    ws = new_workspace(x.device)
    V = virtual_rows("qkv", q.out_features, k.out_features, v.out_features)
    return _tune(lambda pk, rho: fused_qkv(x, q, k, v, pk, rho, workspace=ws), B, V, q.in_features, q.bits,
                 "qkv", warmup, iters, max_split, exhaustive)


def tune_gate_up(x, gate: PackedLinearWeight, up: PackedLinearWeight, warmup: int = 10, iters: int = 50,
                 max_split: Optional[int] = None, exhaustive: bool = False) -> TuneReport:
    B = x.reshape(-1, x.shape[-1]).shape[0]
    ws = new_workspace(x.device)
    return _tune(lambda pk, rho: fused_gate_up(x, gate, up, pk, rho, workspace=ws), B,
                 virtual_rows("gate_up", gate.out_features), gate.in_features, gate.bits, "gate_up",
                 warmup, iters, max_split, exhaustive)


@torch.no_grad()
def tune_model(model: torch.nn.Module, batch: int = 1, warmup: int = 10, iters: int = 50,
               max_split: Optional[int] = None, dtype=torch.float16, verbose: bool = False,
               exhaustive: bool = False) -> Dict[str, Dict]:
    
    from .integration import FusedLlamaMLP, FusedQKV, _blocks
    cache: Dict[tuple, TuneReport] = {}
    tuning: Dict[str, Dict] = {}

    def x_for(K, device):
        return torch.randn(batch, K, device=device, dtype=dtype)

    def record(key, fn):
        if key not in cache:
            cache[key] = fn()
        return cache[key].best

    for i, block in enumerate(_blocks(model)):
        owned = set()
        for m in block.modules():
            if isinstance(m, FusedQKV):
                owned.update(id(x) for x in (m.q, m.k, m.v))
            elif isinstance(m, FusedLlamaMLP):
                owned.update(id(x) for x in (m.gate_proj, m.up_proj))
        for name, mod in block.named_modules():
            if isinstance(mod, FusedQKV):
                ws = [m.weight_spec() for m in (mod.q, mod.k, mod.v)]
                key = ("qkv", ws[0].bits, ws[0].in_features, *(w.out_features for w in ws))
                r = record(key, lambda: tune_qkv(x_for(ws[0].in_features, ws[0].packed.device), *ws, warmup=warmup,
                                                 iters=iters, max_split=max_split, exhaustive=exhaustive))
                mod.split_k, mod.rho = r.split_k, r.rho
                tuning[f"block.{i}.self_attn.qkv"] = {"split_k": r.split_k, "rho": r.rho}
            elif isinstance(mod, FusedLlamaMLP):
                g, u = mod.gate_proj.weight_spec(), mod.up_proj.weight_spec()
                key = ("gate_up", g.bits, g.in_features, g.out_features)
                r = record(key, lambda: tune_gate_up(x_for(g.in_features, g.packed.device), g, u, warmup=warmup,
                                                     iters=iters, max_split=max_split, exhaustive=exhaustive))
                mod.split_k, mod.rho = r.split_k, r.rho
                tuning[f"block.{i}.mlp.gate_up"] = {"split_k": r.split_k, "rho": r.rho}
            elif isinstance(mod, PackedLinear) and id(mod) not in owned:
                w = mod.weight_spec()
                key = ("gemv", w.bits, w.in_features, w.out_features, w.bias is not None, mod.act)
                r = record(key, lambda: tune_gemv(x_for(w.in_features, w.packed.device), w, warmup=warmup, iters=iters,
                                                  max_split=max_split, act=mod.act, exhaustive=exhaustive))
                mod.split_k, mod.rho = r.split_k, r.rho
                tuning[f"block.{i}.{name}"] = {"split_k": r.split_k, "rho": r.rho}
        if verbose:
            print(f"block {i} tuned")
    if verbose:
        for k, rep in cache.items():
            print(k, "best", asdict(rep.best), "oracle", asdict(rep.oracle), f"regret {rep.regret:+.1%}")
    return tuning
