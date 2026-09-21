from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math

from .surrogate import EnergySurrogate, BITS


@dataclass
class AllocationResult:
    bitmap: list[int]
    predicted_energy: float
    objective: float
    budget: float
    effective_budget: float


def solve_mckp_dp(
    quality: Dict[tuple[int, int], float],
    surrogate: EnergySurrogate,
    energy_budget: float,
    resolution_j: float = 0.1,
) -> AllocationResult:
    """Exact-over-discretized-cost multi-choice knapsack solver.

    Maximizes sum_l v[l,b] subject to
        E0 + sum_l c[l,b_l] <= E_budget - delta.

    Per-block ridge costs are shifted to nonnegative values before DP; the
    shift is absorbed into E0 and therefore leaves predicted energies intact.
    """
    s = surrogate.shifted_nonnegative()
    cap_j = energy_budget - s.delta - s.intercept
    if cap_j < -1e-9:
        raise ValueError(
            f"Budget infeasible even before block costs: capacity={cap_j:.3f} J"
        )
    cap = max(0, int(math.floor(cap_j / resolution_j + 1e-9)))
    n = s.n_blocks

    # sparse DP: cost_bin -> (value, bitmap_prefix)
    dp = {0: (0.0, [])}
    for l in range(n):
        nxt = {}
        for used, (val, path) in dp.items():
            for b in BITS:
                c = s.costs[(l, b)]
                cb = int(math.ceil(max(0.0, c) / resolution_j - 1e-12))
                nu = used + cb
                if nu > cap:
                    continue
                nv = val + float(quality[(l, b)])
                if nu not in nxt or nv > nxt[nu][0]:
                    nxt[nu] = (nv, path + [b])
        if not nxt:
            raise ValueError(f"No feasible MCKP choice after block {l}")
        # Pareto prune dominated states: increasing cost must improve value.
        best = -float("inf")
        pruned = {}
        for cbin in sorted(nxt):
            val, path = nxt[cbin]
            if val > best + 1e-15:
                pruned[cbin] = (val, path)
                best = val
        dp = pruned

    _, (obj, bitmap) = max(dp.items(), key=lambda kv: kv[1][0])
    pred = surrogate.predict(bitmap)
    return AllocationResult(
        bitmap=bitmap,
        predicted_energy=pred,
        objective=obj,
        budget=energy_budget,
        effective_budget=energy_budget - surrogate.delta,
    )
