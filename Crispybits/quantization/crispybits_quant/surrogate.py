from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import json

import numpy as np


BITS = (2, 3, 4)


@dataclass
class EnergySurrogate:
    intercept: float
    costs: Dict[tuple[int, int], float]
    delta: float
    alpha: float
    n_blocks: int

    def predict(self, bitmap: Sequence[int]) -> float:
        if len(bitmap) != self.n_blocks:
            raise ValueError("bitmap length does not match surrogate")
        return self.intercept + sum(self.costs[(l, int(b))] for l, b in enumerate(bitmap))

    def shifted_nonnegative(self) -> "EnergySurrogate":
        """Per-block cost shift preserving every predicted configuration energy.

        Ridge coefficients can be negative. For MCKP/DP it is convenient to use
        non-negative per-choice costs. For each block l:
            m_l = min_b c_l,b
            c'_l,b = c_l,b - m_l
            E0' = E0 + sum_l m_l
        """
        costs = dict(self.costs)
        intercept = self.intercept
        for l in range(self.n_blocks):
            m = min(costs[(l, b)] for b in BITS)
            intercept += m
            for b in BITS:
                costs[(l, b)] -= m
        return EnergySurrogate(intercept, costs, self.delta, self.alpha, self.n_blocks)

    def to_json(self, path: str) -> None:
        obj = {
            "intercept": self.intercept,
            "delta": self.delta,
            "alpha": self.alpha,
            "n_blocks": self.n_blocks,
            "costs": {f"{l}:{b}": v for (l, b), v in self.costs.items()},
        }
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "EnergySurrogate":
        with open(path) as f:
            obj = json.load(f)
        costs = {}
        for k, v in obj["costs"].items():
            l, b = map(int, k.split(":"))
            costs[(l, b)] = float(v)
        return cls(float(obj["intercept"]), costs, float(obj["delta"]), float(obj["alpha"]), int(obj["n_blocks"]))


def one_hot_bitmaps(bitmaps: np.ndarray, n_blocks: int) -> np.ndarray:
    
    X = np.zeros((len(bitmaps), n_blocks * 2), dtype=np.float64)
    for i, row in enumerate(bitmaps):
        for l, b in enumerate(row):
            if int(b) == 3:
                X[i, 2*l] = 1.0
            elif int(b) == 4:
                X[i, 2*l + 1] = 1.0
            elif int(b) != 2:
                raise ValueError(f"invalid bit {b}")
    return X


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(A, Xc.T @ yc)
    intercept = y_mean - x_mean @ coef
    return float(intercept), coef


def _metrics(y, pred):
    err = pred - y
    ae = np.abs(err)
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p95_abs": float(np.quantile(ae, 0.95)),
    }


def fit_energy_surrogate(
    bitmaps: np.ndarray,
    energies: np.ndarray,
    alpha: float = 1.0,
    seed: int = 0,
    split=(0.70, 0.15, 0.15),
):
    
    bitmaps = np.asarray(bitmaps, dtype=np.int64)
    energies = np.asarray(energies, dtype=np.float64)
    if bitmaps.ndim != 2 or energies.ndim != 1 or len(bitmaps) != len(energies):
        raise ValueError("bitmaps must be [K,L] and energies [K]")
    n, n_blocks = bitmaps.shape
    if not np.isclose(sum(split), 1.0):
        raise ValueError("split must sum to 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = int(round(split[0] * n))
    nva = int(round(split[1] * n))
    tr = idx[:ntr]
    va = idx[ntr:ntr+nva]
    te = idx[ntr+nva:]

    X = one_hot_bitmaps(bitmaps, n_blocks)
    intercept, coef = _ridge_fit(X[tr], energies[tr], alpha)

    costs: Dict[tuple[int, int], float] = {}
    for l in range(n_blocks):
        costs[(l, 2)] = 0.0
        costs[(l, 3)] = float(coef[2*l])
        costs[(l, 4)] = float(coef[2*l + 1])

    def pred(rows):
        return intercept + X[rows] @ coef

    val_pred = pred(va)
    residual_under = energies[va] - val_pred
    delta = max(0.0, float(np.quantile(residual_under, 0.95))) if len(va) else 0.0

    model = EnergySurrogate(intercept, costs, delta, alpha, n_blocks)
    report = {
        "train": _metrics(energies[tr], pred(tr)),
        "validation": _metrics(energies[va], val_pred) if len(va) else {},
        "test": _metrics(energies[te], pred(te)) if len(te) else {},
        "delta_q95_underprediction": delta,
        "sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
    }
    return model, report
