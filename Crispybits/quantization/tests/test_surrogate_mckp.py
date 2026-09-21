import numpy as np
from crispybits_quant.surrogate import fit_energy_surrogate
from crispybits_quant.mckp import solve_mckp_dp


def test_surrogate_and_allocator():
    rng = np.random.default_rng(0)
    L, K = 5, 300
    bitmaps = rng.choice([2,3,4], size=(K,L))
    true_cost = {(l,b): (b-2)*(1.0+0.2*l) for l in range(L) for b in (2,3,4)}
    energies = np.array([100 + sum(true_cost[(l,int(b))] for l,b in enumerate(row)) for row in bitmaps])
    model, report = fit_energy_surrogate(bitmaps, energies, alpha=1e-6)
    quality = {(l,b): float(b-2) for l in range(L) for b in (2,3,4)}
    res = solve_mckp_dp(quality, model, energy_budget=108.0, resolution_j=0.01)
    assert len(res.bitmap) == L
    assert res.predicted_energy <= 108.0 - model.delta + 0.05
