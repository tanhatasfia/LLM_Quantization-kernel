#!/usr/bin/env python3
import argparse, json
import numpy as np
from crispybits_quant.surrogate import fit_energy_surrogate

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True, help="NPZ with bitmaps[K,L], energies[K]")
p.add_argument("--out", default="energy_surrogate.json")
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

d = np.load(a.npz)
model, report = fit_energy_surrogate(d["bitmaps"], d["energies"], alpha=a.alpha, seed=a.seed)
model.to_json(a.out)
print(json.dumps(report, indent=2))
print(f"saved {a.out}")
