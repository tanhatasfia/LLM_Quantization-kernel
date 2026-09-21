#!/usr/bin/env python3
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--blocks", type=int, required=True)
p.add_argument("--k", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="bitmaps.npy")
a = p.parse_args()
rng = np.random.default_rng(a.seed)
arr = rng.choice(np.array([2,3,4], dtype=np.int8), size=(a.k,a.blocks), replace=True)
np.save(a.out, arr)
print(a.out, arr.shape)
