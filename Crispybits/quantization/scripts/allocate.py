#!/usr/bin/env python3
import argparse, json, csv
from crispybits_quant.surrogate import EnergySurrogate
from crispybits_quant.mckp import solve_mckp_dp

p = argparse.ArgumentParser()
p.add_argument("--trps-csv", required=True, help="columns: block,bits,trps")
p.add_argument("--surrogate", required=True)
p.add_argument("--budget", required=True, type=float)
p.add_argument("--resolution-j", type=float, default=0.1)
p.add_argument("--out", default="allocation.json")
a = p.parse_args()

scores = {}
with open(a.trps_csv) as f:
    for r in csv.DictReader(f):
        scores[(int(r["block"]), int(r["bits"]))] = float(r["trps"])
quality = {}
for l in sorted({x[0] for x in scores}):
    s2 = scores[(l, 2)]
    for b in (2,3,4):
        quality[(l,b)] = 0.0 if b == 2 else s2 - scores[(l,b)]

sur = EnergySurrogate.from_json(a.surrogate)
res = solve_mckp_dp(quality, sur, a.budget, a.resolution_j)
obj = res.__dict__
with open(a.out, "w") as f: json.dump(obj, f, indent=2)
print(json.dumps(obj, indent=2))
