#!/usr/bin/env python3

import argparse, csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from crispybits_quant.trps import calibrate_trps_batch, aggregate_trps

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--dataset", default="wikitext")
p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
p.add_argument("--split", default="train")
p.add_argument("--samples", type=int, default=500)
p.add_argument("--seq-len", type=int, default=2048)
p.add_argument("--group-size", type=int, default=128)
p.add_argument("--device", default="cuda")
p.add_argument("--out", default="trps.csv")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(a.device).eval()
ds = load_dataset(a.dataset, a.dataset_config, split=a.split)
text_col = "text" if "text" in ds.column_names else ds.column_names[0]
all_records = []
count = 0
for ex in ds:
    txt = ex[text_col]
    if not isinstance(txt, str) or not txt.strip():
        continue
    ids = tok(txt, return_tensors="pt", truncation=True, max_length=a.seq_len).input_ids
    if ids.shape[1] < 8:
        continue
    inputs = {"input_ids": ids.to(a.device), "use_cache": False}
    rec = calibrate_trps_batch(model, inputs, group_size=a.group_size)
    all_records.extend(rec)
    count += 1
    print(f"sample {count}/{a.samples}")
    if count >= a.samples:
        break
agg = aggregate_trps(all_records)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["block","bits","mean_local","cvar_tail","mean_prop","trps"])
    w.writeheader(); [w.writerow(r.to_dict()) for r in agg]
print(f"saved {a.out}")
