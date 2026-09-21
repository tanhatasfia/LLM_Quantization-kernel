#!/usr/bin/env python3

import argparse, csv, json, random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from crispybits_quant.trps import trps_token_stats, TRPSAccumulator

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--dataset", default="wikitext")
p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
p.add_argument("--split", default="train")
p.add_argument("--samples", type=int, default=500)
p.add_argument("--seq-len", type=int, default=2048)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--corpus-token-factor", type=float, default=4.0,
               help="stop reading the dataset once the corpus has roughly "
                    "factor*samples*seq_len tokens (keeps huge corpora like C4 cheap)")
p.add_argument("--group-size", type=int, default=128)
p.add_argument("--tau", type=float, default=0.10)
p.add_argument("--lambda-tail", type=float, default=0.25)
p.add_argument("--lambda-prop", type=float, default=0.25)
p.add_argument("--prepend-bos", action="store_true",
               help="prepend BOS to every window (off by default, like GPTQ windows)")
p.add_argument("--skip-first-tokens", type=int, default=0,
               help="exclude the first N positions of each window from the statistics "
                    "(e.g. 1 to drop the attention-sink/BOS position)")
p.add_argument("--device", default="cuda")
p.add_argument("--out", default="trps.csv")
a = p.parse_args()

random.seed(a.seed)
torch.manual_seed(a.seed)

tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(a.device).eval()


ds = load_dataset(a.dataset, a.dataset_config, split=a.split, streaming=True)
target_tokens = int(a.corpus_token_factor * a.samples * a.seq_len)
char_budget = target_tokens * 6          
texts, n_chars = [], 0
for ex in ds:
    txt = ex.get("text") if isinstance(ex, dict) else None
    if txt is None:
        txt = next((v for v in ex.values() if isinstance(v, str)), None)
    if not txt or not txt.strip():
        continue
    texts.append(txt)
    n_chars += len(txt)
    if n_chars >= char_budget:
        break
stream = tok("\n\n".join(texts), return_tensors="pt", add_special_tokens=False).input_ids[0]
del texts

win = a.seq_len - (1 if a.prepend_bos else 0)
if stream.numel() <= win:
    raise SystemExit(f"Corpus has only {stream.numel()} tokens; need > {win}.")
max_start = stream.numel() - win
if a.samples * win > stream.numel():
    print(f"warning: {a.samples} x {win} tokens exceeds corpus ({stream.numel()}); windows will overlap")
starts = [random.randint(0, max_start) for _ in range(a.samples)]


acc = TRPSAccumulator(a.tau, a.lambda_tail, a.lambda_prop)
for i, s0 in enumerate(starts, 1):
    ids = stream[s0:s0 + win]
    if a.prepend_bos:
        ids = torch.cat([torch.tensor([tok.bos_token_id]), ids])
    inputs = {"input_ids": ids.unsqueeze(0).to(a.device), "use_cache": False}
    acc.add(trps_token_stats(model, inputs, group_size=a.group_size,
                             skip_first_tokens=a.skip_first_tokens))
    print(f"sample {i}/{a.samples}")

records = acc.finalize()
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["block", "bits", "mean_local", "cvar_tail", "mean_prop", "trps"])
    w.writeheader()
    for r in records:
        w.writerow(r.to_dict())
meta = vars(a) | {"corpus_tokens": int(stream.numel()), "pooled_tokens": acc.n_tokens}
with open(a.out + ".meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"saved {a.out} (pooled over {acc.n_tokens} calibration tokens)")
