#!/usr/bin/env python3
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM
from crispybits_quant.model_utils import get_transformer_blocks, pack_block

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--bitmap", required=True, help="allocation.json or JSON list of per-block bits")
p.add_argument("--out", required=True)
p.add_argument("--group-size", type=int, default=128)
p.add_argument("--dtype", choices=["float16","bfloat16"], default="float16")
a = p.parse_args()

dtype = torch.float16 if a.dtype == "float16" else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map="cpu")
with open(a.bitmap) as f: obj = json.load(f)
bitmap = obj["bitmap"] if isinstance(obj, dict) else obj
blocks = get_transformer_blocks(model)
assert len(blocks) == len(bitmap)

os.makedirs(a.out, exist_ok=True)
manifest = {"model": a.model, "group_size": a.group_size, "bitmap": bitmap, "blocks": []}
for i,(block,bits) in enumerate(zip(blocks, bitmap)):
    packed = pack_block(block, int(bits), a.group_size)
    path = os.path.join(a.out, f"block_{i:03d}.pt")
    torch.save(packed, path)
    manifest["blocks"].append(os.path.basename(path))
with open(os.path.join(a.out, "manifest.json"), "w") as f: json.dump(manifest, f, indent=2)
print(f"packed {len(blocks)} blocks -> {a.out}")
