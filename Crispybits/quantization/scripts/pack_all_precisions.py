#!/usr/bin/env python3
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM
from crispybits_quant.model_utils import get_transformer_blocks, pack_block

p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--out",required=True);p.add_argument("--group-size",type=int,default=128);p.add_argument("--dtype",choices=["float16","bfloat16"],default="float16");a=p.parse_args()
dtype=torch.float16 if a.dtype=="float16" else torch.bfloat16
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,device_map="cpu").eval();blocks=get_transformer_blocks(model);os.makedirs(a.out,exist_ok=True)
for i,block in enumerate(blocks):
    for bits in (2,3,4):
        path=os.path.join(a.out,f"block_{i:03d}_w{bits}.pt")
        torch.save(pack_block(block,bits,a.group_size),path)
        print(path)
with open(os.path.join(a.out,"manifest.json"),"w") as f: json.dump({"model":a.model,"blocks":len(blocks),"group_size":a.group_size,"precisions":[2,3,4]},f,indent=2)
