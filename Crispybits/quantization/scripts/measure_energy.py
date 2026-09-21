#!/usr/bin/env python3

import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crispybits_quant.energy import NVMLPowerSampler, JetsonSysfsPowerSampler, integrate_energy
from crispybits_kernels.integration import install_bitmap, load_tuning

p=argparse.ArgumentParser()
p.add_argument("--model",required=True);p.add_argument("--packed-dir",required=True);p.add_argument("--bitmaps",required=True)
p.add_argument("--out",default="calibration.npz");p.add_argument("--prompt-file");p.add_argument("--prompt-len",type=int,default=1024);p.add_argument("--new-tokens",type=int,default=1024)
p.add_argument("--warmup",type=int,default=1);p.add_argument("--runs",type=int,default=3);p.add_argument("--hz",type=float,default=50.0);p.add_argument("--device-index",type=int,default=0)
p.add_argument("--jetson-power-path");p.add_argument("--jetson-scale",type=float,default=1e-3);p.add_argument("--tuning-json");p.add_argument("--seed",type=int,default=0)
a=p.parse_args()

torch.manual_seed(a.seed);device=f"cuda:{a.device_index}";dtype=torch.float16
tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
bitmaps=np.load(a.bitmaps);tuning=load_tuning(a.tuning_json)
text=Path(a.prompt_file).read_text() if a.prompt_file else ("The purpose of this calibration request is to measure packed low bit language model inference energy. "*300)
ids=tok(text,return_tensors="pt",truncation=True,max_length=a.prompt_len).input_ids.to(device)

if ids.shape[1]<a.prompt_len:
    reps=(a.prompt_len+ids.shape[1]-1)//ids.shape[1];ids=ids.repeat(1,reps)[:,:a.prompt_len]
else: ids=ids[:,:a.prompt_len]
if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id

def request():
    with torch.inference_mode():
        model.generate(ids,max_new_tokens=a.new_tokens,min_new_tokens=a.new_tokens,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)

sampler=JetsonSysfsPowerSampler(a.jetson_power_path,a.jetson_scale) if a.jetson_power_path else NVMLPowerSampler(a.device_index)
means=[];stds=[];all_runs=[]
for i,bm in enumerate(bitmaps):
    install_bitmap(model,a.packed_dir,bm.tolist(),device=device,dtype=dtype,tuning=tuning)
    torch.cuda.empty_cache()
    for _ in range(a.warmup): request()
    torch.cuda.synchronize()
    runs=[]
    for r in range(a.runs):
        m=integrate_energy(request,sampler,hz=a.hz,synchronize=torch.cuda.synchronize)
        runs.append(m.joules);print(f"config {i+1}/{len(bitmaps)} run {r+1}: {m.joules:.3f} J ({m.seconds:.3f}s)")
    means.append(float(np.mean(runs)));stds.append(float(np.std(runs,ddof=1)) if len(runs)>1 else 0.0);all_runs.append(runs)
    np.savez(a.out,bitmaps=bitmaps,energies=np.asarray(means),stds=np.asarray(stds),runs=np.asarray(all_runs,dtype=float))
print(f"saved {a.out}")
