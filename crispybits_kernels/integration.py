from __future__ import annotations
import json, os
from pathlib import Path
import torch
import torch.nn as nn

from .io import from_packed_dict
from .ops import PackedLinear


def _blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    raise ValueError("Supported integration targets are Hugging Face LLaMA-family and OPT causal LMs")


def _set_by_dotted_name(root: nn.Module, name: str, module: nn.Module):
    parts=name.split('.')
    parent=root
    for p in parts[:-1]: parent=getattr(parent,p)
    setattr(parent,parts[-1],module)


def install_bitmap(model: nn.Module, packed_dir: str, bitmap, device="cuda", dtype=torch.float16, tuning: dict|None=None):
   
    blocks=_blocks(model)
    if len(blocks)!=len(bitmap): raise ValueError("bitmap length does not match model")
    tuning=tuning or {}
    for i,(block,bits) in enumerate(zip(blocks,bitmap)):
        data=torch.load(Path(packed_dir)/f"block_{i:03d}_w{int(bits)}.pt",map_location="cpu",weights_only=False)
        for name,d in data.items():
            w=from_packed_dict(d,device=device,dtype=dtype)
            cfg=tuning.get(f"block.{i}.{name}",{})
            mod=PackedLinear(w,split_k=int(cfg.get("split_k",1)),rho=int(cfg.get("rho",1)))
            _set_by_dotted_name(block,name,mod)
    return model


def load_tuning(path: str|None):
    if not path: return {}
    with open(path) as f: return json.load(f)
