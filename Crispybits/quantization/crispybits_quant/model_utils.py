from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List

import torch
import torch.nn as nn

from .affine import fake_quantize_weight, affine_quantize_weight
from .packing import pack_quantized


def get_transformer_blocks(model: nn.Module) -> List[nn.Module]:
    """Return decoder blocks for Hugging Face LLaMA-family or OPT models."""
    candidates = [
        ("model", "layers"),               # LlamaForCausalLM.model.layers
        ("model", "decoder", "layers"),   # OPTForCausalLM.model.decoder.layers
        ("transformer", "h"),              # fallback GPT-like
    ]
    for path in candidates:
        obj = model
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)):
            return list(obj)
    raise ValueError("Could not locate Transformer decoder blocks for this model")


def iter_block_linears(block: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


@contextmanager
def fake_quantized_block(block: nn.Module, bits: int, group_size: int = 128):
    """Temporarily fake-quantize all Linear weights in a block, then restore."""
    saved = []
    try:
        for _, linear in iter_block_linears(block):
            saved.append((linear, linear.weight.data))
            qdq = fake_quantize_weight(linear.weight.data, bits, group_size)
            linear.weight.data = qdq
        yield block
    finally:
        for linear, original in saved:
            linear.weight.data = original


def pack_block(block: nn.Module, bits: int, group_size: int = 128) -> dict:
    """Pack every Linear weight in a Transformer block for the CUDA backend."""
    result = {}
    for name, linear in iter_block_linears(block):
        qt = affine_quantize_weight(linear.weight.detach(), bits, group_size)
        pw = pack_quantized(qt)
        result[name] = {
            "packed": pw.packed.cpu(),
            "scales": pw.scales.cpu(),
            "zeros": pw.zeros.cpu(),
            "bits": bits,
            "group_size": group_size,
            "out_features": pw.out_features,
            "in_features": pw.in_features,
            "bias": None if linear.bias is None else linear.bias.detach().cpu(),
        }
    return result
