from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple
import copy

import torch
import torch.nn as nn

from .model_utils import get_transformer_blocks, fake_quantized_block


@dataclass
class TRPSRecord:
    block: int
    bits: int
    mean_local: float
    cvar_tail: float
    mean_prop: float
    trps: float

    def to_dict(self):
        return asdict(self)


def _first_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return x[0]
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    raise TypeError(f"Cannot extract hidden tensor from block output type {type(x)}")


def tokenwise_residual_normalized_distortion(
    h_test: torch.Tensor,
    h_ref: torch.Tensor,
    h_in: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Eq. (1) in the paper, returning one value per token."""
    num = (h_test.float() - h_ref.float()).pow(2).sum(dim=-1)
    den = (h_ref.float() - h_in.float()).pow(2).sum(dim=-1) + eps
    return num / den


def cvar(values: torch.Tensor, tau: float = 0.10) -> torch.Tensor:
    """Average of the worst tau fraction (upper-tail CVaR)."""
    x = values.reshape(-1)
    k = max(1, int(round(tau * x.numel())))
    return torch.topk(x, k=k, largest=True).values.mean()


def _clone_args(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    # Inputs are only read by decoder blocks; shallow clone container structure.
    return tuple(a for a in args)


def _capture_block_inputs(model: nn.Module, model_inputs: Dict[str, torch.Tensor]):
    blocks = get_transformer_blocks(model)
    captured: Dict[int, Tuple[Tuple[Any, ...], Dict[str, Any], torch.Tensor]] = {}
    handles = []

    def make_pre(i):
        def hook(module, args, kwargs):
            hidden = args[0].detach() if args and torch.is_tensor(args[0]) else kwargs["hidden_states"].detach()
            captured[i] = (_clone_args(args), dict(kwargs), hidden)
        return hook

    for i, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(make_pre(i), with_kwargs=True))
    with torch.inference_mode():
        model(**model_inputs)
    for h in handles:
        h.remove()
    return captured


def _call_block_with_hidden(block: nn.Module, args, kwargs, hidden: torch.Tensor):
    args = list(args)
    kwargs = dict(kwargs)
    if args and torch.is_tensor(args[0]):
        args[0] = hidden
    elif "hidden_states" in kwargs:
        kwargs["hidden_states"] = hidden
    else:
        args = [hidden] + args
    return _first_tensor(block(*tuple(args), **kwargs))


def calibrate_trps_batch(
    model: nn.Module,
    model_inputs: Dict[str, torch.Tensor],
    bits_set=(2, 3, 4),
    group_size: int = 128,
    tau: float = 0.10,
    lambda_tail: float = 0.25,
    lambda_prop: float = 0.25,
    eps: float = 1e-8,
) -> List[TRPSRecord]:
    
    model.eval()
    blocks = get_transformer_blocks(model)
    cap = _capture_block_inputs(model, model_inputs)
    records: List[TRPSRecord] = []

    with torch.inference_mode():
        # Compute the FP block outputs once from captured inputs.
        fp_out = []
        for l, block in enumerate(blocks):
            args, kwargs, h_in = cap[l]
            fp_out.append(_call_block_with_hidden(block, args, kwargs, h_in))

        for l, block in enumerate(blocks):
            args, kwargs, h_in = cap[l]
            h_fp = fp_out[l]
            for bits in bits_set:
                with fake_quantized_block(block, bits, group_size):
                    h_q = _call_block_with_hidden(block, args, kwargs, h_in)

                d = tokenwise_residual_normalized_distortion(h_q, h_fp, h_in, eps)
                mean_local = d.mean()
                tail = cvar(d, tau)

                if l + 1 < len(blocks):
                    n_args, n_kwargs, _ = cap[l + 1]
                    next_block = blocks[l + 1]
                    h_next_fp = _call_block_with_hidden(next_block, n_args, n_kwargs, h_fp)
                    h_next_q = _call_block_with_hidden(next_block, n_args, n_kwargs, h_q)
                    p = tokenwise_residual_normalized_distortion(
                        h_next_q, h_next_fp, h_fp, eps
                    )
                    mean_prop = p.mean()
                else:
                    mean_prop = torch.zeros((), device=h_fp.device)

                score = mean_local + lambda_tail * tail + lambda_prop * mean_prop
                records.append(TRPSRecord(
                    block=l,
                    bits=bits,
                    mean_local=float(mean_local.item()),
                    cvar_tail=float(tail.item()),
                    mean_prop=float(mean_prop.item()),
                    trps=float(score.item()),
                ))
    return records


def aggregate_trps(records: Iterable[TRPSRecord]) -> List[TRPSRecord]:
    """Average repeated batch records for every (block,bits) pair."""
    acc: Dict[tuple[int, int], list[TRPSRecord]] = {}
    for r in records:
        acc.setdefault((r.block, r.bits), []).append(r)
    out = []
    for (block, bits), rs in sorted(acc.items()):
        n = len(rs)
        out.append(TRPSRecord(
            block=block,
            bits=bits,
            mean_local=sum(x.mean_local for x in rs) / n,
            cvar_tail=sum(x.cvar_tail for x in rs) / n,
            mean_prop=sum(x.mean_prop for x in rs) / n,
            trps=sum(x.trps for x in rs) / n,
        ))
    return out


def trps_quality_values(records: Iterable[TRPSRecord]) -> Dict[tuple[int, int], float]:
    by = {(r.block, r.bits): r.trps for r in records}
    blocks = sorted({b for b, _ in by})
    values = {}
    for l in blocks:
        base = by[(l, 2)]
        for bits in (2, 3, 4):
            values[(l, bits)] = 0.0 if bits == 2 else base - by[(l, bits)]
    return values
