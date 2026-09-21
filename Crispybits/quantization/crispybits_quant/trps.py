from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
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


def trps_token_stats(
    model: nn.Module,
    model_inputs: Dict[str, torch.Tensor],
    bits_set=(2, 3, 4),
    group_size: int = 128,
    eps: float = 1e-8,
    skip_first_tokens: int = 0,
) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Token-level TRPS ingredients for one calibration batch.

    Returns {(block, bits): (d_tokens, p_tokens)} where both are 1-D float32
    CPU tensors with one entry per calibration token (p_tokens is None for
    the last block). Pool these across all calibration samples with
    `TRPSAccumulator` so that the mean and CVaR are taken over *calibration
    tokens*, as defined in the paper, rather than per sample.

    Strategy:
      1. Run the full model once to capture the arguments entering every block.
      2. Re-run only block l with its captured FP input after fake quantizing it.
      3. Feed both FP and quantized block-l outputs through the next FP block.

    Propagated distortion definition used here:
        p_l,b,t = ||F_{l+1}(h_l^b)-F_{l+1}(h_l^fp)||^2 /
                  (||F_{l+1}(h_l^fp)-h_l^fp||^2 + eps)

    IMPORTANT: the current manuscript names p_l,b,t but does not print its
    equation. If your experimental code used a different normalization, modify
    `p` below and add that exact equation to the paper.
    """
    model.eval()
    blocks = get_transformer_blocks(model)
    cap = _capture_block_inputs(model, model_inputs)
    s0 = int(skip_first_tokens)

    def flat(x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq] -> drop leading positions, flatten, move to CPU fp32
        return x[..., s0:].reshape(-1).float().cpu()

    out: Dict[Tuple[int, int], Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    with torch.inference_mode():
        fp_out = []
        for l, block in enumerate(blocks):
            args, kwargs, h_in = cap[l]
            fp_out.append(_call_block_with_hidden(block, args, kwargs, h_in))

        for l, block in enumerate(blocks):
            args, kwargs, h_in = cap[l]
            h_fp = fp_out[l]
            has_next = l + 1 < len(blocks)
            if has_next:
                n_args, n_kwargs, _ = cap[l + 1]
                next_block = blocks[l + 1]
                # Equals fp_out[l+1] up to kernel nondeterminism; recomputed
                # with the same call path as h_next_q so the two are comparable.
                h_next_fp = _call_block_with_hidden(next_block, n_args, n_kwargs, h_fp)
            for bits in bits_set:
                with fake_quantized_block(block, bits, group_size):
                    h_q = _call_block_with_hidden(block, args, kwargs, h_in)
                d = tokenwise_residual_normalized_distortion(h_q, h_fp, h_in, eps)
                p = None
                if has_next:
                    h_next_q = _call_block_with_hidden(next_block, n_args, n_kwargs, h_q)
                    p = flat(tokenwise_residual_normalized_distortion(
                        h_next_q, h_next_fp, h_fp, eps))
                out[(l, bits)] = (flat(d), p)
    return out


class TRPSAccumulator:
    """Pools token-level distortions across calibration samples.

    S_TRPS(l,b) = E_t[d] + lambda_tail * CVaR_tau(d) + lambda_prop * E_t[p]

    where the expectation and the CVaR run over *all* calibration tokens.
    d values are kept (needed for the exact CVaR); p only needs a running sum.
    Memory: ~4 bytes x tokens x blocks x |bits|, e.g. ~400 MB for
    500 x 2048 tokens on a 32-block model.
    """

    def __init__(self, tau: float = 0.10, lambda_tail: float = 0.25, lambda_prop: float = 0.25):
        self.tau = tau
        self.lambda_tail = lambda_tail
        self.lambda_prop = lambda_prop
        self._d: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self._p_sum: Dict[Tuple[int, int], float] = {}
        self._p_n: Dict[Tuple[int, int], int] = {}

    def add(self, stats: Dict[Tuple[int, int], Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
        for key, (d, p) in stats.items():
            self._d.setdefault(key, []).append(d)
            if p is not None:
                self._p_sum[key] = self._p_sum.get(key, 0.0) + float(p.double().sum())
                self._p_n[key] = self._p_n.get(key, 0) + int(p.numel())

    @property
    def n_tokens(self) -> int:
        if not self._d:
            return 0
        return sum(int(x.numel()) for x in next(iter(self._d.values())))

    def finalize(self) -> List[TRPSRecord]:
        records = []
        for (l, bits) in sorted(self._d):
            d = torch.cat(self._d[(l, bits)])
            mean_local = float(d.double().mean())
            tail = float(cvar(d, self.tau).double())
            n_p = self._p_n.get((l, bits), 0)
            mean_prop = self._p_sum[(l, bits)] / n_p if n_p else 0.0
            score = mean_local + self.lambda_tail * tail + self.lambda_prop * mean_prop
            records.append(TRPSRecord(l, bits, mean_local, tail, mean_prop, score))
        return records


def calibrate_trps_batch(
    model: nn.Module,
    model_inputs: Dict[str, torch.Tensor],
    bits_set=(2, 3, 4),
    group_size: int = 128,
    tau: float = 0.10,
    lambda_tail: float = 0.25,
    lambda_prop: float = 0.25,
    eps: float = 1e-8,
    skip_first_tokens: int = 0,
) -> List[TRPSRecord]:
    """TRPS for a single batch (all tokens of this batch pooled).

    For a full calibration set, do NOT average the outputs of this function
    across batches: use `trps_token_stats` + `TRPSAccumulator` instead, so the
    CVaR is taken over the worst tau fraction of all calibration tokens.
    """
    acc = TRPSAccumulator(tau, lambda_tail, lambda_prop)
    acc.add(trps_token_stats(model, model_inputs, bits_set, group_size, eps, skip_first_tokens))
    return acc.finalize()


def aggregate_trps(records: Iterable[TRPSRecord]) -> List[TRPSRecord]:
    """Average repeated batch records for every (block,bits) pair.

    Deprecated for calibration: averaging per-batch CVaRs is not the CVaR of
    the pooled calibration tokens, and it weights short batches as much as
    long ones. Kept only for backward compatibility; use TRPSAccumulator.
    """
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
