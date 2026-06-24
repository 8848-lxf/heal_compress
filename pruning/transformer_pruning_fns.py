"""Transformer-specific physical pruning functions.

These functions intentionally operate on existing ``nn.Linear`` modules and
small parent-module metadata only. They do not depend on torch-pruning and they
avoid native ``nn.MultiheadAttention`` surgery by default.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .pruning_fns import _index


def _tag(fn: Callable, name: str, supports: Callable[[nn.Module], bool]) -> Callable:
    fn.__name__ = name  # type: ignore[attr-defined]
    fn.supports = supports  # type: ignore[attr-defined]
    return fn


def _linear_supports(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def _prune_linear_out_transformer(module: nn.Linear, keep: list[int]) -> dict[str, Any]:
    before = module.out_features
    idx = _index(keep, module.weight.device)
    module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
    module.out_features = len(keep)
    return {"axis": "transformer_linear_out", "before": before, "after": len(keep)}


def _prune_linear_in_transformer(module: nn.Linear, keep: list[int]) -> dict[str, Any]:
    before = module.in_features
    idx = _index(keep, module.weight.device)
    module.weight = nn.Parameter(module.weight.data.index_select(1, idx).clone())
    module.in_features = len(keep)
    return {"axis": "transformer_linear_in", "before": before, "after": len(keep)}


prune_transformer_linear_out = _tag(
    _prune_linear_out_transformer, "prune_transformer_linear_out", _linear_supports
)
prune_transformer_linear_in = _tag(
    _prune_linear_in_transformer, "prune_transformer_linear_in", _linear_supports
)


def make_set_attention_metadata_fn(
    *,
    num_heads_after: int,
    inner_dim_after: int,
    head_dim: int,
) -> Callable[[nn.Module, list[int]], dict[str, Any]]:
    """Create a metadata update pruning_fn for custom attention modules.

    The returned function ignores ``keep`` and updates common attributes used by
    custom attention implementations. It is attached to a parent module as a
    group item so metadata changes remain atomic with Q/K/V/out-proj slicing.
    """

    def _set_attention_metadata(module: nn.Module, keep: list[int]) -> dict[str, Any]:
        before_heads = getattr(module, "num_heads", None)
        before_inner = getattr(module, "inner_dim", getattr(module, "all_head_size", None))
        for attr in ("num_heads", "n_heads", "heads"):
            if hasattr(module, attr):
                setattr(module, attr, int(num_heads_after))
        for attr in ("inner_dim", "all_head_size", "embed_inner_dim"):
            if hasattr(module, attr):
                setattr(module, attr, int(inner_dim_after))
        if hasattr(module, "head_dim"):
            setattr(module, "head_dim", int(head_dim))
        return {
            "axis": "attention_metadata",
            "before_heads": before_heads,
            "after_heads": int(num_heads_after),
            "before_inner_dim": before_inner,
            "after_inner_dim": int(inner_dim_after),
            "head_dim": int(head_dim),
        }

    _set_attention_metadata.__name__ = "set_attention_metadata"  # type: ignore[attr-defined]
    _set_attention_metadata.supports = lambda module: True  # type: ignore[attr-defined]
    return _set_attention_metadata

