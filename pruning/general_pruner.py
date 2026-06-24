"""General-purpose structured channel pruner (Torch-Pruning style orchestrator).

High-level API:
    1. :func:`prune_model` — trace → build groups → select keep indices → prune
       (all-in-one, returns the pruned model + report).
    2. :class:`GeneralPruner` — stateful builder for multi-step workflows.

The pruner orchestrates:
    * Runtime tracing (:mod:`heal_compress.tracer.generic_tracer`)
    * Op-graph construction (:func:`heal_compress.tracer.op_graph.build_op_graph`)
    * Propagation (:class:`heal_compress.pruning.propagation.GroupBuilder`)
    * Keep-index selection (importance-driven or uniform)
    * Group checking (:func:`heal_compress.pruning.group_checker.check_pruning_group`)
    * Atomic pruning (:meth:`PruningGroup.prune <heal_compress.tracer.pruning_group.PruningGroup.prune>`)
    * Post-surgery legality (:func:`heal_compress.pruning.group_checker.check_model_legality`)

Usage::

    from heal_compress.pruning.general_pruner import prune_model

    pruned, report = prune_model(
        model,
        sample_input,
        prune_ratio=0.25,
        importance_scores=None,  # or dict{layer_name: per-channel scores}
        align=16,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..tracer.generic_tracer import trace_model
from ..tracer.op_graph import build_op_graph
from ..tracer.pruning_group import PruningGroup
from .group_checker import check_model_legality, check_pruning_group
from .propagation import GroupBuilder

logger = logging.getLogger(__name__)


def _aligned_keep_count(channels: int, prune_ratio: float, align: int, min_channels: int) -> int:
    """Compute keep count for a channel dimension respecting alignment."""
    if channels <= 0:
        raise ValueError(f"invalid channels: {channels}")
    target = int(round(channels * (1.0 - prune_ratio)))
    target = max(min_channels, min(channels, target))
    if align > 1 and target >= align:
        aligned = target - (target % align)
        if aligned >= min_channels:
            target = aligned
    if target <= 0:
        target = min(channels, max(1, min_channels))
    return min(channels, target)


def _importance_keep_indices(importance: torch.Tensor, keep_count: int) -> List[int]:
    """Select top-k channels by importance score."""
    keep_count = int(keep_count)
    if keep_count >= importance.numel():
        return list(range(int(importance.numel())))
    _, top_idx = torch.topk(importance.float().cpu(), keep_count, largest=True, sorted=False)
    return sorted(int(v) for v in top_idx.tolist())


def _group_aligned_keep_indices(
    group: PruningGroup,
    prune_ratio: float,
    align: int,
    min_channels: int,
    importance_scores: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Select keep indices for a group, respecting grouped-conv constraints.

    For grouped keep_groups: returns a set of indices that uses the same local
    pattern across all groups. For other groups: uses importance if available,
    else uniform L1 fallback.
    """
    c = group.num_channels
    if c <= min_channels:
        return list(range(c))

    # Check if any item is a grouped keep_groups handler -> enforce shared local.
    groups_val = 1
    for item in group.items:
        if getattr(item.pruning_fn, "__name__", "") == "prune_grouped_keep_groups":
            groups_val = item.module.groups
            break

    target = _aligned_keep_count(c, prune_ratio, max(align, groups_val), min_channels)

    # Gather importance if available.
    scores: List[torch.Tensor] = []
    if importance_scores:
        for item in group.items:
            if item.direction == "out" and item.name in importance_scores:
                raw = importance_scores[item.name]
                if isinstance(raw, (list, tuple)):
                    raw = torch.as_tensor(raw, dtype=torch.float32)
                if torch.is_tensor(raw) and int(raw.numel()) == c:
                    scores.append(raw.detach().float().cpu())

    if scores:
        agg = torch.stack(scores).mean(dim=0)
    else:
        # Fallback: uniform or L1 from first root module.
        agg = torch.arange(c, dtype=torch.float32)
        for item in group.items:
            if item.direction == "out" and item.reason in ("root_out", "grouped_conv:keep_groups"):
                module = item.module
                if isinstance(module, nn.Conv2d):
                    agg = module.weight.detach().abs().sum(dim=(1, 2, 3)).cpu()
                elif isinstance(module, nn.Linear):
                    agg = module.weight.detach().abs().sum(dim=1).cpu()
                break

    # For grouped keep_groups, enforce shared local pattern.
    if groups_val > 1 and c % groups_val == 0 and target % groups_val == 0:
        per = c // groups_val
        keep_per = target // groups_val
        # Aggregate scores within each group to pick the top local indices.
        local_scores = agg.view(groups_val, per).sum(dim=0)
        _, local_idx = torch.topk(local_scores, keep_per, largest=True, sorted=False)
        local_keep = sorted(int(v) for v in local_idx.tolist())
        keep: List[int] = []
        for gi in range(groups_val):
            keep.extend(gi * per + idx for idx in local_keep)
        return sorted(keep)

    return _importance_keep_indices(agg, target)


class GeneralPruner:
    """Stateful general pruner for multi-step workflows.

    Args:
        model: The model to prune (modified in place).
        prune_ratio: Target fraction of channels to remove (0.0–1.0).
        align: Hardware channel alignment constraint.
        min_channels: Minimum channels per dimension (safety floor).
        grouped_conv_mode: ``"keep_groups"`` or ``"remove_groups"``.
        protected_layers: Explicit layer names whose output must not change.
    """

    def __init__(
        self,
        model: nn.Module,
        prune_ratio: float = 0.0,
        align: int = 16,
        min_channels: int = 8,
        grouped_conv_mode: str = "keep_groups",
        protected_layers: Optional[List[str]] = None,
    ):
        self.model = model
        self.prune_ratio = float(prune_ratio)
        self.align = int(align)
        self.min_channels = int(min_channels)
        self.grouped_conv_mode = grouped_conv_mode
        self.protected_layers = protected_layers or []
        self.groups: List[PruningGroup] = []
        self.report: Dict[str, Any] = {}

    def trace_and_build_groups(
        self,
        sample_input: Any,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    ) -> List[PruningGroup]:
        """Step 1: trace the model and build pruning groups."""
        logger.info("Tracing model...")
        trace = trace_model(self.model, sample_input, forward_fn=forward_fn)
        op_graph = build_op_graph(trace, self.model, protected_layers=self.protected_layers)
        logger.info("Building pruning groups...")
        builder = GroupBuilder(op_graph, align=self.align, grouped_conv_mode=self.grouped_conv_mode)
        self.groups = builder.build()
        return self.groups

    def prune(
        self,
        importance_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Step 2: select keep indices, check, and atomically prune each group.

        Returns a report dict with ``applied``, ``skipped``, and ``legality``.
        """
        applied: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        for group in self.groups:
            if group.protected:
                skipped.append({
                    "group_id": group.group_id,
                    "reason": group.protected_reason,
                    "num_channels": group.num_channels,
                })
                continue
            keep = _group_aligned_keep_indices(
                group, self.prune_ratio, self.align, self.min_channels, importance_scores
            )
            if len(keep) >= group.num_channels:
                # No pruning needed.
                continue
            check = check_pruning_group(group, keep)
            if not check["legal"]:
                skipped.append({
                    "group_id": group.group_id,
                    "reason": "check_failed",
                    "issues": check["issues"],
                    "num_channels": group.num_channels,
                })
                continue
            result = group.prune(keep)
            if result["applied"]:
                applied.append({
                    "group_id": group.group_id,
                    "num_channels": group.num_channels,
                    "kept": len(keep),
                    "operations": result["operations"],
                })
            else:
                skipped.append({
                    "group_id": group.group_id,
                    "reason": result.get("skipped_reason", "unknown"),
                    "num_channels": group.num_channels,
                })

        legality = check_model_legality(self.model)
        self.report = {
            "applied": applied,
            "skipped": skipped,
            "num_groups_applied": len(applied),
            "num_groups_skipped": len(skipped),
            "legality": legality,
        }
        logger.info(
            "Pruning complete: %d groups applied, %d skipped. Model legal: %s",
            len(applied), len(skipped), legality["legal"],
        )
        return self.report


def prune_model(
    model: nn.Module,
    sample_input: Any,
    prune_ratio: float = 0.25,
    importance_scores: Optional[Dict[str, Any]] = None,
    align: int = 16,
    min_channels: int = 8,
    grouped_conv_mode: str = "keep_groups",
    protected_layers: Optional[List[str]] = None,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
) -> tuple[nn.Module, Dict[str, Any]]:
    """One-shot general channel pruning (trace → build → prune).

    Args:
        model: The model to prune (modified in place).
        sample_input: Example input for tracing.
        prune_ratio: Fraction of channels to remove (0.0–1.0).
        importance_scores: Optional dict mapping layer name -> per-channel scores
            (tensor or list). If None, falls back to L1 weight magnitudes.
        align: Hardware channel alignment.
        min_channels: Minimum channels per dimension.
        grouped_conv_mode: ``"keep_groups"`` or ``"remove_groups"``.
        protected_layers: Explicit protected layer names.
        forward_fn: Custom forward callable ``(model, sample) -> output``.

    Returns:
        (pruned_model, report_dict)
    """
    pruner = GeneralPruner(
        model,
        prune_ratio=prune_ratio,
        align=align,
        min_channels=min_channels,
        grouped_conv_mode=grouped_conv_mode,
        protected_layers=protected_layers,
    )
    pruner.trace_and_build_groups(sample_input, forward_fn=forward_fn)
    report = pruner.prune(importance_scores=importance_scores)
    return model, report
