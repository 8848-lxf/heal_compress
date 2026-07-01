"""Formal structured pruning tools and compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PhysicalPruner",
    "ChannelAlignmentChecker",
    "StructureLegalityChecker",
    "GeneralPruner",
    "prune_model",
    "GroupBuilder",
    "check_pruning_group",
    "check_model_legality",
    "get_pruning_fn",
    "is_prunable_module",
    "classify_grouped_conv",
    "grouped_conv_pruning_fn",
    "CoupledChannelUnit",
    "AtomicPruneUnit",
    "ConcreteCoupledPruningGroup",
    "SelectionConfig",
    "PruningPlan",
    "build_pruning_plan",
    "check_transformer_group",
    "check_transformer_model_legality",
    "prune_transformer_linear_in",
    "prune_transformer_linear_out",
]

_EXPORTS = {
    "PhysicalPruner": ".physical_pruner",
    "ChannelAlignmentChecker": ".alignment_checker",
    "StructureLegalityChecker": ".legality_checker",
    "GeneralPruner": ".general_pruner",
    "prune_model": ".general_pruner",
    "GroupBuilder": ".propagation",
    "check_pruning_group": ".group_checker",
    "check_model_legality": ".group_checker",
    "get_pruning_fn": ".pruning_fns",
    "is_prunable_module": ".pruning_fns",
    "classify_grouped_conv": ".grouped_conv",
    "grouped_conv_pruning_fn": ".grouped_conv",
    "SelectionConfig": ".selection",
    "PruningPlan": ".selection",
    "build_pruning_plan": ".selection",
    "AtomicPruneUnit": ".units",
    "ConcreteCoupledPruningGroup": ".units",
    "CoupledChannelUnit": ".units",
    "check_transformer_group": ".transformer_checker",
    "check_transformer_model_legality": ".transformer_checker",
    "prune_transformer_linear_in": ".transformer_pruning_fns",
    "prune_transformer_linear_out": ".transformer_pruning_fns",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    return getattr(module, name)
