"""Physical pruning execution, channel alignment, and structure legality checking."""

from .physical_pruner import PhysicalPruner
from .alignment_checker import ChannelAlignmentChecker
from .legality_checker import StructureLegalityChecker

# General TP-style pruner (new)
from .general_pruner import GeneralPruner, prune_model
from .propagation import GroupBuilder
from .group_checker import check_pruning_group, check_model_legality
from .pruning_fns import get_pruning_fn, is_prunable_module
from .grouped_conv import classify_grouped_conv, grouped_conv_pruning_fn
from .selection import PruningPlan, SelectionConfig, build_pruning_plan
from .units import AtomicPruneUnit, ConcreteCoupledPruningGroup, CoupledChannelUnit
from .transformer_checker import check_transformer_group, check_transformer_model_legality
from .transformer_pruning_fns import (
    prune_transformer_linear_in,
    prune_transformer_linear_out,
)

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
