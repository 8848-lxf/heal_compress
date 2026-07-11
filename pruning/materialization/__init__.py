"""One-shot physical pruning planning, legalization, execution and replay."""

from .executor import materialize_pruning
from .grouped_conv import expand_group_keep_map, validate_grouped_materialization, validate_grouped_plan_entry
from .ledger import build_application_ledger
from .legalizer import legalize_dense_keep_count, legalize_pruning_plan
from .planner import build_physical_pruning_plan, module_axis_size
from .replay import replay_group_keep_map, replay_pruning

__all__ = [
    "build_application_ledger",
    "build_physical_pruning_plan",
    "expand_group_keep_map",
    "legalize_dense_keep_count",
    "legalize_pruning_plan",
    "materialize_pruning",
    "module_axis_size",
    "replay_group_keep_map",
    "replay_pruning",
    "validate_grouped_materialization",
    "validate_grouped_plan_entry",
]
