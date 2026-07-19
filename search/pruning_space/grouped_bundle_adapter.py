"""Expand legal pruning actions into formal SamplingPruningRequest objects."""

from __future__ import annotations

from typing import Sequence

from pruning.types import SamplingPruningEntry, SamplingPruningRequest

from .action_catalog import PruningSearchAction


def request_from_pruning_actions(actions: Sequence[PruningSearchAction]) -> SamplingPruningRequest:
    entries: list[SamplingPruningEntry] = []
    channel_cost = 0
    for action in actions:
        grouped_module_path = str(
            action.constraints.get("grouped_module_path")
            or (action.root_module_path if action.kind == "grouped_bundle" else "")
        )
        closure_rows = list(action.closure_entries) or [
            {
                "module_path": action.root_module_path,
                "axis": action.root_axis,
                "indices": list(action.root_indices),
                "dependency_types": ["root_output"],
            }
        ]
        root_seen = False
        for order, closure in enumerate(sorted(closure_rows, key=lambda row: (str(row.get("module_path", "")), str(row.get("axis", ""))))):
            module_path = str(closure.get("module_path", ""))
            axis = str(closure.get("axis", ""))
            is_root = module_path == action.root_module_path and axis == action.root_axis
            root_seen = root_seen or is_root
            carries_group_map = (
                action.kind == "grouped_bundle"
                and bool(grouped_module_path)
                and module_path == grouped_module_path
                and axis in {"in", "out", "channel"}
            )
            entries.append(
                SamplingPruningEntry(
                    request_id=f"search_action::{action.action_id}::{order:04d}",
                    scope_id=action.scope_id,
                    module_path=module_path,
                    axis=axis,
                    prune_indices=[int(value) for value in closure.get("indices", [])],
                    source_atomic_unit_ids=list(action.source_atomic_unit_ids),
                    group_keep_map=dict(action.group_keep_map) if carries_group_map else {},
                    group_prune_map=dict(action.group_prune_map) if carries_group_map else {},
                    metadata={
                        "legal_pruning_action_id": action.action_id,
                        "source_coupled_unit_ids": list(action.source_coupled_unit_ids),
                        "constraints": dict(action.constraints),
                        "dependency_types": list(closure.get("dependency_types", [])),
                        "closure_index_map": {
                            int(root): [int(value) for value in values]
                            for root, values in dict(closure.get("closure_index_map", {}) or {}).items()
                        },
                    },
                )
            )
        if not root_seen:
            entries.append(
                SamplingPruningEntry(
                    request_id=f"search_action::{action.action_id}::root",
                    scope_id=action.scope_id,
                    module_path=action.root_module_path,
                    axis=action.root_axis,
                    prune_indices=list(action.root_indices),
                    source_atomic_unit_ids=list(action.source_atomic_unit_ids),
                    group_keep_map=dict(action.group_keep_map) if action.kind == "grouped_bundle" else {},
                    group_prune_map=dict(action.group_prune_map) if action.kind == "grouped_bundle" else {},
                    metadata={"legal_pruning_action_id": action.action_id, "constraints": dict(action.constraints)},
                )
            )
        channel_cost += len(action.root_indices)
    return SamplingPruningRequest(
        entries=entries,
        selected_atomic_unit_ids=[action.action_id for action in actions],
        requested_channel_cost=channel_cost,
        requested_parameter_cost=0,
        one_shot=True,
        selector="two_stage_joint_search_legal_actions",
    )
