"""Deterministic global normalized-importance selection."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from ..config import AlignmentConfig, GroupedConvConfig, SelectionConfig
from ..exceptions import PruningLegalityError
from ..types import AtomicPruneUnit, SamplingPruningEntry, SamplingPruningRequest
from .grouped_conv import select_grouped_conv_channels


def _bundle_grouped_units(
    units: Sequence[AtomicPruneUnit],
    config: GroupedConvConfig,
) -> list[AtomicPruneUnit]:
    """Convert raw grouped channels into legal, replayable scope atoms.

    One candidate is emitted for every smaller allowed width. Each candidate
    independently ranks local positions per group and carries exact keep/prune
    maps. The global selector later enforces at most one candidate per scope.
    """

    dense: list[AtomicPruneUnit] = []
    grouped: dict[str, list[AtomicPruneUnit]] = defaultdict(list)
    for unit in units:
        if bool(unit.constraints.get("grouped_conv")) and not bool(unit.constraints.get("depthwise")):
            grouped[unit.scope_id].append(unit)
        else:
            dense.append(unit)
    bundles: list[AtomicPruneUnit] = []
    for scope_id, rows in sorted(grouped.items()):
        group_counts = {int(row.constraints.get("groups") or 0) for row in rows}
        widths = {int(row.constraints.get("channels_per_group") or 0) for row in rows}
        roots = {(row.root_module_path, row.root_axis) for row in rows}
        if len(group_counts) != 1 or len(widths) != 1 or len(roots) != 1:
            raise PruningLegalityError(f"inconsistent grouped atomic metadata in scope {scope_id}")
        groups = group_counts.pop()
        width = widths.pop()
        if groups <= 0 or width <= 0 or len(rows) != groups * width:
            raise PruningLegalityError(
                f"grouped scope {scope_id} is incomplete: units={len(rows)}, groups={groups}, width={width}"
            )
        by_index: dict[int, AtomicPruneUnit] = {}
        scores: dict[int, list[float]] = {group: [float("inf")] * width for group in range(groups)}
        for row in rows:
            if len(row.root_indices) != 1:
                raise PruningLegalityError(f"raw grouped atom must own one logical channel: {row.stable_id}")
            absolute = int(row.root_indices[0])
            if absolute in by_index or not 0 <= absolute < groups * width:
                raise PruningLegalityError(f"invalid grouped logical index in scope {scope_id}: {absolute}")
            by_index[absolute] = row
            group, local = divmod(absolute, width)
            scores[group][local] = float(row.normalized_score)
        module_path, axis = roots.pop()
        for final_width in sorted(
            (value for value in config.allowed_channels_per_group if value < width),
            reverse=True,
        ):
            decision = select_grouped_conv_channels(
                scores,
                final_channels_per_group=final_width,
                config=config,
            )
            prune_indices = [
                group * width + local
                for group in range(groups)
                for local in decision.group_prune_map[group]
            ]
            selected_rows = [by_index[index] for index in prune_indices]
            closure_by_axis: dict[tuple[str, str], dict[str, object]] = {}
            for selected_row in selected_rows:
                members = list(getattr(selected_row, "members", []) or [])
                if not members:
                    members = list(
                        dict(getattr(selected_row, "metadata", {}) or {}).get(
                            "closure_members", []
                        )
                    )
                for member in members:
                    member_row = (
                        dict(member.to_dict())
                        if hasattr(member, "to_dict")
                        else dict(member)
                        if isinstance(member, dict)
                        else dict(vars(member))
                    )
                    key = (str(member_row.get("module_path", "")), str(member_row.get("axis", "")))
                    if not all(key):
                        continue
                    aggregate = closure_by_axis.setdefault(
                        key,
                        {
                            "module_path": key[0],
                            "axis": key[1],
                            "indices": [],
                            "dependency_types": [],
                            "closure_index_map": {},
                        },
                    )
                    aggregate["indices"] = sorted(
                        set(aggregate["indices"]) | {int(value) for value in member_row.get("indices", [])}
                    )
                    dependency_type = str(member_row.get("dependency_type", ""))
                    if dependency_type:
                        aggregate["dependency_types"] = sorted(
                            set(aggregate["dependency_types"]) | {dependency_type}
                        )
                    raw_index_map = member_row.get("closure_index_map") or member_row.get("index_map") or {}
                    for root_index, local_indices in dict(raw_index_map).items():
                        root = int(root_index)
                        local = sorted({int(value) for value in local_indices})
                        existing = aggregate["closure_index_map"].get(root)
                        if existing is not None and existing != local:
                            raise PruningLegalityError(
                                "conflicting_grouped_closure_index_map:"
                                f"{key[0]}:{key[1]}:root={root}:old={existing}:new={local}"
                            )
                        aggregate["closure_index_map"][root] = local
            normalized = sum(float(row.normalized_score) for row in selected_rows) / max(len(selected_rows), 1)
            finite_raw = [float(row.raw_score) for row in selected_rows if row.raw_score is not None]
            raw = sum(finite_raw) / len(finite_raw) if finite_raw else None
            bundles.append(
                AtomicPruneUnit(
                    scope_id=scope_id,
                    root_module_path=module_path,
                    root_axis=axis,
                    root_indices=prune_indices,
                    source_coupled_unit_ids=sorted(
                        {
                            source
                        for row in selected_rows
                        for source in row.source_coupled_unit_ids
                        }
                    ),
                    normalized_score=normalized,
                    raw_score=raw,
                    parameter_cost=sum(max(int(row.parameter_cost), 0) for row in selected_rows),
                    channel_cost=len(prune_indices),
                    group_keep_map=decision.group_keep_map,
                    group_prune_map=decision.group_prune_map,
                    constraints={
                        "grouped_conv": True,
                        "grouped_module_path": rows[0].constraints.get("grouped_module_path"),
                        "groups": groups,
                        "original_channel_count": groups * width,
                        "channels_per_group_before": width,
                        "channels_per_group_after": final_width,
                        "selection_policy": decision.selection_policy,
                    },
                    metadata={
                        "grouped_choice_scope": scope_id,
                        "selection_decision": decision.to_dict(),
                        "closure_entries": [
                            closure_by_axis[key]
                            for key in sorted(closure_by_axis)
                        ],
                    },
                )
            )
    return dense + bundles


def select_global_units(
    units: Sequence[AtomicPruneUnit],
    *,
    channel_budget: int | None = None,
    parameter_budget: int | None = None,
    grouped_config: GroupedConvConfig | None = None,
    selection_config: SelectionConfig | None = None,
    alignment_config: AlignmentConfig | None = None,
) -> SamplingPruningRequest:
    """Build an immutable one-shot request without changing a model."""

    if channel_budget is None and parameter_budget is None:
        raise ValueError("one global channel_budget or parameter_budget is required")
    if channel_budget is not None and channel_budget < 0:
        raise ValueError("channel_budget must be non-negative")
    if parameter_budget is not None and parameter_budget < 0:
        raise ValueError("parameter_budget must be non-negative")
    unscored = [
        str(getattr(unit, "stable_id", "<unknown>"))
        for unit in units
        if not hasattr(unit, "normalized_score")
    ]
    if unscored:
        raise PruningLegalityError(
            "trace_time_atomic_units_require_importance_scoring_before_global_selection:"
            f"{unscored[:8]}"
        )
    candidates = _bundle_grouped_units(units, grouped_config or GroupedConvConfig())
    selection_policy = selection_config or SelectionConfig()
    alignment_policy = alignment_config or AlignmentConfig()
    declared_widths: dict[str, int] = defaultdict(int)
    for unit in units:
        declared = int(unit.constraints.get("original_channel_count") or 0)
        if declared > 0:
            declared_widths[unit.scope_id] = max(declared_widths[unit.scope_id], declared)
    maximum_pruned: dict[str, int] = {}
    for scope_id, width in declared_widths.items():
        minimum = max(
            int(selection_policy.minimum_retained_channels),
            int(alignment_policy.dense_conv_channel_alignment),
        )
        by_minimum = max(width - minimum, 0)
        by_sparsity = max(int(math.floor(width * float(selection_policy.per_domain_max_sparsity))), 0)
        maximum_pruned[scope_id] = min(by_minimum, by_sparsity)
    ordered = sorted(
        (
            unit
            for unit in candidates
            if not unit.protected and math.isfinite(float(unit.normalized_score))
        ),
        key=lambda unit: (float(unit.normalized_score), unit.stable_id),
    )
    selected: list[AtomicPruneUnit] = []
    selected_grouped_scopes: set[str] = set()
    selected_scope_indices: dict[str, set[int]] = defaultdict(set)
    channel_cost = 0
    parameter_cost = 0
    for unit in ordered:
        grouped_scope = str(unit.metadata.get("grouped_choice_scope", ""))
        if grouped_scope and grouped_scope in selected_grouped_scopes:
            continue
        unit_channels = max(int(unit.channel_cost), len(unit.root_indices), 1)
        unit_parameters = max(int(unit.parameter_cost), 0)
        candidate_indices = set(int(value) for value in unit.root_indices)
        scope_limit = maximum_pruned.get(unit.scope_id)
        if scope_limit is not None and len(selected_scope_indices[unit.scope_id] | candidate_indices) > scope_limit:
            continue
        if channel_budget is not None and channel_cost + unit_channels > channel_budget:
            continue
        if parameter_budget is not None and parameter_cost + unit_parameters > parameter_budget:
            continue
        selected.append(unit)
        if grouped_scope:
            selected_grouped_scopes.add(grouped_scope)
        selected_scope_indices[unit.scope_id].update(candidate_indices)
        channel_cost += unit_channels
        parameter_cost += unit_parameters
        if channel_budget is not None and channel_cost == channel_budget:
            break
        if parameter_budget is not None and parameter_cost == parameter_budget:
            break
    entries: list[SamplingPruningEntry] = []
    for unit in selected:
        closure_rows = list(unit.metadata.get("closure_entries") or unit.metadata.get("closure_members") or [])
        if not closure_rows:
            closure_rows = [
                {
                    "module_path": unit.root_module_path,
                    "axis": unit.root_axis,
                    "indices": list(unit.root_indices),
                    "dependency_types": ["root_output"],
                }
            ]
        grouped_module_path = str(
            unit.constraints.get("grouped_module_path")
            or (unit.root_module_path if unit.constraints.get("grouped_conv") else "")
        )
        merged: dict[tuple[str, str], dict[str, object]] = {}
        for closure in closure_rows:
            row = dict(closure)
            key = (str(row.get("module_path", "")), str(row.get("axis", "")))
            if not all(key):
                continue
            aggregate = merged.setdefault(
                key,
                {
                    "indices": [],
                    "dependency_types": [],
                    "closure_index_map": {},
                },
            )
            aggregate["indices"] = sorted(
                set(aggregate["indices"]) | {int(value) for value in row.get("indices", [])}
            )
            dependency_types = row.get("dependency_types") or [row.get("dependency_type", "")]
            aggregate["dependency_types"] = sorted(
                set(aggregate["dependency_types"]) | {str(value) for value in dependency_types if value}
            )
            raw_index_map = row.get("closure_index_map") or row.get("index_map") or {}
            for root_index, local_indices in dict(raw_index_map).items():
                root = int(root_index)
                local = sorted({int(value) for value in local_indices})
                existing = aggregate["closure_index_map"].get(root)
                if existing is not None and existing != local:
                    raise RuntimeError(
                        "conflicting_global_closure_index_map:"
                        f"{key[0]}:{key[1]}:root={root}:old={existing}:new={local}"
                    )
                aggregate["closure_index_map"][root] = local
        if (unit.root_module_path, unit.root_axis) not in merged:
            merged[(unit.root_module_path, unit.root_axis)] = {
                "indices": list(unit.root_indices),
                "dependency_types": ["root_output"],
                "closure_index_map": {},
            }
        for ordering, ((module_path, axis), closure) in enumerate(sorted(merged.items())):
            is_active_root = module_path == unit.root_module_path and axis == unit.root_axis
            carries_group_map = bool(grouped_module_path and module_path == grouped_module_path and axis in {"in", "out", "channel"})
            entries.append(
                SamplingPruningEntry(
                    request_id=f"request::{unit.stable_id}::{ordering:04d}",
                    scope_id=unit.scope_id,
                    module_path=module_path,
                    axis=axis,
                    prune_indices=list(closure["indices"]),
                    source_atomic_unit_ids=[unit.stable_id],
                    group_keep_map=dict(unit.group_keep_map) if carries_group_map else {},
                    group_prune_map=dict(unit.group_prune_map) if carries_group_map else {},
                    protection_reason=unit.protection_reason,
                    metadata={
                        **unit.metadata,
                        "constraints": dict(unit.constraints),
                        "normalized_score": unit.normalized_score,
                        "raw_score": unit.raw_score,
                        "dependency_driven": not is_active_root,
                        "dependency_types": list(closure["dependency_types"]),
                        "closure_index_map": dict(sorted(closure["closure_index_map"].items())),
                        "active_root_module_path": unit.root_module_path,
                    },
                )
            )
    return SamplingPruningRequest(
        entries=entries,
        selected_atomic_unit_ids=[unit.stable_id for unit in selected],
        requested_channel_cost=channel_cost,
        requested_parameter_cost=parameter_cost,
        one_shot=True,
        selector="global_one_shot",
    )


__all__ = ["select_global_units"]
