"""Build legal pruning actions from formal trace atomic units."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from pruning.config import GroupedConvConfig, GroupedConvSelectionPolicy
from pruning.selection.grouped_conv import select_grouped_conv_channels

DEFAULT_GROUPED_CHANNELS_PER_GROUP = (4, 8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class PruningSearchAction:
    action_id: str
    kind: str
    source_atomic_unit_ids: tuple[str, ...]
    source_coupled_unit_ids: tuple[str, ...]
    root_module_path: str
    root_axis: str
    root_indices: tuple[int, ...]
    scope_id: str
    group_keep_map: dict[int, list[int]] = field(default_factory=dict)
    group_prune_map: dict[int, list[int]] = field(default_factory=dict)
    closure_entries: tuple[dict[str, Any], ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "source_atomic_unit_ids": list(self.source_atomic_unit_ids),
            "source_coupled_unit_ids": list(self.source_coupled_unit_ids),
            "root_module_path": self.root_module_path,
            "root_axis": self.root_axis,
            "root_indices": list(self.root_indices),
            "scope_id": self.scope_id,
            "group_keep_map": {str(key): list(values) for key, values in self.group_keep_map.items()},
            "group_prune_map": {str(key): list(values) for key, values in self.group_prune_map.items()},
            "closure_entries": list(self.closure_entries),
            "constraints": dict(self.constraints),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PruningActionCatalog:
    actions: tuple[PruningSearchAction, ...]
    raw_grouped_unit_ids: tuple[str, ...]
    grouped_action_count: int
    shared_local_mean_bundle_count: int
    independent_group_topk_bundle_count: int
    unbundleable_grouped_scopes: tuple[str, ...] = ()

    @property
    def action_ids(self) -> list[str]:
        return [action.action_id for action in self.actions]


def _member_to_dict(member: Any) -> dict[str, Any]:
    if hasattr(member, "to_dict"):
        return dict(member.to_dict())
    if isinstance(member, dict):
        return dict(member)
    return dict(vars(member))


def _closure_entries(units: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for unit in units:
        members = list(getattr(unit, "members", []) or [])
        if not members:
            members = list((getattr(unit, "metadata", {}) or {}).get("closure_members", []) or [])
        for member in members:
            row = _member_to_dict(member)
            key = (str(row.get("module_path", "")), str(row.get("axis", "")))
            if not all(key):
                continue
            out = merged.setdefault(key, {"module_path": key[0], "axis": key[1], "indices": [], "dependency_types": []})
            out["indices"] = sorted(set(out["indices"]) | {int(value) for value in row.get("indices", [])})
            dep = str(row.get("dependency_type", ""))
            if dep:
                out["dependency_types"] = sorted(set(out["dependency_types"]) | {dep})
    return tuple(merged[key] for key in sorted(merged))


def _dense_action(unit: Any) -> PruningSearchAction:
    stable_id = str(getattr(unit, "stable_id"))
    return PruningSearchAction(
        action_id=stable_id,
        kind="atomic",
        source_atomic_unit_ids=(stable_id,),
        source_coupled_unit_ids=tuple(str(value) for value in getattr(unit, "source_coupled_unit_ids", []) or []),
        root_module_path=str(getattr(unit, "root_module_path", "")),
        root_axis=str(getattr(unit, "root_axis", "")),
        root_indices=tuple(int(value) for value in getattr(unit, "root_indices", []) or []),
        scope_id=str(getattr(unit, "scope_id", "")),
        closure_entries=_closure_entries([unit]),
        constraints=dict(getattr(unit, "constraints", {}) or {}),
        metadata={"normalized_score": float(getattr(unit, "normalized_score", 0.0))},
    )


def _grouped_actions(
    units: Sequence[Any],
    *,
    mode: str,
    align: int,
    allowed_channels_per_group: Sequence[int],
) -> tuple[list[PruningSearchAction], str | None]:
    if not units:
        return [], "empty_grouped_scope"
    rows = list(units)
    constraints = dict(getattr(rows[0], "constraints", {}) or {})
    groups = int(constraints.get("groups") or 0)
    width = int(constraints.get("channels_per_group") or 0)
    if groups <= 0 or width <= 0:
        return [], "invalid_group_metadata"
    target_width = int(align)
    allowed = tuple(sorted({int(value) for value in allowed_channels_per_group}))
    if target_width not in allowed:
        return [], f"target_channels_per_group_not_allowed:{target_width}"
    if width < target_width:
        return [], f"channels_per_group_below_target:{width}<target{target_width}"
    by_index: dict[int, Any] = {}
    scores: dict[int, list[float]] = {group: [float("inf")] * width for group in range(groups)}
    for unit in rows:
        root_indices = list(getattr(unit, "root_indices", []) or [])
        if len(root_indices) != 1:
            continue
        absolute = int(root_indices[0])
        if not 0 <= absolute < groups * width:
            continue
        by_index[absolute] = unit
        group, local = divmod(absolute, width)
        scores[group][local] = float(getattr(unit, "normalized_score", 0.0))
    if len(by_index) != groups * width:
        return [], "incomplete_grouped_scope"
    policy = GroupedConvSelectionPolicy.SHARED_LOCAL_MEAN if mode == "shared_local_mean" else GroupedConvSelectionPolicy.INDEPENDENT_GROUP_TOPK
    config = GroupedConvConfig(allowed_channels_per_group=allowed, selection_policy=policy)
    decision = select_grouped_conv_channels(scores, final_channels_per_group=target_width, config=config)
    prune_indices = [
        group * width + local
        for group in range(groups)
        for local in decision.group_prune_map[group]
    ]
    selected = [by_index[index] for index in prune_indices]
    action_id = f"bundle_{str(getattr(rows[0], 'scope_id', 'grouped'))}_{decision.selection_policy}_{width}_to_{align}"
    return [
        PruningSearchAction(
            action_id=action_id,
            kind="grouped_bundle",
            source_atomic_unit_ids=tuple(str(getattr(unit, "stable_id")) for unit in selected),
            source_coupled_unit_ids=tuple(sorted({source for unit in selected for source in getattr(unit, "source_coupled_unit_ids", []) or []})),
            root_module_path=str(getattr(rows[0], "root_module_path", "")),
            root_axis=str(getattr(rows[0], "root_axis", "")),
            root_indices=tuple(prune_indices),
            scope_id=str(getattr(rows[0], "scope_id", "")),
            group_keep_map=decision.group_keep_map,
            group_prune_map=decision.group_prune_map,
            closure_entries=_closure_entries(selected),
            constraints={
                **constraints,
                "grouped_conv": True,
                "selection_policy": decision.selection_policy,
                "channels_per_group_before": width,
                "channels_per_group_after": target_width,
                "allowed_channels_per_group": list(allowed),
            },
            metadata={"selection_decision": decision.to_dict()},
        )
    ], None


def build_pruning_action_catalog(
    atomic_units: Sequence[Any],
    *,
    grouped_conv_mode: str = "shared_local_mean",
    grouped_conv_align: int = 8,
    grouped_allowed_channels_per_group: Sequence[int] = DEFAULT_GROUPED_CHANNELS_PER_GROUP,
) -> PruningActionCatalog:
    dense: list[PruningSearchAction] = []
    grouped: dict[str, list[Any]] = defaultdict(list)
    raw_grouped_ids: list[str] = []
    for unit in atomic_units:
        if bool(getattr(unit, "protected", False)):
            continue
        constraints = dict(getattr(unit, "constraints", {}) or {})
        if constraints.get("grouped_conv") and not constraints.get("depthwise"):
            grouped[str(getattr(unit, "scope_id", ""))].append(unit)
            raw_grouped_ids.append(str(getattr(unit, "stable_id")))
        else:
            dense.append(_dense_action(unit))
    bundles: list[PruningSearchAction] = []
    unbundleable: list[str] = []
    for _scope, rows in sorted(grouped.items()):
        made, reason = _grouped_actions(
            rows,
            mode=grouped_conv_mode,
            align=grouped_conv_align,
            allowed_channels_per_group=grouped_allowed_channels_per_group,
        )
        bundles.extend(made)
        if reason:
            unbundleable.append(_scope)
    actions = tuple(sorted([*dense, *bundles], key=lambda row: row.action_id))
    return PruningActionCatalog(
        actions=actions,
        raw_grouped_unit_ids=tuple(sorted(raw_grouped_ids)),
        grouped_action_count=len(bundles),
        shared_local_mean_bundle_count=sum(1 for action in bundles if action.constraints.get("selection_policy") == "shared_local_mean"),
        independent_group_topk_bundle_count=sum(1 for action in bundles if action.constraints.get("selection_policy") == "independent_group_topk"),
        unbundleable_grouped_scopes=tuple(sorted(unbundleable)),
    )
