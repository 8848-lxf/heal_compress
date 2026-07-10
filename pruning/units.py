"""Fine-grained pruning units built from dependency scopes.

The existing :class:`PruningGroup` remains the dependency recipe: it defines a
reference channel space plus item-local index transforms. The dataclasses here
materialize TP-style concrete search units on top of that recipe.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from ..tracer.pruning_group import GroupItem, PruningGroup


@dataclass
class CoupledChannelUnit:
    unit_id: str
    scope_id: str
    root_idx: int
    root_node: str = ""
    root_module: str = ""
    root_axis: str = "out_channels"
    root_channel_index: int = 0
    members: list[dict[str, Any]] = field(default_factory=list)
    dependency_types: list[str] = field(default_factory=list)
    is_grouped_conv_related: bool = False
    grouped_conv_info: dict[str, Any] | None = None
    group_id_in_grouped_conv: int | None = None
    local_idx_in_group: int | None = None
    local_indices_by_item: dict[str, list[int]] = field(default_factory=dict)
    item_modules: list[str] = field(default_factory=list)
    item_directions: list[str] = field(default_factory=list)
    importance: float | None = None
    importance_mode: str | None = None
    params_removed: int = 0
    flops_removed: float | None = None
    protected: bool = False
    protected_reason: str | None = None
    is_minimal_proven: bool = False
    proof_edges: list[dict[str, Any]] = field(default_factory=list)
    unsupported_reason: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicPruneUnit:
    candidate_id: str
    scope_id: str
    candidate_type: str
    source_coupled_units: list[str] = field(default_factory=list)
    ref_indices: list[int] = field(default_factory=list)
    local_indices_by_item: dict[str, list[int]] = field(default_factory=dict)
    importance: float | None = None
    importance_mode: str | None = None
    params_removed: int = 0
    flops_removed: float | None = None
    protected: bool = False
    protected_reason: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcreteCoupledPruningGroup:
    concrete_group_id: str
    scope_id: str
    prune_indices: list[int]
    keep_indices: list[int]
    items: list[GroupItem]
    local_prune_indices_by_item: dict[str, list[int]] = field(default_factory=dict)
    local_keep_indices_by_item: dict[str, list[int]] = field(default_factory=dict)
    source_coupled_units: list[str] = field(default_factory=list)
    source_atomic_units: list[str] = field(default_factory=list)
    importance_sum: float = 0.0
    params_removed: int = 0
    protected: bool = False
    protected_reason: str | None = None
    check_group_passed: bool = False


def item_key(item: GroupItem) -> str:
    return f"{item.name}:{item.direction}"


def _scope_id(scope: Any) -> str:
    return str(getattr(scope, "group_id", getattr(scope, "scope_id", "")))


def _is_grouped_conv(module: Any) -> bool:
    return isinstance(module, nn.Conv2d) and int(getattr(module, "groups", 1)) > 1


def grouped_conv_info(scope: PruningGroup) -> dict[str, Any]:
    for item in scope.items:
        module = item.module
        if not _is_grouped_conv(module):
            continue
        groups = int(module.groups)
        channels = int(getattr(scope, "num_channels", 0))
        if channels <= 0 or channels % groups != 0:
            per_group = None
        else:
            per_group = channels // groups
        return {
            "has_grouped_conv": True,
            "module_name": item.name,
            "module": module,
            "groups": groups,
            "per_group": per_group,
            "fn_name": getattr(item.pruning_fn, "__name__", ""),
        }
    return {"has_grouped_conv": False}


def scope_constraints(scope: PruningGroup) -> dict[str, Any]:
    meta = getattr(scope, "meta", {}) or {}
    gt = str(meta.get("group_type", ""))
    grouped = grouped_conv_info(scope)
    return {
        "has_grouped_conv": bool(grouped.get("has_grouped_conv", False)),
        "grouped_conv_module": grouped.get("module_name"),
        "groups": grouped.get("groups"),
        "per_group": grouped.get("per_group"),
        "has_residual": gt == "add",
        "has_concat": gt == "cat",
        "has_transformer": gt.startswith("transformer") or "mha" in gt or "ffn" in gt,
        "group_type": gt,
    }


def _importance_at(values: Sequence[float] | torch.Tensor | None, idx: int) -> float | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        value = float(values.detach().float().cpu()[idx])
    else:
        value = float(values[idx])
    return value


def _axis_for_item(item: GroupItem) -> str:
    module = item.module
    if isinstance(module, nn.BatchNorm2d):
        return "bn_channel"
    if isinstance(module, nn.Linear):
        return "linear_out" if item.direction == "out" else "linear_in"
    if item.direction == "in":
        return "in_channels"
    return "out_channels"


def _dependency_type(scope: PruningGroup, item: GroupItem) -> str:
    reason = str(item.reason or "")
    group_type = str((scope.meta or {}).get("group_type", ""))
    if reason:
        if "concat" in reason or "cat" in reason:
            return "concat_branch_offset" if item.direction == "out" else "concat_out_to_next_conv_in"
        if "shortcut" in reason:
            return "projection_shortcut"
        if "bn" in reason:
            return "conv_out_to_bn"
        if "next" in reason or item.direction == "in":
            return "conv_out_to_next_conv_in"
        if "add" in reason or "residual" in reason:
            return "residual_add"
        return reason
    if group_type == "add":
        return "residual_add"
    if group_type == "cat":
        return "concat_branch_offset" if item.direction == "out" else "concat_out_to_next_conv_in"
    if isinstance(item.module, nn.BatchNorm2d):
        return "conv_out_to_bn"
    if item.direction == "in":
        return "conv_out_to_next_conv_in"
    return "root_channel"


def _grouped_conv_role(item: GroupItem) -> str:
    module = item.module
    if not isinstance(module, nn.Conv2d) or int(getattr(module, "groups", 1)) <= 1:
        return ""
    if int(module.groups) == int(module.in_channels) == int(module.out_channels):
        return "depthwise_conv"
    if item.direction == "in":
        return "ordinary_grouped_conv_input"
    return "ordinary_grouped_conv_output"


def _transpose_conv_role(item: GroupItem) -> str:
    if not isinstance(item.module, nn.ConvTranspose2d):
        return ""
    return "convtranspose_input" if item.direction == "in" else "convtranspose_output"


def _residual_add_id(scope: PruningGroup, item: GroupItem, dep: str) -> str:
    meta = scope.meta or {}
    if str(meta.get("group_type", "")) != "add" and "add" not in dep and "residual" not in dep:
        return ""
    reasons = meta.get("reasons", []) or []
    for reason in reasons:
        text = str(reason)
        if text.startswith("add:"):
            return text.split("add:", 1)[1]
    return str(meta.get("add_node", ""))


def _member_rows(scope: PruningGroup, root_idx: int, item: GroupItem, local_indices: Sequence[int]) -> list[dict[str, Any]]:
    group_type = str((scope.meta or {}).get("group_type", ""))
    axis = _axis_for_item(item)
    dep = _dependency_type(scope, item)
    meta = scope.meta or {}
    rows: list[dict[str, Any]] = []
    for local in local_indices:
        layout = (meta.get("cat_layout", {}) or {}).get(item.name, {}) if group_type == "cat" else {}
        concat_offset = int(layout.get("offset", 0) or 0) if group_type == "cat" and item.direction == "out" else 0
        member_axis = axis
        if group_type == "add":
            member_axis = "add_channel" if "Add" in item.name or dep == "residual_add" and item.direction == "out" else axis
        if group_type == "cat" and item.direction == "out" and concat_offset:
            member_axis = "concat_output_channel"
        rows.append(
            {
                "node": item.name,
                "module": item.name,
                "module_name": item.name,
                "op_type": item.module.__class__.__name__,
                "module_type": item.module.__class__.__name__,
                "axis": member_axis,
                "index": int(local),
                "local_index": int(local),
                "dependency_type": dep,
                "producer_tensor": str(meta.get("cat_node", "")) if group_type == "cat" and item.direction == "in" else str(meta.get("roots", [""])[0] if meta.get("roots") else ""),
                "consumer_tensor": item.name,
                "branch_id": str((scope.meta or {}).get("branch_id", "")),
                "concat_offset": int(concat_offset),
                "residual_add_id": _residual_add_id(scope, item, dep),
                "grouped_conv_role": _grouped_conv_role(item),
                "transpose_conv_role": _transpose_conv_role(item),
            }
        )
    return rows


def _proof_edges_for_item(scope: PruningGroup, root_idx: int, item: GroupItem, local_indices: Sequence[int]) -> list[dict[str, Any]]:
    root = scope.items[0].name if scope.items else _scope_id(scope)
    dep = _dependency_type(scope, item)
    return [
        {
            "src": root,
            "dst": item.name,
            "root_index": int(root_idx),
            "local_index": int(local),
            "axis": _axis_for_item(item),
            "direction": item.direction,
            "dependency_type": dep,
            "reason": item.reason,
            "proof_basis": "pruning_group_recipe",
        }
        for local in local_indices
    ]


def expand_coupled_channel_units(
    scope: PruningGroup,
    scope_importance: Sequence[float] | torch.Tensor | None = None,
    *,
    importance_mode: str | None = None,
) -> list[CoupledChannelUnit]:
    """Expand one dependency scope into one unit per root channel index."""

    sid = _scope_id(scope)
    constraints = scope_constraints(scope)
    grouped = grouped_conv_info(scope)
    groups = grouped.get("groups")
    per_group = grouped.get("per_group")
    units: list[CoupledChannelUnit] = []
    for root_idx in range(int(scope.num_channels)):
        local_by_item: dict[str, list[int]] = {}
        members: list[dict[str, Any]] = []
        proof_edges: list[dict[str, Any]] = []
        dependency_types: list[str] = []
        item_modules: list[str] = []
        item_directions: list[str] = []
        for item in scope.items:
            local = sorted(int(v) for v in item.local_keep([root_idx]))
            if local:
                local_by_item[item_key(item)] = local
                members.extend(_member_rows(scope, root_idx, item, local))
                proof_edges.extend(_proof_edges_for_item(scope, root_idx, item, local))
                dependency_types.append(_dependency_type(scope, item))
                item_modules.append(item.name)
                item_directions.append(item.direction)
        if not members:
            root_name = scope.items[0].name if scope.items else sid
            members.append(
                {
                    "node": root_name,
                    "module": root_name,
                    "op_type": "Unknown",
                    "module_name": root_name,
                    "module_type": "Unknown",
                    "axis": "out_channels",
                    "index": int(root_idx),
                    "local_index": int(root_idx),
                    "dependency_type": "root_channel",
                    "producer_tensor": "",
                    "consumer_tensor": root_name,
                    "branch_id": "",
                    "concat_offset": 0,
                    "residual_add_id": "",
                    "grouped_conv_role": "",
                    "transpose_conv_role": "",
                }
            )
            dependency_types.append("root_channel")
            proof_edges.append(
                {
                    "src": root_name,
                    "dst": root_name,
                    "root_index": int(root_idx),
                    "local_index": int(root_idx),
                    "axis": "out_channels",
                    "direction": "out",
                    "dependency_type": "root_channel",
                    "reason": "fallback_empty_members",
                    "proof_basis": "fallback",
                }
            )
        importance = _importance_at(scope_importance, root_idx)
        protected = bool(scope.protected)
        protected_reason = scope.protected_reason or None
        if importance is not None and not math.isfinite(importance):
            protected = True
            protected_reason = protected_reason or "invalid_importance"
        unsupported_reason = protected_reason or ""
        group_id: int | None = None
        local_idx: int | None = None
        if groups and per_group:
            group_id = root_idx // int(per_group)
            local_idx = root_idx % int(per_group)
        units.append(
            CoupledChannelUnit(
                unit_id=f"{sid}::idx{root_idx}",
                scope_id=sid,
                root_idx=root_idx,
                root_node=sid,
                root_module=scope.items[0].name if scope.items else sid,
                root_axis="out_channels",
                root_channel_index=root_idx,
                members=members,
                dependency_types=sorted(set(dependency_types)),
                is_grouped_conv_related=bool(grouped.get("has_grouped_conv", False)),
                grouped_conv_info={k: v for k, v in grouped.items() if k != "module"} if grouped.get("has_grouped_conv") else None,
                group_id_in_grouped_conv=group_id,
                local_idx_in_group=local_idx,
                local_indices_by_item=local_by_item,
                item_modules=item_modules,
                item_directions=item_directions,
                importance=importance,
                importance_mode=importance_mode,
                protected=protected,
                protected_reason=protected_reason,
                is_minimal_proven=bool(not protected and proof_edges),
                proof_edges=proof_edges,
                unsupported_reason=str(unsupported_reason),
                constraints=dict(constraints),
                metadata={
                    "group_type": (scope.meta or {}).get("group_type", ""),
                    "num_scope_items": len(scope.items),
                    "proof_scope": "single_trace_dependency_recipe",
                },
            )
        )
    return units


def instantiate_concrete_pruning_group(
    scope: PruningGroup,
    prune_indices: Iterable[int],
    *,
    coupled_units: Sequence[CoupledChannelUnit] | Mapping[str, CoupledChannelUnit] | None = None,
    atomic_units: Sequence[AtomicPruneUnit] | None = None,
    check_group_passed: bool = False,
) -> ConcreteCoupledPruningGroup:
    sid = _scope_id(scope)
    prune = sorted({int(idx) for idx in prune_indices if 0 <= int(idx) < int(scope.num_channels)})
    prune_set = set(prune)
    keep = [idx for idx in range(int(scope.num_channels)) if idx not in prune_set]
    local_prune = {item_key(item): item.local_keep(prune) for item in scope.items}
    local_keep = {item_key(item): item.local_keep(keep) for item in scope.items}

    if isinstance(coupled_units, Mapping):
        unit_list = list(coupled_units.values())
    else:
        unit_list = list(coupled_units or [])
    source_units = [u.unit_id for u in unit_list if int(u.root_idx) in prune_set]
    importance_sum = sum(float(u.importance or 0.0) for u in unit_list if int(u.root_idx) in prune_set)
    selected_atomic = list(atomic_units or [])
    source_atomic = [u.candidate_id for u in selected_atomic]
    params_removed = sum(int(u.params_removed or 0) for u in selected_atomic)

    return ConcreteCoupledPruningGroup(
        concrete_group_id=f"{sid}::prune{len(prune)}",
        scope_id=sid,
        prune_indices=prune,
        keep_indices=keep,
        items=list(scope.items),
        local_prune_indices_by_item=local_prune,
        local_keep_indices_by_item=local_keep,
        source_coupled_units=source_units,
        source_atomic_units=source_atomic,
        importance_sum=float(importance_sum),
        params_removed=params_removed,
        protected=bool(scope.protected),
        protected_reason=scope.protected_reason or None,
        check_group_passed=bool(check_group_passed),
    )


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def dataclass_to_json_dict(obj: Any) -> dict[str, Any]:
    data = asdict(obj)
    data.pop("items", None)
    return _jsonable(data)


def dependency_scope_rows(scopes: Sequence[PruningGroup]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        constraints = scope_constraints(scope)
        rows.append(
            {
                "scope_id": _scope_id(scope),
                "root_module": scope.items[0].name if scope.items else "",
                "num_channels": int(scope.num_channels),
                "group_type": (scope.meta or {}).get("group_type", ""),
                "num_items": len(scope.items),
                "item_modules": ";".join(item.name for item in scope.items),
                "has_grouped_conv": constraints["has_grouped_conv"],
                "has_residual": constraints["has_residual"],
                "has_concat": constraints["has_concat"],
                "has_transformer": constraints["has_transformer"],
                "protected": bool(scope.protected),
                "protected_reason": scope.protected_reason,
            }
        )
    return rows


def coupled_channel_unit_rows(units: Sequence[CoupledChannelUnit]) -> list[dict[str, Any]]:
    fields = [
        "unit_id",
        "scope_id",
        "root_idx",
        "root_node",
        "root_module",
        "root_axis",
        "root_channel_index",
        "members",
        "dependency_types",
        "is_grouped_conv_related",
        "grouped_conv_info",
        "group_id_in_grouped_conv",
        "local_idx_in_group",
        "local_indices_by_item",
        "importance",
        "importance_mode",
        "params_removed",
        "protected",
        "protected_reason",
        "is_minimal_proven",
        "proof_edges",
        "unsupported_reason",
        "constraints",
        "metadata",
    ]
    return [{field: _jsonable(getattr(unit, field)) for field in fields} for unit in units]


def atomic_prune_unit_rows(units: Sequence[AtomicPruneUnit]) -> list[dict[str, Any]]:
    fields = [
        "candidate_id",
        "scope_id",
        "candidate_type",
        "source_coupled_units",
        "ref_indices",
        "importance",
        "importance_mode",
        "params_removed",
        "protected",
        "protected_reason",
        "constraints",
        "metadata",
    ]
    return [{field: _jsonable(getattr(unit, field)) for field in fields} for unit in units]


def concrete_pruning_group_rows(groups: Sequence[ConcreteCoupledPruningGroup]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        rows.append(
            {
                "concrete_group_id": group.concrete_group_id,
                "scope_id": group.scope_id,
                "source_coupled_units": _jsonable(group.source_coupled_units),
                "source_atomic_units": _jsonable(group.source_atomic_units),
                "prune_indices": _jsonable(group.prune_indices),
                "keep_indices": _jsonable(group.keep_indices),
                "num_pruned_indices": len(group.prune_indices),
                "num_kept_indices": len(group.keep_indices),
                "importance_sum": group.importance_sum,
                "params_removed": group.params_removed,
                "protected": group.protected,
                "protected_reason": group.protected_reason,
                "check_group_passed": group.check_group_passed,
            }
        )
    return rows
