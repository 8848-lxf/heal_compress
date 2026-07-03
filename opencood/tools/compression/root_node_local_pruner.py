"""Root-node local pruning utilities used by latency-LUT pruning smoke tests.

This module is intentionally independent of HEAL source code.  It provides the
artifact schema and deterministic selection semantics required by the v8
pruning pipeline, while the existing physical surgery still goes through the
formal `heal_compress.pruning` path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class ChannelMember:
    node: str
    module: str
    op_type: str
    axis: str
    index: int
    branch_id: str = ""
    concat_offset: int = 0


@dataclass
class CoupledChannelUnitV8:
    unit_id: str
    root_node: str
    root_module: str
    root_axis: str
    root_channel_index: int
    members: list[ChannelMember] = field(default_factory=list)
    dependency_types: list[str] = field(default_factory=list)
    is_grouped_conv_related: bool = False
    grouped_conv_info: dict[str, Any] | None = None
    importance: float | None = None


@dataclass
class RootNodeLocalPruningDomain:
    domain_id: str
    root_node: str
    root_module: str
    root_axis: str
    unit_ids: list[str]
    num_units: int
    ranking_scope: str = "root_node_local"
    cross_domain_ranking: bool = False


@dataclass
class GroupItemSpec:
    name: str
    op_type: str = "Conv"
    direction: str = "out"
    reason: str = ""
    branch_id: str = ""
    concat_offset: int = 0


@dataclass
class ScopeSpec:
    group_id: str
    root_node: str
    root_module: str
    num_channels: int
    group_type: str = ""
    items: list[GroupItemSpec] = field(default_factory=list)
    protected: bool = False
    protected_reason: str = ""
    grouped_conv_info: dict[str, Any] | None = None


def _axis_for_item(item: GroupItemSpec, group_type: str) -> str:
    if item.op_type in {"BatchNorm", "BatchNorm2d"}:
        return "bn_channel"
    if item.op_type in {"Linear", "Gemm", "MatMul"}:
        return "linear_out" if item.direction == "out" else "linear_in"
    if item.op_type in {"Add", "Sum"}:
        return "add_channel"
    if item.op_type == "Concat":
        return "concat_output_channel"
    if item.direction == "in":
        return "in_channels"
    return "out_channels"


def _member_index(item: GroupItemSpec, root_idx: int) -> int:
    if item.op_type == "Concat" or item.reason == "concat_out_to_next_conv_in":
        return int(item.concat_offset) + int(root_idx)
    return int(root_idx)


def _dependency_type(item: GroupItemSpec, group_type: str) -> str:
    reason = str(item.reason or "")
    if reason:
        return reason
    if group_type == "add":
        return "residual_add"
    if group_type == "cat":
        return "concat_branch_offset"
    if item.op_type in {"BatchNorm", "BatchNorm2d"}:
        return "conv_out_to_bn"
    if item.direction == "in":
        return "conv_out_to_next_conv_in"
    return "root_channel"


def build_coupled_channel_units_from_scope_specs(scopes: Sequence[ScopeSpec]) -> list[CoupledChannelUnitV8]:
    units: list[CoupledChannelUnitV8] = []
    for scope in scopes:
        for root_idx in range(int(scope.num_channels)):
            members: list[ChannelMember] = []
            deps: list[str] = []
            for item in scope.items:
                dep = _dependency_type(item, scope.group_type)
                axis = _axis_for_item(item, scope.group_type)
                members.append(
                    ChannelMember(
                        node=item.name,
                        module=item.name,
                        op_type=item.op_type,
                        axis=axis,
                        index=_member_index(item, root_idx),
                        branch_id=item.branch_id,
                        concat_offset=int(item.concat_offset),
                    )
                )
                deps.append(dep)
            grouped = scope.grouped_conv_info or None
            units.append(
                CoupledChannelUnitV8(
                    unit_id=f"{scope.root_node}::root_ch_{root_idx}",
                    root_node=scope.root_node,
                    root_module=scope.root_module,
                    root_axis="out_channels",
                    root_channel_index=root_idx,
                    members=members,
                    dependency_types=sorted(set(deps)),
                    is_grouped_conv_related=bool(grouped),
                    grouped_conv_info=grouped,
                )
            )
    return units


def build_root_node_local_domains(units: Sequence[CoupledChannelUnitV8]) -> list[RootNodeLocalPruningDomain]:
    by_root: dict[str, list[CoupledChannelUnitV8]] = {}
    for unit in units:
        by_root.setdefault(unit.root_node, []).append(unit)
    domains: list[RootNodeLocalPruningDomain] = []
    for root_node, root_units in sorted(by_root.items()):
        root_units = sorted(root_units, key=lambda u: int(u.root_channel_index))
        domains.append(
            RootNodeLocalPruningDomain(
                domain_id=f"root_node::{root_node}",
                root_node=root_node,
                root_module=root_units[0].root_module if root_units else root_node,
                root_axis="out_channels",
                unit_ids=[u.unit_id for u in root_units],
                num_units=len(root_units),
            )
        )
    return domains


def _aligned_keep_count(num_units: int, keep_ratio: float, align: int, min_keep_ratio: float) -> int:
    if num_units <= 0:
        return 0
    target = int(round(float(num_units) * float(keep_ratio)))
    min_keep = int(math.ceil(float(num_units) * float(min_keep_ratio)))
    target = max(min_keep, target)
    target = max(1, min(num_units, target))
    if align > 1 and target >= align and target % align != 0:
        target = min(num_units, ((target + align - 1) // align) * align)
    return max(1, min(num_units, target))


def select_units_by_domain_local_ratio(
    domains: Sequence[RootNodeLocalPruningDomain],
    units: Sequence[CoupledChannelUnitV8],
    *,
    keep_ratio: float | None = None,
    prune_ratio: float | None = None,
    align: int = 8,
    min_keep_ratio: float = 0.0,
) -> dict[str, Any]:
    if keep_ratio is None:
        keep_ratio = 1.0 - float(prune_ratio or 0.0)
    unit_by_id = {unit.unit_id: unit for unit in units}
    rows: list[dict[str, Any]] = []
    for domain in domains:
        domain_units = [unit_by_id[uid] for uid in domain.unit_ids if uid in unit_by_id]
        ranked = sorted(domain_units, key=lambda u: float(u.importance if u.importance is not None else 0.0))
        num_keep = _aligned_keep_count(len(ranked), float(keep_ratio), int(align), float(min_keep_ratio))
        num_prune = max(0, len(ranked) - num_keep)
        pruned = ranked[:num_prune]
        kept = sorted(ranked[num_prune:], key=lambda u: int(u.root_channel_index))
        rows.append(
            {
                "domain_id": domain.domain_id,
                "root_node": domain.root_node,
                "num_units": len(ranked),
                "requested_keep_ratio": float(keep_ratio),
                "requested_prune_ratio": 1.0 - float(keep_ratio),
                "align": int(align),
                "min_keep_ratio": float(min_keep_ratio),
                "actual_num_keep": len(kept),
                "actual_num_prune": len(pruned),
                "actual_keep_ratio": len(kept) / len(ranked) if ranked else 1.0,
                "kept_unit_ids": [u.unit_id for u in kept],
                "pruned_unit_ids": [u.unit_id for u in pruned],
            }
        )
    return {
        "selection_mode": "root_node_local_unit_ratio",
        "global_ranking": False,
        "module_stage_based_domain": False,
        "domains": rows,
    }


def select_grouped_conv_units(
    *,
    scores_by_group: Sequence[Sequence[float]],
    keep_ratio: float,
    mode: str = "independent_group_topk",
    align: int = 8,
) -> dict[str, Any]:
    groups = len(scores_by_group)
    per_group = len(scores_by_group[0]) if groups else 0
    if groups == 0 or per_group == 0:
        return {"mode": mode, "violations": ["empty_grouped_conv_scores"]}
    keep_count = _aligned_keep_count(per_group, keep_ratio, align, 0.0)
    keep_count = min(per_group, max(1, keep_count))
    group_keep_map: dict[str, list[int]] = {}
    if mode == "shared_local_mean":
        local_scores = []
        for local_idx in range(per_group):
            local_scores.append(sum(float(scores_by_group[g][local_idx]) for g in range(groups)) / groups)
        keep = sorted(sorted(range(per_group), key=lambda i: local_scores[i], reverse=True)[:keep_count])
        group_keep_map = {str(g): list(keep) for g in range(groups)}
    elif mode == "independent_group_topk":
        for g, scores in enumerate(scores_by_group):
            keep = sorted(sorted(range(per_group), key=lambda i: float(scores[i]), reverse=True)[:keep_count])
            group_keep_map[str(g)] = keep
    else:
        return {"mode": mode, "violations": [f"unsupported_grouped_conv_mode:{mode}"]}

    counts = {group: len(keep) for group, keep in group_keep_map.items()}
    violations: list[str] = []
    if len(set(counts.values())) > 1:
        violations.append("per_group_keep_count_not_equal")
    if any(count <= 0 for count in counts.values()):
        violations.append("group_would_be_empty")
    if align > 1 and keep_count >= align and any(count % align != 0 for count in counts.values()):
        violations.append("per_group_keep_count_not_aligned")
    return {
        "mode": mode,
        "groups_preserved": True,
        "groups_before": groups,
        "groups_after": groups,
        "per_group_before": per_group,
        "per_group_after": keep_count,
        "group_keep_map": group_keep_map,
        "per_group_keep_count_equal": len(set(counts.values())) == 1,
        "violations": violations,
    }


def units_to_json(units: Sequence[CoupledChannelUnitV8]) -> list[dict[str, Any]]:
    return [asdict(unit) for unit in units]


def domains_to_json(domains: Sequence[RootNodeLocalPruningDomain]) -> dict[str, Any]:
    return {"domains": [asdict(domain) for domain in domains]}
