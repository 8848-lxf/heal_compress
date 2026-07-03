from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .full_engine_precision import normalize_precision_label, split_precision_config
from .precision_constraint_graph import (
    FIXED_PRECISION,
    MUST_SAME_DTYPE_BEFORE_OP,
    MUST_SAME_PRECISION,
    PrecisionConstraintGraph,
)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in list(self.parent):
            out.setdefault(self.find(item), []).append(item)
        return out


def _priority(precision: str) -> int:
    return {"FP16": 1, "FP32": 2, "INT8": 3}[precision]


def _best_float_precision(values: set[str]) -> str:
    return "FP32" if "FP32" in values else "FP16"


def resolve_precision_constraints(
    graph: PrecisionConstraintGraph,
    candidate: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    atomic_inventory: list[dict[str, Any]] | None = None,
    scale_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default, overrides = split_precision_config(candidate.get("precision_config") or {})
    node_ids = sorted(graph.nodes)
    uf = _UnionFind(node_ids)
    merge_reasons: dict[tuple[str, str], str] = {}

    for edge in graph.edges:
        if edge.edge_type in {MUST_SAME_PRECISION, MUST_SAME_DTYPE_BEFORE_OP}:
            uf.union(edge.src, edge.dst)
            merge_reasons[(edge.src, edge.dst)] = edge.reason or edge.edge_type

    requested: dict[str, str] = {node_id: default for node_id in node_ids}
    for node_id, node in graph.nodes.items():
        if node.precision:
            requested[node_id] = normalize_precision_label(node.precision)
    for key, value in overrides.items():
        requested[str(key)] = normalize_precision_label(value)
        if str(key) not in uf.parent:
            uf.parent[str(key)] = str(key)

    fixed_precision_regions: list[dict[str, Any]] = []
    for edge in graph.edges:
        if edge.edge_type == FIXED_PRECISION:
            requested[edge.src] = normalize_precision_label(edge.metadata.get("precision", requested.get(edge.src, default)))
            fixed_precision_regions.append({"region": edge.src, "precision": requested[edge.src], "reason": edge.reason})

    resolved = dict(requested)
    auto_promoted: list[dict[str, Any]] = []
    unsupported_int8: list[dict[str, Any]] = []
    unsupported_precision_regions: list[dict[str, Any]] = []

    for root, members in uf.groups().items():
        values = {requested.get(member, default) for member in members}
        if "INT8" in values and len(values) > 1:
            unsupported_int8.append(
                {
                    "region": root,
                    "members": sorted(members),
                    "requested": sorted(values, key=_priority),
                    "reason": "int8_residual_merge_not_supported",
                }
            )
            continue
        target = "INT8" if values == {"INT8"} else _best_float_precision(values)
        if len(values) > 1:
            reason = "must_same_dtype_before_op"
            for (src, dst), edge_reason in merge_reasons.items():
                if src in members and dst in members:
                    reason = edge_reason
                    break
            auto_promoted.append(
                {
                    "region": root,
                    "members": sorted(members),
                    "from": sorted(values, key=_priority),
                    "to": target,
                    "reason": reason,
                }
            )
        for member in members:
            resolved[member] = target

    inventory_by_unit = {str(unit.get("unit_id")): unit for unit in atomic_inventory or []}
    scale_units = (scale_cache or {}).get("units") or {}
    for unit_id, precision in requested.items():
        if precision != "INT8":
            continue
        unit = inventory_by_unit.get(str(unit_id))
        if unit is not None and not unit.get("int8_supported", False):
            unsupported_precision_regions.append(
                {
                    "region": unit_id,
                    "requested": "INT8",
                    "reason": "int8_unit_not_supported",
                    "unit_type": unit.get("unit_type"),
                    "int8_reason": unit.get("int8_reason"),
                }
            )
            continue
        unit_scale = scale_units.get(str(unit_id))
        if scale_cache is not None and not unit_scale:
            unsupported_precision_regions.append(
                {
                    "region": unit_id,
                    "requested": "INT8",
                    "reason": "int8_activation_scale_missing",
                }
            )
            continue
        if unit_scale is not None and not (unit_scale.get("weight_tensors") or {}):
            unsupported_precision_regions.append(
                {
                    "region": unit_id,
                    "requested": "INT8",
                    "reason": "int8_weight_scale_missing",
                }
            )

    status = "success"
    if unsupported_int8:
        status = "int8_residual_merge_not_supported"
    if unsupported_precision_regions and status == "success":
        status = str(unsupported_precision_regions[0]["reason"])
    report = {
        "candidate_id": candidate.get("candidate_id"),
        "success": not unsupported_int8 and not unsupported_precision_regions,
        "status": status,
        "requested_precision_config": candidate.get("precision_config") or {},
        "resolved_precision_config": resolved,
        "default_precision": default,
        "auto_promoted_precision_regions": auto_promoted,
        "unsupported_int8_regions": unsupported_int8,
        "unsupported_precision_regions": unsupported_precision_regions,
        "fixed_precision_regions": fixed_precision_regions,
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
