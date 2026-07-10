"""Precision coupling tracer for mixed-precision TensorRT/QDQ workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch.nn as nn


PRECISIONS = ("fp32", "fp16", "int8")
UNSUPPORTED_INT8_KEYWORDS = ("scatter", "bev_pool", "bevpool", "warp", "plugin", "voxel", "pillar_vfe")
HEAD_KEYWORDS = ("head", "cls", "reg", "dir", "obj")


@dataclass
class PrecisionGroup:
    precision_group_id: str
    member_modules: list[str]
    reason: str
    allowed_precisions: list[str]
    default_precision: str
    force_same_precision: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _node_by_name(dependency_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node.get("name") or node.get("node_id")): node for node in dependency_graph.get("nodes", [])}


def _producer_modules(name: str, nodes: dict[str, dict[str, Any]], seen: set[str] | None = None) -> list[str]:
    seen = seen or set()
    if name in seen:
        return []
    seen.add(name)
    node = nodes.get(name)
    if not node:
        return []
    module = str(node.get("module_name") or "")
    if module:
        return [module]
    out: list[str] = []
    for parent in node.get("inputs", []) or []:
        out.extend(_producer_modules(str(parent), nodes, seen))
    return out


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _module_supported_precisions(name: str, module: nn.Module, *, allow_head_int8: bool) -> tuple[list[str], str, str]:
    low = name.lower()
    if any(key in low for key in UNSUPPORTED_INT8_KEYWORDS):
        return ["fp32", "fp16"], "fp16", "unsupported_int8"
    if any(key in low for key in HEAD_KEYWORDS) and not allow_head_int8:
        return ["fp32", "fp16"], "fp16", "head_constraint"
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return list(PRECISIONS), "fp16", "user_constraint"
    return ["fp32", "fp16"], "fp16", "unsupported_int8"


def build_precision_coupling_groups(
    model: nn.Module,
    dependency_graph: dict[str, Any],
    sample_batch: Any | None = None,
    *,
    allow_head_int8: bool = False,
) -> list[PrecisionGroup]:
    nodes = _node_by_name(dependency_graph)
    groups: list[PrecisionGroup] = []
    covered: set[str] = set()
    idx = 0
    for node in dependency_graph.get("nodes", []) or []:
        op_type = str(node.get("op_type", "")).lower()
        if "add" not in op_type and "concat" not in op_type:
            continue
        members = _unique(
            module
            for input_name in node.get("inputs", []) or []
            for module in _producer_modules(str(input_name), nodes)
        )
        if len(members) < 2:
            continue
        reason = "concat" if "concat" in op_type else "residual"
        group = PrecisionGroup(
            precision_group_id=f"pg_{idx:04d}",
            member_modules=members,
            reason=reason,
            allowed_precisions=list(PRECISIONS),
            default_precision="fp16",
            force_same_precision=False,
        )
        groups.append(group)
        covered.update(members)
        idx += 1

    for name, module in model.named_modules():
        if not name or name in covered:
            continue
        allowed, default, reason = _module_supported_precisions(name, module, allow_head_int8=allow_head_int8)
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.Identity)):
            continue
        groups.append(
            PrecisionGroup(
                precision_group_id=f"pg_{idx:04d}",
                member_modules=[name],
                reason=reason,
                allowed_precisions=allowed,
                default_precision=default,
                force_same_precision=True,
            )
        )
        idx += 1
    return groups


def precision_groups_to_json(groups: list[PrecisionGroup]) -> list[dict[str, Any]]:
    return [group.to_dict() for group in groups]
