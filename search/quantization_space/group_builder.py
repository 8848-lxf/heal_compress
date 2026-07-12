"""Build search variables from existing precision-coupling groups."""

from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

from .types import QuantizationSearchGroup


def _upper_precisions(values: Sequence[str]) -> tuple[str, ...]:
    allowed = []
    for value in values:
        text = str(value).upper()
        if text in {"FP32", "FP16", "INT8"} and text not in allowed:
            allowed.append(text)
    return tuple(allowed or ["FP16"])


def _module_parameter_count(model: nn.Module, module_paths: Sequence[str]) -> int:
    modules = dict(model.named_modules())
    total = 0
    for name in module_paths:
        module = modules.get(str(name))
        if module is None:
            continue
        total += sum(int(param.numel()) for param in module.parameters(recurse=False))
    return total


def _module_weight_macs_proxy(model: nn.Module, module_paths: Sequence[str]) -> float:
    modules = dict(model.named_modules())
    total = 0.0
    for name in module_paths:
        weight = getattr(modules.get(str(name)), "weight", None)
        if weight is not None:
            total += float(weight.numel())
    return total


def build_quantization_search_groups(
    model: nn.Module,
    *,
    precision_groups: Sequence[Any],
    origin_map: Any | None = None,
) -> list[QuantizationSearchGroup]:
    """Convert tracer precision groups into stable GA search variables."""

    canonical_by_module: dict[str, list[str]] = {}
    if origin_map is not None:
        for entry in getattr(origin_map, "entries", []) or []:
            canonical_by_module.setdefault(str(entry.module_path), []).append(str(entry.canonical_node_name))
    groups: list[QuantizationSearchGroup] = []
    seen: set[str] = set()
    for ordering, raw in enumerate(precision_groups):
        group_id = str(getattr(raw, "precision_group_id", getattr(raw, "group_id", "")))
        if not group_id:
            raise RuntimeError("missing_precision_group_id")
        if group_id.startswith("search::"):
            raise RuntimeError(f"synthetic_precision_group_forbidden:{group_id}")
        if group_id in seen:
            raise RuntimeError(f"duplicate_precision_group_id:{group_id}")
        seen.add(group_id)
        modules = tuple(str(value) for value in getattr(raw, "member_modules", []))
        if not modules:
            raise RuntimeError(f"empty_precision_group:{group_id}")
        allowed = _upper_precisions(getattr(raw, "allowed_precisions", ["fp16"]))
        protected = "INT8" not in allowed
        protection_reason = "" if not protected else str(getattr(raw, "reason", "int8_not_allowed"))
        canonical_nodes = tuple(
            node
            for module_path in modules
            for node in canonical_by_module.get(module_path, [])
        )
        groups.append(
            QuantizationSearchGroup(
                group_id=group_id,
                module_paths=modules,
                canonical_node_ids=canonical_nodes,
                allowed_precisions=allowed,
                protected=protected,
                protection_reason=protection_reason,
                ordering=ordering,
                parameter_count=_module_parameter_count(model, modules),
                baseline_macs=_module_weight_macs_proxy(model, modules),
                metadata={
                    "reason": str(getattr(raw, "reason", "")),
                    "default_precision": str(getattr(raw, "default_precision", "fp16")).upper(),
                    "force_same_precision": bool(getattr(raw, "force_same_precision", True)),
                },
            )
        )
    return sorted(groups, key=lambda row: row.ordering)
