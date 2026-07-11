"""Construct a single physical plan before any model mutation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from math import prod
from typing import Any

import torch.nn as nn

from ..exceptions import PruningLegalityError, PruningPlanError
from ..policies.protection import build_protection_registry, require_direction_allowed
from ..types import (
    PhysicalPruningPlan,
    PhysicalPruningPlanEntry,
    SamplingPruningEntry,
    SamplingPruningRequest,
)


def module_axis_size(module: nn.Module, axis: str) -> int:
    """Read a logical channel axis from the unmodified live model."""

    direction = str(axis)
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if direction == "in":
            return int(module.in_channels)
        if direction in {"out", "channel"}:
            return int(module.out_channels)
    if isinstance(module, nn.Linear):
        if direction == "in":
            return int(module.in_features)
        if direction in {"out", "channel"}:
            return int(module.out_features)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        if direction in {"in", "out", "channel"}:
            return int(module.num_features)
    if isinstance(module, nn.LayerNorm) and len(module.normalized_shape) == 1:
        if direction in {"in", "out", "channel"}:
            return int(module.normalized_shape[0])
    raise PruningPlanError(f"unsupported physical pruning axis {axis!r} for {type(module).__name__}")


def _merge_group_map(
    rows: list[SamplingPruningEntry],
    attribute: str,
) -> dict[int, list[int]]:
    maps = [getattr(row, attribute) for row in rows if getattr(row, attribute)]
    if not maps:
        return {}
    canonical = [
        {int(key): sorted({int(value) for value in values}) for key, values in mapping.items()}
        for mapping in maps
    ]
    if any(mapping != canonical[0] for mapping in canonical[1:]):
        raise PruningPlanError(f"conflicting {attribute} values for one module axis")
    return canonical[0]


def _merge_closure_index_map(rows: list[SamplingPruningEntry]) -> dict[int, list[int]]:
    """Merge an optional root-to-local mapping saved by the dependency tracer."""

    mappings: list[dict[int, list[int]]] = []
    for row in rows:
        raw = (
            row.metadata.get("closure_index_map")
            or row.metadata.get("root_to_local_map")
            or row.metadata.get("index_map")
        )
        if not raw:
            continue
        mappings.append(
            {int(key): sorted({int(value) for value in values}) for key, values in raw.items()}
        )
    if not mappings:
        return {}
    if any(mapping != mappings[0] for mapping in mappings[1:]):
        raise PruningPlanError("conflicting closure index maps for one module axis")
    return mappings[0]


def build_physical_pruning_plan(
    model: nn.Module,
    request: SamplingPruningRequest,
) -> PhysicalPruningPlan:
    """Merge closures and freeze every logical index against one model state."""

    if not request.one_shot:
        raise PruningPlanError("formal physical materialization requires a one-shot request")
    modules = dict(model.named_modules())
    protection = build_protection_registry(model)
    grouped: dict[tuple[str, str], list[SamplingPruningEntry]] = defaultdict(list)
    for row in request.entries:
        grouped[(row.module_path, row.axis)].append(row)
    plan_entries: list[PhysicalPruningPlanEntry] = []
    for (module_path, axis), rows in sorted(grouped.items()):
        module = modules.get(module_path)
        if module is None:
            raise PruningPlanError(f"request references unknown module: {module_path}")
        policy = protection[module_path]
        allowed, reason = require_direction_allowed(
            policy,
            axis,
            dependency_driven=(str(axis) == "in"),
        )
        if not allowed:
            raise PruningLegalityError(
                f"protected module axis cannot be pruned: {module_path}:{axis}: {reason}"
            )
        size = module_axis_size(module, axis)
        prune = sorted({index for row in rows for index in row.prune_indices})
        if prune and (prune[0] < 0 or prune[-1] >= size):
            raise PruningPlanError(
                f"prune indices out of bounds for {module_path}:{axis}, size={size}: {prune}"
            )
        if len(prune) >= size:
            raise PruningLegalityError(f"request would remove every channel from {module_path}:{axis}")
        keep = [index for index in range(size) if index not in set(prune)]
        plan_entries.append(
            PhysicalPruningPlanEntry(
                module_path=module_path,
                axis=axis,
                prune_indices=prune,
                keep_indices=keep,
                original_axis_size=size,
                source_request_ids=sorted({row.request_id for row in rows}),
                group_keep_map=_merge_group_map(rows, "group_keep_map"),
                group_prune_map=_merge_group_map(rows, "group_prune_map"),
                metadata={
                    "scope_ids": sorted({row.scope_id for row in rows}),
                    "fixed_output_contract": policy.fixed_output_contract,
                    "protection_reason": policy.protection_reason,
                    "dependency_driven": str(axis) == "in",
                    "closure_index_map": _merge_closure_index_map(rows),
                },
            )
        )
    return PhysicalPruningPlan(
        entries=plan_entries,
        source_request=request,
        indices_frozen_before_materialization=True,
    )


def estimate_physical_parameter_count(model: nn.Module, plan: PhysicalPruningPlan) -> int:
    """Predict snapshot-v2 parameter count from a frozen plan without mutation."""

    by_module: dict[str, dict[str, PhysicalPruningPlanEntry]] = defaultdict(dict)
    for entry in plan.entries:
        by_module[entry.module_path][entry.axis] = entry
    total = 0
    supported = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.modules.batchnorm._BatchNorm)
    for module_path, module in model.named_modules():
        if not module_path or not isinstance(module, supported):
            continue
        axes = by_module.get(module_path, {})
        input_entry = axes.get("in")
        output_entry = axes.get("out") or axes.get("channel")
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            input_width = len(input_entry.keep_indices) if input_entry is not None else int(module.in_channels)
            output_width = len(output_entry.keep_indices) if output_entry is not None else int(module.out_channels)
            groups = int(module.groups)
            if groups == int(module.in_channels) == int(module.out_channels) and (input_entry or output_entry):
                groups = input_width
            if groups <= 0 or input_width % groups or output_width % groups:
                raise PruningLegalityError(
                    f"predicted grouped shape is illegal for {module_path}: in={input_width}, out={output_width}, groups={groups}"
                )
            kernel = int(prod(int(value) for value in module.weight.shape[2:]))
            if isinstance(module, nn.ConvTranspose2d):
                total += input_width * (output_width // groups) * kernel
            else:
                total += output_width * (input_width // groups) * kernel
            if module.bias is not None:
                total += output_width
        elif isinstance(module, nn.Linear):
            input_width = len(input_entry.keep_indices) if input_entry is not None else int(module.in_features)
            output_width = len(output_entry.keep_indices) if output_entry is not None else int(module.out_features)
            total += input_width * output_width
            if module.bias is not None:
                total += output_width
        else:
            row = axes.get("channel") or axes.get("out") or axes.get("in")
            width = len(row.keep_indices) if row is not None else int(module.num_features)
            if module.weight is not None:
                total += width
            if module.bias is not None:
                total += width
    return int(total)


__all__ = ["build_physical_pruning_plan", "estimate_physical_parameter_count", "module_axis_size"]
