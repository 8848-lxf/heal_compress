"""Resolve trace units to exact parameter slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class ParameterSlice:
    parameter_name: str
    module_path: str
    axis: int
    indices: tuple[int, ...]
    operation: str


def _weight_axis(module: nn.Module, logical_axis: str) -> int | None:
    axis = str(logical_axis)
    if isinstance(module, nn.ConvTranspose2d):
        if axis == "in":
            return 0
        if axis in {"out", "channel"}:
            return 1
    if isinstance(module, nn.Conv2d):
        if axis in {"out", "channel"}:
            return 0
        if axis == "in":
            return 1
    if isinstance(module, nn.Linear):
        if axis in {"out", "channel"}:
            return 0
        if axis == "in":
            return 1
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        if axis in {"in", "out", "channel"}:
            return 0
    return None


def _sanitize_indices(module: nn.Module, axis: int, indices: tuple[int, ...], *, constraints: dict[str, object] | None = None) -> tuple[int, ...]:
    weight = getattr(module, "weight", None)
    if weight is None or int(axis) >= int(weight.ndim):
        return ()
    axis_size = int(weight.shape[int(axis)])
    if axis_size <= 0:
        return ()
    values = tuple(int(value) for value in indices)
    grouped = int(getattr(module, "groups", 1) or 1) > 1 or bool((constraints or {}).get("grouped_conv"))
    if grouped and int(axis) == 1 and any(value >= axis_size or value < 0 for value in values):
        values = tuple(value % axis_size for value in values)
    return tuple(sorted({value for value in values if 0 <= value < axis_size}))


def build_unit_parameter_slices(model: nn.Module, atomic_units: list[Any]) -> dict[str, list[ParameterSlice]]:
    modules = dict(model.named_modules())
    result: dict[str, list[ParameterSlice]] = {}
    for unit in atomic_units:
        rows: list[ParameterSlice] = []
        constraints = dict(getattr(unit, "constraints", {}) or {})
        for member in getattr(unit, "members", []):
            module_path = str(getattr(member, "module_path", ""))
            axis_name = str(getattr(member, "axis", ""))
            indices = tuple(int(value) for value in getattr(member, "indices", []))
            module = modules.get(module_path)
            if module is None or not indices:
                continue
            axis = _weight_axis(module, axis_name)
            if axis is None:
                continue
            if getattr(module, "weight", None) is not None:
                weight_indices = _sanitize_indices(module, axis, indices, constraints=constraints)
                if weight_indices:
                    rows.append(ParameterSlice(f"{module_path}.weight", module_path, axis, weight_indices, "prune_weight_slice"))
            if axis == 0 and getattr(module, "bias", None) is not None:
                bias_size = int(module.bias.shape[0])
                bias_indices = tuple(sorted({index for index in indices if 0 <= index < bias_size}))
                if bias_indices:
                    rows.append(ParameterSlice(f"{module_path}.bias", module_path, 0, bias_indices, "prune_bias_slice"))
        result[str(getattr(unit, "stable_id"))] = rows
    return result
