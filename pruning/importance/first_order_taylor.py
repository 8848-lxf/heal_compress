"""Normalized first-order Taylor importance for formal pruning units."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ..config import ImportanceConfig
from ..types import ImportanceResult
from .aggregation import coupled_dependency_mean
from .normalization import normalize_scope_scores


IMPLEMENTATION_VERSION = "first-order-taylor-v1"


def _gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    saved = getattr(parameter, "_importance_grad", None)
    return saved if saved is not None else parameter.grad


def _slice_score(parameter: torch.Tensor, indices: Sequence[int], axis: int) -> float:
    values = sorted({int(value) for value in indices})
    if not values:
        return 0.0
    if min(values) < 0 or max(values) >= int(parameter.shape[axis]):
        return float("inf")
    gradient = _gradient(parameter)
    if gradient is None:
        return float("inf")
    index = torch.as_tensor(values, dtype=torch.long, device=parameter.device)
    weight_slice = parameter.index_select(axis, index)
    gradient_slice = gradient.index_select(axis, index)
    return float((weight_slice * gradient_slice).abs().sum().detach().cpu())


def _vector_score(parameter: torch.Tensor | None, indices: Sequence[int]) -> float:
    if parameter is None:
        return 0.0
    return _slice_score(parameter, indices, 0)


def _grouped_conv_input_score(module: nn.Conv2d, indices: Sequence[int]) -> float:
    """Map logical grouped-convolution inputs to stored local weight columns."""

    groups = int(module.groups)
    if groups <= 1 or module.in_channels % groups or module.out_channels % groups:
        return float("inf")
    gradient = _gradient(module.weight)
    if gradient is None:
        return float("inf")
    in_per_group = module.in_channels // groups
    out_per_group = module.out_channels // groups
    total = module.weight.new_zeros(())
    for absolute_index in sorted({int(value) for value in indices}):
        if absolute_index < 0 or absolute_index >= module.in_channels:
            return float("inf")
        group_id, local_index = divmod(absolute_index, in_per_group)
        start = group_id * out_per_group
        stop = start + out_per_group
        total = total + (
            module.weight[start:stop, local_index : local_index + 1]
            * gradient[start:stop, local_index : local_index + 1]
        ).abs().sum()
    return float(total.detach().cpu())


def score_dependency_member(module: nn.Module, axis: str, indices: Sequence[int]) -> float:
    """Return ``sum(abs(w * dL/dw))`` for one dependency member."""

    direction = str(axis)
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
        weight_score = _vector_score(getattr(module, "weight", None), indices)
        bias_score = _vector_score(getattr(module, "bias", None), indices)
        return weight_score + bias_score
    parameter = getattr(module, "weight", None)
    if parameter is None:
        return float("inf")
    if isinstance(module, nn.Conv2d) and module.groups > 1 and direction == "in":
        return _grouped_conv_input_score(module, indices)
    if direction in {"out", "channel"}:
        parameter_axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
    elif direction == "in":
        parameter_axis = 0 if isinstance(module, nn.ConvTranspose2d) else 1
    else:
        return float("inf")
    score = _slice_score(parameter, indices, parameter_axis)
    if direction in {"out", "channel"} and getattr(module, "bias", None) is not None:
        score += _vector_score(module.bias, indices)
    return score


def estimate_dependency_parameter_cost(module: nn.Module, axis: str, indices: Sequence[int]) -> int:
    """Estimate parameters removed by one logical member channel slice."""

    count = len({int(value) for value in indices})
    if count <= 0:
        return 0
    direction = str(axis)
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
        width = int(getattr(module, "num_features", 0) or (module.normalized_shape[0] if len(module.normalized_shape) == 1 else 0))
        if width <= 0:
            return 0
        return sum(int(parameter.numel()) * count // width for parameter in module.parameters(recurse=False))
    parameter = getattr(module, "weight", None)
    if parameter is None:
        return 0
    if direction == "in":
        width = int(getattr(module, "in_channels", getattr(module, "in_features", 0)) or 0)
    elif direction in {"out", "channel"}:
        width = int(getattr(module, "out_channels", getattr(module, "out_features", 0)) or 0)
    else:
        return 0
    if width <= 0:
        return 0
    cost = int(parameter.numel()) * count // width
    if direction in {"out", "channel"} and getattr(module, "bias", None) is not None:
        cost += int(module.bias.numel()) * count // width
    return int(cost)


def estimate_unit_parameter_cost(model: nn.Module, unit: Any) -> int:
    """Sum unique dependency-member slice costs for one coupled unit."""

    modules = dict(model.named_modules())
    total = 0
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    for member in getattr(unit, "members", []):
        module_path = str(member.module_path)
        key = (module_path, str(member.axis), tuple(sorted({int(value) for value in member.indices})))
        if key in seen or module_path not in modules:
            continue
        seen.add(key)
        total += estimate_dependency_parameter_cost(modules[module_path], str(member.axis), member.indices)
    return int(total)


def score_first_order_taylor(
    model: nn.Module,
    units: Sequence[Any],
    *,
    config: ImportanceConfig | None = None,
    calibration_batches: int = 0,
    task_loss: str = "",
) -> ImportanceResult:
    """Score coupled units and normalize within each dependency scope.

    A member contributes the sum of elementwise ``abs(weight * gradient)``.
    The raw coupled-unit score is the arithmetic mean across its dependency
    members. Raw unit scores are divided by the finite mean of their scope.
    """

    cfg = config or ImportanceConfig()
    modules = dict(model.named_modules())
    raw_scores: dict[str, float] = {}
    scope_ids: dict[str, list[str]] = defaultdict(list)
    unit_rows: list[dict[str, Any]] = []
    for unit in units:
        dependency_rows: list[dict[str, Any]] = []
        member_scores: list[float] = []
        for member in getattr(unit, "members", []):
            module_path = str(member.module_path)
            module = modules.get(module_path)
            score = (
                score_dependency_member(module, str(member.axis), member.indices)
                if module is not None
                else float("inf")
            )
            dependency_rows.append(
                {
                    "module_path": module_path,
                    "axis": str(member.axis),
                    "indices": list(member.indices),
                    "raw_score": score,
                    "finite": math.isfinite(score),
                }
            )
            if math.isfinite(score):
                member_scores.append(score)
        stable_id = str(unit.stable_id)
        scope_id = str(unit.scope_id)
        raw_scores[stable_id] = coupled_dependency_mean(member_scores)
        scope_ids[scope_id].append(stable_id)
        unit_rows.append(
            {
                "stable_id": stable_id,
                "scope_id": scope_id,
                "root_module_path": str(unit.root_module_path),
                "root_channel_index": int(getattr(unit, "root_channel_index", -1)),
                "dependency_scores": dependency_rows,
            }
        )
    scope_values = {
        scope_id: [raw_scores[stable_id] for stable_id in stable_ids]
        for scope_id, stable_ids in scope_ids.items()
    }
    normalized_by_scope = normalize_scope_scores(scope_values, config=cfg.normalization)
    normalized_scores: dict[str, float] = {}
    for scope_id, stable_ids in scope_ids.items():
        for stable_id, score in zip(stable_ids, normalized_by_scope[scope_id]):
            normalized_scores[stable_id] = score
    for row in unit_rows:
        row["raw_score"] = raw_scores[row["stable_id"]]
        row["normalized_score"] = normalized_scores[row["stable_id"]]
    unit_parameter_costs = {
        str(unit.stable_id): estimate_unit_parameter_cost(model, unit)
        for unit in units
    }
    return ImportanceResult(
        mode=cfg.mode.value,
        normalization=cfg.normalization.strategy.value,
        aggregation=cfg.aggregation.value,
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
        unit_parameter_costs=unit_parameter_costs,
        unit_scores=unit_rows,
        calibration_batches=int(calibration_batches),
        task_loss=task_loss,
        gradient_accumulation=cfg.gradient_accumulation,
        implementation_version=IMPLEMENTATION_VERSION,
    )


__all__ = [
    "IMPLEMENTATION_VERSION",
    "estimate_dependency_parameter_cost",
    "estimate_unit_parameter_cost",
    "score_dependency_member",
    "score_first_order_taylor",
]
