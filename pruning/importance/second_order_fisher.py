"""Second-order Fisher Taylor importance scorer."""

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
from .first_order_taylor import estimate_unit_parameter_cost
from .normalization import normalize_scope_scores


IMPLEMENTATION_VERSION = "second-order-fisher-taylor-v2"


def _gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    saved = getattr(parameter, "_importance_grad", None)
    return saved if saved is not None else parameter.grad


def _fisher_diag(parameter: torch.Tensor) -> torch.Tensor | None:
    saved = getattr(parameter, "_importance_fisher_diag", None)
    if saved is not None:
        return saved
    saved = getattr(parameter, "_importance_empirical_fisher", None)
    if saved is not None:
        return saved
    gradient = _gradient(parameter)
    return None if gradient is None else gradient.square()


def _slice_score(parameter: torch.Tensor, indices: Sequence[int], axis: int) -> float:
    values = sorted({int(value) for value in indices})
    if not values:
        return 0.0
    if min(values) < 0 or max(values) >= int(parameter.shape[axis]):
        return float("inf")
    gradient = _gradient(parameter)
    fisher_diag = _fisher_diag(parameter)
    if gradient is None or fisher_diag is None:
        return float("inf")
    index = torch.as_tensor(values, dtype=torch.long, device=parameter.device)
    weight_slice = parameter.index_select(axis, index)
    gradient_slice = gradient.to(parameter.device).index_select(axis, index)
    fisher_slice = fisher_diag.to(parameter.device).index_select(axis, index)
    score = (weight_slice * gradient_slice).abs() + 0.5 * fisher_slice * weight_slice.square()
    return float(score.sum().detach().cpu())


def _vector_score(parameter: torch.Tensor | None, indices: Sequence[int]) -> float:
    if parameter is None:
        return 0.0
    return _slice_score(parameter, indices, 0)


def _grouped_conv_input_score(module: nn.Conv2d, indices: Sequence[int]) -> float:
    groups = int(module.groups)
    if groups <= 1 or module.in_channels % groups or module.out_channels % groups:
        return float("inf")
    gradient = _gradient(module.weight)
    fisher_diag = _fisher_diag(module.weight)
    if gradient is None or fisher_diag is None:
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
        weight = module.weight[start:stop, local_index : local_index + 1]
        grad = gradient[start:stop, local_index : local_index + 1]
        fisher = fisher_diag[start:stop, local_index : local_index + 1]
        total = total + ((weight * grad).abs() + 0.5 * fisher * weight.square()).sum()
    return float(total.detach().cpu())


def score_fisher_dependency_member(module: nn.Module, axis: str, indices: Sequence[int]) -> float:
    """Return ``sum(|g*w| + 0.5*h*w^2)`` for one dependency member."""

    direction = str(axis)
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
        return _vector_score(getattr(module, "weight", None), indices) + _vector_score(getattr(module, "bias", None), indices)
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


def fisher_parameter_slice(parameter: torch.Tensor, indices: Sequence[int], axis: int) -> float:
    """Return ``sum(|g*w| + 0.5*h*w^2)`` for one parameter slice."""

    return _slice_score(parameter, indices, axis)


def score_second_order_fisher(
    model: nn.Module,
    units: Sequence[Any],
    *,
    config: ImportanceConfig | None = None,
    calibration_batches: int = 0,
    task_loss: str = "",
) -> ImportanceResult:
    """Score coupled units with empirical Fisher diagonal Taylor loss."""

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
                score_fisher_dependency_member(module, str(member.axis), member.indices)
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
    normalized_by_scope = normalize_scope_scores(
        {scope_id: [raw_scores[stable_id] for stable_id in stable_ids] for scope_id, stable_ids in scope_ids.items()},
        config=cfg.normalization,
    )
    normalized_scores: dict[str, float] = {}
    for scope_id, stable_ids in scope_ids.items():
        for stable_id, score in zip(stable_ids, normalized_by_scope[scope_id]):
            normalized_scores[stable_id] = score
    for row in unit_rows:
        row["raw_score"] = raw_scores[row["stable_id"]]
        row["normalized_score"] = normalized_scores[row["stable_id"]]
    return ImportanceResult(
        mode=cfg.mode.value,
        normalization=cfg.normalization.strategy.value,
        aggregation=cfg.aggregation.value,
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
        unit_parameter_costs={str(unit.stable_id): estimate_unit_parameter_cost(model, unit) for unit in units},
        unit_scores=unit_rows,
        calibration_batches=int(calibration_batches),
        task_loss=task_loss,
        gradient_accumulation=cfg.gradient_accumulation,
        implementation_version=IMPLEMENTATION_VERSION,
    )


__all__ = [
    "IMPLEMENTATION_VERSION",
    "fisher_parameter_slice",
    "score_fisher_dependency_member",
    "score_second_order_fisher",
]
