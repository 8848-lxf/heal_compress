"""First-order Taylor importance for v10.8 coupled channel units."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from ..tracer.pruning_group import PruningGroup


@dataclass
class TaylorScopeImportance:
    scope_id: str
    raw_scores: torch.Tensor
    normalized_scores: torch.Tensor
    unit_rows: list[dict[str, Any]]
    dependency_score_rows: list[dict[str, Any]]
    skipped_units: list[dict[str, Any]]


@dataclass
class TaylorImportanceRun:
    scope_scores: dict[str, torch.Tensor]
    raw_scope_scores: dict[str, torch.Tensor]
    unit_rows: list[dict[str, Any]]
    report: dict[str, Any]


def _grad_for(param: torch.Tensor) -> torch.Tensor | None:
    grad = getattr(param, "_importance_grad", None)
    if grad is None:
        grad = param.grad
    return grad


def _take_axis(value: torch.Tensor, axis: int, indices: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor([int(v) for v in indices], dtype=torch.long, device=value.device)
    return value.index_select(axis, idx)


def _score_param_slice(param: torch.Tensor, indices: Sequence[int], *, axis: int) -> tuple[float | None, str]:
    if not indices:
        return 0.0, ""
    dim = int(param.shape[axis])
    if min(indices) < 0 or max(indices) >= dim:
        return None, "importance_index_out_of_bounds"
    grad = _grad_for(param)
    if grad is None:
        return None, "missing_gradient"
    selected = _take_axis(param, axis, indices)
    selected_grad = _take_axis(grad, axis, indices)
    return float((selected * selected_grad).abs().sum().detach().cpu()), ""


def _score_vector_param(param: torch.Tensor | None, indices: Sequence[int]) -> tuple[float, str]:
    if param is None:
        return 0.0, ""
    if not indices:
        return 0.0, ""
    if min(indices) < 0 or max(indices) >= int(param.shape[0]):
        return 0.0, "importance_index_out_of_bounds"
    grad = _grad_for(param)
    if grad is None:
        return 0.0, "missing_gradient"
    idx = torch.as_tensor([int(v) for v in indices], dtype=torch.long, device=param.device)
    selected = param.index_select(0, idx)
    selected_grad = grad.index_select(0, idx)
    return float((selected * selected_grad).abs().sum().detach().cpu()), ""


def _score_grouped_conv_input(module: nn.Conv2d, indices: Sequence[int]) -> tuple[float | None, str]:
    if not indices:
        return 0.0, ""
    groups = int(module.groups)
    if int(module.in_channels) % groups or int(module.out_channels) % groups:
        return None, "grouped_conv_divisibility"
    if min(indices) < 0 or max(indices) >= int(module.in_channels):
        return None, "importance_index_out_of_bounds"
    grad = _grad_for(module.weight)
    if grad is None:
        return None, "missing_gradient"
    in_per = int(module.in_channels) // groups
    out_per = int(module.out_channels) // groups
    scores = []
    for abs_idx in sorted({int(v) for v in indices}):
        group_idx = abs_idx // in_per
        local_idx = abs_idx % in_per
        out_start = group_idx * out_per
        out_end = out_start + out_per
        w = module.weight[out_start:out_end, local_idx : local_idx + 1]
        g = grad[out_start:out_end, local_idx : local_idx + 1]
        scores.append((w * g).abs().sum())
    if not scores:
        return 0.0, ""
    return float(torch.stack([v.float() for v in scores]).sum().detach().cpu()), ""


def _dependency_taylor_score(item: Any, local_indices: Sequence[int]) -> tuple[float | None, str]:
    module = item.module
    direction = str(getattr(item, "direction", "out"))
    indices = [int(v) for v in local_indices]
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
        total = 0.0
        for param in (getattr(module, "weight", None), getattr(module, "bias", None)):
            value, reason = _score_vector_param(param, indices)
            if reason:
                return None, reason
            total += value
        return total, ""
    if not hasattr(module, "weight") or module.weight is None:
        return None, "no_weight_parameter"
    if isinstance(module, nn.Conv2d) and int(module.groups) > 1 and direction == "in":
        return _score_grouped_conv_input(module, indices)
    if direction == "out":
        axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
    elif direction == "in":
        axis = 0 if isinstance(module, nn.ConvTranspose2d) else 1
    else:
        return None, f"unsupported_direction:{direction}"
    value, reason = _score_param_slice(module.weight, indices, axis=axis)
    if reason:
        return None, reason
    assert value is not None
    if direction == "out" and hasattr(module, "bias") and module.bias is not None:
        bias_value, bias_reason = _score_vector_param(module.bias, indices)
        if bias_reason:
            return None, bias_reason
        value += bias_value
    return float(value), ""


def _is_grouped_output_scope(scope: PruningGroup) -> bool:
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if (
            isinstance(module, nn.Conv2d)
            and int(module.groups) > 1
            and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
            and getattr(item, "direction", "") == "out"
        ):
            return True
    return False


def _grouped_info(scope: PruningGroup) -> dict[str, Any]:
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if isinstance(module, nn.Conv2d) and int(module.groups) > 1:
            groups = int(module.groups)
            channels = int(getattr(scope, "num_channels", module.out_channels))
            per_group = channels // groups if groups and channels % groups == 0 else None
            return {
                "module_name": item.name,
                "groups": groups,
                "per_group": per_group,
            }
    return {}


def compute_taylor_importance_for_scope(scope: PruningGroup, *, eps: float = 1e-12) -> TaylorScopeImportance:
    """Score every root channel in one pruning domain.

    Per dependency score is ``sum(abs(w * grad))`` over the local channel
    slice. A CoupledChannelUnit raw score is the mean of dependency scores, and
    normalized score divides by the domain mean raw score.
    """

    num_channels = int(getattr(scope, "num_channels", 0))
    raw = torch.zeros(num_channels, dtype=torch.float32)
    dependency_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    grouped = _grouped_info(scope)
    is_grouped = _is_grouped_output_scope(scope) or bool(grouped)
    for root_idx in range(num_channels):
        dep_scores: list[float] = []
        dep_reasons: list[str] = []
        for item in getattr(scope, "items", []):
            local = sorted(int(v) for v in item.local_keep([root_idx]))
            if not local:
                continue
            score, reason = _dependency_taylor_score(item, local)
            row = {
                "pruning_domain_id": getattr(scope, "group_id", ""),
                "root_channel_index": root_idx,
                "module_name": item.name,
                "direction": getattr(item, "direction", ""),
                "local_indices": local,
                "dependency_score": score,
                "skipped_reason": reason,
            }
            dependency_rows.append(row)
            if reason or score is None or not math.isfinite(float(score)):
                dep_reasons.append(reason or "invalid_importance")
                continue
            dep_scores.append(float(score))
        if dep_scores:
            raw[root_idx] = float(sum(dep_scores) / len(dep_scores))
        else:
            raw[root_idx] = float("inf")
            skipped.append(
                {
                    "pruning_domain_id": getattr(scope, "group_id", ""),
                    "root_channel_index": root_idx,
                    "skip_reasons": dep_reasons or ["no_dependency_scores"],
                }
            )
    finite = raw[torch.isfinite(raw)]
    domain_mean = float(finite.mean().item()) if int(finite.numel()) else 0.0
    normalized = raw / float(domain_mean + eps)
    root_name = scope.items[0].name if scope.items else getattr(scope, "group_id", "")
    per_group = grouped.get("per_group")
    unit_rows: list[dict[str, Any]] = []
    for idx in range(num_channels):
        group_index = int(idx // per_group) if per_group else None
        local_index = int(idx % per_group) if per_group else None
        unit_rows.append(
            {
                "pruning_domain_id": getattr(scope, "group_id", ""),
                "root_module_name": root_name,
                "root_dim": "out",
                "num_root_channels": num_channels,
                "coupled_unit_id": f"{getattr(scope, 'group_id', '')}::idx{idx}",
                "root_channel_index": idx,
                "is_grouped_conv": bool(is_grouped),
                "grouped_local_unit_id": f"{getattr(scope, 'group_id', '')}::g{group_index}::l{local_index}" if group_index is not None else None,
                "group_index": group_index,
                "local_channel_index": local_index,
                "importance_raw": float(raw[idx].item()),
                "importance_normalized": float(normalized[idx].item()),
                "selected_for_pruning": False,
                "protected_reason": getattr(scope, "protected_reason", "") if getattr(scope, "protected", False) else "",
                "skipped_reason": "",
            }
        )
    return TaylorScopeImportance(
        scope_id=getattr(scope, "group_id", ""),
        raw_scores=raw,
        normalized_scores=normalized,
        unit_rows=unit_rows,
        dependency_score_rows=dependency_rows,
        skipped_units=skipped,
    )


def compute_taylor_importance_for_scopes(
    scopes: Iterable[PruningGroup],
    *,
    calibration_batches: int = 0,
    loss_terms_used: Sequence[str] | None = None,
    eps: float = 1e-12,
) -> TaylorImportanceRun:
    scope_scores: dict[str, torch.Tensor] = {}
    raw_scores: dict[str, torch.Tensor] = {}
    unit_rows: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for scope in scopes:
        result = compute_taylor_importance_for_scope(scope, eps=eps)
        scope_scores[result.scope_id] = result.normalized_scores
        raw_scores[result.scope_id] = result.raw_scores
        unit_rows.extend(result.unit_rows)
        dependency_rows.extend(result.dependency_score_rows)
        skipped.extend(result.skipped_units)
    finite_scores = [
        float(row["importance_normalized"])
        for row in unit_rows
        if math.isfinite(float(row.get("importance_normalized", float("inf"))))
    ]
    tensor = torch.as_tensor(finite_scores, dtype=torch.float32)
    bottom = sorted(unit_rows, key=lambda r: float(r.get("importance_normalized", float("inf"))))[:100]
    top = sorted(unit_rows, key=lambda r: float(r.get("importance_normalized", float("-inf"))), reverse=True)[:100]
    skip_counts: dict[str, int] = {}
    for row in skipped:
        for reason in row.get("skip_reasons", []) or []:
            key = str(reason)
            skip_counts[key] = skip_counts.get(key, 0) + 1
    report = {
        "calibration_batches": int(calibration_batches),
        "loss_terms_used": list(loss_terms_used or []),
        "num_pruning_domains": len(scope_scores),
        "num_coupled_units_scored": len(unit_rows),
        "num_grouped_units_scored": sum(1 for row in unit_rows if row.get("is_grouped_conv")),
        "score_min": float(tensor.min().item()) if int(tensor.numel()) else 0.0,
        "score_max": float(tensor.max().item()) if int(tensor.numel()) else 0.0,
        "score_mean": float(tensor.mean().item()) if int(tensor.numel()) else 0.0,
        "score_std": float(tensor.std(unbiased=False).item()) if int(tensor.numel()) else 0.0,
        "top_100_highest": top,
        "bottom_100_lowest": bottom,
        "skipped_units": skipped,
        "skip_reasons": skip_counts,
        "dependency_score_rows": dependency_rows,
    }
    return TaylorImportanceRun(scope_scores=scope_scores, raw_scope_scores=raw_scores, unit_rows=unit_rows, report=report)
