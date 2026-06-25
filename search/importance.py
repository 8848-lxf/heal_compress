"""Group-level structured pruning importance.

Importance is computed from each ``PruningGroup`` item using the item's local
channel indices and pruning direction. This avoids layer-scalar scores and keeps
CNN, FFN, QKV, fused-QKV and projection-input pruning on the same path.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from ..pruning.units import AtomicPruneUnit, item_key

logger = logging.getLogger(__name__)


class ImportanceEstimator:
    """Estimate scalar importance for each ``PruningGroup``.

    Supported methods:
    ``l1_norm``, ``l2_norm``, ``first_order_taylor``, ``second_order_fisher``.
    Taylor/Fisher require gradients collected via :meth:`compute_gradients`.
    """

    METHODS = ("l1_norm", "l2_norm", "first_order_taylor", "second_order_fisher")

    def __init__(
        self,
        model: nn.Module,
        groups: list[Any],
        method: str = "l1_norm",
        aggregation: str = "sum",
        *,
        strict_grad: bool = False,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown importance method '{method}'. Must be one of {self.METHODS}")
        self.model = model
        self.groups = groups
        self.method = method
        self.aggregation = aggregation
        self.strict_grad = bool(strict_grad)
        self._gradients: dict[str, torch.Tensor] = {}
        self._fisher_diag: dict[str, torch.Tensor] = {}
        self.records: list[dict[str, Any]] = []

    def compute_gradients(
        self,
        forward_fn: Any,
        calibration_data: Any,
        loss_fn: Any,
        num_samples: int = 200,
    ) -> None:
        self._gradients = {}
        self._fisher_diag = {}
        self.model.eval()
        count = 0
        for batch in calibration_data:
            if count >= num_samples:
                break
            self.model.zero_grad(set_to_none=True)
            outputs = forward_fn(self.model, batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                self._gradients.setdefault(name, torch.zeros_like(param.data))
                self._fisher_diag.setdefault(name, torch.zeros_like(param.data))
                self._gradients[name] += grad
                self._fisher_diag[name] += grad.pow(2)
            count += 1
        if count > 0:
            for name in self._gradients:
                self._gradients[name] /= count
                self._fisher_diag[name] /= count
        logger.info("Computed gradients on %d samples for %d parameters", count, len(self._gradients))

    def estimate(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        self.records = []
        for group in self.groups:
            if getattr(group, "protected", getattr(group, "is_protected", False)):
                score = float("inf")
                risk = ""
            else:
                score, risk = self._compute_group_importance(group)
                if risk:
                    try:
                        group.protect(risk)
                    except AttributeError:
                        pass
                    score = float("inf")
            scores[group.group_id] = score
            rec = self._record(group, score, risk)
            self.records.append(rec)
        return scores

    def estimate_with_records(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        scores = self.estimate()
        return scores, self.records

    def _compute_group_importance(self, group: Any) -> tuple[float, str]:
        values: list[torch.Tensor] = []
        missing_grads: list[str] = []
        risk_reasons: list[str] = []
        for item in getattr(group, "items", []):
            module = item.module
            if not hasattr(module, "weight") or module.weight is None:
                continue
            if getattr(item, "idx_transform", None) is None and getattr(item, "idxs", None):
                reference = list(item.idxs)
            else:
                reference = list(range(int(getattr(group, "num_channels", 0))))
            keep = item.local_keep(reference)
            if not keep:
                continue
            value, missing, risk = self._item_importance(item.name, module, item.direction, keep)
            if risk:
                risk_reasons.append(risk)
                continue
            if missing:
                missing_grads.append(item.name)
            if value is not None:
                values.append(value)
        if risk_reasons:
            group.meta["importance_risk_reasons"] = risk_reasons
            return float("inf"), "high_risk_protected"
        if missing_grads and self.strict_grad:
            raise RuntimeError(f"Missing gradients for group {group.group_id}: {missing_grads}")
        if not values:
            return 0.0, ""
        if self.method == "l2_norm":
            score_tensor = torch.stack([v.float().pow(2) for v in values]).sum().sqrt()
        else:
            score_tensor = torch.stack([v.float() for v in values]).sum()
        score = float(score_tensor.detach().cpu())
        if math.isnan(score) or math.isinf(score):
            return score, "high_risk_protected"
        return score, ""

    def _item_importance(
        self,
        layer_name: str,
        module: nn.Module,
        direction: str,
        indices: list[int],
    ) -> tuple[torch.Tensor | None, bool, str]:
        weight = module.weight
        if direction == "out":
            axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
            dim_size = int(weight.shape[axis])
        elif direction == "in":
            axis = 0 if isinstance(module, nn.ConvTranspose2d) else 1
            dim_size = int(weight.shape[axis])
        else:
            return None, False, ""

        if not indices:
            return None, False, "empty_importance_indices"
        min_idx = min(indices)
        max_idx = max(indices)
        if min_idx < 0 or max_idx >= dim_size:
            msg = (
                f"importance_index_out_of_bounds:{layer_name}:"
                f"direction={direction}:axis={axis}:dim={dim_size}:"
                f"min={min_idx}:max={max_idx}:num_indices={len(indices)}"
            )
            logger.warning(msg)
            return None, False, msg

        # Build the index on CPU first so invalid mappings are caught above
        # before any CUDA index_select can trigger a device-side assert.
        idx = torch.as_tensor(indices, dtype=torch.long, device=weight.device)
        if direction == "out":
            w = weight.index_select(axis, idx)
        else:
            w = weight.index_select(axis, idx)

        if self.method == "l1_norm":
            return w.detach().abs().sum(), False, ""
        if self.method == "l2_norm":
            return w.detach().pow(2).sum(), False, ""

        param_name = f"{layer_name}.weight"
        grad = self._gradients.get(param_name)
        fisher = self._fisher_diag.get(param_name)
        if grad is None:
            logger.warning("Missing gradient for %s; falling back to L1 for this item", param_name)
            if self.strict_grad:
                return None, True, ""
            return w.detach().abs().sum(), True, ""
        if direction == "out":
            axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
            g = grad.index_select(axis, idx)
            f = fisher.index_select(axis, idx) if fisher is not None else None
        else:
            axis = 1
            if isinstance(module, nn.ConvTranspose2d):
                axis = 0
            g = grad.index_select(axis, idx)
            f = fisher.index_select(axis, idx) if fisher is not None else None
        first = (g * w).abs().sum()
        if self.method == "first_order_taylor":
            return first.detach(), False, ""
        second = 0.5 * (f * w.pow(2)).sum() if f is not None else torch.zeros((), device=w.device)
        return (first + second).detach(), fisher is None, ""

    def _record(self, group: Any, score: float, risk: str) -> dict[str, Any]:
        meta = getattr(group, "meta", {}) or {}
        module_names = ";".join(item.name for item in getattr(group, "items", []))
        protected = bool(getattr(group, "protected", getattr(group, "is_protected", False)))
        return {
            "group_id": getattr(group, "group_id", ""),
            "group_type": meta.get("group_type", ""),
            "module_names": module_names,
            "num_channels": int(getattr(group, "num_channels", 0)),
            "protected": protected,
            "protected_reason": getattr(group, "protected_reason", "") or risk,
            "importance_risk_reasons": ";".join(meta.get("importance_risk_reasons", [])),
            "importance_score": score,
            "importance_mode": self.method,
            "transformer_block_name": meta.get("transformer_block_name", ""),
            "attention_type": meta.get("attention_type", ""),
            "qkv_type": meta.get("qkv_type", ""),
            "head_id": meta.get("head_id", ""),
            "num_heads_before": meta.get("num_heads_before", ""),
            "num_heads_after": meta.get("num_heads_after", ""),
            "head_dim": meta.get("head_dim", ""),
            "inner_dim_before": meta.get("inner_dim_before", ""),
            "inner_dim_after": meta.get("inner_dim_after", ""),
            "ffn_dim_before": meta.get("ffn_dim_before", ""),
            "ffn_dim_after": meta.get("ffn_dim_after", ""),
            "hidden_dim": meta.get("hidden_dim", ""),
        }


def compute_group_importance(
    model: nn.Module,
    groups: list[Any],
    *,
    method: str = "l1_norm",
    forward_fn: Any | None = None,
    calibration_data: Any | None = None,
    loss_fn: Any | None = None,
    num_calib_batches: int = 0,
    strict_grad: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    estimator = ImportanceEstimator(model, groups, method=method, strict_grad=strict_grad)
    if method in ("first_order_taylor", "second_order_fisher") and calibration_data is not None:
        if forward_fn is None or loss_fn is None:
            raise ValueError("Taylor/Fisher importance requires forward_fn and loss_fn")
        estimator.compute_gradients(forward_fn, calibration_data, loss_fn, num_samples=num_calib_batches)
    return estimator.estimate_with_records()


def _select_by_axis(weight: torch.Tensor, axis: int, indices: list[int]) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long, device=weight.device)
    return weight.index_select(axis, idx)


def _weight_importance_value(
    weight: torch.Tensor,
    *,
    method: str,
    grad: torch.Tensor | None = None,
    fisher: torch.Tensor | None = None,
) -> torch.Tensor:
    if method == "l1_norm":
        return weight.detach().abs().sum()
    if method == "l2_norm":
        return weight.detach().pow(2).sum().sqrt()
    if grad is None:
        return torch.full((), float("inf"), dtype=weight.dtype, device=weight.device)
    first = (grad * weight).abs().sum()
    if method == "first_order_taylor":
        return first.detach()
    if fisher is None:
        return torch.full((), float("inf"), dtype=weight.dtype, device=weight.device)
    second = 0.5 * (fisher * weight.pow(2)).sum()
    return (first + second).detach()


def _item_local_importance(
    item: Any,
    local_indices: list[int],
    *,
    method: str,
) -> tuple[float, str]:
    module = item.module
    if not local_indices:
        return 0.0, ""

    # BatchNorm / LayerNorm have direct per-channel affine vectors.
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
        values: list[torch.Tensor] = []
        for attr in ("weight", "bias"):
            param = getattr(module, attr, None)
            if param is None:
                continue
            if max(local_indices) >= int(param.shape[0]) or min(local_indices) < 0:
                return float("inf"), "importance_index_out_of_bounds"
            idx = torch.as_tensor(local_indices, dtype=torch.long, device=param.device)
            selected = param.index_select(0, idx)
            grad = param.grad.index_select(0, idx) if param.grad is not None else None
            fisher = grad.pow(2) if grad is not None else None
            values.append(_weight_importance_value(selected, method=method, grad=grad, fisher=fisher))
        if not values:
            return 0.0, ""
        value = torch.stack([v.float() for v in values]).sum()
        return float(value.detach().cpu()), ""

    if not hasattr(module, "weight") or module.weight is None:
        return 0.0, ""

    weight = module.weight
    grad = weight.grad
    fisher = grad.pow(2) if grad is not None else None

    if isinstance(module, nn.Conv2d) and module.groups > 1 and item.direction == "in":
        if module.in_channels % module.groups != 0 or module.out_channels % module.groups != 0:
            return float("inf"), "grouped_conv_divisibility"
        in_per = module.in_channels // module.groups
        out_per = module.out_channels // module.groups
        pieces: list[torch.Tensor] = []
        grad_pieces: list[torch.Tensor] = []
        fisher_pieces: list[torch.Tensor] = []
        for abs_idx in local_indices:
            if abs_idx < 0 or abs_idx >= module.in_channels:
                return float("inf"), "importance_index_out_of_bounds"
            group_id = abs_idx // in_per
            local = abs_idx % in_per
            row_start = group_id * out_per
            row_end = row_start + out_per
            pieces.append(weight[row_start:row_end, local:local + 1])
            if grad is not None:
                grad_pieces.append(grad[row_start:row_end, local:local + 1])
            if fisher is not None:
                fisher_pieces.append(fisher[row_start:row_end, local:local + 1])
        selected = torch.cat([p.reshape(-1) for p in pieces])
        selected_grad = torch.cat([p.reshape(-1) for p in grad_pieces]) if grad_pieces else None
        selected_fisher = torch.cat([p.reshape(-1) for p in fisher_pieces]) if fisher_pieces else None
        value = _weight_importance_value(selected, method=method, grad=selected_grad, fisher=selected_fisher)
        return float(value.detach().cpu()), ""

    if item.direction == "out":
        axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
        dim_size = int(weight.shape[axis])
    elif item.direction == "in":
        axis = 0 if isinstance(module, nn.ConvTranspose2d) else 1
        dim_size = int(weight.shape[axis])
    else:
        return 0.0, ""

    if min(local_indices) < 0 or max(local_indices) >= dim_size:
        return float("inf"), "importance_index_out_of_bounds"

    selected = _select_by_axis(weight, axis, local_indices)
    selected_grad = _select_by_axis(grad, axis, local_indices) if grad is not None else None
    selected_fisher = _select_by_axis(fisher, axis, local_indices) if fisher is not None else None
    value = _weight_importance_value(selected, method=method, grad=selected_grad, fisher=selected_fisher)

    if item.direction == "out" and hasattr(module, "bias") and module.bias is not None:
        bias = module.bias
        if max(local_indices) < int(bias.shape[0]):
            bias_sel = bias.index_select(0, torch.as_tensor(local_indices, dtype=torch.long, device=bias.device))
            bias_grad = bias.grad.index_select(0, torch.as_tensor(local_indices, dtype=torch.long, device=bias.device)) if bias.grad is not None else None
            value = value + _weight_importance_value(bias_sel, method=method, grad=bias_grad)

    return float(value.detach().cpu()), ""


def compute_scope_channel_importance(
    scope: Any,
    *,
    method: str = "l1_norm",
    importance_reduction: str = "sum",
    group_reduction: str = "sum",
    channel_group_reduction: str = "sum",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute scope-level per-root-channel importance.

    The result shape is ``[scope.num_channels]`` and each element corresponds to
    one ``CoupledChannelUnit`` rooted at that index. Contributions are gathered
    from every mapped ``GroupItem`` in the dependency scope.
    """
    if method not in ImportanceEstimator.METHODS:
        raise ValueError(f"Unknown importance method '{method}'. Must be one of {ImportanceEstimator.METHODS}")
    channels = int(getattr(scope, "num_channels", 0))
    scores = torch.zeros(channels, dtype=torch.float32)
    source_items: list[str] = []
    invalid_roots: dict[int, list[str]] = {}
    missing_grad_items: list[str] = []

    for item in getattr(scope, "items", []):
        contributed = False
        key = item_key(item)
        if method in ("first_order_taylor", "second_order_fisher"):
            module = item.module
            if hasattr(module, "weight") and module.weight is not None and module.weight.grad is None:
                missing_grad_items.append(key)
        for root_idx in range(channels):
            local = sorted(int(v) for v in item.local_keep([root_idx]))
            if not local:
                continue
            value, risk = _item_local_importance(item, local, method=method)
            if risk or math.isnan(value) or math.isinf(value):
                invalid_roots.setdefault(root_idx, []).append(risk or "invalid_importance")
                scores[root_idx] = float("inf")
                continue
            if not math.isinf(float(scores[root_idx])):
                scores[root_idx] += float(value)
            contributed = True
        if contributed:
            source_items.append(key)

    record = {
        "scope_id": getattr(scope, "group_id", ""),
        "num_channels": channels,
        "importance_mode": method,
        "importance_shape": [channels],
        "importance_source_items": source_items,
        "importance_reduction": importance_reduction,
        "group_reduction": group_reduction,
        "channel_group_reduction": channel_group_reduction,
        "invalid_root_indices": {str(k): v for k, v in invalid_roots.items()},
        "missing_grad_items": missing_grad_items,
    }
    return scores, record


def compute_scope_channel_importance_map(
    groups: list[Any],
    *,
    method: str = "l1_norm",
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    scope_scores: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for group in groups:
        score, record = compute_scope_channel_importance(group, method=method)
        scope_scores[getattr(group, "group_id", "")] = score
        records.append(record)
    return scope_scores, records


def compute_candidate_importance(
    candidate: AtomicPruneUnit,
    coupled_units_by_id: dict[str, Any],
    *,
    reduction: str = "sum",
) -> float:
    values = [
        float(coupled_units_by_id[unit_id].importance)
        for unit_id in candidate.source_coupled_units
        if unit_id in coupled_units_by_id and coupled_units_by_id[unit_id].importance is not None
    ]
    if not values:
        return 0.0
    if reduction == "mean":
        return float(sum(values) / len(values))
    if reduction == "max":
        return float(max(values))
    return float(sum(values))


def compute_layer_channel_importance(
    groups: list[Any],
    *,
    method: str = "l1_norm",
) -> dict[str, torch.Tensor]:
    """Compute per-layer channel scores for group keep-index selection.

    The group-level score ranks which group to prune. This helper keeps the
    channel-level scores needed to decide which channels inside a selected group
    should survive.
    """
    if method not in ("l1_norm", "l2_norm"):
        return {}
    scores: dict[str, torch.Tensor] = {}
    for group in groups:
        if getattr(group, "protected", getattr(group, "is_protected", False)):
            continue
        for item in getattr(group, "items", []):
            module = item.module
            if not hasattr(module, "weight") or module.weight is None:
                continue
            weight = module.weight.detach()
            if item.direction == "out":
                axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
            elif item.direction == "in":
                axis = 0 if isinstance(module, nn.ConvTranspose2d) else 1
            else:
                continue
            reduce_dims = tuple(dim for dim in range(weight.dim()) if dim != axis)
            if method == "l2_norm":
                value = weight.pow(2).sum(dim=reduce_dims).sqrt().float().cpu()
            else:
                value = weight.abs().sum(dim=reduce_dims).float().cpu()
            scores[item.name] = value
    return scores
