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
