"""Channel importance estimation for coupled channel groups.

Supports L1-norm, L2-norm, first-order Taylor, and second-order Fisher
importance metrics. Importance is aggregated at the group level.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ImportanceEstimator:
    """Estimates per-group channel importance for pruning decisions.

    Supported methods:
    - 'l1_norm': I_g = sum(|w_i|)
    - 'l2_norm': I_g = sqrt(sum(w_i^2))
    - 'first_order_taylor' (default): I_g = sum(|g_i * w_i|)
    - 'second_order_fisher': I_g = sum(|g_i * w_i|) + 0.5 * sum(h_i * w_i^2)

    For Taylor/Fisher methods, gradients must be computed on calibration data
    with full annotations (forward + backward pass).

    Args:
        model: The HEAL model.
        groups: List of coupled channel groups.
        method: Importance estimation method name.
        aggregation: How to aggregate within a group ('mean' or 'sum').
    """

    METHODS = ("l1_norm", "l2_norm", "first_order_taylor", "second_order_fisher")

    def __init__(
        self,
        model: nn.Module,
        groups: list[Any],
        method: str = "first_order_taylor",
        aggregation: str = "mean",
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown importance method '{method}'. Must be one of {self.METHODS}")
        self.model = model
        self.groups = groups
        self.method = method
        self.aggregation = aggregation
        self._gradients: dict[str, torch.Tensor] = {}
        self._fisher_diag: dict[str, torch.Tensor] = {}

    def compute_gradients(
        self,
        forward_fn: Any,
        calibration_data: Any,
        loss_fn: Any,
        num_samples: int = 200,
    ) -> None:
        """Compute parameter gradients on calibration data.

        Runs forward + backward on calibration samples and accumulates
        gradients (for Taylor) and squared gradients (for Fisher diagonal).

        Args:
            forward_fn: Callable(model, batch) -> outputs.
            calibration_data: Iterable of calibration batches.
            loss_fn: Callable(outputs, batch) -> loss scalar.
            num_samples: Maximum number of samples to use.
        """
        self._gradients = {}
        self._fisher_diag = {}
        self.model.eval()

        count = 0
        for batch in calibration_data:
            if count >= num_samples:
                break
            self.model.zero_grad()
            outputs = forward_fn(self.model, batch)
            loss = loss_fn(outputs, batch)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if name not in self._gradients:
                    self._gradients[name] = torch.zeros_like(param.data)
                    self._fisher_diag[name] = torch.zeros_like(param.data)
                self._gradients[name] += grad
                self._fisher_diag[name] += grad ** 2

            count += 1

        # Average
        if count > 0:
            for name in self._gradients:
                self._gradients[name] /= count
                self._fisher_diag[name] /= count

        logger.info(
            f"Computed gradients on {count} samples for "
            f"{len(self._gradients)} parameters"
        )

    def estimate(self) -> dict[str, float]:
        """Estimate importance for each coupled channel group.

        Returns:
            Map of group_id -> importance score.
        """
        modules = dict(self.model.named_modules())
        scores: dict[str, float] = {}

        for group in self.groups:
            if group.is_protected:
                scores[group.group_id] = float("inf")
                continue
            group_score = self._compute_group_importance(
                group.source_modules, modules
            )
            scores[group.group_id] = group_score

        return scores

    def _compute_group_importance(
        self,
        layer_names: list[str],
        modules: dict[str, nn.Module],
    ) -> float:
        """Compute importance for a single group by aggregating over its layers.

        Args:
            layer_names: Layer names in the group.
            modules: Map of name -> module.

        Returns:
            Aggregated importance score.
        """
        values: list[float] = []

        for layer_name in layer_names:
            module = modules.get(layer_name)
            if module is None or not hasattr(module, "weight"):
                continue
            weight = module.weight.detach()
            importance = self._compute_layer_importance(layer_name, weight)
            values.append(importance)

        if not values:
            return 0.0
        if self.aggregation == "sum":
            return sum(values)
        return sum(values) / len(values)

    def _compute_layer_importance(
        self,
        layer_name: str,
        weight: torch.Tensor,
    ) -> float:
        """Compute importance for a single layer.

        Args:
            layer_name: Fully qualified layer name (for gradient lookup).
            weight: The layer's weight tensor.

        Returns:
            Scalar importance score.
        """
        if self.method == "l1_norm":
            return float(weight.abs().sum().cpu())

        if self.method == "l2_norm":
            return float(weight.pow(2).sum().sqrt().cpu())

        # Find the parameter name for gradient lookup
        param_name = f"{layer_name}.weight"
        grad = self._gradients.get(param_name)
        fisher = self._fisher_diag.get(param_name)

        if self.method == "first_order_taylor":
            if grad is None:
                return float(weight.abs().sum().cpu())
            return float((grad * weight).abs().sum().cpu())

        if self.method == "second_order_fisher":
            first = float((grad * weight).abs().sum().cpu()) if grad is not None else 0.0
            second = float((fisher * weight.pow(2)).sum().cpu()) * 0.5 if fisher is not None else 0.0
            return first + second

        return 0.0
