"""Joint pruning-plus-weight-quantization Taylor task-loss proxy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..candidate import CandidatePhenotype
from .candidate_perturbation import (
    parameter_slices_for_phenotype,
    pseudo_quantize_tensor,
    retained_mask_for_parameter,
)
from .fisher_proxy import FisherStatistics
from .parameter_slice_resolver import ParameterSlice


class JointWeightTaylorProxy:
    """Score the exact combined weight perturbation on one normalized scale.

    Pruned elements use ``delta_w=-w``. Retained elements use
    ``delta_w=Q_precision(w)-w``. The same first-plus-diagonal-Fisher Taylor
    expression is applied once to the combined delta, so pruning and weight
    quantization cannot be double counted. Activation Taylor is deliberately
    excluded from this version.
    """

    def __init__(
        self,
        model: Any,
        *,
        statistics: FisherStatistics,
        unit_to_parameter_slices: Mapping[str, Sequence[ParameterSlice]],
        epsilon: float = 1.0e-12,
        strict: bool = True,
    ) -> None:
        self.model = model
        self.statistics = statistics
        self.unit_to_parameter_slices = {
            str(key): list(values) for key, values in unit_to_parameter_slices.items()
        }
        self.epsilon = float(epsilon)
        self.strict = bool(strict)
        self._denominator_by_precision_universe: dict[tuple[str, ...], float] = {}

    @staticmethod
    def _module_path(parameter_name: str) -> str:
        return str(parameter_name).rsplit(".", 1)[0]

    def _statistics_for(
        self,
        parameter_name: str,
        parameter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        gradient = self.statistics.gradients.get(parameter_name)
        fisher = self.statistics.fisher_diag.get(parameter_name)
        if gradient is None or fisher is None:
            if self.strict:
                raise RuntimeError(f"joint_weight_taylor_statistics_missing:{parameter_name}")
            return None
        return (
            gradient.detach().to(device=parameter.device, dtype=parameter.dtype),
            fisher.detach().to(device=parameter.device, dtype=parameter.dtype),
        )

    @staticmethod
    def _cost(
        delta: torch.Tensor,
        gradient: torch.Tensor,
        fisher: torch.Tensor,
    ) -> torch.Tensor:
        return (gradient * delta).abs() + 0.5 * fisher * delta.square()

    def _normalization_denominator(self, precision_layers: tuple[str, ...]) -> float:
        cached = self._denominator_by_precision_universe.get(precision_layers)
        if cached is not None:
            return cached
        prunable_parameters = {
            str(row.parameter_name)
            for rows in self.unit_to_parameter_slices.values()
            for row in rows
        }
        quantized_weights = {f"{layer}.weight" for layer in precision_layers}
        searchable = prunable_parameters | quantized_weights
        total = 0.0
        for name, parameter in self.model.named_parameters():
            if name not in searchable:
                continue
            statistics = self._statistics_for(name, parameter.detach())
            if statistics is None:
                continue
            gradient, fisher = statistics
            delta = -parameter.detach()
            total += float(self._cost(delta, gradient, fisher).sum().detach().cpu())
        if total <= self.epsilon:
            raise RuntimeError("joint_weight_taylor_normalization_nonpositive")
        self._denominator_by_precision_universe[precision_layers] = total
        return total

    def evaluate_breakdown(self, phenotype: CandidatePhenotype) -> dict[str, float | str | bool]:
        if not self.statistics.gradients or not self.statistics.fisher_diag:
            raise RuntimeError("joint_weight_taylor_statistics_missing")
        modules = dict(self.model.named_modules())
        slices_by_parameter = parameter_slices_for_phenotype(
            phenotype,
            self.unit_to_parameter_slices,
        )
        precision_layers = tuple(sorted(phenotype.realized_precision_profile))
        denominator = self._normalization_denominator(precision_layers)
        joint_total = 0.0
        pruning_only_total = 0.0
        retained_quant_total = 0.0
        for name, parameter in self.model.named_parameters():
            module_path = self._module_path(name)
            is_quantized_weight = name.endswith(".weight") and module_path in phenotype.realized_precision_profile
            rows = slices_by_parameter.get(name, [])
            if not is_quantized_weight and not rows:
                continue
            weight = parameter.detach()
            statistics = self._statistics_for(name, weight)
            if statistics is None:
                continue
            gradient, fisher = statistics
            retained = retained_mask_for_parameter(weight, rows)
            if is_quantized_weight:
                precision = phenotype.realized_precision_profile[module_path]
                quantized = pseudo_quantize_tensor(
                    weight,
                    precision,
                    module=modules.get(module_path),
                )
            else:
                quantized = weight
            effective = torch.where(retained, quantized, torch.zeros_like(weight))
            joint_delta = effective - weight
            pruning_delta = torch.where(retained, torch.zeros_like(weight), -weight)
            quant_delta = torch.where(retained, quantized - weight, torch.zeros_like(weight))
            joint_total += float(self._cost(joint_delta, gradient, fisher).sum().detach().cpu())
            pruning_only_total += float(
                self._cost(pruning_delta, gradient, fisher).sum().detach().cpu()
            )
            retained_quant_total += float(
                self._cost(quant_delta, gradient, fisher).sum().detach().cpu()
            )
        return {
            "L_joint_weight_taylor": joint_total / max(denominator, self.epsilon),
            "L_joint_weight_taylor_raw": joint_total,
            "L_pruning_only_taylor": pruning_only_total / max(denominator, self.epsilon),
            "L_retained_weight_quant_taylor": retained_quant_total / max(denominator, self.epsilon),
            "joint_weight_taylor_denominator": denominator,
            "joint_weight_taylor_formula": "sum(abs(g*delta_w)+0.5*E[g^2]*delta_w^2)/all_searchable_weight_removal_mass",
            "activation_taylor_included": False,
            "weight_quantization_granularity": "production_layout_aware_per_channel",
        }

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        return float(self.evaluate_breakdown(phenotype)["L_joint_weight_taylor"])
