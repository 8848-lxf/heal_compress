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
        self._pruning_element_terms: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._quantized_weights: dict[tuple[str, str], torch.Tensor] = {}
        self._quantization_element_terms: dict[
            tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @staticmethod
    def _module_path(parameter_name: str) -> str:
        return str(parameter_name).rsplit(".", 1)[0]

    def _statistics_for(
        self,
        parameter_name: str,
        parameter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        gradient = self.statistics.gradients.get(parameter_name)
        absolute_gradient = self.statistics.absolute_gradients.get(parameter_name)
        fisher = self.statistics.fisher_diag.get(parameter_name)
        if gradient is None or fisher is None:
            if self.strict:
                raise RuntimeError(f"joint_weight_taylor_statistics_missing:{parameter_name}")
            return None
        return (
            (absolute_gradient if absolute_gradient is not None else gradient.abs())
            .detach()
            .to(device=parameter.device, dtype=parameter.dtype),
            fisher.detach().to(device=parameter.device, dtype=parameter.dtype),
        )

    @staticmethod
    def _cost_terms(
        delta: torch.Tensor,
        gradient: torch.Tensor,
        fisher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError("weight_taylor_delta_nonfinite")
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(fisher).all()):
            raise RuntimeError("weight_taylor_statistics_nonfinite")
        first = (gradient * delta).abs()
        second = 0.5 * (fisher * delta.square()).abs()
        score = first + second
        if bool((score < 0).any()) or not bool(torch.isfinite(score).all()):
            raise RuntimeError("weight_taylor_element_score_invalid")
        return first, second, score

    @classmethod
    def _cost(
        cls,
        delta: torch.Tensor,
        gradient: torch.Tensor,
        fisher: torch.Tensor,
    ) -> torch.Tensor:
        return cls._cost_terms(delta, gradient, fisher)[2]

    def _slices_by_parameter(
        self, phenotype: CandidatePhenotype
    ) -> dict[str, list[ParameterSlice]]:
        return parameter_slices_for_phenotype(
            phenotype,
            self.unit_to_parameter_slices,
        )

    def pruning_action_breakdown(
        self,
        current: CandidatePhenotype,
        successor: CandidatePhenotype,
    ) -> dict[str, float | int | str | bool]:
        """Score only elements newly removed by one legal width action."""
        current_slices = self._slices_by_parameter(current)
        successor_slices = self._slices_by_parameter(successor)
        newly_pruned_units = sorted(
            set(successor.pruned_unit_ids) - set(current.pruned_unit_ids)
        )
        changed_slices = parameter_slices_for_phenotype(
            CandidatePhenotype(pruned_unit_ids=newly_pruned_units),
            self.unit_to_parameter_slices,
        )
        first_total = 0.0
        second_total = 0.0
        element_count = 0
        tensor_count = 0
        parameters = dict(self.model.named_parameters())
        for name in sorted(changed_slices):
            parameter = parameters.get(name)
            if parameter is None:
                continue
            weight = parameter.detach()
            statistics = self._statistics_for(name, weight)
            if statistics is None:
                continue
            gradient, fisher = statistics
            retained_before = retained_mask_for_parameter(
                weight, current_slices.get(name, [])
            )
            retained_after = retained_mask_for_parameter(
                weight, successor_slices.get(name, [])
            )
            newly_removed = retained_before & ~retained_after
            if not bool(newly_removed.any()):
                continue
            cached = self._pruning_element_terms.get(name)
            if cached is None:
                first, second, _score = self._cost_terms(-weight, gradient, fisher)
                cached = (first.detach(), second.detach())
                self._pruning_element_terms[name] = cached
            first, second = cached
            first_total += float(first[newly_removed].sum().detach().cpu())
            second_total += float(second[newly_removed].sum().detach().cpu())
            element_count += int(newly_removed.sum().detach().cpu())
            tensor_count += 1
        total = first_total + second_total
        if total < 0.0:
            raise RuntimeError("pruning_action_taylor_negative")
        return {
            "delta_J_prune": total,
            "delta_J_WQ": 0.0,
            "first_order_abs_sum": first_total,
            "second_order_abs_sum": second_total,
            "newly_pruned_parameter_count": element_count,
            "touched_parameter_tensor_count": tensor_count,
            "risk_refund": 0.0,
            "formula": "sum_newly_removed(abs(g*(-w))+0.5*abs(h*w^2))",
            "elementwise_abs_before_reduction": True,
        }

    def weight_quantization_action_breakdown(
        self,
        current: CandidatePhenotype,
        successor: CandidatePhenotype,
    ) -> dict[str, float | int | str | bool]:
        """Score retained weights for the adjacent current-to-next precision."""
        current_slices = self._slices_by_parameter(current)
        modules = dict(self.model.named_modules())
        parameters = dict(self.model.named_parameters())
        first_total = 0.0
        second_total = 0.0
        element_count = 0
        tensor_count = 0
        changed_paths = sorted(
            path
            for path, precision in successor.realized_precision_profile.items()
            if current.realized_precision_profile.get(path, "FP32") != precision
        )
        for module_path in changed_paths:
            name = f"{module_path}.weight"
            weight = parameters.get(name)
            if weight is None:
                continue
            value = weight.detach()
            statistics = self._statistics_for(name, value)
            if statistics is None:
                continue
            gradient, fisher = statistics
            current_precision = current.realized_precision_profile.get(
                module_path, "FP32"
            )
            next_precision = successor.realized_precision_profile[module_path]
            transition_key = (name, current_precision, next_precision)
            cached_terms = self._quantization_element_terms.get(transition_key)
            if cached_terms is None:
                current_key = (name, current_precision)
                next_key = (name, next_precision)
                current_quantized = self._quantized_weights.get(current_key)
                if current_quantized is None:
                    current_quantized = pseudo_quantize_tensor(
                        value, current_precision, module=modules.get(module_path)
                    ).detach()
                    self._quantized_weights[current_key] = current_quantized
                next_quantized = self._quantized_weights.get(next_key)
                if next_quantized is None:
                    next_quantized = pseudo_quantize_tensor(
                        value, next_precision, module=modules.get(module_path)
                    ).detach()
                    self._quantized_weights[next_key] = next_quantized
                delta = next_quantized - current_quantized
                first, second, _score = self._cost_terms(delta, gradient, fisher)
                cached_terms = (first.detach(), second.detach())
                self._quantization_element_terms[transition_key] = cached_terms
            retained = retained_mask_for_parameter(
                value, current_slices.get(name, [])
            )
            first, second = cached_terms
            first_total += float(first[retained].sum().detach().cpu())
            second_total += float(second[retained].sum().detach().cpu())
            element_count += int(retained.sum().detach().cpu())
            tensor_count += 1
        total = first_total + second_total
        if total < 0.0:
            raise RuntimeError("weight_quantization_action_taylor_negative")
        return {
            "delta_J_prune": 0.0,
            "delta_J_WQ": total,
            "first_order_abs_sum": first_total,
            "second_order_abs_sum": second_total,
            "retained_quantized_parameter_count": element_count,
            "touched_parameter_tensor_count": tensor_count,
            "risk_refund": 0.0,
            "formula": "sum_retained(abs(g*(Q_next(w)-Q_current(w)))+0.5*abs(h*(Q_next(w)-Q_current(w))^2))",
            "elementwise_abs_before_reduction": True,
        }

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
