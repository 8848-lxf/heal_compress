"""Fisher/Taylor proxy with reusable per-parameter statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ..candidate import CandidatePhenotype
from .candidate_perturbation import parameter_slices_for_phenotype, retained_mask_for_parameter


@dataclass
class FisherStatistics:
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    fisher_diag: dict[str, torch.Tensor] = field(default_factory=dict)
    manifest_hash: str = ""
    statistics_version: str = "fisher-diagonal-v1"
    manifest: dict[str, Any] = field(default_factory=dict)


class FisherTaylorProxy:
    """Compute sum |g * dW| + 0.5 * h * dW^2 without mutating the model."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        statistics: FisherStatistics | None = None,
        unit_to_parameter_names: Mapping[str, list[Any]] | None = None,
        normalize_by_total: bool = True,
    ) -> None:
        self.model = model
        self.statistics = statistics or FisherStatistics()
        self.unit_to_parameter_names = {str(key): list(values) for key, values in (unit_to_parameter_names or {}).items()}
        self.normalize_by_total = bool(normalize_by_total)

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        if self.model is None:
            raise RuntimeError("fisher_statistics_missing:model")
        if not self.statistics.gradients or not self.statistics.fisher_diag:
            raise RuntimeError("fisher_statistics_missing")
        params = dict(self.model.named_parameters())
        slices_by_parameter = parameter_slices_for_phenotype(phenotype, self.unit_to_parameter_names)  # type: ignore[arg-type]
        pruned = self._cost_for_slices(params, slices_by_parameter)
        if not self.normalize_by_total:
            return float(pruned)
        total_slices = self._all_unit_slices_by_parameter()
        total = self._cost_for_slices(params, total_slices)
        if total <= 0.0:
            raise RuntimeError("fisher_unit_score_total_nonpositive")
        return float(pruned / max(total, 1.0e-12))

    def _all_unit_slices_by_parameter(self) -> dict[str, list[Any]]:
        by_parameter: dict[str, list[Any]] = {}
        seen: set[tuple[str, int, tuple[int, ...]]] = set()
        for rows in self.unit_to_parameter_names.values():
            for row in rows:
                key = (row.parameter_name, int(row.axis), tuple(int(value) for value in row.indices))
                if key in seen:
                    continue
                seen.add(key)
                by_parameter.setdefault(row.parameter_name, []).append(row)
        return by_parameter

    def _cost_for_slices(self, params: dict[str, torch.Tensor], slices_by_parameter: Mapping[str, list[Any]]) -> float:
        total = 0.0
        for name in sorted(slices_by_parameter):
            param = params.get(name)
            if param is None:
                continue
            mask = retained_mask_for_parameter(param.detach(), slices_by_parameter[name])
            delta = torch.where(mask, torch.zeros_like(param.detach()), -param.detach())
            grad = self.statistics.gradients.get(name)
            fisher = self.statistics.fisher_diag.get(name)
            if grad is not None:
                grad = grad.detach().to(device=delta.device, dtype=delta.dtype)
                total += float((grad.detach() * delta).abs().sum().cpu())
            if fisher is not None:
                fisher = fisher.detach().to(device=delta.device, dtype=delta.dtype)
                total += 0.5 * float((fisher.detach() * delta.pow(2)).sum().cpu())
        return float(total)
