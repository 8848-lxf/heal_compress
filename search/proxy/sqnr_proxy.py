"""SQNR quantization-error proxy."""

from __future__ import annotations

from typing import Any

import torch

from ..candidate import CandidatePhenotype
from .candidate_perturbation import parameter_slices_for_phenotype, pseudo_quantize_tensor, retained_mask_for_parameter
from .parameter_slice_resolver import ParameterSlice


class SQNRProxy:
    """Noise-energy ratio proxy using realized precision."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        epsilon: float = 1.0e-12,
        unit_to_parameter_slices: dict[str, list[ParameterSlice]] | None = None,
    ) -> None:
        self.model = model
        self.epsilon = float(epsilon)
        self.unit_to_parameter_slices = unit_to_parameter_slices or {}

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        if self.model is None:
            return 0.0
        modules = dict(self.model.named_modules())
        parameter_pruned = parameter_slices_for_phenotype(phenotype, self.unit_to_parameter_slices)
        total = 0.0
        for layer, precision in phenotype.realized_precision_profile.items():
            module = modules.get(layer)
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            w = weight.detach()
            q = pseudo_quantize_tensor(w, precision)
            mask = retained_mask_for_parameter(w, parameter_pruned.get(f"{layer}.weight", []))
            diff = torch.where(mask, q - w, torch.zeros_like(w))
            signal = torch.where(mask, w, torch.zeros_like(w))
            numerator = float(diff.pow(2).sum().cpu())
            denominator = float(signal.pow(2).sum().cpu()) + self.epsilon
            total += numerator / denominator
        return float(total)
