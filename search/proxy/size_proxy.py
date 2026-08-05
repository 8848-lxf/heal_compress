"""Mixed-precision model-size retention proxy."""

from __future__ import annotations

from typing import Any

from ..candidate import CandidatePhenotype
from .parameter_slice_resolver import ParameterSlice
from .virtual_shape_resolver import resolve_virtual_shapes


BIT_WIDTHS = {"FP32": 32, "FP16": 16, "INT8": 8}


class SizeProxy:
    """Estimate retained weight bits relative to original FP32 weights."""

    def __init__(
        self,
        model: Any | None = None,
        layer_parameter_counts: dict[str, int] | None = None,
        unit_to_parameter_slices: dict[str, list[ParameterSlice]] | None = None,
        default_precision: str = "FP16",
        include_constant_parameters_in_size: bool = False,
        virtual_shape_cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.unit_to_parameter_slices = unit_to_parameter_slices or {}
        self.default_precision = str(default_precision).upper()
        self.include_constant_parameters_in_size = bool(
            include_constant_parameters_in_size
        )
        self.base_parameter_count = 0
        if layer_parameter_counts is None and model is not None:
            layer_parameter_counts = {
                name: int(module.weight.numel())
                for name, module in model.named_modules()
                if getattr(module, "weight", None) is not None
            }
            self.base_parameter_count = sum(
                int(parameter.numel()) for parameter in model.parameters()
            )
        self.layer_parameter_counts = dict(layer_parameter_counts or {})
        if self.base_parameter_count <= 0:
            self.base_parameter_count = sum(self.layer_parameter_counts.values()) or 1
        self.base_bits = sum(count * 32 for count in self.layer_parameter_counts.values()) or 1
        self.base_fp16_bits = sum(count * 16 for count in self.layer_parameter_counts.values()) or 1
        if self.include_constant_parameters_in_size:
            self.base_bits = self.base_parameter_count * 32
            self.base_fp16_bits = self.base_parameter_count * 16
        self._virtual_shape_cache = (
            virtual_shape_cache if virtual_shape_cache is not None else {}
        )

    def _virtual_shapes(self, phenotype: CandidatePhenotype) -> dict[str, Any]:
        key = tuple(phenotype.pruned_unit_ids)
        cached = self._virtual_shape_cache.get(key)
        if cached is None:
            cached = resolve_virtual_shapes(
                self.model, phenotype, self.unit_to_parameter_slices
            )
            self._virtual_shape_cache[key] = cached
        return cached

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        return float(self.evaluate_breakdown(phenotype)["R_size_vs_fp32"])

    def evaluate_breakdown(self, phenotype: CandidatePhenotype) -> dict[str, float]:
        if self.model is not None and self.unit_to_parameter_slices:
            shapes = self._virtual_shapes(phenotype)
            total_bits = 0
            parameter_count_after = 0
            parameter_count_before = 0
            for layer, shape in shapes.items():
                precision = phenotype.realized_precision_profile.get(
                    layer, self.default_precision
                )
                total_bits += int(shape.parameter_count_after) * BIT_WIDTHS.get(str(precision).upper(), 16)
                parameter_count_after += int(shape.parameter_count_after)
                parameter_count_before += int(shape.parameter_count_before)
            # Parameters outside the virtual shape universe are not affected by
            # structural channel pruning.  Keep them in both sides of the
            # physical parameter-retention ratio instead of silently dropping
            # them from the original-model reference.
            constant_parameter_count = max(
                0,
                int(self.base_parameter_count) - int(parameter_count_before),
            )
            if self.include_constant_parameters_in_size:
                total_bits += constant_parameter_count * BIT_WIDTHS.get(
                    self.default_precision, 32
                )
            parameter_count_after += constant_parameter_count
            parameter_retention = float(
                parameter_count_after / max(self.base_parameter_count, 1)
            )
            return {
                "R_size_vs_fp32": float(total_bits / self.base_bits),
                "R_size_vs_fp16_deploy": float(total_bits / self.base_fp16_bits),
                "size_bits_total": float(total_bits),
                "R_parameter_retention": parameter_retention,
                "parameter_pruning_rate": 1.0 - parameter_retention,
                "parameter_count_base": float(self.base_parameter_count),
                "parameter_count_after": float(parameter_count_after),
                "constant_untracked_parameter_count": float(
                    constant_parameter_count
                ),
            }
        total = 0
        for layer, count in self.layer_parameter_counts.items():
            precision = phenotype.realized_precision_profile.get(
                layer, self.default_precision
            )
            total += int(count) * BIT_WIDTHS.get(str(precision).upper(), 16)
        return {
            "R_size_vs_fp32": float(total / self.base_bits),
            "R_size_vs_fp16_deploy": float(total / self.base_fp16_bits),
            "size_bits_total": float(total),
            "R_parameter_retention": 1.0,
            "parameter_pruning_rate": 0.0,
            "parameter_count_base": float(self.base_parameter_count),
            "parameter_count_after": float(self.base_parameter_count),
            "constant_untracked_parameter_count": 0.0,
        }
