"""BOPS retention proxy."""

from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

from .size_proxy import BIT_WIDTHS
from ..candidate import CandidatePhenotype
from .parameter_slice_resolver import ParameterSlice
from .runtime_shape_profiler import RuntimeLayerShape
from .virtual_shape_resolver import resolve_virtual_shapes


class BOPSProxy:
    """Approximate BOPS with retained layer op counts and realized precision."""

    def __init__(
        self,
        model: Any | None = None,
        layer_ops: dict[str, int] | None = None,
        activation_bits: int = 16,
        unit_to_parameter_slices: dict[str, list[ParameterSlice]] | None = None,
        runtime_shapes: Sequence[RuntimeLayerShape] | None = None,
        include_module_paths: Sequence[str] | None = None,
        exclude_module_paths: Sequence[str] = (),
        default_precision: str = "FP16",
        virtual_shape_cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.unit_to_parameter_slices = unit_to_parameter_slices or {}
        included = None if include_module_paths is None else {str(value) for value in include_module_paths}
        excluded = {str(value) for value in exclude_module_paths}
        self.runtime_shapes = tuple(
            row
            for row in (runtime_shapes or ())
            if (included is None or str(row.module_path) in included)
            and str(row.module_path) not in excluded
        )
        self.default_precision = str(default_precision).upper()
        if layer_ops is None and model is not None:
            layer_ops = {
                name: int(module.weight.numel())
                for name, module in model.named_modules()
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and getattr(module, "weight", None) is not None
                and (included is None or name in included)
                and name not in excluded
            }
        self.layer_ops = {
            str(name): int(value)
            for name, value in dict(layer_ops or {}).items()
            if (included is None or str(name) in included) and str(name) not in excluded
        }
        self.activation_bits = int(activation_bits)
        self.base_bops = sum(count * 32 * 32 for count in self.layer_ops.values()) or 1
        self.base_fp16_bops = self._base_fp16_bops()
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

    def _base_fp16_bops(self) -> float:
        if self.runtime_shapes:
            return sum(float(shape.macs) * 16.0 * 16.0 for shape in self.runtime_shapes) or 1.0
        return sum(float(count) * 16.0 * 16.0 for count in self.layer_ops.values()) or 1.0

    @staticmethod
    def _activation_bits_for_precision(precision: str) -> int:
        return BIT_WIDTHS.get(str(precision).upper(), 16)

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        return float(self.evaluate_breakdown(phenotype)["R_bops_vs_fp32"])

    def evaluate_breakdown(self, phenotype: CandidatePhenotype) -> dict[str, Any]:
        if self.runtime_shapes:
            virtual = (
                self._virtual_shapes(phenotype)
                if self.model is not None and self.unit_to_parameter_slices
                else {}
            )
            total = 0.0
            int8_macs = 0.0
            total_macs = 0.0
            rows = []
            counted: set[tuple[str, int]] = set()
            for shape in self.runtime_shapes:
                key = (shape.module_path, shape.call_index)
                if key in counted:
                    continue
                counted.add(key)
                precision = phenotype.realized_precision_profile.get(
                    shape.module_path, self.default_precision
                )
                weight_bits = BIT_WIDTHS.get(str(precision).upper(), 16)
                activation_bits = self._activation_bits_for_precision(str(precision))
                vshape = virtual.get(shape.module_path)
                c_in_after = int(getattr(vshape, "c_in_after", shape.c_in or 1) or 1)
                c_out_after = int(getattr(vshape, "c_out_after", shape.c_out or 1) or 1)
                groups_after = int(getattr(vshape, "groups_after", shape.groups) or 1)
                if vshape is None and float(shape.macs) > 0:
                    macs = float(shape.macs)
                elif shape.module_type in {"Conv2d", "ConvTranspose2d"}:
                    kh, kw = shape.kernel_size or (1, 1)
                    macs = float(shape.h_out * shape.w_out * kh * kw * c_in_after * c_out_after / max(groups_after, 1))
                elif shape.module_type == "Linear":
                    macs = float(c_in_after * c_out_after)
                else:
                    continue
                bops = macs * weight_bits * activation_bits
                total += bops
                total_macs += macs
                if str(precision).upper() == "INT8":
                    int8_macs += macs
                rows.append(
                    {
                        "module_path": shape.module_path,
                        "call_index": shape.call_index,
                        "precision_group_id": shape.precision_group_id,
                        "stage1_realized_precision": str(precision).upper(),
                        "MACs": macs,
                        "weight_bits": weight_bits,
                        "activation_bits": activation_bits,
                        "BOPS": bops,
                        "C_in_before": shape.c_in,
                        "C_in_after": c_in_after,
                        "C_out_before": shape.c_out,
                        "C_out_after": c_out_after,
                        "H_out": shape.h_out,
                        "W_out": shape.w_out,
                        "groups_before": shape.groups,
                        "groups_after": groups_after,
                    }
                )
            fp16_base = self.base_fp16_bops
            fp32_base = sum(float(shape.macs) * 32.0 * 32.0 for shape in self.runtime_shapes) or 1.0
            return {
                "R_bops_vs_fp16_deploy": float(total / fp16_base),
                "R_bops_vs_fp32": float(total / fp32_base),
                "R_bops": float(total / fp16_base),
                "int8_macs_ratio": float(int8_macs / max(total_macs, 1.0)),
                "bops_total": float(total),
                "bops_fp16_baseline": float(fp16_base),
                "bops_fp32_baseline": float(fp32_base),
                "breakdown": rows,
            }
        if self.model is not None and self.unit_to_parameter_slices:
            total = 0
            for layer, shape in self._virtual_shapes(phenotype).items():
                precision = phenotype.realized_precision_profile.get(
                    layer, self.default_precision
                )
                weight_bits = BIT_WIDTHS.get(str(precision).upper(), 16)
                activation_bits = self._activation_bits_for_precision(str(precision))
                if shape.module_type in {"Conv2d", "ConvTranspose2d"}:
                    kh, kw = shape.kernel_size or (1, 1)
                    macs = int(shape.h_out * shape.w_out * kh * kw * (shape.c_in_after or 1) * (shape.c_out_after or 1) / max(shape.groups_after, 1))
                elif shape.module_type == "Linear":
                    macs = int((shape.c_in_after or 1) * (shape.c_out_after or 1))
                else:
                    continue
                total += macs * weight_bits * activation_bits
            return {
                "R_bops_vs_fp16_deploy": float(total / self.base_fp16_bops),
                "R_bops_vs_fp32": float(total / self.base_bops),
                "R_bops": float(total / self.base_fp16_bops),
                "int8_macs_ratio": 0.0,
                "bops_total": float(total),
                "bops_fp16_baseline": float(self.base_fp16_bops),
                "bops_fp32_baseline": float(self.base_bops),
                "breakdown": [],
            }
        total = 0
        for layer, ops in self.layer_ops.items():
            precision = phenotype.realized_precision_profile.get(
                layer, self.default_precision
            )
            weight_bits = BIT_WIDTHS.get(str(precision).upper(), 16)
            activation_bits = self._activation_bits_for_precision(str(precision))
            total += int(ops) * weight_bits * activation_bits
        return {
            "R_bops_vs_fp16_deploy": float(total / self.base_fp16_bops),
            "R_bops_vs_fp32": float(total / self.base_bops),
            "R_bops": float(total / self.base_fp16_bops),
            "int8_macs_ratio": 0.0,
            "bops_total": float(total),
            "bops_fp16_baseline": float(self.base_fp16_bops),
            "bops_fp32_baseline": float(self.base_bops),
            "breakdown": [],
        }
