"""Candidate weight perturbation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from ..candidate import CandidatePhenotype
from .parameter_slice_resolver import ParameterSlice


def pseudo_quantize_tensor(weight: torch.Tensor, precision: str) -> torch.Tensor:
    precision = str(precision).upper()
    if precision == "FP32":
        return weight
    if precision == "FP16":
        return weight.to(torch.float16).to(weight.dtype)
    if precision == "INT8":
        amax = weight.detach().abs().amax()
        if float(amax) <= 0.0:
            return weight.clone()
        scale = amax / 127.0
        return torch.clamp(torch.round(weight / scale), -127, 127).to(weight.dtype) * scale
    raise ValueError(f"unsupported precision: {precision}")


def _slice_mask_like(value: torch.Tensor, axis: int, indices: tuple[int, ...]) -> torch.Tensor:
    mask = torch.ones_like(value, dtype=torch.bool)
    index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
    selector = [slice(None)] * value.ndim
    selector[int(axis)] = index
    mask[tuple(selector)] = False
    return mask


def retained_mask_for_parameter(
    parameter: torch.Tensor,
    slices: list[ParameterSlice],
) -> torch.Tensor:
    mask = torch.ones_like(parameter, dtype=torch.bool)
    for row in slices:
        index = torch.as_tensor(row.indices, dtype=torch.long, device=parameter.device)
        selector = [slice(None)] * parameter.ndim
        selector[int(row.axis)] = index
        mask[tuple(selector)] = False
    return mask


def parameter_slices_for_phenotype(
    phenotype: CandidatePhenotype,
    unit_to_parameter_slices: dict[str, list[ParameterSlice]],
) -> dict[str, list[ParameterSlice]]:
    by_parameter: dict[str, list[ParameterSlice]] = defaultdict(list)
    for unit_id in phenotype.pruned_unit_ids:
        for row in unit_to_parameter_slices.get(unit_id, []):
            by_parameter[row.parameter_name].append(row)
    return dict(by_parameter)


def effective_delta(
    parameter_name: str,
    parameter: torch.Tensor,
    phenotype: CandidatePhenotype,
    parameter_slices: list[ParameterSlice],
    *,
    layer_name: str | None = None,
) -> torch.Tensor:
    module_path = layer_name or parameter_name.rsplit(".", 1)[0]
    precision = phenotype.realized_precision_profile.get(module_path, "FP32")
    quantized = pseudo_quantize_tensor(parameter.detach(), precision)
    mask = retained_mask_for_parameter(parameter.detach(), parameter_slices)
    effective = torch.where(mask, quantized, torch.zeros_like(parameter.detach()))
    return effective - parameter.detach()
