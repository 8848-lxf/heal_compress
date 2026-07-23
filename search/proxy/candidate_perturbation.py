"""Candidate weight perturbation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from ..candidate import CandidatePhenotype
from .parameter_slice_resolver import ParameterSlice


def pseudo_quantize_tensor(
    weight: torch.Tensor,
    precision: str,
    *,
    module: Any | None = None,
) -> torch.Tensor:
    """Mirror production weight granularity on the PyTorch parameter layout."""

    precision = str(precision).upper()
    if precision == "FP32":
        return weight
    if precision == "FP16":
        return weight.to(torch.float16).to(weight.dtype)
    if precision == "INT8":
        if weight.ndim == 0:
            amax = weight.detach().abs()
            scale = amax / 127.0
        else:
            # ONNX Conv/ConvTranspose/MatMul use output-channel axes 0/1/1.
            # On the source PyTorch layout Linear is [out,in], hence axis 0.
            axis = 1 if isinstance(module, torch.nn.ConvTranspose2d) else 0
            if axis >= weight.ndim:
                axis = 0
            reduce_axes = tuple(index for index in range(weight.ndim) if index != axis)
            amax = weight.detach().abs().amax(dim=reduce_axes, keepdim=True) if reduce_axes else weight.detach().abs()
            scale = amax / 127.0
        scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        return torch.clamp(torch.round(weight / scale), -127, 127).to(weight.dtype) * scale
    raise ValueError(f"unsupported precision: {precision}")


def deployment_precision(value: str) -> str:
    """Map deployment labels to the internal coupled precision state."""

    text = str(value).upper()
    aliases = {
        "W32A32": "FP32",
        "A32": "FP32",
        "W16A16": "FP16",
        "A16": "FP16",
        "W8A8": "INT8",
        "A8": "INT8",
    }
    return aliases.get(text, text)


def pseudo_quantize_activation(
    activation: torch.Tensor,
    precision: str,
    *,
    channel_axis: int | None = None,
) -> torch.Tensor:
    """Deterministic activation fake quantization for the Stage-1 proxy."""

    precision = deployment_precision(precision)
    if precision == "FP32":
        return activation
    if precision == "FP16":
        return activation.to(torch.float16).to(activation.dtype)
    if precision == "INT8":
        if not activation.is_floating_point():
            raise TypeError("activation_fake_quant_requires_floating_tensor")
        if channel_axis is None:
            amax = activation.detach().abs().amax()
        else:
            axis = int(channel_axis) % activation.ndim
            reduce_axes = tuple(index for index in range(activation.ndim) if index != axis)
            amax = (
                activation.detach().abs().amax(dim=reduce_axes, keepdim=True)
                if reduce_axes
                else activation.detach().abs()
            )
        scale = amax / 127.0
        scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        return torch.clamp(torch.round(activation / scale), -127, 127).to(activation.dtype) * scale
    raise ValueError(f"unsupported activation precision: {precision}")


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
    module: Any | None = None,
) -> torch.Tensor:
    module_path = layer_name or parameter_name.rsplit(".", 1)[0]
    precision = phenotype.realized_precision_profile.get(module_path, "FP32")
    quantized = pseudo_quantize_tensor(parameter.detach(), precision, module=module)
    mask = retained_mask_for_parameter(parameter.detach(), parameter_slices)
    effective = torch.where(mask, quantized, torch.zeros_like(parameter.detach()))
    return effective - parameter.detach()
