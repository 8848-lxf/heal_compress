"""Scan physical truth directly from a live model and its state dictionary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..exceptions import PhysicalStructureMismatchError
from ..policies.protection import infer_directional_protection
from ..types import PhysicalModuleSnapshot, PhysicalStructureSnapshot
from .schemas import SNAPSHOT_SCHEMA_VERSION


SUPPORTED_PHYSICAL_MODULES = (
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.Linear,
    nn.modules.batchnorm._BatchNorm,
)


def _integer_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (tuple, list, torch.Size)):
        return [int(item) for item in value]
    return []


def _state_key(state_dict: Mapping[str, Any], module_name: str, parameter_name: str) -> str:
    exact = f"{module_name}.{parameter_name}"
    if exact in state_dict:
        return exact
    matches = sorted(str(key) for key in state_dict if str(key).endswith(exact))
    return matches[0] if len(matches) == 1 else ""


def build_physical_structure_snapshot(
    model: nn.Module,
    *,
    state_dict: Mapping[str, Any] | None = None,
) -> PhysicalStructureSnapshot:
    """Build snapshot v2 from live attributes, parameters and buffers."""

    state = state_dict if state_dict is not None else model.state_dict()
    rows: list[PhysicalModuleSnapshot] = []
    for order, (name, module) in enumerate(
        (item for item in model.named_modules() if item[0] and isinstance(item[1], SUPPORTED_PHYSICAL_MODULES))
    ):
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        weight_key = _state_key(state, name, "weight")
        bias_key = _state_key(state, name, "bias")
        if weight is not None:
            if not weight_key:
                raise PhysicalStructureMismatchError(f"live weight is absent from state_dict: {name}.weight")
            if tuple(state[weight_key].shape) != tuple(weight.shape):
                raise PhysicalStructureMismatchError(
                    f"live/state weight shape mismatch for {name}: {tuple(weight.shape)} != {tuple(state[weight_key].shape)}"
                )
        if bias is not None:
            if not bias_key or tuple(state[bias_key].shape) != tuple(bias.shape):
                raise PhysicalStructureMismatchError(f"live/state bias shape mismatch for {name}")
        direct_parameters = list(module.parameters(recurse=False))
        policy = infer_directional_protection(name, module)
        rows.append(
            PhysicalModuleSnapshot(
                canonical_module_name=name,
                module_type=type(module).__name__,
                canonical_order=order,
                in_channels=int(module.in_channels) if hasattr(module, "in_channels") else None,
                out_channels=int(module.out_channels) if hasattr(module, "out_channels") else None,
                in_features=int(module.in_features) if hasattr(module, "in_features") else None,
                out_features=int(module.out_features) if hasattr(module, "out_features") else None,
                num_features=int(module.num_features) if hasattr(module, "num_features") else None,
                groups=int(getattr(module, "groups", 1) or 1),
                kernel_size=_integer_list(getattr(module, "kernel_size", None)),
                stride=_integer_list(getattr(module, "stride", None)),
                padding=_integer_list(getattr(module, "padding", None)),
                dilation=_integer_list(getattr(module, "dilation", None)),
                output_padding=_integer_list(getattr(module, "output_padding", None)),
                weight_shape=list(weight.shape) if torch.is_tensor(weight) else [],
                bias_shape=list(bias.shape) if torch.is_tensor(bias) else [],
                parameter_count=sum(int(parameter.numel()) for parameter in direct_parameters),
                parameter_size_bytes=sum(
                    int(parameter.numel() * parameter.element_size()) for parameter in direct_parameters
                ),
                source_state_dict_key=weight_key,
                source_bias_state_dict_key=bias_key,
                protection_policy={
                    "root_pruning_allowed": policy.root_pruning_allowed,
                    "input_dependency_pruning_allowed": policy.input_dependency_pruning_allowed,
                    "output_dependency_pruning_allowed": policy.output_dependency_pruning_allowed,
                    "fixed_output_contract": policy.fixed_output_contract,
                    "protection_reason": policy.protection_reason,
                },
            )
        )
    return PhysicalStructureSnapshot(
        modules=rows,
        parameter_count=sum(row.parameter_count for row in rows),
        parameter_size_bytes=sum(row.parameter_size_bytes for row in rows),
        weighted_module_count=sum(bool(row.weight_shape) for row in rows),
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
        generated_from="live_model_and_state_dict",
    )


__all__ = ["SUPPORTED_PHYSICAL_MODULES", "build_physical_structure_snapshot"]
