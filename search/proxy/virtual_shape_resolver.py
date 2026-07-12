"""Virtual shape and BOPS/size estimation from pruning phenotype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from ..candidate import CandidatePhenotype
from .parameter_slice_resolver import ParameterSlice


@dataclass(frozen=True)
class VirtualLayerShape:
    module_path: str
    module_type: str
    c_in_before: int | None
    c_in_after: int | None
    c_out_before: int | None
    c_out_after: int | None
    groups_before: int
    groups_after: int
    weight_shape_before: tuple[int, ...]
    parameter_count_before: int
    parameter_count_after: int
    kernel_size: tuple[int, ...] = ()
    h_out: int = 1
    w_out: int = 1


def _module_parameter_count(module: nn.Module, c_in: int | None, c_out: int | None, groups: int) -> int:
    if isinstance(module, nn.Conv2d):
        kh, kw = tuple(int(v) for v in module.kernel_size)
        return int((c_out or module.out_channels) * ((c_in or module.in_channels) // max(groups, 1)) * kh * kw + (c_out or module.out_channels if module.bias is not None else 0))
    if isinstance(module, nn.ConvTranspose2d):
        kh, kw = tuple(int(v) for v in module.kernel_size)
        return int((c_in or module.in_channels) * ((c_out or module.out_channels) // max(groups, 1)) * kh * kw + (c_out or module.out_channels if module.bias is not None else 0))
    if isinstance(module, nn.Linear):
        return int((c_in or module.in_features) * (c_out or module.out_features) + (c_out or module.out_features if module.bias is not None else 0))
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        width = c_out or module.num_features
        return int(width * int(module.weight is not None) + width * int(module.bias is not None))
    return sum(int(p.numel()) for p in module.parameters(recurse=False))


def resolve_virtual_shapes(
    model: nn.Module,
    phenotype: CandidatePhenotype,
    unit_to_parameter_slices: dict[str, list[ParameterSlice]],
) -> dict[str, VirtualLayerShape]:
    pruned: dict[tuple[str, str], set[int]] = {}
    modules = dict(model.named_modules())
    for unit_id in phenotype.pruned_unit_ids:
        for row in unit_to_parameter_slices.get(unit_id, []):
            module = modules.get(row.module_path)
            if isinstance(module, nn.ConvTranspose2d) and row.parameter_name.endswith(".weight"):
                axis_name = "in" if row.axis == 0 else "out"
            else:
                axis_name = "out" if row.axis == 0 else "in"
            if row.parameter_name.endswith(".bias"):
                axis_name = "out"
            pruned.setdefault((row.module_path, axis_name), set()).update(row.indices)
    result = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            c_in_before = int(module.in_channels)
            c_out_before = int(module.out_channels)
            groups_before = int(module.groups)
            c_in_after = c_in_before - len(pruned.get((name, "in"), set()))
            c_out_after = c_out_before - len(pruned.get((name, "out"), set()))
            groups_after = c_out_after if groups_before == c_in_before == c_out_before else groups_before
            kernel_size = tuple(int(v) for v in module.kernel_size)
        elif isinstance(module, nn.Linear):
            c_in_before = int(module.in_features)
            c_out_before = int(module.out_features)
            groups_before = groups_after = 1
            c_in_after = c_in_before - len(pruned.get((name, "in"), set()))
            c_out_after = c_out_before - len(pruned.get((name, "out"), set()))
            kernel_size = ()
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            c_in_before = c_out_before = int(module.num_features)
            groups_before = groups_after = 1
            c_in_after = c_out_after = c_out_before - len(pruned.get((name, "out"), set()))
            kernel_size = ()
        else:
            continue
        if c_in_after <= 0 or c_out_after <= 0:
            raise RuntimeError(f"missing_virtual_shape_mapping:empty_width:{name}")
        result[name] = VirtualLayerShape(
            module_path=name,
            module_type=type(module).__name__,
            c_in_before=c_in_before,
            c_in_after=c_in_after,
            c_out_before=c_out_before,
            c_out_after=c_out_after,
            groups_before=groups_before,
            groups_after=groups_after,
            weight_shape_before=tuple(int(v) for v in weight.shape),
            parameter_count_before=sum(int(p.numel()) for p in module.parameters(recurse=False)),
            parameter_count_after=_module_parameter_count(module, c_in_after, c_out_after, groups_after),
            kernel_size=kernel_size,
        )
    return result
