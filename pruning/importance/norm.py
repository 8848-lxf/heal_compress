"""L1/L2 importance compatibility implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def norm_parameter_slice(parameter: torch.Tensor, indices: Sequence[int], axis: int, *, order: int) -> float:
    """Compute an L1 or L2 norm on one indexed parameter slice."""

    index = torch.as_tensor([int(value) for value in indices], device=parameter.device, dtype=torch.long)
    selected = parameter.detach().index_select(axis, index)
    if order == 1:
        return float(selected.abs().sum().cpu())
    if order == 2:
        return float(selected.square().sum().sqrt().cpu())
    raise ValueError(f"unsupported norm order: {order}")


def score_norm_units(model: torch.nn.Module, units: Sequence[Any], *, order: int) -> dict[str, float]:
    """Score root parameter slices for optional L1/L2 modes."""

    modules = dict(model.named_modules())
    result: dict[str, float] = {}
    for unit in units:
        module = modules.get(str(unit.root_module_path))
        if module is None or getattr(module, "weight", None) is None:
            result[str(unit.stable_id)] = float("inf")
            continue
        axis_name = str(unit.root_axis)
        if module.__class__.__name__ == "ConvTranspose2d":
            axis = 1 if axis_name in {"out", "channel"} else 0
        else:
            axis = 0 if axis_name in {"out", "channel"} else 1
        indices = getattr(unit, "root_indices", [getattr(unit, "root_channel_index", 0)])
        result[str(unit.stable_id)] = norm_parameter_slice(module.weight, indices, axis, order=order)
    return result


__all__ = ["norm_parameter_slice", "score_norm_units"]
