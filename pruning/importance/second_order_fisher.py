"""Second-order Fisher compatibility scorer."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def fisher_parameter_slice(parameter: torch.Tensor, indices: Sequence[int], axis: int) -> float:
    """Return ``sum((w * gradient) ** 2)`` for one parameter slice."""

    gradient = getattr(parameter, "_importance_grad", None)
    if gradient is None:
        gradient = parameter.grad
    if gradient is None:
        return float("inf")
    index = torch.as_tensor(sorted({int(value) for value in indices}), device=parameter.device)
    value = parameter.index_select(axis, index) * gradient.index_select(axis, index)
    return float(value.square().sum().detach().cpu())


__all__ = ["fisher_parameter_slice"]
