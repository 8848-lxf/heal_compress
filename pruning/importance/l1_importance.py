from __future__ import annotations

from typing import Any


def l1_scores_for_module(module: Any) -> list[float]:
    weight = getattr(module, "weight", None)
    if weight is None:
        return []
    tensor = weight.detach().abs()
    if tensor.ndim >= 2:
        return tensor.reshape(tensor.shape[0], -1).sum(dim=1).cpu().tolist()
    return tensor.cpu().tolist()
