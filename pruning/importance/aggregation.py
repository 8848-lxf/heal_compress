"""Dependency-member to coupled-unit score aggregation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def coupled_dependency_mean(scores: Sequence[float]) -> float:
    """Return the arithmetic mean of finite dependency-member scores."""

    finite = [float(value) for value in scores if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("inf")


__all__ = ["coupled_dependency_mean"]
