"""Versioned importance normalization used for global comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..config import ImportanceNormalizationConfig, ImportanceNormalizationStrategy


FORMAL_NORMALIZATION_NAME = "coupled_dependency_mean_then_scope_mean_v1"


def normalize_scope_scores(
    scope_scores: Mapping[str, Sequence[float]],
    *,
    config: ImportanceNormalizationConfig | None = None,
) -> dict[str, list[float]]:
    r"""Normalize each dependency scope by its finite arithmetic mean.

    For raw coupled-unit scores :math:`r_{s,i}`, this implements
    :math:`\hat r_{s,i}=r_{s,i}/(\operatorname{mean}_{j\in F_s}r_{s,j}+\epsilon)`,
    where :math:`F_s` contains finite scores in scope ``s``. Non-finite
    values remain ``inf`` so they cannot be selected by lowest-score ranking.
    """

    cfg = config or ImportanceNormalizationConfig()
    normalized: dict[str, list[float]] = {}
    for scope_id, values in scope_scores.items():
        rows = [float(value) for value in values]
        if cfg.strategy is ImportanceNormalizationStrategy.NONE:
            normalized[str(scope_id)] = rows
            continue
        finite = [value for value in rows if math.isfinite(value)]
        if not finite:
            normalized[str(scope_id)] = [float("inf") for _ in rows]
            continue
        denominator = sum(finite) / len(finite)
        # Keep an exactly-zero scope comparable and deterministic. Such a
        # scope contains no observed Taylor signal, hence all finite scores 0.
        if abs(denominator) <= cfg.epsilon:
            normalized[str(scope_id)] = [0.0 if math.isfinite(value) else float("inf") for value in rows]
        else:
            normalized[str(scope_id)] = [
                value / (denominator + cfg.epsilon) if math.isfinite(value) else float("inf")
                for value in rows
            ]
    return normalized


__all__ = ["FORMAL_NORMALIZATION_NAME", "normalize_scope_scores"]
