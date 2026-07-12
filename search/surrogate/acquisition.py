"""Exploration acquisition helpers."""

from __future__ import annotations

from typing import Any


def expected_improvement_or_diversity(prediction: dict[str, float], diversity_score: float) -> float:
    if not prediction.get("available"):
        return float(diversity_score)
    return float(-prediction.get("F2", 0.0) + 0.1 * diversity_score)
