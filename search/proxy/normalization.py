"""Fixed robust normalization statistics for Stage-1 objective terms."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class NormalizationStats:
    medians: dict[str, float] = field(default_factory=dict)
    version: str = "fixed-median-v1"

    def normalize(self, key: str, value: float) -> float:
        denom = float(self.medians.get(key, 1.0) or 1.0)
        return float(value) / max(abs(denom), 1.0e-12)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "medians": dict(self.medians)}


def build_normalization_stats(rows: Iterable[dict[str, float]], keys: list[str]) -> NormalizationStats:
    collected = list(rows)
    medians = {}
    for key in keys:
        values = [float(row[key]) for row in collected if key in row]
        medians[key] = float(statistics.median(values)) if values else 1.0
    return NormalizationStats(medians=medians)
