"""CPU-pure latency statistics."""

from __future__ import annotations

import math
import statistics
from typing import Iterable

from ..types import LatencySummary


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latency(values: Iterable[float]) -> LatencySummary:
    """Summarize milliseconds with linear-interpolated percentiles."""

    rows = [float(value) for value in values]
    if not rows:
        return LatencySummary(0, None, None, None, None, None, None, None)
    return LatencySummary(
        count=len(rows),
        mean_ms=float(statistics.mean(rows)),
        p50_ms=_percentile(rows, 0.50),
        p90_ms=_percentile(rows, 0.90),
        p95_ms=_percentile(rows, 0.95),
        p99_ms=_percentile(rows, 0.99),
        minimum_ms=min(rows),
        maximum_ms=max(rows),
    )
