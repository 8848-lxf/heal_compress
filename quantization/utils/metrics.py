from __future__ import annotations

from typing import Any


def numeric_summary(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "mean": None, "max": None}
    ordered = sorted(float(v) for v in values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
        return float(ordered[int(idx)])

    return {
        "min": ordered[0],
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }
