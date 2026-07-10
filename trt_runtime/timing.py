from __future__ import annotations

import statistics
from typing import Any, Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(q)))))
    return float(ordered[index])


def summarize_latency(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def series(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get("success") and row.get(key) is not None]

    out: dict[str, Any] = {}
    for prefix, key in (
        ("total", "total_latency_ms"),
        ("forward", "forward_latency_ms"),
        ("postprocess", "postprocess_latency_ms"),
        ("data_to_gpu", "data_to_gpu_latency_ms"),
        ("unaccounted", "unaccounted_time_ms"),
    ):
        values = series(key)
        out[f"{prefix}_mean_ms"] = float(statistics.mean(values)) if values else None
        out[f"{prefix}_p50_ms"] = percentile(values, 0.50)
        out[f"{prefix}_p90_ms"] = percentile(values, 0.90)
        out[f"{prefix}_p95_ms"] = percentile(values, 0.95)
        out[f"{prefix}_p99_ms"] = percentile(values, 0.99)
    return out
