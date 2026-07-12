"""Optional lightweight surrogate predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LightweightPredictor:
    min_samples: int = 20
    trained: bool = False

    def fit(self, rows: list[dict[str, Any]]) -> "LightweightPredictor":
        self.trained = len(rows) >= self.min_samples
        return self

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        if not self.trained:
            return {"available": 0.0, "mAP": 0.0, "latency": 0.0, "F2": 0.0}
        proxy = features.get("L_fisher", 0.0) + features.get("L_sqnr", 0.0)
        return {"available": 1.0, "mAP": max(0.0, 1.0 - proxy), "latency": features.get("R_bops", 1.0), "F2": proxy + features.get("R_bops", 1.0)}
