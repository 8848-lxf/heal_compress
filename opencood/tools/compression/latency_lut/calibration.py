from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CalibrationSample:
    candidate_id: str
    predicted_lut_ms: float
    real_engine_p50_ms: float
    real_engine_p90_ms: float | None = None
    deploy_mode: str = "single_engine_maxK"
    fixed_K: int = 29696
    channel_config_hash: str | None = None
    quant_config_hash: str | None = None
    features: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationSample":
        return cls(
            candidate_id=str(data.get("candidate_id")),
            predicted_lut_ms=float(data.get("predicted_lut_ms", data.get("latency_lut_raw_ms", 0.0))),
            real_engine_p50_ms=float(data.get("real_engine_p50_ms", data.get("real_latency_ms", 0.0))),
            real_engine_p90_ms=float(data["real_engine_p90_ms"]) if data.get("real_engine_p90_ms") is not None else None,
            deploy_mode=str(data.get("deploy_mode", "single_engine_maxK")),
            fixed_K=int(data.get("fixed_K", 29696)),
            channel_config_hash=data.get("channel_config_hash"),
            quant_config_hash=data.get("quant_config_hash"),
            features={str(k): float(v) for k, v in dict(data.get("features") or {}).items()},
        )


class IdentityCalibrationModel:
    model_type = "identity"

    def predict(self, latency_lut_ms: float, features: dict[str, Any] | None = None) -> float:
        return float(latency_lut_ms)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type}

    @classmethod
    def from_dict(cls, _data: dict[str, Any]) -> "IdentityCalibrationModel":
        return cls()


class LinearCalibrationModel:
    model_type = "linear"

    def __init__(self, feature_names: list[str] | None = None, coefficients: list[float] | None = None, intercept: float = 0.0) -> None:
        self.feature_names = feature_names or []
        self.coefficients = coefficients or [1.0]
        self.intercept = float(intercept)

    @classmethod
    def fit(cls, samples: list[CalibrationSample]) -> "LinearCalibrationModel":
        if not samples:
            return cls()
        feature_names = sorted({name for sample in samples for name in sample.features.keys()})
        import numpy as np

        x_rows = []
        y = []
        for sample in samples:
            x_rows.append([float(sample.predicted_lut_ms)] + [float(sample.features.get(name, 0.0)) for name in feature_names] + [1.0])
            y.append(float(sample.real_engine_p50_ms))
        x = np.asarray(x_rows, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        coef, *_ = np.linalg.lstsq(x, y_arr, rcond=None)
        return cls(feature_names=feature_names, coefficients=[float(v) for v in coef[:-1]], intercept=float(coef[-1]))

    def predict(self, latency_lut_ms: float, features: dict[str, Any] | None = None) -> float:
        features = features or {}
        values = [float(latency_lut_ms)] + [float(features.get(name, 0.0)) for name in self.feature_names]
        return float(sum(c * v for c, v in zip(self.coefficients, values)) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearCalibrationModel":
        return cls(
            feature_names=list(data.get("feature_names") or []),
            coefficients=[float(v) for v in data.get("coefficients", [1.0])],
            intercept=float(data.get("intercept", 0.0)),
        )


def load_calibration_samples(path: str | Path) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    if not Path(path).is_file():
        return samples
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(CalibrationSample.from_dict(json.loads(line)))
    return samples


def load_calibration_model(path: str | Path | None) -> IdentityCalibrationModel | LinearCalibrationModel:
    if path is None or not Path(path).is_file():
        return IdentityCalibrationModel()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("model_type") == "linear":
        return LinearCalibrationModel.from_dict(data)
    return IdentityCalibrationModel.from_dict(data)


def regression_metrics(samples: list[CalibrationSample], predictions: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"MAE": None, "MAPE": None, "RMSE": None, "Pearson": None, "Spearman": None}
    y = [float(sample.real_engine_p50_ms) for sample in samples]
    pred = [float(v) for v in predictions]
    err = [p - t for p, t in zip(pred, y)]
    mae = sum(abs(v) for v in err) / len(err)
    mape = sum(abs(p - t) / max(abs(t), 1e-9) for p, t in zip(pred, y)) / len(y) * 100.0
    rmse = math.sqrt(sum(v * v for v in err) / len(err))
    return {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "Pearson": _pearson(pred, y),
        "Spearman": _spearman(pred, y),
    }


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    for rank, (_value, idx) in enumerate(ordered, start=1):
        ranks[idx] = float(rank)
    return ranks


def _spearman(a: list[float], b: list[float]) -> float | None:
    return _pearson(_rank(a), _rank(b))
