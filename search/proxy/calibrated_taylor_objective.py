"""Frozen robust calibration for the unified structural/WQ/AQ Taylor proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .taylor_convergence import spearman_rank


FEATURE_KEYS = ("J_struct_gate", "J_WQ", "J_AQ", "J_WQ_x_J_AQ")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _positive_scale(values: Sequence[float]) -> float:
    positive = np.asarray([float(value) for value in values if float(value) > 0.0])
    if positive.size == 0:
        return 1.0
    q25, q75 = np.percentile(positive, (25.0, 75.0))
    iqr = float(q75 - q25)
    median = float(np.median(positive))
    return max(iqr, median, 1.0e-12)


def _features(row: Mapping[str, Any], scales: Mapping[str, float]) -> np.ndarray:
    struct = float(row.get("J_struct_gate", 0.0)) / float(scales["J_struct_gate"])
    weight = float(row.get("J_WQ", 0.0)) / float(scales["J_WQ"])
    activation = float(row.get("J_AQ", 0.0)) / float(scales["J_AQ"])
    return np.asarray(
        [struct, weight, activation, weight * activation], dtype=np.float64
    )


def _huber_nonnegative_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    delta: float = 1.35,
    iterations: int = 25,
) -> np.ndarray:
    from scipy.optimize import lsq_linear

    weights = np.ones(target.shape[0], dtype=np.float64)
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    for _ in range(int(iterations)):
        root = np.sqrt(weights)
        result = lsq_linear(
            matrix * root[:, None],
            target * root,
            bounds=(0.0, np.inf),
            method="trf",
        )
        if not result.success:
            raise RuntimeError(f"taylor_huber_nnls_failed:{result.message}")
        updated = np.asarray(result.x, dtype=np.float64)
        residual = target - matrix @ updated
        scale = max(float(np.median(np.abs(residual))) / 0.67448975, 1.0e-12)
        ratio = np.abs(residual) / (float(delta) * scale)
        next_weights = np.ones_like(ratio)
        selected = ratio > 1.0
        next_weights[selected] = 1.0 / ratio[selected]
        if np.allclose(updated, coefficients, rtol=1.0e-8, atol=1.0e-12):
            coefficients = updated
            break
        coefficients = updated
        weights = next_weights
    if not bool(np.isfinite(coefficients).all()) or bool((coefficients < 0.0).any()):
        raise RuntimeError("taylor_huber_nnls_coefficients_invalid")
    return coefficients


def _topk_recall(actual: Sequence[float], predicted: Sequence[float], k: int) -> float:
    count = min(max(1, int(k)), len(actual))
    actual_top = set(sorted(range(len(actual)), key=lambda i: (actual[i], i))[:count])
    predicted_top = set(
        sorted(range(len(predicted)), key=lambda i: (predicted[i], i))[:count]
    )
    return float(len(actual_top & predicted_top) / count)


def _validation_metrics(rows: Sequence[Mapping[str, Any]], predicted: Sequence[float]) -> dict[str, Any]:
    actual = [float(row["delta_task_loss"]) for row in rows]
    if len(rows) < 2:
        correlation = 1.0
    else:
        correlation = spearman_rank(actual, predicted)
    bins: list[dict[str, Any]] = []
    ordered = sorted(
        range(len(rows)), key=lambda index: float(rows[index]["R_bops_vs_fp32"])
    )
    for indices in np.array_split(np.asarray(ordered, dtype=np.int64), min(4, len(rows))):
        selected = [int(value) for value in indices]
        if not selected:
            continue
        bin_actual = [actual[index] for index in selected]
        bin_predicted = [predicted[index] for index in selected]
        bins.append(
            {
                "minimum_bops": min(float(rows[index]["R_bops_vs_fp32"]) for index in selected),
                "maximum_bops": max(float(rows[index]["R_bops_vs_fp32"]) for index in selected),
                "candidate_count": len(selected),
                "spearman": (
                    spearman_rank(bin_actual, bin_predicted)
                    if len(selected) >= 2
                    else 1.0
                ),
            }
        )
    modes: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("candidate_mode", "unknown"))
        modes[mode] = modes.get(mode, 0) + 1
    return {
        "candidate_count": len(rows),
        "spearman": float(correlation),
        "top_k": min(5, len(rows)),
        "top_k_good_candidate_recall": _topk_recall(actual, predicted, min(5, len(rows))),
        "bops_bins": bins,
        "candidate_mode_coverage": dict(sorted(modes.items())),
        "structural_candidate_count": sum(
            int(str(row.get("candidate_mode")) in {"pruning", "mixed"}) for row in rows
        ),
        "pure_quantization_candidate_count": sum(
            int(str(row.get("candidate_mode")) in {"weight_quant", "activation_quant"})
            for row in rows
        ),
    }


@dataclass(frozen=True)
class FrozenTaylorObjectiveCalibration:
    scales: dict[str, float]
    coefficients: dict[str, float]
    fit_candidate_hashes: tuple[str, ...]
    validation_candidate_hashes: tuple[str, ...]
    fit_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    raw_sum_fit_metrics: dict[str, Any]
    raw_sum_validation_metrics: dict[str, Any]
    calibration_data_hash: str
    schema_version: str = "frozen-task-loss-taylor-huber-nnls-v2"

    def score(self, *, j_struct: float, j_wq: float, j_aq: float) -> float:
        row = {
            "J_struct_gate": float(j_struct),
            "J_WQ": float(j_wq),
            "J_AQ": float(j_aq),
        }
        vector = _features(row, self.scales)
        coefficient = np.asarray(
            [float(self.coefficients[key]) for key in FEATURE_KEYS], dtype=np.float64
        )
        value = float(vector @ coefficient)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("calibrated_taylor_objective_invalid")
        return value

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["formula"] = (
            "alpha_s*(J_struct/s_struct)+alpha_w*(J_WQ/s_WQ)+"
            "alpha_a*(J_AQ/s_AQ)+alpha_wa*(J_WQ/s_WQ)*(J_AQ/s_AQ)"
        )
        payload["calibration_hash"] = _hash(payload)
        return payload


def fit_frozen_taylor_objective(
    fit_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    calibration_data_hash: str,
) -> FrozenTaylorObjectiveCalibration:
    """Fit fixed non-negative coefficients and audit them on disjoint candidates."""

    fit = tuple(dict(row) for row in fit_rows)
    validation = tuple(dict(row) for row in validation_rows)
    if len(fit) < len(FEATURE_KEYS) or not validation:
        raise ValueError("taylor_objective_calibration_candidate_count_insufficient")
    required_modes = {"pruning", "weight_quant", "activation_quant", "mixed"}
    for label, rows in (("fit", fit), ("validation", validation)):
        missing = required_modes - {str(row.get("candidate_mode")) for row in rows}
        if missing:
            raise ValueError(f"taylor_objective_{label}_candidate_modes_missing:{sorted(missing)}")
    scales = {
        key: _positive_scale([float(row.get(key, 0.0)) for row in fit])
        for key in FEATURE_KEYS[:3]
    }
    scales["J_WQ_x_J_AQ"] = 1.0
    matrix = np.stack([_features(row, scales) for row in fit])
    target = np.asarray([float(row["delta_task_loss"]) for row in fit], dtype=np.float64)
    coefficients_array = _huber_nonnegative_fit(matrix, target)
    coefficients = {
        key: float(coefficients_array[index]) for index, key in enumerate(FEATURE_KEYS)
    }
    fit_predicted = [float(value) for value in matrix @ coefficients_array]
    validation_matrix = np.stack([_features(row, scales) for row in validation])
    validation_predicted = [float(value) for value in validation_matrix @ coefficients_array]
    raw_fit_predicted = [
        float(row.get("J_struct_gate", 0.0))
        + float(row.get("J_WQ", 0.0))
        + float(row.get("J_AQ", 0.0))
        for row in fit
    ]
    raw_validation_predicted = [
        float(row.get("J_struct_gate", 0.0))
        + float(row.get("J_WQ", 0.0))
        + float(row.get("J_AQ", 0.0))
        for row in validation
    ]
    return FrozenTaylorObjectiveCalibration(
        scales=scales,
        coefficients=coefficients,
        fit_candidate_hashes=tuple(str(row["candidate_hash"]) for row in fit),
        validation_candidate_hashes=tuple(
            str(row["candidate_hash"]) for row in validation
        ),
        fit_metrics=_validation_metrics(fit, fit_predicted),
        validation_metrics=_validation_metrics(validation, validation_predicted),
        raw_sum_fit_metrics=_validation_metrics(fit, raw_fit_predicted),
        raw_sum_validation_metrics=_validation_metrics(
            validation, raw_validation_predicted
        ),
        calibration_data_hash=str(calibration_data_hash),
    )


__all__ = [
    "FEATURE_KEYS",
    "FrozenTaylorObjectiveCalibration",
    "fit_frozen_taylor_objective",
]
