"""Calibrate and freeze the exponential joint-Taylor task-score scale."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..hashing import canonical_json_hash


class TauCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProxyScale:
    mapping: str
    formula: str
    tau: float
    max_allowed_absolute_mAP_drop: float
    mAP_reference: float
    mAP_min_allowed: float
    safe_anchor_candidate_hash: str
    safe_anchor_physical_hash: str
    safe_anchor_precision_profile_hash: str
    safe_anchor_engine_hash: str
    safe_anchor_mAP: float
    safe_anchor_delta_mAP: float
    safe_anchor_L_joint: float
    safe_anchor_R_prune: float
    safe_anchor_R_BOPS: float
    calibration_pool: tuple[dict[str, Any], ...]
    tau_boundary_underresolved: bool
    calibration_manifest_hash: str
    validation_manifest_hash: str
    code_commit: str
    created_at: str
    calibration_passed: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["calibration_pool"] = [dict(row) for row in self.calibration_pool]
        return payload


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile_requires_values")
    location = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = location - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def task_score_distribution(losses: Sequence[float], *, tau: float) -> dict[str, Any]:
    scale = float(tau)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("joint_taylor_tau_must_be_finite_positive")
    exponents = [float(value) / scale for value in losses]
    scores = sorted(math.exp(-min(value, 80.0)) for value in exponents)
    if not scores:
        raise ValueError("task_score_distribution_requires_values")
    low_cluster = sum(value <= 0.05 for value in scores) / len(scores)
    high_cluster = sum(value >= 0.95 for value in scores) / len(scores)
    return {
        "count": len(scores),
        "S_task_min": scores[0],
        "S_task_max": scores[-1],
        "S_task_mean": sum(scores) / len(scores),
        "S_task_median": _quantile(scores, 0.50),
        "S_task_P10": _quantile(scores, 0.10),
        "S_task_P25": _quantile(scores, 0.25),
        "S_task_P75": _quantile(scores, 0.75),
        "S_task_P90": _quantile(scores, 0.90),
        "S_task_le_0_01_count": sum(value <= 0.01 for value in scores),
        "S_task_ge_0_99_count": sum(value >= 0.99 for value in scores),
        "saturated_count": sum(value > 80.0 for value in exponents),
        "mapping_resolution_warning": bool(low_cluster >= 0.80 or high_cluster >= 0.80),
    }


def _tau_pool(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"strict_fp16", "maximal_legal_int8"}
    return [
        dict(row)
        for row in rows
        if bool(row.get("valid_for_tau", False))
        and str(row.get("precision_variant", "")) in allowed
        and math.isfinite(float(row.get("mAP", float("nan"))))
        and math.isfinite(float(row.get("L_joint", float("nan"))))
    ]


def calibrate_tau(
    anchor_rows: Sequence[Mapping[str, Any]],
    *,
    mAP_reference: float,
    max_allowed_absolute_mAP_drop: float = 0.1,
    code_commit: str = "",
    created_at: str | None = None,
) -> ProxyScale:
    reference = float(mAP_reference)
    max_drop = float(max_allowed_absolute_mAP_drop)
    if not math.isfinite(reference) or not math.isfinite(max_drop) or max_drop <= 0.0:
        raise TauCalibrationError("invalid_map_reference_or_absolute_drop")
    minimum = reference - max_drop
    pool = _tau_pool(anchor_rows)
    for row in pool:
        row["delta_mAP"] = reference - float(row["mAP"])
    safe = [
        row
        for row in pool
        if float(row["delta_mAP"]) <= max_drop + 1.0e-12
        and float(row["L_joint"]) > 0.0
    ]
    if not safe:
        raise TauCalibrationError("no_positive_loss_safe_anchor")
    safe.sort(
        key=lambda row: (
            abs(max_drop - float(row["delta_mAP"])),
            -float(row.get("R_prune", 0.0)),
            float(row.get("R_BOPS", float("inf"))),
            str(row.get("candidate_hash", "")),
        )
    )
    selected = safe[0]
    safe_loss = float(selected["L_joint"])
    tau = safe_loss / math.log(2.0)
    if not math.isfinite(tau) or tau <= 0.0:
        raise TauCalibrationError("safe_anchor_loss_invalid")
    positive_pool = [row for row in pool if float(row["L_joint"]) > 0.0]
    unsafe = [row for row in positive_pool if float(row["delta_mAP"]) > max_drop + 1.0e-12]
    underresolved = bool(positive_pool and not unsafe)
    diagnostics = task_score_distribution(
        [float(row["L_joint"]) for row in pool], tau=tau
    )
    return ProxyScale(
        mapping="exponential",
        formula="exp(-L_joint/tau)",
        tau=tau,
        max_allowed_absolute_mAP_drop=max_drop,
        mAP_reference=reference,
        mAP_min_allowed=minimum,
        safe_anchor_candidate_hash=str(selected.get("candidate_hash", "")),
        safe_anchor_physical_hash=str(selected.get("physical_hash", "")),
        safe_anchor_precision_profile_hash=str(
            selected.get("precision_profile_hash", "")
        ),
        safe_anchor_engine_hash=str(selected.get("engine_hash", "")),
        safe_anchor_mAP=float(selected["mAP"]),
        safe_anchor_delta_mAP=float(selected["delta_mAP"]),
        safe_anchor_L_joint=safe_loss,
        safe_anchor_R_prune=float(selected.get("R_prune", 0.0)),
        safe_anchor_R_BOPS=float(selected.get("R_BOPS", float("nan"))),
        calibration_pool=tuple(pool),
        tau_boundary_underresolved=underresolved,
        calibration_manifest_hash=str(selected.get("calibration_manifest_hash", "")),
        validation_manifest_hash=str(selected.get("validation_manifest_hash", "")),
        code_commit=str(code_commit),
        created_at=created_at or datetime.now(timezone.utc).astimezone().isoformat(),
        diagnostics=diagnostics,
    )


def proxy_scale_hash(payload: Mapping[str, Any] | ProxyScale) -> str:
    data = payload.to_dict() if isinstance(payload, ProxyScale) else dict(payload)
    data.pop("proxy_scale_hash", None)
    return canonical_json_hash(data)


def write_proxy_scale(path: str | Path, scale: ProxyScale) -> str:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = scale.to_dict()
    digest = proxy_scale_hash(payload)
    payload["proxy_scale_hash"] = digest
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(target, 0o444)
    return digest


def validate_fixed_proxy_scale(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    generation: int,
) -> dict[str, Any]:
    expected_hash = proxy_scale_hash(expected)
    observed_hash = proxy_scale_hash(observed)
    if expected_hash != observed_hash:
        raise RuntimeError(f"proxy_scale_changed_during_ga:generation={int(generation)}")
    return {
        "passed": True,
        "generation": int(generation),
        "proxy_scale_hash": expected_hash,
        "tau": float(expected["tau"]),
    }

