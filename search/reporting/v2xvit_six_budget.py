"""Accuracy, compression, and GA-admission rules for six-budget V2X-ViT runs."""

from __future__ import annotations

import math
from typing import Any, Mapping


def classify_ap_drop(reference_map: float, candidate_map: float) -> dict[str, Any]:
    reference = float(reference_map)
    candidate = float(candidate_map)
    if reference <= 0.0 or not all(math.isfinite(value) for value in (reference, candidate)):
        raise ValueError("ap_collapse_metric_invalid")
    drop = reference - candidate
    retention = candidate / reference
    if retention < 0.50:
        label = "SEVERE_COLLAPSE"
        catastrophic = True
    elif drop > 0.10 or retention < 0.80:
        label = "SEVERE_COLLAPSE"
        catastrophic = False
    elif drop > 0.03:
        label = "SIGNIFICANT_DROP"
        catastrophic = False
    elif drop > 0.01:
        label = "MILD_DROP"
        catastrophic = False
    else:
        label = "SAFE"
        catastrophic = False
    return {
        "reference_mAP": reference,
        "candidate_mAP": candidate,
        "absolute_drop": drop,
        "mAP_retention": retention,
        "classification": label,
        "catastrophic_collapse": catastrophic,
    }


def classify_ga_admission(
    *,
    budget_reached: bool,
    s32_drop: float | None,
    jmix_drop: float | None,
    jmix_engine_built: bool,
    requested_realized_exact: bool,
    precision_conflict_count: int,
    fallback_count: int,
    latency_batch_valid: bool,
    jmix_speedup: float | None,
    taylor_convergence_passed: bool,
    framework_tests_passed: bool,
) -> str:
    if not budget_reached:
        return "GA_PROXY_UNRELIABLE"
    if not jmix_engine_built or not requested_realized_exact or precision_conflict_count or fallback_count:
        return "GA_DEPLOYMENT_INVALID"
    if s32_drop is None or jmix_drop is None:
        return "GA_PROXY_UNRELIABLE"
    if float(s32_drop) > 0.01 or float(jmix_drop) > 0.01:
        return "GA_UNSAFE"
    if not taylor_convergence_passed or not framework_tests_passed:
        return "GA_PROXY_UNRELIABLE"
    if not latency_batch_valid or jmix_speedup is None or float(jmix_speedup) <= 1.0:
        return "GA_UNSAFE"
    return "GA_ADMISSIBLE"


def compression_metrics(
    *,
    bops_retention: float,
    parameter_retention: float,
    mixed_weight_retention: float,
) -> Mapping[str, float]:
    values = tuple(float(value) for value in (bops_retention, parameter_retention, mixed_weight_retention))
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("compression_retention_invalid")
    return {
        "R_BOPS": values[0],
        "BOPS_compression": 1.0 / values[0],
        "parameter_retention": values[1],
        "parameter_compression": 1.0 / values[1],
        "mixed_weight_retention": values[2],
        "mixed_weight_compression": 1.0 / values[2],
    }


__all__ = ["classify_ap_drop", "classify_ga_admission", "compression_metrics"]
