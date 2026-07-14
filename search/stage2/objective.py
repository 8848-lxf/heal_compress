"""Stage-2 real objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Stage2ObjectiveConfig:
    eta_map: float = 1.0
    eta_latency: float = 1.0
    latency_metric: str = "forward_mean_ms"
    tau_ap: float | None = None
    max_map_drop: float | None = None
    min_map: float | None = None
    min_ap07: float | None = None
    required_evaluated_frames: int | None = None
    required_skipped_frames: int | None = None
    r_mac_floor: float | None = None
    int8_mac_share_min: float | None = None
    int8_mac_share_max: float | None = None
    failure_score: float = float("inf")
    epsilon: float = 1.0e-12
    large_penalty: float = 1.0e6


FAILURE_STATUSES = {
    "prune_failed",
    "onnx_export_failed",
    "qdq_failed",
    "engine_build_failed",
    "engine_structure_validation_failed",
    "precision_realization_validation_failed",
    "evaluation_failed",
}


def compute_stage2_score(
    evaluation: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    config: Stage2ObjectiveConfig | None = None,
) -> dict[str, Any]:
    policy = config or Stage2ObjectiveConfig()
    status = str(evaluation.get("status", "ok"))
    if status in FAILURE_STATUSES or status.endswith("_failed"):
        return {"F2": policy.failure_score, "status": status}
    base_map = float(baseline.get("mAP", baseline.get("map", 0.0)) or 0.0)
    cand_map = float(evaluation.get("mAP", evaluation.get("map", 0.0)) or 0.0)
    base_latency = float(baseline.get(policy.latency_metric, 0.0) or 0.0)
    cand_latency = float(evaluation.get(policy.latency_metric, 0.0) or 0.0)
    ap_scale = float(policy.tau_ap) if policy.tau_ap is not None else (base_map if base_map > 0.0 else policy.epsilon)
    loss_map = max(0.0, base_map - cand_map) / max(ap_scale, policy.epsilon)
    latency_ratio = cand_latency / (base_latency if base_latency > 0.0 else policy.epsilon)
    hard_gate_reasons: list[str] = []
    if policy.required_evaluated_frames is not None and int(
        evaluation.get("num_evaluated_frames", -1)
    ) != int(policy.required_evaluated_frames):
        hard_gate_reasons.append("evaluated_frame_count_mismatch")
    if policy.required_skipped_frames is not None and int(
        evaluation.get("num_skipped_frames", -1)
    ) != int(policy.required_skipped_frames):
        hard_gate_reasons.append("skipped_frame_count_mismatch")
    if policy.min_map is not None and cand_map < float(policy.min_map):
        hard_gate_reasons.append("mAP_below_minimum")
    ap07 = float(evaluation.get("AP@0.7", evaluation.get("AP07", 0.0)) or 0.0)
    if policy.min_ap07 is not None and ap07 < float(policy.min_ap07):
        hard_gate_reasons.append("AP07_below_minimum")
    if hard_gate_reasons:
        return {
            "F2": policy.failure_score,
            "status": "accuracy_hard_gate_failed",
            "accuracy_admission_passed": False,
            "failure_reasons": hard_gate_reasons,
            "L_map_real": float(loss_map),
            "R_latency_real": float(latency_ratio),
        }
    if policy.max_map_drop is not None and cand_map < base_map - float(policy.max_map_drop):
        return {
            "F2": policy.failure_score,
            "status": "accuracy_hard_gate_failed",
            "accuracy_admission_passed": False,
            "failure_reasons": ["mAP_drop_exceeds_maximum"],
            "L_map_real": float(loss_map),
            "R_latency_real": float(latency_ratio),
        }
    score = policy.eta_map * loss_map + policy.eta_latency * latency_ratio
    return {
        "F2": float(score),
        "status": status,
        "accuracy_admission_passed": True,
        "failure_reasons": [],
        "L_map_real": float(loss_map),
        "R_latency_real": float(latency_ratio),
    }
