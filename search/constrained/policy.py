"""Fail-closed policy for the approved 4090 BOPS 0.21 search region."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..candidate import CandidateGenotype, normalize_precision
from ..hashing import canonical_json_hash


@dataclass(frozen=True)
class ConstrainedStageAPolicy:
    r_mac_floor: float = 0.95
    int8_mac_share_min: float = 0.14
    int8_mac_share_max: float = 0.22
    bops_target: float = 0.21
    bops_tolerance: float = 0.005
    min_map: float = 0.705088
    min_ap07: float = 0.564003
    allowed_precision_values: tuple[str, ...] = ("FP16", "INT8")

    @property
    def bops_interval(self) -> tuple[float, float]:
        return (
            float(self.bops_target) - float(self.bops_tolerance),
            float(self.bops_target) + float(self.bops_tolerance),
        )


def canonical_int8_mac_share(
    precision_profile: Mapping[str, str],
    canonical_macs: Mapping[str, float],
) -> float:
    total = sum(max(0.0, float(value)) for value in canonical_macs.values())
    if total <= 0.0:
        raise ValueError("canonical_MACs_empty")
    int8 = sum(
        max(0.0, float(canonical_macs.get(name, 0.0)))
        for name, precision in precision_profile.items()
        if normalize_precision(precision) == "INT8"
    )
    return float(int8 / total)


def constrained_resource_admission(
    metrics: Mapping[str, Any],
    policy: ConstrainedStageAPolicy,
) -> dict[str, Any]:
    r_mac = float(metrics.get("R_MAC", metrics.get("MAC_retention", float("-inf"))))
    int8_share = float(
        metrics.get(
            "int8_macs_share_full",
            metrics.get("INT8_MAC_share", float("inf")),
        )
    )
    bops = float(
        metrics.get(
            "R_bops_vs_fp32",
            metrics.get("BOPS_retention", metrics.get("R_BOPS_realized", float("inf"))),
        )
    )
    lower, upper = policy.bops_interval
    reasons: list[str] = []
    if r_mac < float(policy.r_mac_floor):
        reasons.append("R_MAC_below_floor")
    if not float(policy.int8_mac_share_min) <= int8_share <= float(policy.int8_mac_share_max):
        reasons.append("INT8_MAC_share_out_of_range")
    if not lower <= bops <= upper:
        reasons.append("legalized_BOPS_out_of_budget")
    return {
        "passed": not reasons,
        "status": "passed" if not reasons else "constrained_resource_gate_failed",
        "failure_reasons": reasons,
        "R_MAC": r_mac,
        "R_MAC_floor": float(policy.r_mac_floor),
        "int8_macs_share_full": int8_share,
        "int8_macs_share_interval": [
            float(policy.int8_mac_share_min),
            float(policy.int8_mac_share_max),
        ],
        "BOPS_retention": bops,
        "BOPS_legal_interval": [lower, upper],
    }


def validate_precision_genes(
    precision_genes: Mapping[str, str],
    *,
    int8_allowlist: set[str] | frozenset[str],
    policy: ConstrainedStageAPolicy,
) -> dict[str, Any]:
    allowed = {normalize_precision(value) for value in policy.allowed_precision_values}
    reasons: list[str] = []
    for group_id, raw_precision in sorted(precision_genes.items()):
        precision = normalize_precision(raw_precision)
        if precision == "FP32":
            reasons.append(f"FP32_precision_gene_forbidden:{group_id}")
        elif precision not in allowed:
            reasons.append(f"precision_gene_forbidden:{group_id}:{precision}")
        elif precision == "INT8" and str(group_id) not in int8_allowlist:
            reasons.append(f"INT8_group_not_allowlisted:{group_id}")
    return {
        "passed": not reasons,
        "status": "passed" if not reasons else "precision_gene_illegal",
        "failure_reasons": reasons,
        "allowed_precision_values": sorted(allowed),
        "int8_allowlist": sorted(str(value) for value in int8_allowlist),
    }


def _precision_gene_hash(candidate: CandidateGenotype) -> str:
    return canonical_json_hash(
        {
            str(group_id): normalize_precision(precision)
            for group_id, precision in sorted(candidate.precision_genes.items())
        }
    )


def precision_repair_identity(
    raw: CandidateGenotype,
    repaired: CandidateGenotype,
) -> dict[str, Any]:
    raw_hash = _precision_gene_hash(raw)
    repaired_hash = _precision_gene_hash(repaired)
    passed = raw_hash == repaired_hash
    return {
        "passed": passed,
        "status": "passed" if passed else "precision_repair_identity_failure",
        "failure_reason": "" if passed else "pruning_repair_modified_precision_genes",
        "raw_precision_gene_hash": raw_hash,
        "repaired_precision_gene_hash": repaired_hash,
    }


def smoke10_admission(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if str(evaluation.get("status", "")) != "ok":
        reasons.append("smoke_status_not_ok")
    if int(evaluation.get("num_evaluated_frames", -1)) != 10:
        reasons.append("smoke_evaluated_not_10")
    if int(evaluation.get("num_skipped_frames", -1)) != 0:
        reasons.append("smoke_skipped_nonzero")
    required_finite = ("mAP", "AP@0.7", "forward_p50_ms")
    if any(
        value is None or not math.isfinite(float(value))
        for value in (evaluation.get(key) for key in required_finite)
    ):
        reasons.append("smoke_output_nonfinite")
    elif float(evaluation.get("mAP", 0.0) or 0.0) <= 0.0:
        reasons.append("smoke_detection_complete_collapse")
    return {
        "passed": not reasons,
        "status": "passed" if not reasons else "smoke10_admission_failed",
        "failure_reasons": reasons,
        "formal_ap_gate_applied": False,
        "evaluated": int(evaluation.get("num_evaluated_frames", -1)),
        "skipped": int(evaluation.get("num_skipped_frames", -1)),
    }


def constrained_smoke_unlock(
    candidates: Sequence[Mapping[str, Any]],
    *,
    workers_stopped: bool,
    residual_gpu_processes: Sequence[Any],
    required: int = 5,
) -> dict[str, Any]:
    rows = [dict(row) for row in candidates]
    reasons: list[str] = []
    if len(rows) != int(required):
        reasons.append(f"accepted_candidate_count:{len(rows)}!={int(required)}")
    identity_fields = ("candidate_hash", "physical_hash", "deployment_hash")
    for field in identity_fields:
        values = [str(row.get(field, "")) for row in rows]
        if any(not value for value in values):
            reasons.append(f"{field}_missing")
        elif len(set(values)) != len(values):
            reasons.append(f"{field}_not_unique")
    required_flags = (
        "resource_admission_passed",
        "accuracy_admission_passed",
        "precision_identity_passed",
        "deployment_audits_passed",
    )
    for index, row in enumerate(rows):
        if str(row.get("status", "")) != "ok":
            reasons.append(f"candidate_{index}_status_not_ok")
        if int(row.get("evaluated", -1)) != 200:
            reasons.append(f"candidate_{index}_evaluated_not_200")
        if int(row.get("skipped", -1)) != 0:
            reasons.append(f"candidate_{index}_skipped_nonzero")
        for flag in required_flags:
            if not bool(row.get(flag, False)):
                reasons.append(f"candidate_{index}_{flag}_false")
    if not workers_stopped:
        reasons.append("stage2_workers_not_stopped")
    if residual_gpu_processes:
        reasons.append("residual_gpu_processes")
    passed = not reasons
    return {
        "status": "passed" if passed else "constrained_top5_smoke_failed",
        "failure_reasons": reasons,
        "accepted_candidate_count": len(rows),
        "MULTIGPU_TOP5_SMOKE_PASS": passed,
        "STAGE_A_ALLOWED": passed,
        "STAGE_A_STARTED": False,
        "STAGE_B_ALLOWED": False,
    }
