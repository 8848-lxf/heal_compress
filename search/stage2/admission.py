"""Stage-2 admission and deployment uniqueness helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..candidate import CandidatePhenotype
from ..hashing import canonical_json_hash


def _normalized_profile(profile: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value).upper() for key, value in sorted(dict(profile).items(), key=lambda item: str(item[0]))}


def realized_precision_profile_hash(profile: Mapping[str, Any]) -> str:
    """Stable hash for actual canonical-layer precision realization."""

    return canonical_json_hash({"realized_precision_profile": _normalized_profile(profile)})


def deployment_signature(physical_hash: str, realized_profile_hash: str) -> str:
    """Global deployment identity: physical structure plus actual precision mapping."""

    return canonical_json_hash(
        {
            "physical_hash": str(physical_hash),
            "realized_precision_profile_hash": str(realized_profile_hash),
            "signature_version": "physical-plus-realized-profile-v1",
        }
    )


def legalized_int8_group_count(phenotype: CandidatePhenotype) -> int:
    metadata = dict(phenotype.metadata or {})
    profile = (
        metadata.get("stage1_legalized_group_profile")
        or metadata.get("legalized_group_profile")
        or metadata.get("requested_group_profile")
        or {}
    )
    return sum(1 for value in dict(profile).values() if str(value).upper() == "INT8")


def realized_int8_layer_count(*, qdq_summary: Mapping[str, Any] | None = None, realized_profile: Mapping[str, Any] | None = None) -> int:
    summary = dict(qdq_summary or {})
    if "realized_int8_layer_count" in summary:
        return int(summary.get("realized_int8_layer_count") or 0)
    return sum(1 for value in dict(realized_profile or {}).values() if str(value).upper() == "INT8")


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str = "accepted"
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": bool(self.accepted), "reason": self.reason, "details": dict(self.details or {})}


@dataclass(frozen=True)
class Stage2AdmissionPolicy:
    bops_target: float
    tolerance: float = 0.005

    def check_repaired_candidate(self, phenotype: CandidatePhenotype, metrics: Mapping[str, Any]) -> AdmissionDecision:
        bops = _metric_float(metrics, "R_BOPS", "R_bops", "R_bops_vs_fp32", "R_BOPS_repaired")
        limit = float(self.bops_target) + float(self.tolerance)
        if bops is None:
            return AdmissionDecision(False, "missing_repaired_bops")
        if bops > limit:
            return AdmissionDecision(False, "repaired_bops_over_budget", {"R_BOPS": bops, "limit": limit})
        pruned_count = len(phenotype.pruned_unit_ids)
        int8_groups = legalized_int8_group_count(phenotype)
        if pruned_count <= 0 and int8_groups <= 0:
            return AdmissionDecision(False, "control_only_repaired_candidate")
        return AdmissionDecision(
            True,
            "accepted",
            {"R_BOPS": bops, "limit": limit, "pruned_unit_count": pruned_count, "legalized_int8_group_count": int8_groups},
        )

    def check_realized_candidate(
        self,
        *,
        pruned_unit_count: int,
        realized_int8_layers: int,
        r_bops_realized: float | None,
    ) -> AdmissionDecision:
        if int(pruned_unit_count) <= 0 and int(realized_int8_layers) <= 0:
            return AdmissionDecision(False, "control_only")
        if r_bops_realized is None:
            return AdmissionDecision(False, "missing_realized_bops")
        limit = float(self.bops_target) + float(self.tolerance)
        if float(r_bops_realized) > limit:
            return AdmissionDecision(False, "realized_bops_over_budget", {"R_BOPS_realized": float(r_bops_realized), "limit": limit})
        return AdmissionDecision(True, "accepted", {"R_BOPS_realized": float(r_bops_realized), "limit": limit})


@dataclass(frozen=True)
class DeploymentRegistration:
    accepted: bool
    deployment_signature: str
    realized_precision_profile_hash: str
    reason: str = "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "deployment_signature": self.deployment_signature,
            "realized_precision_profile_hash": self.realized_precision_profile_hash,
            "reason": self.reason,
        }


class DeploymentSignatureRegistry:
    """In-memory global uniqueness registry for deployed candidates."""

    def __init__(self, seen: set[str] | None = None) -> None:
        self.seen = set(seen or set())

    def register(self, physical_hash: str, realized_profile: Mapping[str, Any]) -> DeploymentRegistration:
        profile_hash = realized_precision_profile_hash(realized_profile)
        signature = deployment_signature(physical_hash, profile_hash)
        if signature in self.seen:
            return DeploymentRegistration(False, signature, profile_hash, "duplicate_deployment_signature")
        self.seen.add(signature)
        return DeploymentRegistration(True, signature, profile_hash)


def _metric_float(metrics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name not in metrics or metrics[name] in (None, ""):
            continue
        try:
            return float(metrics[name])
        except (TypeError, ValueError):
            return None
    return None
