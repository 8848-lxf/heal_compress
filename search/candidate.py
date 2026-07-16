"""Candidate genotype and phenotype schemas for joint search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


PRECISION_VALUES = ("FP32", "FP16", "INT8")


def normalize_precision(value: str, *, default: str = "FP16") -> str:
    text = str(value or default).upper()
    return text if text in PRECISION_VALUES else str(default).upper()


@dataclass(frozen=True)
class PrecisionDecision:
    """Requested and legalized precision for one canonical weighted layer."""

    requested_precision: str
    realized_precision: str
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_precision", normalize_precision(self.requested_precision))
        object.__setattr__(self, "realized_precision", normalize_precision(self.realized_precision))
        object.__setattr__(self, "fallback_reason", str(self.fallback_reason or ""))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrecisionDecision":
        return cls(
            requested_precision=str(payload.get("requested_precision", "FP16")),
            realized_precision=str(payload.get("realized_precision", payload.get("realized_request_precision", "FP16"))),
            fallback_reason=str(payload.get("fallback_reason", "")),
        )


@dataclass(frozen=True)
class CandidateGenotype:
    """Raw GA-produced genes before repair/legalization."""

    pruning_genes: dict[str, int] = field(default_factory=dict)
    precision_genes: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    pruning_width_genes: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pruning_genes",
            {str(key): 1 if int(value) else 0 for key, value in self.pruning_genes.items()},
        )
        object.__setattr__(
            self,
            "precision_genes",
            {str(key): normalize_precision(value) for key, value in self.precision_genes.items()},
        )
        object.__setattr__(self, "meta", dict(self.meta))
        object.__setattr__(
            self,
            "pruning_width_genes",
            {str(key): int(value) for key, value in self.pruning_width_genes.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pruning_genes": {key: self.pruning_genes[key] for key in sorted(self.pruning_genes)},
            "pruning_width_genes": {
                key: self.pruning_width_genes[key] for key in sorted(self.pruning_width_genes)
            },
            "precision_genes": {key: self.precision_genes[key] for key in sorted(self.precision_genes)},
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateGenotype":
        return cls(
            pruning_genes=dict(payload.get("pruning_genes") or payload.get("prune_vars") or {}),
            precision_genes=dict(payload.get("precision_genes") or payload.get("bitwidth_vars") or {}),
            meta=dict(payload.get("meta") or {}),
            pruning_width_genes=dict(
                payload.get("pruning_width_genes") or payload.get("domain_width_genes") or {}
            ),
        )


@dataclass(frozen=True)
class CandidatePhenotype:
    """Legalized candidate used for artifact generation and hashing."""

    pruned_unit_ids: list[str] = field(default_factory=list)
    precision_profile: dict[str, PrecisionDecision] = field(default_factory=dict)
    pruning_policy_version: str = "formal-plan-first-v1"
    precision_policy_version: str = "explicit-qdq-canonical-fp16-int8-v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pruned_unit_ids", sorted({str(value) for value in self.pruned_unit_ids}))
        profile = {
            str(key): value if isinstance(value, PrecisionDecision) else PrecisionDecision.from_dict(value)
            for key, value in self.precision_profile.items()
        }
        object.__setattr__(self, "precision_profile", {key: profile[key] for key in sorted(profile)})
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def realized_precision_profile(self) -> dict[str, str]:
        return {
            key: self.precision_profile[key].realized_precision
            for key in sorted(self.precision_profile)
        }

    @property
    def requested_precision_profile(self) -> dict[str, str]:
        return {
            key: self.precision_profile[key].requested_precision
            for key in sorted(self.precision_profile)
        }

    @property
    def fallback_report(self) -> dict[str, str]:
        return {
            key: decision.fallback_reason
            for key, decision in self.precision_profile.items()
            if decision.fallback_reason
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pruned_unit_ids": list(self.pruned_unit_ids),
            "precision_profile": {
                key: self.precision_profile[key].to_dict()
                for key in sorted(self.precision_profile)
            },
            "requested_precision_profile": self.requested_precision_profile,
            "realized_precision_profile": self.realized_precision_profile,
            "fallback_report": self.fallback_report,
            "pruning_policy_version": self.pruning_policy_version,
            "precision_policy_version": self.precision_policy_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidatePhenotype":
        profile_payload = payload.get("precision_profile") or {}
        if profile_payload and all(isinstance(value, str) for value in profile_payload.values()):
            profile = {
                str(key): PrecisionDecision(str(value), str(value), "")
                for key, value in profile_payload.items()
            }
        else:
            profile = {
                str(key): PrecisionDecision.from_dict(value)
                for key, value in dict(profile_payload).items()
            }
        return cls(
            pruned_unit_ids=[str(value) for value in payload.get("pruned_unit_ids", [])],
            precision_profile=profile,
            pruning_policy_version=str(payload.get("pruning_policy_version", "formal-plan-first-v1")),
            precision_policy_version=str(payload.get("precision_policy_version", "explicit-qdq-canonical-fp16-int8-v1")),
            metadata=dict(payload.get("metadata") or {}),
        )
