"""Evidence gates between model-family capability discovery and formal search.

Capability discovery is intentionally optimistic (what a model *could* use).
This module is conservative (what the current production path has actually
proved).  GA/greedy orchestration must consume the latter before exposing a
model-family gene.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .contracts import ModelFamilyAudit


QUANTIZATION_REQUIRED_EVIDENCE = (
    "strict_state_dict_load",
    "onnx_export",
    "onnx_checker",
    "canonical_weight_mapping",
    "semantic_qdq_boundaries",
    "plugin_contract",
    "strongly_typed_parser",
    "precision_realization",
    "tensor_parity",
)

PRUNING_REQUIRED_EVIDENCE = (
    "strict_state_dict_load",
    "full_runtime_trace_coverage",
    "domain_ranking",
    "physical_materializer",
    "physical_strict_reload",
    "physical_forward",
)


@dataclass(frozen=True)
class ModelFamilySearchReadiness:
    schema_version: str
    family_id: str
    audit_hash: str
    evidence: dict[str, bool]
    missing_quantization_evidence: tuple[str, ...]
    missing_pruning_evidence: tuple[str, ...]
    potential_precision_gene_ids: tuple[str, ...]
    ready_precision_gene_ids: tuple[str, ...]
    potential_pruning_domain_ids: tuple[str, ...]
    ready_pruning_domain_ids: tuple[str, ...]
    protected_weighted_op_ids: tuple[str, ...]
    blocked_pruning_domain_ids: tuple[str, ...]

    @property
    def quantization_search_ready(self) -> bool:
        return not self.missing_quantization_evidence and bool(self.ready_precision_gene_ids)

    @property
    def pruning_search_ready(self) -> bool:
        return not self.missing_pruning_evidence and bool(self.ready_pruning_domain_ids)

    @property
    def joint_search_ready(self) -> bool:
        return self.quantization_search_ready and self.pruning_search_ready

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "audit_hash": self.audit_hash,
            "evidence": dict(sorted(self.evidence.items())),
            "missing_quantization_evidence": list(self.missing_quantization_evidence),
            "missing_pruning_evidence": list(self.missing_pruning_evidence),
            "potential_precision_gene_ids": list(self.potential_precision_gene_ids),
            "ready_precision_gene_ids": list(self.ready_precision_gene_ids),
            "potential_pruning_domain_ids": list(self.potential_pruning_domain_ids),
            "ready_pruning_domain_ids": list(self.ready_pruning_domain_ids),
            "protected_weighted_op_ids": list(self.protected_weighted_op_ids),
            "blocked_pruning_domain_ids": list(self.blocked_pruning_domain_ids),
            "quantization_search_ready": self.quantization_search_ready,
            "pruning_search_ready": self.pruning_search_ready,
            "joint_search_ready": self.joint_search_ready,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["readiness_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload


def build_model_family_search_readiness(
    audit: ModelFamilyAudit,
    evidence: Mapping[str, bool] | None = None,
) -> ModelFamilySearchReadiness:
    observed = {str(key): bool(value) for key, value in (evidence or {}).items()}
    missing_quantization = tuple(
        key for key in QUANTIZATION_REQUIRED_EVIDENCE if not observed.get(key, False)
    )
    missing_pruning = tuple(
        key for key in PRUNING_REQUIRED_EVIDENCE if not observed.get(key, False)
    )
    potential_precision = tuple(
        sorted(row.canonical_id for row in audit.weighted_ops if "INT8" in row.potential_precisions)
    )
    protected = tuple(
        sorted(
            row.canonical_id
            for row in audit.weighted_ops
            if not row.production_enabled or "INT8" not in row.allowed_precisions
        )
    )
    ready_precision = (
        tuple(
            sorted(
                row.canonical_id
                for row in audit.weighted_ops
                if row.production_enabled and "INT8" in row.allowed_precisions
            )
        )
        if not missing_quantization
        else ()
    )
    potential_pruning = tuple(sorted(row.domain_id for row in audit.pruning_domains))
    blocked_pruning = tuple(
        sorted(row.domain_id for row in audit.pruning_domains if not row.production_enabled)
    )
    ready_pruning = (
        tuple(sorted(row.domain_id for row in audit.pruning_domains if row.production_enabled))
        if not missing_pruning
        else ()
    )
    return ModelFamilySearchReadiness(
        schema_version="heal-model-family-search-readiness-v1",
        family_id=audit.family_id,
        audit_hash=audit.to_dict()["audit_hash"],
        evidence=observed,
        missing_quantization_evidence=missing_quantization,
        missing_pruning_evidence=missing_pruning,
        potential_precision_gene_ids=potential_precision,
        ready_precision_gene_ids=ready_precision,
        potential_pruning_domain_ids=potential_pruning,
        ready_pruning_domain_ids=ready_pruning,
        protected_weighted_op_ids=protected,
        blocked_pruning_domain_ids=blocked_pruning,
    )


__all__ = [
    "ModelFamilySearchReadiness",
    "PRUNING_REQUIRED_EVIDENCE",
    "QUANTIZATION_REQUIRED_EVIDENCE",
    "build_model_family_search_readiness",
]
