"""Stable candidate serialization helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .candidate import CandidateGenotype, CandidatePhenotype


def encode_candidate(candidate: Any) -> dict[str, Any]:
    """Encode a candidate with sorted keys for stable JSON output."""

    return candidate.to_dict()


def decode_candidate(payload: Mapping[str, Any]) -> Any:
    """Decode a genotype payload."""

    if str(payload.get("structure_gene_type", "")) == "legal_keep_width" or "width_genes" in payload:
        from .encoding.legal_width_genotype import LegalWidthGenotype

        return LegalWidthGenotype(
            width_genes=dict(payload.get("width_genes") or {}),
            precision_genes=dict(
                payload.get("layer_bitwidth")
                or payload.get("precision_genes")
                or {}
            ),
            meta=dict(payload.get("meta") or {}),
        )
    return CandidateGenotype.from_dict(payload)


def decode_phenotype(payload: Mapping[str, Any]) -> CandidatePhenotype:
    """Decode a phenotype payload."""

    return CandidatePhenotype.from_dict(payload)
