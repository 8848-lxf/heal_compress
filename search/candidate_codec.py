"""Stable candidate serialization helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .candidate import CandidateGenotype, CandidatePhenotype


def encode_candidate(candidate: CandidateGenotype | CandidatePhenotype) -> dict[str, Any]:
    """Encode a candidate with sorted keys for stable JSON output."""

    return candidate.to_dict()


def decode_candidate(payload: Mapping[str, Any]) -> CandidateGenotype:
    """Decode a genotype payload."""

    return CandidateGenotype.from_dict(payload)


def decode_phenotype(payload: Mapping[str, Any]) -> CandidatePhenotype:
    """Decode a phenotype payload."""

    return CandidatePhenotype.from_dict(payload)
