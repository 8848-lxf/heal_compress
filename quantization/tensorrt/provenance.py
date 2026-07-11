"""Deployment provenance capture and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..artifacts.io import file_sha256
from ..types import ProvenanceValidationResult, ValidationIssue


PROVENANCE_FIELDS = (
    "physical_structure_hash",
    "precision_profile_hash",
    "canonical_mapping_hash",
    "base_onnx_hash",
    "qdq_onnx_hash",
    "engine_hash",
    "plugin_hash",
    "tensorrt_version",
    "build_policy_version",
)


def build_engine_provenance(
    *,
    physical_structure_hash: str,
    precision_profile_hash: str,
    canonical_mapping_hash: str,
    base_onnx_path: str | Path,
    qdq_onnx_path: str | Path,
    engine_path: str | Path,
    plugin_path: str | Path | None = None,
    tensorrt_version: str,
    build_policy_version: str,
) -> dict[str, str]:
    """Capture hashes for every formal deployment stage."""

    return {
        "physical_structure_hash": str(physical_structure_hash),
        "precision_profile_hash": str(precision_profile_hash),
        "canonical_mapping_hash": str(canonical_mapping_hash),
        "base_onnx_hash": file_sha256(base_onnx_path),
        "qdq_onnx_hash": file_sha256(qdq_onnx_path),
        "engine_hash": file_sha256(engine_path),
        "plugin_hash": file_sha256(plugin_path) if plugin_path is not None else "",
        "tensorrt_version": str(tensorrt_version),
        "build_policy_version": str(build_policy_version),
    }


def validate_engine_provenance(
    provenance: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> ProvenanceValidationResult:
    """Validate completeness and optional expected hashes without path fallbacks."""

    optional_empty = {"plugin_hash"}
    missing = [field for field in PROVENANCE_FIELDS if field not in provenance or (not provenance.get(field) and field not in optional_empty)]
    issues: list[ValidationIssue] = []
    if missing:
        issues.append(ValidationIssue("provenance_fields_missing", "required provenance fields are absent", details={"fields": missing}))
    for key, value in (expected or {}).items():
        if str(provenance.get(key, "")) != str(value):
            issues.append(
                ValidationIssue(
                    "provenance_value_mismatch",
                    f"provenance value differs for {key}",
                    details={"expected": value, "actual": provenance.get(key)},
                )
            )
    return ProvenanceValidationResult(passed=not issues, missing_fields=missing, issues=issues)
