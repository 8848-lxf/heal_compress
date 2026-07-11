"""Canonical TensorRT engine structure checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import TensorRTValidationConfig
from ..types import (
    CanonicalPrecisionMappingResult,
    EngineStructureValidationResult,
    ValidationIssue,
    stable_json_hash,
)
from .layer_info import has_canonical_identity, is_weighted_compute_layer, layer_metadata, load_layer_info


def _snapshot_payload(snapshot: Any | None) -> tuple[str, str, dict[str, dict[str, Any]]]:
    if snapshot is None:
        return "", "", {}
    payload = snapshot if isinstance(snapshot, Mapping) else snapshot.to_dict()
    version = str(payload.get("snapshot_schema_version") or payload.get("schema_version") or "")
    rows = payload.get("modules", [])
    if isinstance(rows, Mapping):
        normalized = {str(name): dict(row) for name, row in rows.items() if isinstance(row, Mapping)}
    else:
        normalized = {
            str(row.get("canonical_module_name") or row.get("module_path") or row.get("name") or ""): dict(row)
            for row in rows
            if isinstance(row, Mapping)
            and str(row.get("canonical_module_name") or row.get("module_path") or row.get("name") or "")
        }
    return version, stable_json_hash({"snapshot_schema_version": version, "modules": normalized}), normalized


def _layer_weight_shape(row: Mapping[str, Any]) -> tuple[int, ...]:
    weights = row.get("Weights") or row.get("weights") or {}
    raw = weights.get("Dimensions") or weights.get("dimensions") or weights.get("Shape") or weights.get("shape") if isinstance(weights, Mapping) else None
    if raw is None:
        raw = row.get("KernelShape") or row.get("kernel_shape")
    if isinstance(raw, str):
        cleaned = raw.strip("[]() ").replace("x", ",")
        try:
            return tuple(int(value.strip()) for value in cleaned.split(",") if value.strip())
        except ValueError:
            return ()
    if isinstance(raw, (list, tuple)):
        return tuple(int(value) for value in raw)
    return ()


def validate_engine_structure(
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    precision_mapping: CanonicalPrecisionMappingResult,
    *,
    physical_snapshot: Any | None = None,
    config: TensorRTValidationConfig | None = None,
) -> EngineStructureValidationResult:
    """Require canonical weighted layers to appear in engine layer metadata."""

    policy = config or TensorRTValidationConfig()
    rows = load_layer_info(layer_info)
    metadata = [layer_metadata(row) for row in rows]
    missing: list[str] = []
    ambiguous: list[str] = []
    matched = 0
    issues: list[ValidationIssue] = []
    snapshot_version, snapshot_hash, physical_rows = _snapshot_payload(physical_snapshot)
    if policy.require_physical_snapshot_v2 and physical_snapshot is None:
        issues.append(ValidationIssue("physical_snapshot_missing", "physical_structure_snapshot_v2 is required"))
    if physical_snapshot is not None and snapshot_version != "physical-structure-snapshot-v2":
        issues.append(
            ValidationIssue(
                "physical_snapshot_schema_invalid",
                f"expected physical-structure-snapshot-v2, got {snapshot_version or 'missing'}",
            )
        )
    if physical_snapshot is not None:
        for entry in precision_mapping.entries:
            if entry.module_path not in physical_rows:
                issues.append(ValidationIssue("physical_module_missing", "canonical module is absent from physical snapshot", entry.module_path))
    shape_checks: list[dict[str, Any]] = []
    for entry in precision_mapping.entries:
        matching_rows = [row for row in rows if has_canonical_identity(row, entry.canonical_node_name)]
        compute_rows = [row for row in matching_rows if is_weighted_compute_layer(row)]
        if not compute_rows:
            missing.append(entry.canonical_node_name)
        elif len(compute_rows) > 1:
            ambiguous.append(entry.canonical_node_name)
        else:
            matched += 1
            compute_row = compute_rows[0]
            hits = [layer_metadata(compute_row)]
            physical_row = physical_rows.get(entry.module_path, {})
            engine_weight_shape = _layer_weight_shape(compute_row)
            physical_weight_shape = tuple(int(value) for value in (physical_row.get("weight_shape") or ()))
            groups = int(physical_row.get("groups", 1) or 1)
            type_expected = str(entry.onnx_op_type)
            parameter_text = " ".join(hits).lower()
            type_passed = (
                (type_expected in {"Conv", "ConvTranspose"} and "conv" in parameter_text)
                or (type_expected in {"Gemm", "MatMul"} and any(token in parameter_text for token in ("gemm", "matmul", "matrix", "fully")))
                or not parameter_text
            )
            weight_passed = not engine_weight_shape or not physical_weight_shape or engine_weight_shape == physical_weight_shape
            group_passed = groups > 0 and (not physical_weight_shape or groups == 1 or (
                len(physical_weight_shape) >= 2
                and ((type_expected == "Conv" and physical_weight_shape[0] % groups == 0) or (type_expected == "ConvTranspose" and physical_weight_shape[0] % groups == 0))
            ))
            shape_check = {
                "module_path": entry.module_path,
                "canonical_node_name": entry.canonical_node_name,
                "onnx_op_type": type_expected,
                "physical_weight_shape": list(physical_weight_shape),
                "engine_weight_shape": list(engine_weight_shape),
                "groups": groups,
                "type_passed": type_passed,
                "weight_shape_passed": weight_passed,
                "group_legality_passed": group_passed,
                "passed": bool(type_passed and weight_passed and group_passed),
            }
            shape_checks.append(shape_check)
            if not shape_check["passed"]:
                issues.append(
                    ValidationIssue(
                        "engine_physical_shape_mismatch",
                        "TensorRT layer metadata disagrees with physical structure",
                        entry.module_path,
                        shape_check,
                    )
                )
    if missing:
        issues.append(ValidationIssue("canonical_layers_missing", "engine is missing canonical weighted layers", details={"layers": missing}))
    if ambiguous and policy.reject_ambiguous_metadata:
        issues.append(ValidationIssue("canonical_layer_ambiguous", "canonical names match multiple compute layers", details={"layers": ambiguous}))
    return EngineStructureValidationResult(
        passed=not issues,
        matched_canonical_count=matched,
        expected_canonical_count=len(precision_mapping.entries),
        missing_canonical_layers=missing,
        ambiguous_layers=ambiguous,
        issues=issues,
        physical_snapshot_schema_version=snapshot_version,
        physical_snapshot_hash=snapshot_hash,
        shape_checks=shape_checks,
    )
