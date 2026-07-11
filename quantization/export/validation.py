"""Canonical ONNX and physical-snapshot validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..types import OnnxOriginMapResult, OnnxValidationResult, ValidationIssue
from .origin_trace import build_weight_trace_index, trace_compute_node_weight


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError("physical snapshot must be a mapping or expose to_dict()")


def _snapshot_rows(snapshot: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    payload = _as_mapping(snapshot)
    version = str(payload.get("snapshot_schema_version") or payload.get("schema_version") or "")
    raw_rows = payload.get("modules", [])
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(raw_rows, Mapping):
        iterator = raw_rows.items()
    else:
        iterator = ((None, row) for row in raw_rows)
    for key, raw in iterator:
        row = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        name = str(row.get("canonical_module_name") or row.get("module_path") or row.get("name") or key or "")
        if name:
            rows[name] = row
    return version, rows


def _effective_weight_shape(trace: Mapping[str, Any]) -> tuple[int, ...]:
    shape = tuple(int(value) for value in trace.get("root_initializer_shape", ()))
    for permutation in reversed(list(trace.get("transpose_permutations", []))):
        perm = tuple(int(value) for value in permutation)
        if len(perm) != len(shape) or sorted(perm) != list(range(len(shape))):
            return ()
        shape = tuple(shape[index] for index in perm)
    return shape


def _shape_contract(entry: Any, row: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    physical = tuple(int(value) for value in (row.get("weight_shape") or ()))
    root = tuple(int(value) for value in trace.get("root_initializer_shape", ()))
    effective = _effective_weight_shape(trace)
    groups = int(row.get("groups", entry.groups) or 1)
    op_type = str(entry.onnx_op_type)
    attrs_in = row.get("in_channels")
    attrs_out = row.get("out_channels")
    passed = bool(trace.get("success")) and bool(physical)
    layout = ""
    logical_in = None
    logical_out = None
    expected_consumed = physical
    if op_type == "Conv":
        layout = "[C_out,C_in/groups,kH,kW]"
        passed = passed and len(physical) >= 2 and root == physical and effective == physical
        if len(physical) >= 2:
            logical_in = physical[1] * groups
            logical_out = physical[0]
            passed = passed and logical_out % groups == 0
    elif op_type == "ConvTranspose":
        layout = "[C_in,C_out/groups,kH,kW]"
        passed = passed and len(physical) >= 2 and root == physical and effective == physical
        if len(physical) >= 2:
            logical_in = physical[0]
            logical_out = physical[1] * groups
            passed = passed and logical_in % groups == 0
    elif op_type in {"MatMul", "Gemm"}:
        layout = "matrix"
        passed = passed and len(physical) == 2
        trans_b = int(trace.get("trans_b", 0) or 0)
        if len(physical) == 2:
            expected_consumed = physical if op_type == "Gemm" and trans_b else tuple(reversed(physical))
            passed = passed and effective == expected_consumed
            logical_out, logical_in = physical
    else:
        passed = False
    if attrs_in is not None:
        passed = passed and logical_in == int(attrs_in)
    if attrs_out is not None:
        passed = passed and logical_out == int(attrs_out)
    return {
        "passed": bool(passed),
        "layout": layout,
        "groups": groups,
        "logical_in_channels_or_features": logical_in,
        "logical_out_channels_or_features": logical_out,
        "physical_weight_shape": physical,
        "root_initializer_shape": root,
        "effective_consumed_weight_shape": effective,
        "expected_consumed_weight_shape": expected_consumed,
        "trans_b": int(trace.get("trans_b", 0) or 0),
        "transpose_permutations": [tuple(value) for value in trace.get("transpose_permutations", [])],
    }


def validate_onnx_against_physical_snapshot(
    onnx_path: str | Path,
    physical_snapshot: Any,
    origin_map: OnnxOriginMapResult,
) -> OnnxValidationResult:
    """Compare live physical weight shapes with canonical ONNX initializers."""

    version, snapshot = _snapshot_rows(physical_snapshot)
    issues: list[ValidationIssue] = []
    if version and version != "physical-structure-snapshot-v2":
        issues.append(ValidationIssue("invalid_snapshot_schema", f"expected physical-structure-snapshot-v2, got {version}"))
    index = build_weight_trace_index(onnx_path)
    nodes_by_name = index["nodes_by_name"]
    checks: list[dict[str, Any]] = []
    for entry in origin_map.entries:
        node = nodes_by_name.get(entry.canonical_node_name) or nodes_by_name.get(entry.original_node_name)
        row = snapshot.get(entry.module_path)
        if node is None:
            issues.append(ValidationIssue("canonical_node_missing", "canonical ONNX node is absent", entry.module_path))
            continue
        if row is None:
            issues.append(ValidationIssue("physical_module_missing", "module is absent from physical snapshot", entry.module_path))
            continue
        trace = trace_compute_node_weight(index, node)
        contract = _shape_contract(entry, row, trace)
        passed = bool(contract["passed"]) and str(trace.get("root_initializer", "")) == entry.weight_initializer
        check = {
            "module_path": entry.module_path,
            "module_type": row.get("module_type", entry.module_type),
            "canonical_node_name": entry.canonical_node_name,
            "onnx_op_type": entry.onnx_op_type,
            "groups": int(row.get("groups", entry.groups) or 1),
            "weight_layout": contract["layout"],
            "logical_in_channels_or_features": contract["logical_in_channels_or_features"],
            "logical_out_channels_or_features": contract["logical_out_channels_or_features"],
            "physical_weight_shape": list(contract["physical_weight_shape"]),
            "root_initializer": trace.get("root_initializer", ""),
            "root_initializer_shape": list(contract["root_initializer_shape"]),
            "effective_consumed_weight_shape": list(contract["effective_consumed_weight_shape"]),
            "expected_consumed_weight_shape": list(contract["expected_consumed_weight_shape"]),
            "trans_b": contract["trans_b"],
            "transpose_permutations": [list(value) for value in contract["transpose_permutations"]],
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            issues.append(
                ValidationIssue(
                    "physical_weight_chain_mismatch",
                    "physical snapshot, canonical initializer, and ONNX weight shape disagree",
                    entry.module_path,
                    check,
                )
            )
    return OnnxValidationResult(passed=not issues, checks=checks, issues=issues)
