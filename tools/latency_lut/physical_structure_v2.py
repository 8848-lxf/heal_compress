#!/usr/bin/env python3
"""Physical structure v2 schemas, hashes, ledgers, and ONNX preflight."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


SNAPSHOT_SCHEMA_VERSION = "physical-structure-snapshot-v2"
HASH_SCHEMA_VERSION = "physical-structure-v2"
SAMPLING_POLICY_VERSION = "deployment-aware-random-request-v2"
LEDGER_SCHEMA_VERSION = "physical-pruning-application-ledger-v2"
QDQ_POLICY_VERSION = "explicit-qdq-canonical-v2"
CANONICAL_MAPPING_VERSION = "canonical-v2-origin-map"
TRT_BUILD_POLICY_VERSION = "trt-v12-obey-constraints"
TERMINAL_STATUSES = {"applied", "repaired", "merged", "skipped"}
PASSTHROUGH_WEIGHT_OPS = {
    "QuantizeLinear",
    "DequantizeLinear",
    "Cast",
    "Identity",
    "Transpose",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
}

STRUCTURE_HASH_FIELDS = (
    "canonical_module_name",
    "canonical_order",
    "module_type",
    "groups",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "output_padding",
)
SHAPE_HASH_FIELDS = (
    "canonical_module_name",
    "module_type",
    "in_channels",
    "out_channels",
    "in_features",
    "out_features",
    "num_features",
    "groups",
    "weight_shape",
    "bias_shape",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path, default: Any) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (tuple, list, torch.Size)):
        return [int(item) for item in value]
    return []


def _state_key(state_dict: Mapping[str, Any] | None, module_name: str, parameter: str) -> str:
    if state_dict is None:
        return ""
    exact = f"{module_name}.{parameter}"
    if exact in state_dict:
        return exact
    matches = [str(key) for key in state_dict if str(key).endswith(exact)]
    return matches[0] if len(matches) == 1 else ""


def _protected_contract(name: str) -> dict[str, bool]:
    low = str(name).lower()
    return {
        "fixed_shape_interface": any(token in low for token in ("pillar_vfe", "pfn_layers", "scatter")),
        "deblock_output_contract": "pyramid_backbone.deblocks" in low,
        "head_output_contract": any(token in low for token in ("cls_head", "reg_head", "dir_head")),
    }


def _snapshot_module_row(name: str, module: nn.Module, order: int, state_dict: Mapping[str, Any] | None) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    row: dict[str, Any] = {
        "canonical_module_name": str(name),
        "module_type": type(module).__name__,
        "in_channels": int(getattr(module, "in_channels")) if hasattr(module, "in_channels") else None,
        "out_channels": int(getattr(module, "out_channels")) if hasattr(module, "out_channels") else None,
        "in_features": int(getattr(module, "in_features")) if hasattr(module, "in_features") else None,
        "out_features": int(getattr(module, "out_features")) if hasattr(module, "out_features") else None,
        "num_features": int(getattr(module, "num_features")) if hasattr(module, "num_features") else None,
        "groups": int(getattr(module, "groups", 1) or 1),
        "kernel_size": _as_list(getattr(module, "kernel_size", None)),
        "stride": _as_list(getattr(module, "stride", None)),
        "padding": _as_list(getattr(module, "padding", None)),
        "dilation": _as_list(getattr(module, "dilation", None)),
        "output_padding": _as_list(getattr(module, "output_padding", None)),
        "weight_shape": [int(dim) for dim in weight.shape] if torch.is_tensor(weight) else [],
        "bias_shape": [int(dim) for dim in bias.shape] if torch.is_tensor(bias) else [],
        "parameter_count": int(sum(parameter.numel() for parameter in module.parameters(recurse=False))),
        "parameter_size_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in module.parameters(recurse=False))),
        "canonical_order": int(order),
        "protected_contract": _protected_contract(name),
        "source_state_dict_key": _state_key(state_dict, name, "weight"),
        "source_bias_state_dict_key": _state_key(state_dict, name, "bias"),
    }
    return row


def build_physical_structure_snapshot_v2(
    model: nn.Module,
    *,
    state_dict: Mapping[str, Any] | None = None,
    generated_from: str = "pruned_model_object.pth:model.named_modules+state_dict",
) -> dict[str, Any]:
    state_dict = state_dict if state_dict is not None else model.state_dict()
    supported = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.modules.batchnorm._BatchNorm)
    rows = [
        _snapshot_module_row(name, module, order, state_dict)
        for order, (name, module) in enumerate((item for item in model.named_modules() if item[0] and isinstance(item[1], supported)))
    ]
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_from": generated_from,
        "module_count": len(rows),
        "weighted_module_count": sum(1 for row in rows if row.get("weight_shape")),
        "parameter_count": sum(int(row.get("parameter_count", 0) or 0) for row in rows),
        "parameter_size_bytes": sum(int(row.get("parameter_size_bytes", 0) or 0) for row in rows),
        "modules": rows,
    }


def _hash_rows(snapshot: Mapping[str, Any], fields: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for row in snapshot.get("modules", []):
        if not isinstance(row, Mapping):
            continue
        rows.append({field: row.get(field) for field in fields})
    return sorted(rows, key=lambda row: (int(row.get("canonical_order", 0) or 0), str(row.get("canonical_module_name", ""))))


def compute_physical_hash_v2(
    snapshot: Mapping[str, Any],
    *,
    legacy_structure_hash: str = "",
    legacy_shape_hash: str = "",
) -> dict[str, Any]:
    structure_rows = _hash_rows(snapshot, STRUCTURE_HASH_FIELDS)
    shape_rows = _hash_rows(snapshot, SHAPE_HASH_FIELDS)
    snapshot_core = {
        "snapshot_schema_version": snapshot.get("snapshot_schema_version", SNAPSHOT_SCHEMA_VERSION),
        "modules": sorted(
            [{key: row.get(key) for key in set(STRUCTURE_HASH_FIELDS + SHAPE_HASH_FIELDS)} for row in snapshot.get("modules", []) if isinstance(row, Mapping)],
            key=lambda row: (int(row.get("canonical_order", 0) or 0), str(row.get("canonical_module_name", ""))),
        ),
    }
    return {
        "hash_schema_version": HASH_SCHEMA_VERSION,
        "structure_hash_v2": sha256_payload({"schema": HASH_SCHEMA_VERSION, "structure": structure_rows}),
        "shape_hash_v2": sha256_payload({"schema": HASH_SCHEMA_VERSION, "shape": shape_rows}),
        "snapshot_sha256": sha256_payload(snapshot_core),
        "module_count": len(snapshot.get("modules", [])),
        "generated_from": "physical_structure_snapshot_v2.json",
        "legacy_structure_hash": str(legacy_structure_hash),
        "legacy_shape_hash": str(legacy_shape_hash),
        "physical_metrics_v2": {
            "parameter_count": int(snapshot.get("parameter_count", 0) or 0),
            "parameter_size_bytes": int(snapshot.get("parameter_size_bytes", 0) or 0),
            "weighted_module_count": int(snapshot.get("weighted_module_count", 0) or 0),
            "metric_source": "physical_structure_snapshot_v2",
        },
        "hash_input_summary": {
            "structure_fields": list(STRUCTURE_HASH_FIELDS),
            "shape_fields": list(SHAPE_HASH_FIELDS),
            "sort": "canonical_order,canonical_module_name",
            "json": "sort_keys=True,separators=(',',':'),ensure_ascii=True",
        },
    }


def _row_attrs(row: Mapping[str, Any], stage: str) -> dict[str, Any]:
    raw = row.get(stage) if isinstance(row.get(stage), Mapping) else {}
    if isinstance(raw, Mapping) and isinstance(raw.get("attrs"), Mapping):
        raw = raw["attrs"]
    return {str(key): value for key, value in raw.items()} if isinstance(raw, Mapping) else {}


def build_sampling_structure_request(manifest: Mapping[str, Any]) -> dict[str, Any]:
    requests = []
    rows = manifest.get("before_after_shapes") or []
    if isinstance(rows, Mapping):
        rows = [{"module_name": name, **dict(row)} for name, row in rows.items() if isinstance(row, Mapping)]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        module_name = str(row.get("module_name", ""))
        request_id = str(row.get("request_id") or f"sampling_request_{index:04d}")
        requests.append(
            {
                "request_id": request_id,
                "dependency_domain_id": str(row.get("dependency_domain_id") or row.get("pruning_domain_id") or module_name),
                "module_name": module_name,
                "affected_modules": list(row.get("affected_modules") or [module_name]),
                "requested_before": _row_attrs(row, "before"),
                "requested_after": _row_attrs(row, "after"),
                "requested_pruned_indices": list(row.get("requested_pruned_indices") or []),
                "requested_keep_indices": list(row.get("requested_keep_indices") or []),
                "sampler_constraints": {
                    "round_to": manifest.get("round_to", manifest.get("ordinary_conv_round_to", 4)),
                    "max_channel_prune_ratio": manifest.get("max_channel_prune_ratio"),
                    "min_channel_keep_ratio": manifest.get("min_channel_keep_ratio"),
                    "grouped_conv_safe_per_group": manifest.get("grouped_conv_safe_per_group", []),
                },
                "random_seed": manifest.get("random_seed"),
                "sampling_policy_version": SAMPLING_POLICY_VERSION,
            }
        )
    return {
        "sampling_request_schema_version": "sampling-structure-request-v2",
        "sampling_policy_version": SAMPLING_POLICY_VERSION,
        "request_count": len(requests),
        "requests": requests,
    }


def _snapshot_index(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("canonical_module_name")): dict(row)
        for row in snapshot.get("modules", [])
        if isinstance(row, Mapping) and row.get("canonical_module_name")
    }


def _comparable_subset(actual: Mapping[str, Any], requested: Mapping[str, Any]) -> dict[str, Any]:
    return {key: actual.get(key) for key in requested if key in actual}


def build_physical_application_ledger(
    sampling_request: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    physical_plan: Mapping[str, Any] | None = None,
    legacy_migration: bool = False,
) -> dict[str, Any]:
    modules = _snapshot_index(snapshot)
    plan_entries = list((physical_plan or {}).get("requests", []) or [])
    entries = []
    for request in sampling_request.get("requests", []):
        request = dict(request)
        affected = list(request.get("affected_modules") or [request.get("module_name")])
        applied_modules = [name for name in affected if name in modules]
        actual_rows = {name: modules[name] for name in applied_modules}
        primary = modules.get(str(request.get("module_name", "")), {})
        before = dict(request.get("requested_before") or {})
        after = dict(request.get("requested_after") or {})
        actual_after = _comparable_subset(primary, after)
        actual_before = _comparable_subset(primary, before)
        matching_plan = [row for row in plan_entries if any(str(name) in json.dumps(row, sort_keys=True) for name in affected)]
        if after and actual_after == after:
            status = "applied"
            skip_reason = ""
            repair_reason = ""
        elif before and actual_before == before:
            status = "skipped"
            skip_reason = "unknown_legacy_provenance" if legacy_migration else "not_emitted_to_final_plan"
            repair_reason = ""
        else:
            status = "repaired"
            skip_reason = ""
            repair_reason = "unknown_legacy_provenance" if legacy_migration else "dependency_domain_conflict"
        entries.append(
            {
                "request_id": str(request.get("request_id", "")),
                "dependency_domain_id": str(request.get("dependency_domain_id", "")),
                "requested_modules": affected,
                "requested_before": before,
                "requested_after": after,
                "status": status,
                "applied_modules": applied_modules,
                "applied_before": before,
                "applied_after": actual_rows,
                "applied_pruned_indices": [],
                "applied_keep_indices": [],
                "physical_plan_entries": matching_plan,
                "merged_into_request_id": "",
                "skip_reason": skip_reason,
                "repair_reason": repair_reason,
                "alignment_rule": request.get("sampler_constraints", {}).get("round_to"),
                "protection_rule": "",
                "dependency_closure_summary": {
                    "requested_module_count": len(affected),
                    "materialized_module_count": len(applied_modules),
                    "legacy_migration": bool(legacy_migration),
                },
            }
        )
    return {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "entry_count": len(entries),
        "entries": entries,
        "status_counts": {status: sum(1 for row in entries if row["status"] == status) for status in sorted(TERMINAL_STATUSES)},
    }


def validate_physical_application_ledger(sampling_request: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    request_ids = [str(row.get("request_id", "")) for row in sampling_request.get("requests", [])]
    entries = [row for row in ledger.get("entries", []) if isinstance(row, Mapping)]
    by_id = {str(row.get("request_id", "")): row for row in entries}
    missing = sorted(request_id for request_id in request_ids if request_id not in by_id)
    invalid = sorted(
        request_id
        for request_id, row in by_id.items()
        if str(row.get("status", "")) not in TERMINAL_STATUSES
        or (row.get("status") == "skipped" and not row.get("skip_reason"))
        or (row.get("status") == "repaired" and not row.get("repair_reason"))
    )
    duplicates = sorted({request_id for request_id in request_ids if request_ids.count(request_id) > 1})
    return {
        "passed": not missing and not invalid and not duplicates and len(entries) == len(request_ids),
        "request_count": len(request_ids),
        "ledger_entry_count": len(entries),
        "missing_request_ids": missing,
        "invalid_terminal_entries": invalid,
        "duplicate_request_ids": duplicates,
    }


def interpret_weight_shape(module_type: str, weight_shape: Sequence[int], *, groups: int = 1, trans_b: int = 0, transpose_perms: Sequence[Sequence[int]] = ()) -> dict[str, Any]:
    shape = [int(value) for value in weight_shape]
    groups = int(groups or 1)
    if module_type in {"Conv2d", "Conv"} and len(shape) >= 2:
        return {"layout": "[C_out,C_in/groups,kH,kW]", "logical_in_channels": shape[1] * groups, "logical_out_channels": shape[0], "groups": groups, "weight_shape": shape}
    if module_type in {"ConvTranspose2d", "ConvTranspose"} and len(shape) >= 2:
        return {"layout": "[C_in,C_out/groups,kH,kW]", "logical_in_channels": shape[0], "logical_out_channels": shape[1] * groups, "groups": groups, "weight_shape": shape}
    matrix = list(shape)
    if len(matrix) == 2 and trans_b:
        matrix = [matrix[1], matrix[0]]
    for perm in transpose_perms:
        if list(perm) == [1, 0] and len(matrix) == 2:
            matrix = [matrix[1], matrix[0]]
    return {"layout": "matrix", "logical_matrix_shape": matrix, "transB": int(trans_b), "transpose_perms": [list(item) for item in transpose_perms], "weight_shape": shape}


def _shape_after_transposes(shape: Sequence[int], transpose_perms: Sequence[Sequence[int]]) -> list[int]:
    effective = [int(value) for value in shape]
    for perm in reversed([list(item) for item in transpose_perms]):
        if len(perm) != len(effective) or sorted(perm) != list(range(len(effective))):
            return []
        effective = [effective[index] for index in perm]
    return effective


def onnx_weight_shape_matches_module(module: nn.Module, trace: Mapping[str, Any]) -> dict[str, Any]:
    live_shape = [int(dim) for dim in module.weight.shape] if torch.is_tensor(getattr(module, "weight", None)) else []
    root_shape = [int(dim) for dim in trace.get("root_initializer_shape", [])]
    effective_shape = _shape_after_transposes(root_shape, trace.get("transpose_perms", []))
    op_type = str(trace.get("compute_op_type", ""))
    expected_consumed = list(live_shape)
    if isinstance(module, nn.Linear) and len(live_shape) == 2:
        if op_type == "MatMul":
            expected_consumed = [live_shape[1], live_shape[0]]
        elif op_type == "Gemm" and not int(trace.get("transB", 0) or 0):
            expected_consumed = [live_shape[1], live_shape[0]]
    return {
        "passed": bool(trace.get("success")) and effective_shape == expected_consumed,
        "live_weight_shape": live_shape,
        "root_initializer_shape": root_shape,
        "effective_consumed_weight_shape": effective_shape,
        "expected_consumed_weight_shape": expected_consumed,
        "compute_op_type": op_type,
        "transB": int(trace.get("transB", 0) or 0),
        "transpose_perms": [list(item) for item in trace.get("transpose_perms", [])],
    }


def _onnx_attribute(node: Any, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name != name:
            continue
        if attr.ints:
            return [int(value) for value in attr.ints]
        return int(attr.i)
    return default


def build_onnx_weight_trace_index(onnx_path: str | Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    return {
        "nodes": {str(node.name): node for node in model.graph.node},
        "initializers": {item.name: [int(dim) for dim in item.dims] for item in model.graph.initializer},
        "producers": {str(output): producer for producer in model.graph.node for output in producer.output},
    }


def trace_onnx_weight_from_index(index: Mapping[str, Any], node_name: str, *, original_node_name: str = "") -> dict[str, Any]:
    nodes = index.get("nodes", {})
    node = nodes.get(str(node_name)) or nodes.get(str(original_node_name))
    if node is None or len(node.input) < 2:
        return {"success": False, "failure_reason": "compute_node_or_weight_input_missing", "node_name": node_name}
    initializers = index.get("initializers", {})
    producers = index.get("producers", {})
    consumed = str(node.input[1])
    current = consumed
    chain = []
    transpose_perms = []
    visited = set()
    while current and current not in visited:
        visited.add(current)
        if current in initializers:
            return {
                "success": True,
                "compute_node_name": str(node.name),
                "compute_op_type": str(node.op_type),
                "consumed_weight_tensor": consumed,
                "root_initializer": current,
                "root_initializer_shape": initializers[current],
                "trace_chain": chain,
                "transpose_perms": transpose_perms,
                "transB": int(_onnx_attribute(node, "transB", 0) or 0),
            }
        producer = producers.get(current)
        if producer is None:
            break
        chain.append({"node_name": str(producer.name), "op_type": str(producer.op_type), "output_tensor": current, "inputs": list(producer.input)})
        if producer.op_type == "Transpose":
            transpose_perms.append(list(_onnx_attribute(producer, "perm", []) or []))
        if producer.op_type not in PASSTHROUGH_WEIGHT_OPS or not producer.input:
            break
        current = str(producer.input[0])
    return {
        "success": False,
        "failure_reason": "root_weight_initializer_unresolved",
        "compute_node_name": str(node.name),
        "consumed_weight_tensor": consumed,
        "trace_chain": chain,
        "unresolved_tensor": current,
    }


def trace_onnx_weight_to_initializer(onnx_path: str | Path, node_name: str, *, original_node_name: str = "") -> dict[str, Any]:
    return trace_onnx_weight_from_index(
        build_onnx_weight_trace_index(onnx_path),
        node_name,
        original_node_name=original_node_name,
    )


def compute_deployment_profile_hash_v2(
    *,
    shape_hash_v2: str,
    profile: Mapping[str, Any],
    requested_int8_modules: Sequence[str],
    qdq_policy_version: str = QDQ_POLICY_VERSION,
    canonical_mapping_version: str = CANONICAL_MAPPING_VERSION,
    trt_build_policy_version: str = TRT_BUILD_POLICY_VERSION,
) -> str:
    assignment = {str(key): str(value).lower() for key, value in (profile.get("layer_precision_assignment") or {}).items()}
    payload = {
        "schema": "deployment-profile-v2",
        "shape_hash_v2": str(shape_hash_v2),
        "precision_assignment": assignment,
        "requested_int8_modules": sorted({str(value) for value in requested_int8_modules}),
        "qdq_policy_version": str(qdq_policy_version),
        "canonical_mapping_version": str(canonical_mapping_version),
        "trt_build_policy_version": str(trt_build_policy_version),
    }
    return sha256_payload(payload)


def _load_physical_model(subnet_dir: Path) -> tuple[nn.Module, Mapping[str, Any]]:
    object_payload = torch.load(subnet_dir / "pruned_model_object.pth", map_location="cpu", weights_only=False)
    model = object_payload.get("model_object") if isinstance(object_payload, Mapping) else object_payload
    state_payload = torch.load(subnet_dir / "pruned_state_dict_with_manifest.pth", map_location="cpu", weights_only=False)
    state = state_payload.get("state_dict") if isinstance(state_payload, Mapping) else state_payload
    if not isinstance(model, nn.Module) or not isinstance(state, Mapping):
        raise TypeError("physical model/state_dict artifact missing")
    return model.eval(), state


def run_physical_structure_preflight(
    *,
    subnet_dir: str | Path,
    profile_dir: str | Path,
    require_snapshot: bool = True,
    physical_model: nn.Module | None = None,
    physical_state_dict: Mapping[str, Any] | None = None,
    base_onnx_trace_index: Mapping[str, Any] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    subnet_dir = Path(subnet_dir)
    profile_dir = Path(profile_dir)
    snapshot_path = subnet_dir / "physical_structure_snapshot_v2.json"
    hash_path = subnet_dir / "physical_hash_v2.json"
    snapshot = read_json(snapshot_path, {})
    physical_hash = read_json(hash_path, {})
    reasons = []
    if require_snapshot and not snapshot.get("modules"):
        reasons.append("physical_structure_snapshot_v2_missing")
    if require_snapshot and physical_hash.get("hash_schema_version") != HASH_SCHEMA_VERSION:
        reasons.append("physical_hash_v2_missing")
    if physical_model is None or physical_state_dict is None:
        model, state = _load_physical_model(subnet_dir)
    else:
        model, state = physical_model, physical_state_dict
    modules = dict(model.named_modules())
    snapshot_rows = _snapshot_index(snapshot)
    live_state_snapshot_checks = []
    for module_name, snapshot_row in snapshot_rows.items():
        module = modules.get(module_name)
        state_key = str(snapshot_row.get("source_state_dict_key", ""))
        bias_key = str(snapshot_row.get("source_bias_state_dict_key", ""))
        live_weight = getattr(module, "weight", None) if module is not None else None
        live_bias = getattr(module, "bias", None) if module is not None else None
        live_shape = [int(dim) for dim in live_weight.shape] if torch.is_tensor(live_weight) else []
        live_bias_shape = [int(dim) for dim in live_bias.shape] if torch.is_tensor(live_bias) else []
        state_shape = [int(dim) for dim in state[state_key].shape] if state_key in state and torch.is_tensor(state[state_key]) else []
        state_bias_shape = [int(dim) for dim in state[bias_key].shape] if bias_key in state and torch.is_tensor(state[bias_key]) else []
        snapshot_shape = [int(dim) for dim in snapshot_row.get("weight_shape", [])]
        snapshot_bias_shape = [int(dim) for dim in snapshot_row.get("bias_shape", [])]
        passed = module is not None and live_shape == state_shape == snapshot_shape and live_bias_shape == state_bias_shape == snapshot_bias_shape
        if not passed:
            reasons.append(f"live_state_snapshot_mismatch:{module_name}")
        live_state_snapshot_checks.append(
            {
                "canonical_module_name": module_name,
                "module_type": str(snapshot_row.get("module_type", "")),
                "source_state_dict_key": state_key,
                "source_bias_state_dict_key": bias_key,
                "live_weight_shape": live_shape,
                "saved_state_dict_weight_shape": state_shape,
                "snapshot_weight_shape": snapshot_shape,
                "live_bias_shape": live_bias_shape,
                "saved_state_dict_bias_shape": state_bias_shape,
                "snapshot_bias_shape": snapshot_bias_shape,
                "passed": passed,
            }
        )
    recomputed_hash = compute_physical_hash_v2(snapshot)
    hash_matches_snapshot = bool(physical_hash) and all(
        str(physical_hash.get(key, "")) == str(recomputed_hash.get(key, ""))
        for key in ("hash_schema_version", "structure_hash_v2", "shape_hash_v2", "snapshot_sha256", "module_count")
    )
    if not hash_matches_snapshot:
        reasons.append("physical_hash_v2_does_not_match_snapshot")
    mapping = read_json(profile_dir / "canonical_precision_mapping.json", {})
    mapping_entries = [entry for entry in mapping.get("entries", []) if isinstance(entry, Mapping)]
    canonical_names = [str(entry.get("canonical_module_name", "")) for entry in mapping_entries]
    unique_node_names = [str(entry.get("onnx_node_name_unique", "")) for entry in mapping_entries]
    mapping_unique = (
        int(mapping.get("ambiguous_mapping_count", 0) or 0) == 0
        and all(canonical_names)
        and all(unique_node_names)
        and len(canonical_names) == len(set(canonical_names))
        and len(unique_node_names) == len(set(unique_node_names))
    )
    if not mapping_unique:
        reasons.append("canonical_precision_mapping_not_unique")
    base_onnx = subnet_dir / "onnx/model_signal_maxk.onnx"
    qdq_onnx = profile_dir / "onnx/model_mixed_qdq.onnx"
    try:
        base_trace_index = dict(base_onnx_trace_index) if base_onnx_trace_index is not None else build_onnx_weight_trace_index(base_onnx)
        qdq_trace_index = build_onnx_weight_trace_index(qdq_onnx)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"onnx_weight_trace_index_load_failed:{type(exc).__name__}:{exc}")
        base_trace_index = {"nodes": {}, "initializers": {}, "producers": {}}
        qdq_trace_index = {"nodes": {}, "initializers": {}, "producers": {}}
    checks = []
    for entry in mapping_entries:
        module_name = str(entry.get("canonical_module_name", ""))
        module = modules.get(module_name)
        snapshot_row = snapshot_rows.get(module_name, {})
        if module is None or not hasattr(module, "weight") or not snapshot_row:
            reasons.append(f"canonical_module_missing_from_physical_snapshot:{module_name}")
            checks.append(
                {
                    "canonical_module_name": module_name,
                    "passed": False,
                    "failure_reason": "canonical_module_missing_from_live_or_snapshot",
                }
            )
            continue
        live_shape = [int(dim) for dim in module.weight.shape]
        state_key = str(snapshot_row.get("source_state_dict_key") or f"{module_name}.weight")
        state_shape = [int(dim) for dim in state[state_key].shape] if state_key in state and torch.is_tensor(state[state_key]) else []
        base_trace = trace_onnx_weight_from_index(base_trace_index, str(entry.get("onnx_node_name_unique", "")), original_node_name=str(entry.get("onnx_node_name_original", "")))
        qdq_trace = trace_onnx_weight_from_index(qdq_trace_index, str(entry.get("onnx_node_name_unique", "")), original_node_name=str(entry.get("onnx_node_name_original", "")))
        snapshot_shape = [int(dim) for dim in snapshot_row.get("weight_shape", [])]
        base_shape = [int(dim) for dim in base_trace.get("root_initializer_shape", [])]
        qdq_shape = [int(dim) for dim in qdq_trace.get("root_initializer_shape", [])]
        mapping_initializer = str(entry.get("onnx_weight_initializer", ""))
        base_shape_match = onnx_weight_shape_matches_module(module, base_trace)
        qdq_shape_match = onnx_weight_shape_matches_module(module, qdq_trace)
        passed = (
            live_shape == state_shape == snapshot_shape
            and bool(base_shape_match.get("passed"))
            and bool(qdq_shape_match.get("passed"))
            and mapping_initializer == str(base_trace.get("root_initializer", ""))
            and str(base_trace.get("root_initializer", "")) == str(qdq_trace.get("root_initializer", ""))
        )
        if not passed:
            reasons.append(f"physical_weight_chain_mismatch:{module_name}")
        checks.append(
            {
                "canonical_module_name": module_name,
                "module_type": type(module).__name__,
                "groups": int(getattr(module, "groups", 1) or 1),
                "live_weight_shape": live_shape,
                "saved_state_dict_weight_shape": state_shape,
                "snapshot_weight_shape": snapshot_shape,
                "base_onnx_weight_shape": base_shape,
                "qdq_root_weight_shape": qdq_shape,
                "canonical_mapping_initializer": mapping_initializer,
                "base_root_initializer": base_trace.get("root_initializer", ""),
                "qdq_root_initializer": qdq_trace.get("root_initializer", ""),
                "base_consumed_weight_tensor": base_trace.get("consumed_weight_tensor", ""),
                "qdq_consumed_weight_tensor": qdq_trace.get("consumed_weight_tensor", ""),
                "base_trace_chain": base_trace.get("trace_chain", []),
                "qdq_trace_chain": qdq_trace.get("trace_chain", []),
                "base_weight_shape_interpretation": base_shape_match,
                "qdq_weight_shape_interpretation": qdq_shape_match,
                "weight_layout": interpret_weight_shape(type(module).__name__, live_shape, groups=int(getattr(module, "groups", 1) or 1), trans_b=int(base_trace.get("transB", 0) or 0), transpose_perms=base_trace.get("transpose_perms", [])),
                "passed": passed,
            }
        )
    if not checks:
        reasons.append("canonical_mapping_has_no_weighted_physical_checks")
    report = {
        "preflight_schema_version": "physical-onnx-preflight-v2",
        "preflight_passed": not reasons,
        "failure_reason": ";".join(dict.fromkeys(reasons)),
        "live_state_snapshot_check_count": len(live_state_snapshot_checks),
        "live_state_snapshot_checks": live_state_snapshot_checks,
        "live_state_snapshot_passed": all(row.get("passed") for row in live_state_snapshot_checks),
        "canonical_mapping_unique": mapping_unique,
        "physical_hash_matches_snapshot": hash_matches_snapshot,
        "recomputed_physical_hash_v2": recomputed_hash,
        "check_count": len(checks),
        "checks": checks,
        "physical_hash_v2": physical_hash,
        "provenance": {
            "snapshot_sha256": sha256_file(snapshot_path),
            "base_onnx_sha256": sha256_file(base_onnx),
            "qdq_onnx_sha256": sha256_file(qdq_onnx),
            "canonical_mapping_sha256": sha256_file(profile_dir / "canonical_precision_mapping.json"),
        },
    }
    if write_report:
        atomic_write_json(profile_dir / "physical_structure_preflight_report.json", report)
    return report
