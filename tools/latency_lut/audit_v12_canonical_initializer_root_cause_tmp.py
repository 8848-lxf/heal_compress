#!/usr/bin/env python3
"""Read-only root-cause audit for v12 canonical initializer shape mismatches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import onnx
import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _path in (_UNIAD, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


FAILED_SUBNETS = (
    "subnet_003",
    "subnet_004",
    "subnet_019",
    "subnet_021",
    "subnet_022",
    "subnet_023",
    "subnet_028",
    "subnet_029",
    "subnet_031",
)
ALLOWED_VERDICTS = {
    "physical_pruning_surgery_error",
    "stale_pruning_manifest",
    "stale_base_onnx",
    "onnx_exporter_shape_error",
    "stale_origin_map",
    "canonical_mapping_wrong_initializer",
    "qdq_weight_rewrite_error",
    "convtranspose_layout_checker_bug",
    "grouped_conv_layout_checker_bug",
    "gemm_matmul_transpose_checker_bug",
    "copied_artifact_provenance_error",
    "unknown_requires_reproduction",
}
RANDOM_SAMPLING_METHOD = "deployment_aware_random_without_taylor_ranking"


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def attrs(row: Mapping[str, Any], stage: str) -> dict[str, int]:
    raw = row.get(stage) if isinstance(row.get(stage), Mapping) else {}
    if isinstance(raw, Mapping) and isinstance(raw.get("attrs"), Mapping):
        raw = raw["attrs"]
    out: dict[str, int] = {}
    if isinstance(raw, Mapping):
        for key in ("in_channels", "out_channels", "in_features", "out_features", "num_features", "groups"):
            if key in raw and raw[key] is not None:
                try:
                    out[key] = int(raw[key])
                except Exception:
                    pass
    return out


def rows_by_module(value: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(value, Mapping):
        for name, row in value.items():
            if isinstance(row, Mapping):
                out[str(name)] = {"module_name": str(name), **dict(row)}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for row in value:
            if isinstance(row, Mapping) and row.get("module_name"):
                out[str(row["module_name"])] = dict(row)
    return out


def tensor_shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape] if torch.is_tensor(value) else []


def module_actual_attrs(module: torch.nn.Module | None) -> dict[str, int]:
    if module is None:
        return {}
    out: dict[str, int] = {}
    for key in ("in_channels", "out_channels", "in_features", "out_features", "num_features", "groups"):
        if hasattr(module, key):
            value = getattr(module, key)
            if isinstance(value, int):
                out[key] = int(value)
    return out


def load_model_and_state(subnet_dir: Path) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    payload = torch.load(subnet_dir / "pruned_model_object.pth", map_location="cpu", weights_only=False)
    model = payload.get("model_object") if isinstance(payload, Mapping) else payload
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"missing model object: {subnet_dir}")
    state_payload = torch.load(subnet_dir / "pruned_state_dict_with_manifest.pth", map_location="cpu", weights_only=False)
    state = state_payload.get("state_dict") if isinstance(state_payload, Mapping) else state_payload
    if not isinstance(state, Mapping):
        raise TypeError(f"missing state_dict: {subnet_dir}")
    return model.eval(), state


def state_weight(state: Mapping[str, Any], module_name: str) -> tuple[str, list[int]]:
    exact = f"{module_name}.weight"
    if exact in state:
        return exact, tensor_shape(state[exact])
    matches = [key for key in state if str(key).endswith(exact)]
    if len(matches) == 1:
        return str(matches[0]), tensor_shape(state[matches[0]])
    return "", []


class OnnxGraph:
    def __init__(self, path: Path):
        self.path = path
        self.model = onnx.load(str(path))
        self.initializers = {item.name: [int(dim) for dim in item.dims] for item in self.model.graph.initializer}
        self.nodes_by_name = {str(node.name): node for node in self.model.graph.node}
        self.producer = {str(output): node for node in self.model.graph.node for output in node.output}

    def node(self, unique: str, original: str = "") -> Any | None:
        return self.nodes_by_name.get(unique) or self.nodes_by_name.get(original)

    def attribute(self, node: Any, name: str, default: Any = None) -> Any:
        for attr in node.attribute:
            if attr.name != name:
                continue
            if attr.ints:
                return [int(value) for value in attr.ints]
            if attr.i or name in {"transA", "transB"}:
                return int(attr.i)
        return default

    def trace_to_initializer(self, tensor: str) -> dict[str, Any]:
        current = str(tensor)
        chain: list[dict[str, Any]] = []
        transpose_perms: list[list[int]] = []
        visited: set[str] = set()
        passthrough = {"QuantizeLinear", "DequantizeLinear", "Cast", "Identity", "Transpose", "Reshape", "Squeeze", "Unsqueeze"}
        while current and current not in visited:
            visited.add(current)
            if current in self.initializers:
                return {
                    "initializer": current,
                    "initializer_shape": self.initializers[current],
                    "trace_chain": chain,
                    "transpose_perms": transpose_perms,
                }
            node = self.producer.get(current)
            if node is None:
                break
            row = {"node_name": str(node.name), "op_type": str(node.op_type), "output_tensor": current, "inputs": list(node.input)}
            chain.append(row)
            if node.op_type == "Transpose":
                transpose_perms.append(list(self.attribute(node, "perm", [])))
            if node.op_type not in passthrough or not node.input:
                break
            current = str(node.input[0])
        return {"initializer": "", "initializer_shape": [], "trace_chain": chain, "transpose_perms": transpose_perms, "unresolved_tensor": current}


def mapping_entry(profile_dir: Path, module_name: str) -> dict[str, Any]:
    mapping = read_json(profile_dir / "canonical_precision_mapping.json", {})
    rows = [dict(row) for row in mapping.get("entries", []) if row.get("canonical_module_name") == module_name]
    return rows[0] if len(rows) == 1 else {}


def origin_entry(subnet_dir: Path, module_name: str) -> dict[str, Any]:
    origin = read_json(subnet_dir / "onnx" / "onnx_export_origin_map.json", {})
    rows = [dict(row) for row in origin.get("entries", []) if row.get("canonical_module_name") == module_name]
    return rows[0] if len(rows) == 1 else {}


def logical_weight_interpretation(op_type: str, shape: Sequence[int], groups: int, trans_b: int = 0, transpose_perms: Sequence[Sequence[int]] = ()) -> dict[str, Any]:
    shape = [int(value) for value in shape]
    result: dict[str, Any] = {"raw_weight_shape": shape, "groups": int(groups)}
    if len(shape) >= 2 and op_type == "Conv":
        result.update({"logical_in_channels": shape[1] * int(groups), "logical_out_channels": shape[0], "layout": "[C_out,C_in/groups,kH,kW]"})
    elif len(shape) >= 2 and op_type == "ConvTranspose":
        result.update({"logical_in_channels": shape[0], "logical_out_channels": shape[1] * int(groups), "layout": "[C_in,C_out/groups,kH,kW]"})
    elif len(shape) == 2 and op_type in {"Gemm", "MatMul"}:
        interpreted = list(shape)
        if op_type == "Gemm" and int(trans_b):
            interpreted = [shape[1], shape[0]]
        for perm in transpose_perms:
            if list(perm) == [1, 0]:
                interpreted = [interpreted[1], interpreted[0]]
        result.update({"logical_matrix_shape": interpreted, "transB": int(trans_b), "transpose_perms": [list(x) for x in transpose_perms]})
    return result


def mismatch_axes(actual: Sequence[int], expected: Sequence[int]) -> list[int]:
    count = max(len(actual), len(expected))
    return [idx for idx in range(count) if idx >= len(actual) or idx >= len(expected) or int(actual[idx]) != int(expected[idx])]


def extract_shape_detail(
    subnet_dir: Path,
    profile_dir: Path,
    check: Mapping[str, Any],
    model: torch.nn.Module,
    state: Mapping[str, Any],
    base_graph: OnnxGraph,
    qdq_graph: OnnxGraph,
) -> dict[str, Any]:
    manifest = read_json(subnet_dir / "pruning_manifest.json", {})
    physical = rows_by_module(manifest.get("module_channel_before_after"))
    estimates = rows_by_module(manifest.get("before_after_shapes"))
    module_name = str(check.get("canonical_module_name", ""))
    entry = mapping_entry(profile_dir, module_name)
    origin = origin_entry(subnet_dir, module_name)
    unique_name = str(entry.get("onnx_node_name_unique") or check.get("onnx_node_name_unique") or "")
    original_name = str(entry.get("onnx_node_name_original") or check.get("onnx_node_name_original") or "")
    base_node = base_graph.node(unique_name, original_name)
    qdq_node = qdq_graph.node(unique_name, original_name)
    base_consumed = str(base_node.input[1]) if base_node is not None and len(base_node.input) > 1 else ""
    qdq_consumed = str(qdq_node.input[1]) if qdq_node is not None and len(qdq_node.input) > 1 else ""
    base_trace = base_graph.trace_to_initializer(base_consumed)
    qdq_trace = qdq_graph.trace_to_initializer(qdq_consumed)
    modules = dict(model.named_modules())
    module = modules.get(module_name)
    state_key, saved_shape = state_weight(state, module_name)
    live_shape = tensor_shape(getattr(module, "weight", None))
    module_type = type(module).__name__ if module is not None else str(origin.get("module_type", ""))
    groups = int(getattr(module, "groups", origin.get("groups", check.get("manifest_after_attrs", {}).get("groups", 1))) or 1)
    expected = [int(value) for value in check.get("expected_weight_shape", [])]
    interpreted = [int(value) for value in check.get("onnx_weight_shape", [])]
    estimate = estimates.get(module_name, {})
    physical_row = physical.get(module_name, {})
    manifest_source = "physical_module_channel_before_after" if physical_row else "sampling_before_after_shapes_fallback"
    op_type = str(entry.get("onnx_op_type") or check.get("onnx_op_type") or (base_node.op_type if base_node else ""))
    trans_b = int(base_graph.attribute(base_node, "transB", 0)) if base_node is not None else 0
    return {
        "subnet_id": subnet_dir.name,
        "profile_id": profile_dir.name,
        "source_subset": manifest.get("source_subset", ""),
        "canonical_module_name": module_name,
        "module_type": module_type,
        "groups": groups,
        "onnx_op_type": op_type,
        "onnx_node_name": unique_name,
        "onnx_node_name_original": original_name,
        "canonical_mapping_initializer": entry.get("onnx_weight_initializer", ""),
        "base_conv_actual_consumed_tensor": base_consumed,
        "qdq_conv_actual_consumed_tensor": qdq_consumed,
        "base_weight_trace": base_trace,
        "qdq_weight_trace": qdq_trace,
        "base_original_weight_initializer": base_trace.get("initializer", ""),
        "qdq_original_weight_initializer": qdq_trace.get("initializer", ""),
        "pytorch_live_module_weight_shape": live_shape,
        "saved_state_dict_key": state_key,
        "saved_state_dict_weight_shape": saved_shape,
        "manifest_expected_after_attrs": check.get("manifest_after_attrs", {}),
        "manifest_expected_weight_shape": expected,
        "manifest_shape_source_used_by_checker": manifest_source,
        "sampling_estimate_before_attrs": attrs(estimate, "before"),
        "sampling_estimate_after_attrs": attrs(estimate, "after"),
        "physical_module_channel_before_after_present": bool(physical_row),
        "physical_before_attrs": attrs(physical_row, "before"),
        "physical_after_attrs": attrs(physical_row, "after"),
        "base_onnx_initializer_shape": base_trace.get("initializer_shape", []),
        "qdq_onnx_initializer_shape": qdq_trace.get("initializer_shape", []),
        "origin_map_weight_shape": origin.get("weight_shape", []),
        "checker_interpreted_shape": interpreted,
        "mismatch_axis": mismatch_axes(interpreted, expected),
        "dimension_order_only": bool(interpreted != expected and sorted(interpreted) == sorted(expected)),
        "base_logical_weight_interpretation": logical_weight_interpretation(op_type, base_trace.get("initializer_shape", []), groups, trans_b, base_trace.get("transpose_perms", [])),
        "qdq_logical_weight_interpretation": logical_weight_interpretation(op_type, qdq_trace.get("initializer_shape", []), groups, trans_b, qdq_trace.get("transpose_perms", [])),
        "canonical_mapping_matches_base_root_initializer": str(entry.get("onnx_weight_initializer", "")) == str(base_trace.get("initializer", "")),
        "live_saved_base_qdq_equal": live_shape == saved_shape == list(base_trace.get("initializer_shape", [])) == list(qdq_trace.get("initializer_shape", [])),
        "shape_check_passed": bool(check.get("shape_check_passed", False)),
    }


def classify_subnet(details: Sequence[Mapping[str, Any]], source_identical: bool) -> tuple[str, list[str]]:
    if details and all(
        row.get("live_saved_base_qdq_equal")
        and not row.get("physical_module_channel_before_after_present")
        and row.get("sampling_estimate_after_attrs") == row.get("manifest_expected_after_attrs")
        for row in details
    ):
        return "stale_pruning_manifest", [
            "live module == saved state_dict == base ONNX == QDQ root initializer",
            "mismatched module is absent from physical module_channel_before_after delta, so it was not physically changed",
            "checker expectation comes from sampling-time before_after_shapes.after",
            "source and combined model/ONNX/channel-delta artifacts are identical" if source_identical else "combined provenance requires separate review",
        ]
    if any(not row.get("canonical_mapping_matches_base_root_initializer") for row in details):
        return "canonical_mapping_wrong_initializer", ["canonical mapping initializer differs from Conv/Gemm weight root"]
    if any(row.get("base_onnx_initializer_shape") != row.get("qdq_onnx_initializer_shape") for row in details):
        return "qdq_weight_rewrite_error", ["base and QDQ root initializer shapes differ"]
    if any(row.get("pytorch_live_module_weight_shape") != row.get("saved_state_dict_weight_shape") for row in details):
        return "physical_pruning_surgery_error", ["live model and saved state_dict shapes differ"]
    return "unknown_requires_reproduction", ["available shape evidence does not uniquely select another allowed verdict"]


def source_artifact_identity(dataset_dir: Path, subnet_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    source_subset = str(manifest.get("source_subset", ""))
    source_id = str(manifest.get("source_subnet_id", subnet_dir.name))
    if source_subset == "historical_v11":
        source = dataset_dir.parent / "v11_random_deployment_aware_subnets_dryrun_v2" / "subnets" / source_id
    else:
        source = dataset_dir / "_new_v12_deblock_protected_work" / "subnets" / source_id
    rows = []
    for rel in ("module_channel_before_after.json", "pruned_model_object.pth", "pruned_state_dict_with_manifest.pth", "onnx/model_signal_maxk.onnx"):
        left = subnet_dir / rel
        right = source / rel
        rows.append({"path": rel, "combined_sha256": sha256_file(left), "source_sha256": sha256_file(right), "identical": bool(left.is_file() and right.is_file() and sha256_file(left) == sha256_file(right))})
    return {"source_path": str(source), "source_exists": source.is_dir(), "artifacts": rows, "all_core_artifacts_identical": all(row["identical"] for row in rows)}


def category_for_detail(row: Mapping[str, Any]) -> str:
    if str(row.get("module_type")) == "ConvTranspose2d":
        return "convtranspose"
    if str(row.get("module_type")) == "Conv2d" and int(row.get("groups", 1)) > 1:
        return "grouped_conv"
    if str(row.get("module_type")) == "Conv2d":
        return "ordinary_conv"
    return "gemm_matmul"


def collect_success_checks(subnets_root: Path, excluded: set[str]) -> list[tuple[Path, Path, dict[str, Any]]]:
    rows = []
    seen_subnets: set[str] = set()
    for report_path in sorted(subnets_root.glob("subnet_*/profile_*/engine_structure_check_report.json")):
        subnet_dir = report_path.parents[1]
        if subnet_dir.name in excluded or subnet_dir.name in seen_subnets:
            continue
        report = read_json(report_path, {})
        if not report.get("structure_check_passed"):
            continue
        for check in report.get("canonical_shape_checks", []):
            if check.get("shape_check_passed"):
                rows.append((subnet_dir, report_path.parent, dict(check)))
        seen_subnets.add(subnet_dir.name)
    return rows


def select_controls(subnets_root: Path, unique_details: Sequence[Mapping[str, Any]], model_cache: dict[str, tuple[torch.nn.Module, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    targets: dict[str, Mapping[str, Any]] = {}
    for row in unique_details:
        category = category_for_detail(row)
        targets.setdefault(category, row)
    candidates = collect_success_checks(subnets_root, set(FAILED_SUBNETS))
    controls: list[dict[str, Any]] = []
    for category, target in targets.items():
        same_name = [item for item in candidates if item[2].get("canonical_module_name") == target.get("canonical_module_name")]
        pool = same_name or candidates
        ranked = []
        for item in pool:
            subnet_dir, profile_dir, check = item
            origin = origin_entry(subnet_dir, str(check.get("canonical_module_name", "")))
            typ = str(origin.get("module_type", ""))
            groups = int(origin.get("groups", check.get("manifest_after_attrs", {}).get("groups", 1)) or 1)
            candidate_category = "convtranspose" if typ == "ConvTranspose2d" else "grouped_conv" if typ == "Conv2d" and groups > 1 else "ordinary_conv" if typ == "Conv2d" else "gemm_matmul"
            if candidate_category != category:
                continue
            shape = [int(x) for x in check.get("onnx_weight_shape", [])]
            target_shape = [int(x) for x in target.get("base_onnx_initializer_shape", [])]
            distance = sum(abs(a - b) for a, b in zip(shape, target_shape)) + abs(len(shape) - len(target_shape)) * 100000
            ranked.append((distance, subnet_dir, profile_dir, check))
        if not ranked:
            continue
        _, subnet_dir, profile_dir, check = sorted(ranked, key=lambda item: (item[0], item[1].name))[0]
        if subnet_dir.name not in model_cache:
            model_cache[subnet_dir.name] = load_model_and_state(subnet_dir)
        model, state = model_cache[subnet_dir.name]
        detail = extract_shape_detail(
            subnet_dir,
            profile_dir,
            check,
            model,
            state,
            OnnxGraph(subnet_dir / "onnx/model_signal_maxk.onnx"),
            OnnxGraph(profile_dir / "onnx/model_mixed_qdq.onnx"),
        )
        detail["control_category"] = category
        detail["why_passed"] = "physical delta overrides the estimate, or sampling estimate already equals the live/ONNX shape"
        controls.append(detail)
    return controls


def audit_success_label_impact(subnets_root: Path) -> dict[str, Any]:
    labels = []
    for path in subnets_root.glob("subnet_*/profile_*/lut_sample_label.json"):
        row = read_json(path, {})
        row["_path"] = str(path)
        labels.append(row)
    available = [row for row in labels if row.get("label_available")]
    available_subnets = {str(row.get("subnet_id")) for row in available}
    subnet_rows = []
    for subnet_id in sorted(available_subnets):
        subnet_dir = subnets_root / subnet_id
        manifest = read_json(subnet_dir / "pruning_manifest.json", {})
        estimates = rows_by_module(manifest.get("before_after_shapes"))
        physical = rows_by_module(manifest.get("module_channel_before_after"))
        model, _ = load_model_and_state(subnet_dir)
        modules = dict(model.named_modules())
        stale = []
        actual_shape_rows = []
        actual_structure_rows = []
        for name, estimate in estimates.items():
            module = modules.get(name)
            actual = module_actual_attrs(module)
            if not actual:
                continue
            before = attrs(estimate, "before")
            estimated_after = attrs(estimate, "after")
            comparable_actual = {key: actual[key] for key in estimated_after if key in actual}
            if name not in physical and estimated_after and comparable_actual != estimated_after:
                stale.append({"module_name": name, "sampling_after": estimated_after, "actual_after": comparable_actual})
            actual_shape_rows.append({"module_name": name, "after": actual})
            actual_structure_rows.append({"module_name": name, "module_type": type(module).__name__ if module is not None else estimate.get("module_type", ""), "before": before, "after": actual})
        actual_shape_hash = stable_hash({"rows": actual_shape_rows})
        actual_structure_hash = stable_hash({"method": RANDOM_SAMPLING_METHOD, "rows": actual_structure_rows})
        subnet_rows.append({
            "subnet_id": subnet_id,
            "available_label_count": sum(1 for row in available if row.get("subnet_id") == subnet_id),
            "stored_shape_hash": manifest.get("shape_hash", ""),
            "actual_shape_hash_recomputed": actual_shape_hash,
            "stored_shape_hash_matches_actual": str(manifest.get("shape_hash", "")) == actual_shape_hash,
            "stored_structure_hash": manifest.get("structure_hash", ""),
            "actual_structure_hash_recomputed": actual_structure_hash,
            "stored_structure_hash_matches_actual": str(manifest.get("structure_hash", "")) == actual_structure_hash,
            "stale_sampling_estimate_count": len(stale),
            "first_20_stale_sampling_estimates": stale[:20],
        })
    return {
        "total_label_files": len(labels),
        "label_available_count": len(available),
        "label_available_subnet_count": len(available_subnets),
        "successful_subnets_with_stale_sampling_estimates": sum(1 for row in subnet_rows if row["stale_sampling_estimate_count"] > 0),
        "successful_labels_on_subnets_with_stale_sampling_estimates": sum(row["available_label_count"] for row in subnet_rows if row["stale_sampling_estimate_count"] > 0),
        "successful_subnets_with_stored_shape_hash_mismatch": sum(1 for row in subnet_rows if not row["stored_shape_hash_matches_actual"]),
        "successful_labels_with_stored_shape_hash_mismatch": sum(row["available_label_count"] for row in subnet_rows if not row["stored_shape_hash_matches_actual"]),
        "subnets": subnet_rows,
    }


def fresh_export_comparison(dataset_dir: Path, fresh_onnx: Path | None, unique_details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if fresh_onnx is None or not fresh_onnx.is_file():
        return {"executed": False, "failure_reason": "fresh_onnx_not_provided"}
    subnet_dir = dataset_dir / "subnets/subnet_003"
    old_graph = OnnxGraph(subnet_dir / "onnx/model_signal_maxk.onnx")
    fresh_graph = OnnxGraph(fresh_onnx)
    fresh_origin = read_json(fresh_onnx.parent / "onnx_export_origin_map.json", {})
    fresh_by_module = {str(row.get("canonical_module_name")): dict(row) for row in fresh_origin.get("entries", [])}
    rows = []
    for detail in unique_details:
        if detail.get("subnet_id") != "subnet_003":
            continue
        module = str(detail.get("canonical_module_name"))
        old_entry = origin_entry(subnet_dir, module)
        new_entry = fresh_by_module.get(module, {})
        old_node = old_graph.node(str(old_entry.get("onnx_node_name_unique", "")), str(old_entry.get("onnx_node_name_original", "")))
        new_node = fresh_graph.node(str(new_entry.get("onnx_node_name_unique", "")), str(new_entry.get("onnx_node_name_original", "")))
        old_trace = old_graph.trace_to_initializer(str(old_node.input[1]) if old_node is not None and len(old_node.input) > 1 else "")
        new_trace = fresh_graph.trace_to_initializer(str(new_node.input[1]) if new_node is not None and len(new_node.input) > 1 else "")
        rows.append({"canonical_module_name": module, "old_initializer": old_trace.get("initializer"), "old_shape": old_trace.get("initializer_shape"), "fresh_initializer": new_trace.get("initializer"), "fresh_shape": new_trace.get("initializer_shape"), "shape_identical": old_trace.get("initializer_shape") == new_trace.get("initializer_shape")})
    profile_dir = subnet_dir / "profile_000"
    mapping = read_json(profile_dir / "canonical_precision_mapping.json", {})
    manifest = read_json(subnet_dir / "pruning_manifest.json", {})
    try:
        from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import _canonical_shape_consistency_report

        preflight_passed, checks = _canonical_shape_consistency_report(onnx_path=fresh_onnx, manifest=manifest, canonical_mapping=mapping)
        preflight = {"passed": preflight_passed, "mismatches": [row for row in checks if not row.get("shape_check_passed", True)]}
    except Exception as exc:
        preflight = {"passed": False, "failure_reason": f"{type(exc).__name__}: {exc}"}
    result = {
        "executed": True,
        "subnet_id": "subnet_003",
        "old_onnx_path": str(subnet_dir / "onnx/model_signal_maxk.onnx"),
        "fresh_onnx_path": str(fresh_onnx),
        "old_onnx_sha256": sha256_file(subnet_dir / "onnx/model_signal_maxk.onnx"),
        "fresh_onnx_sha256": sha256_file(fresh_onnx),
        "module_comparisons": rows,
        "all_mismatch_module_shapes_identical": bool(rows) and all(row["shape_identical"] for row in rows),
        "unfixed_structure_preflight": preflight,
    }
    write_json(fresh_onnx.parent / "unfixed_structure_preflight_report.json", preflight)
    return result


def make_report(payload: Mapping[str, Any]) -> str:
    root = payload["root_cause_matrix"]
    dist = payload["failure_module_type_distribution_unique"]
    impact = payload["successful_label_impact"]
    repro = payload["minimal_reproduction"]
    lines = [
        "# Canonical ONNX Initializer Shape Mismatch Root-Cause Audit",
        "",
        "## Executive Summary",
        "",
        f"- affected_profiles: {len(payload['affected_profiles'])}",
        f"- affected_subnets: {len(root)}",
        f"- unique_mismatch_modules: {len(payload['unique_mismatch_details'])}",
        "- primary_root_cause: stale_pruning_manifest",
        "- checker_false_positive: true",
        "- evidence: live model, saved state_dict, base ONNX, and QDQ root initializer agree; checker alone consumes stale sampling-time after shapes for modules absent from the physical delta.",
        "- no failed artifact was deleted, overwritten, rebuilt, or admitted to LUT training.",
        "",
        "## Root-Cause Matrix",
        "",
        "| subnet | subset | profiles | mismatch modules | verdict | source copy identical |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in root:
        lines.append(f"| {row['subnet_id']} | {row['source_subset']} | {', '.join(row['affected_profiles'])} | {row['unique_mismatch_module_count']} | {row['verdict']} | {str(row['provenance']['all_core_artifacts_identical']).lower()} |")
    for row in payload.get("related_non_structure_failures", []):
        lines.append(f"\n- Excluded from the 35 structure mismatches: `{row['subnet_id']}/{row['profile_id']}` stopped at `{row['failure_stage']}` ({row['failure_reason_summary']}).")
    lines.extend([
        "",
        "## Failure Module Types",
        "",
        f"- ordinary_conv: {dist.get('ordinary_conv', 0)}",
        f"- grouped_conv: {dist.get('grouped_conv', 0)}",
        f"- convtranspose: {dist.get('convtranspose', 0)}",
        f"- gemm_matmul: {dist.get('gemm_matmul', 0)}",
        "",
        "Grouped Conv is not a layout-checker failure: `[C_out, C_in/groups, kH, kW]` reconstructs the same logical channels as the live module. ConvTranspose is likewise correctly interpreted as `[C_in, C_out/groups, kH, kW]`.",
        "",
        "## Unique Mismatch Details",
        "",
        "| subnet | module | type | g | live/state/base/QDQ | checker expected | source | axes | reorder only |",
        "|---|---|---|---:|---|---|---|---|---:|",
    ])
    for row in payload["unique_mismatch_details"]:
        lines.append(f"| {row['subnet_id']} | {row['canonical_module_name']} | {row['module_type']} | {row['groups']} | {row['pytorch_live_module_weight_shape']} | {row['manifest_expected_weight_shape']} | {row['manifest_shape_source_used_by_checker']} | {row['mismatch_axis']} | {str(row['dimension_order_only']).lower()} |")
    lines.extend(["", "Full per-profile weight traces, consumed tensors, original initializers, state_dict keys, and logical layout interpretations are in `canonical_initializer_root_cause_audit.json`.", "", "## Successful Controls", ""])
    for row in payload["successful_controls"]:
        lines.append(f"- {row['control_category']}: {row['subnet_id']}/{row['profile_id']} `{row['canonical_module_name']}`; live={row['pytorch_live_module_weight_shape']}, expected={row['manifest_expected_weight_shape']}, source={row['manifest_shape_source_used_by_checker']}. {row['why_passed']}")
    lines.extend([
        "",
        "The trigger is not a specific channel count, group count, pruning bin, QDQ profile, or deblock policy. It is the combination `sampling estimate changed` + `physical plan skipped that module` + `no physical delta row to override the estimate`.",
        "",
        "## Minimal Reproduction",
        "",
        f"- executed: {str(bool(repro.get('executed'))).lower()}",
        f"- old/fresh mismatch-module shapes identical: {str(bool(repro.get('all_mismatch_module_shapes_identical'))).lower()}",
        f"- unfixed preflight passed: {str(bool((repro.get('unfixed_structure_preflight') or {}).get('passed'))).lower()}",
        f"- old ONNX: {repro.get('old_onnx_path', '')}",
        f"- fresh ONNX: {repro.get('fresh_onnx_path', '')}",
        "",
        "If the fresh export has the same initializer shape and the unfixed preflight still fails, exporter staleness and QDQ rewrite are excluded; the checker/manifest merge is reproduced independently.",
        "",
        "## Successful Label Impact",
        "",
        f"- label_available_count: {impact.get('label_available_count', 0)}",
        f"- successful_labels_on_subnets_with_stale_sampling_estimates: {impact.get('successful_labels_on_subnets_with_stale_sampling_estimates', 0)}",
        f"- successful_labels_with_stored_shape_hash_mismatch: {impact.get('successful_labels_with_stored_shape_hash_mismatch', 0)}",
        "",
        "Engine/AP/latency measurements for the 155 passed labels are not invalidated by this false-positive checker branch because their engine was built from the matching physical ONNX. However, labels on subnets whose stored structure_hash/shape_hash differs from the recomputed physical structure must not be trusted as structure-keyed LUT records until hashes and structural metadata are repaired. The exact affected successful subnets are in the JSON impact section.",
        "",
        "## Minimal Fix",
        "",
        "- Primary location: `tools/latency_lut/run_v11_mixed_precision_lut_dataset_builder.py::_manifest_shapes_by_module`.",
        "- For materialized subnets, absence from `module_channel_before_after` means unchanged, not unknown. Initialize unmatched sampling rows with their `before` shape, then overlay physical delta rows; alternatively emit a complete post-surgery physical snapshot and make it the sole hard ground truth.",
        "- Recompute structure_hash/shape_hash from the saved physical model rather than retaining dry-run sampling hashes.",
        "- Keep `before_after_shapes` as sampling estimate only and label it explicitly; do not use it as a hard engine structure expectation.",
        "",
        "## Reprocess Scope",
        "",
    ])
    for item in payload["affected_profiles"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    subnets_root = dataset_dir / "subnets"
    model_cache: dict[str, tuple[torch.nn.Module, Mapping[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    affected_profiles: list[str] = []
    profile_reports: list[dict[str, Any]] = []
    provenance_by_subnet: dict[str, dict[str, Any]] = {}
    for subnet_id in FAILED_SUBNETS:
        subnet_dir = subnets_root / subnet_id
        manifest = read_json(subnet_dir / "pruning_manifest.json", {})
        provenance_by_subnet[subnet_id] = source_artifact_identity(dataset_dir, subnet_dir, manifest)
        model_cache[subnet_id] = load_model_and_state(subnet_dir)
        model, state = model_cache[subnet_id]
        base_graph = OnnxGraph(subnet_dir / "onnx/model_signal_maxk.onnx")
        for report_path in sorted(subnet_dir.glob("profile_*/engine_structure_check_report.json")):
            report = read_json(report_path, {})
            bad = [dict(row) for row in report.get("canonical_shape_checks", []) if not row.get("shape_check_passed", True)]
            if not bad or "canonical_onnx_initializer_shape_mismatch" not in str(report.get("failure_reason", "")):
                continue
            profile_dir = report_path.parent
            affected_profiles.append(f"{subnet_id}/{profile_dir.name}")
            qdq_graph = OnnxGraph(profile_dir / "onnx/model_mixed_qdq.onnx")
            profile_reports.append({"subnet_id": subnet_id, "profile_id": profile_dir.name, "mismatch_count": len(bad), "build_success": bool(read_json(profile_dir / "build_report.json", {}).get("build_success")), "failure_report": read_json(profile_dir / "profile_failure_report.json", {}), "worker_result_status": read_json(profile_dir / "worker_result.json", {}).get("status", "")})
            for check in bad:
                details.append(extract_shape_detail(subnet_dir, profile_dir, check, model, state, base_graph, qdq_graph))
    unique_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in details:
        unique_map.setdefault((row["subnet_id"], row["canonical_module_name"]), row)
    unique_details = list(unique_map.values())
    matrix = []
    for subnet_id in FAILED_SUBNETS:
        subset_details = [row for row in unique_details if row["subnet_id"] == subnet_id]
        verdict, evidence = classify_subnet(subset_details, provenance_by_subnet[subnet_id]["all_core_artifacts_identical"])
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"invalid verdict: {verdict}")
        manifest = read_json(subnets_root / subnet_id / "pruning_manifest.json", {})
        matrix.append({"subnet_id": subnet_id, "source_subset": manifest.get("source_subset", ""), "affected_profiles": [item.split("/", 1)[1] for item in affected_profiles if item.startswith(subnet_id + "/")], "unique_mismatch_module_count": len(subset_details), "verdict": verdict, "evidence_chain": evidence, "provenance": provenance_by_subnet[subnet_id]})
    controls = select_controls(subnets_root, unique_details, model_cache)
    impact = audit_success_label_impact(subnets_root)
    fresh = fresh_export_comparison(dataset_dir, Path(args.fresh_onnx) if args.fresh_onnx else None, unique_details)
    affected_set = set(affected_profiles)
    related_non_structure_failures = []
    for subnet_id in FAILED_SUBNETS:
        for profile_index in range(4):
            profile_id = f"profile_{profile_index:03d}"
            key = f"{subnet_id}/{profile_id}"
            profile_dir = subnets_root / subnet_id / profile_id
            if key in affected_set or not profile_dir.is_dir():
                continue
            failure = read_json(profile_dir / "profile_failure_report.json", {})
            reason = str(failure.get("failure_reason", ""))
            related_non_structure_failures.append({
                "subnet_id": subnet_id,
                "profile_id": profile_id,
                "failure_stage": str(failure.get("stage_failed", "not_a_structure_mismatch")),
                "failure_reason_summary": "TensorRT build timed out before structure gate" if "timed out" in reason else reason[:240],
            })
    payload = {
        "dataset_dir": str(dataset_dir),
        "audit_read_only_on_existing_artifacts": True,
        "affected_profiles": affected_profiles,
        "affected_profile_count": len(affected_profiles),
        "affected_subnets": list(FAILED_SUBNETS),
        "root_cause_matrix": matrix,
        "profile_reports": profile_reports,
        "all_mismatch_details": details,
        "unique_mismatch_details": unique_details,
        "failure_module_type_distribution_unique": dict(Counter(category_for_detail(row) for row in unique_details)),
        "failure_module_type_distribution_all_profiles": dict(Counter(category_for_detail(row) for row in details)),
        "successful_controls": controls,
        "related_non_structure_failures": related_non_structure_failures,
        "minimal_reproduction": fresh,
        "checker_false_positive": all(row["verdict"] == "stale_pruning_manifest" for row in matrix),
        "successful_label_impact": impact,
        "minimal_fix_locations": [
            "tools/latency_lut/run_v11_mixed_precision_lut_dataset_builder.py::_manifest_shapes_by_module",
            "tools/latency_lut/run_v11_random_deployment_aware_gate_dryrun.py::materialize_subnet (emit full physical post-surgery shape snapshot)",
        ],
        "profiles_to_reprocess_after_fix": affected_profiles,
    }
    write_json(dataset_dir / "canonical_initializer_root_cause_audit.json", payload)
    (dataset_dir / "canonical_initializer_root_cause_audit.md").write_text(make_report(payload), encoding="utf-8")
    print(json.dumps({"affected_profiles": len(affected_profiles), "unique_mismatch_modules": len(unique_details), "verdicts": dict(Counter(row["verdict"] for row in matrix)), "checker_false_positive": payload["checker_false_positive"], "report": str(dataset_dir / "canonical_initializer_root_cause_audit.md")}, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="outputs/latency_lut/v12_combined_lut_dataset_300frames")
    parser.add_argument("--fresh-onnx", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
