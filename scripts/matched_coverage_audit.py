#!/usr/bin/env python3
"""Build the canonical 70-compute-layer Legacy/explicit coverage audit.

This script is read-only with respect to all source experiment artifacts.  It
creates a new audit directory containing the matched layer inventory and the
stable mapping for the parameter-free affine-grid MatMul fusion group.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from quantization.export.origin_mapping import build_onnx_origin_map
from quantization.export.origin_trace import build_weight_trace_index, trace_compute_node_weight
from quantization.tensorrt.layer_info import (
    has_canonical_identity,
    is_weighted_compute_layer,
    load_layer_info,
    precision_name,
)


ACCEPTANCE_ROOT = REPO / "outputs/H800_explicit_qdq_acceptance_20260714_023005"
E27 = (
    ACCEPTANCE_ROOT
    / "production_trt_entropy_v4/h800_explicit_qdq_production_acceptance_20260713_131745"
    / "baselines/original_trusted_explicit_qdq_int8"
)
LEGACY_ROOT = REPO / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare"
LEGACY_ONNX = (
    LEGACY_ROOT
    / "artifacts/onnx/fixedK29696/dynamic_agent_single_engine_maxK"
    / "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
)
LEGACY_INSPECTOR = (
    LEGACY_ROOT
    / "artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/int8_train_calib200"
    / "layerinfo_dynamic_agent_single_engine_maxK_int8_train_calib200.json"
)
LEGACY_CACHE = (
    LEGACY_ROOT
    / "artifacts/calibration"
    / "lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK29696_int8_train_calib200.cache"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_inventory(model: Any) -> dict[str, dict[str, Any]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    rows = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    result: dict[str, dict[str, Any]] = {}
    for value in rows:
        tensor_type = value.type.tensor_type
        shape: list[Any] = []
        for dim in tensor_type.shape.dim:
            shape.append(int(dim.dim_value) if dim.dim_value else str(dim.dim_param or "?"))
        result[str(value.name)] = {
            "shape": shape,
            "dtype": int(tensor_type.elem_type),
            "dtype_name": onnx.TensorProto.DataType.Name(int(tensor_type.elem_type)),
        }
    return result


def parameterized_nodes(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    model = onnx.load(str(path))
    index = build_weight_trace_index(model)
    initializers = {str(row.name): numpy_helper.to_array(row) for row in model.graph.initializer}
    rows: list[dict[str, Any]] = []
    for graph_index, node in enumerate(model.graph.node):
        if str(node.op_type) not in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
            continue
        trace = trace_compute_node_weight(index, node)
        if not trace.get("success"):
            continue
        root = str(trace["root_initializer"])
        array = np.asarray(initializers[root])
        rows.append(
            {
                "graph_index": int(graph_index),
                "node": node,
                "node_name": str(node.name),
                "op_type": str(node.op_type),
                "initializer": root,
                "initializer_shape": list(array.shape),
                "initializer_dtype": str(array.dtype),
                "initializer_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return model, rows


def functional_nodes(model: Any) -> list[dict[str, Any]]:
    index = build_weight_trace_index(model)
    result = []
    for graph_index, node in enumerate(model.graph.node):
        if str(node.op_type) != "MatMul":
            continue
        trace = trace_compute_node_weight(index, node)
        if trace.get("success"):
            continue
        result.append(
            {
                "graph_index": int(graph_index),
                "node": node,
                "node_name": str(node.name),
                "inputs": [str(value) for value in node.input],
                "outputs": [str(value) for value in node.output],
            }
        )
    return result


def inspector_tensor(row: dict[str, Any], field: str) -> list[dict[str, Any]]:
    return [
        {
            "name": str(item.get("Name", "")),
            "format_dtype": str(item.get("Format/Datatype", "")),
            "dimensions": str(item.get("Dimensions", "")),
        }
        for item in row.get(field, [])
    ]


def row_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(row.get("Name", "")),
        "layer_type": str(row.get("LayerType", "")),
        "precision": precision_name(row),
        "inputs": inspector_tensor(row, "Inputs"),
        "outputs": inspector_tensor(row, "Outputs"),
        "metadata": str(row.get("Metadata", "")),
    }


def original_calls(origin_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "module_path": str(row["module_path"]),
            "module_type": str(row.get("module_type", "")),
            "call_index": int(row["call_index"]),
            "weight_initializer": str(row["weight_initializer"]),
            "weight_shape": list(row.get("weight_shape", [])),
            "groups": int(row.get("groups", 1)),
            "mapped_onnx_op_type": str(row.get("onnx_op_type", "")),
        }
        for row in origin_payload.get("entries", [])
    ]


def assert_parameterized_graph_equivalence(current: list[dict[str, Any]], legacy: list[dict[str, Any]]) -> None:
    if len(current) != 69 or len(legacy) != 69:
        raise RuntimeError(f"parameterized_weighted_count_mismatch:current={len(current)} legacy={len(legacy)}")
    for index, (left, right) in enumerate(zip(current, legacy)):
        fields = ("op_type", "initializer_shape", "initializer_dtype", "initializer_sha256")
        mismatches = {field: (left[field], right[field]) for field in fields if left[field] != right[field]}
        if mismatches:
            raise RuntimeError(f"base_onnx_parameterized_layer_mismatch:{index}:{mismatches}")


def main(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    current_base = args.current_artifact / "pruned_fp32.onnx"
    current_origin_path = args.current_artifact / "origin_map.json"
    current_inspector_path = args.current_artifact / "engine_layer_info.json"
    for path in (
        current_base,
        current_origin_path,
        current_inspector_path,
        args.legacy_onnx,
        args.legacy_inspector,
        args.legacy_cache,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    current_model, current_parameterized = parameterized_nodes(current_base)
    legacy_model, legacy_parameterized = parameterized_nodes(args.legacy_onnx)
    assert_parameterized_graph_equivalence(current_parameterized, legacy_parameterized)
    current_functional = functional_nodes(current_model)
    legacy_functional = functional_nodes(legacy_model)
    if len(current_functional) != 3 or len(legacy_functional) != 3:
        raise RuntimeError(
            f"affine_grid_functional_matmul_count_mismatch:current={len(current_functional)} legacy={len(legacy_functional)}"
        )

    origin = build_onnx_origin_map(current_base, original_calls(read_json(current_origin_path)))
    if len(origin.entries) != 69 or len(origin.functional_compute_groups) != 1:
        raise RuntimeError(
            f"canonical_origin_count_mismatch:weighted={len(origin.entries)} functional={len(origin.functional_compute_groups)}"
        )
    functional_group = origin.functional_compute_groups[0]
    if len(functional_group.graph_indices) != 3:
        raise RuntimeError("functional_affine_grid_group_member_count_mismatch")
    canonical_by_graph = {int(row.graph_index): row for row in origin.entries}

    current_engine = [row for row in load_layer_info(current_inspector_path) if is_weighted_compute_layer(row)]
    legacy_engine = [row for row in load_layer_info(args.legacy_inspector) if is_weighted_compute_layer(row)]
    if len(current_engine) != 70 or len(legacy_engine) != 70:
        raise RuntimeError(
            f"engine_compute_count_mismatch:explicit={len(current_engine)} legacy={len(legacy_engine)}"
        )

    compute_groups: list[dict[str, Any]] = []
    for current, legacy in zip(current_parameterized, legacy_parameterized):
        entry = canonical_by_graph.get(int(current["graph_index"]))
        if entry is None:
            raise RuntimeError(f"canonical_parameterized_entry_missing:{current['graph_index']}")
        compute_groups.append(
            {
                "order_key": int(current["graph_index"]),
                "kind": "parameterized_weighted",
                "entry": entry,
                "current_nodes": [current],
                "legacy_nodes": [legacy],
            }
        )
    compute_groups.append(
        {
            "order_key": min(int(value) for value in functional_group.graph_indices),
            "kind": "parameter_free_functional_compute",
            "entry": functional_group,
            "current_nodes": current_functional,
            "legacy_nodes": legacy_functional,
        }
    )
    compute_groups.sort(key=lambda row: int(row["order_key"]))
    if len(compute_groups) != 70:
        raise RuntimeError(f"canonical_compute_count_mismatch:{len(compute_groups)}")

    table: list[dict[str, Any]] = []
    used_current_rows: set[int] = set()
    used_legacy_rows: set[int] = set()
    for ordinal, group in enumerate(compute_groups):
        entry = group["entry"]
        canonical = str(entry.canonical_node_name)
        kind = str(group["kind"])
        if kind == "parameterized_weighted":
            current_matches = [
                (index, row)
                for index, row in enumerate(current_engine)
                if has_canonical_identity(row, canonical)
            ]
            if len(current_matches) != 1:
                raise RuntimeError(
                    f"explicit_engine_canonical_match_count:{ordinal}:{canonical}:{len(current_matches)}"
                )
            current_index, current_row = current_matches[0]
        else:
            current_matches = [
                (index, row)
                for index, row in enumerate(current_engine)
                if all(str(node["node_name"]) in str(row.get("Name", "")) for node in current_functional)
            ]
            if len(current_matches) != 1:
                raise RuntimeError(f"explicit_functional_engine_group_match_count:{len(current_matches)}")
            current_index, current_row = current_matches[0]
        if current_index in used_current_rows:
            raise RuntimeError(f"explicit_engine_row_mapped_twice:{current_index}")
        used_current_rows.add(current_index)

        if kind == "parameterized_weighted":
            expected_legacy_name = str(group["legacy_nodes"][0]["node_name"])
            accepted_legacy_names = {
                expected_legacy_name,
                expected_legacy_name.replace("pfn_layers.0", "pfn_layers_0"),
            }
            legacy_matches = [
                (index, row)
                for index, row in enumerate(legacy_engine)
                if any(value in str(row.get("Name", "")) for value in accepted_legacy_names)
            ]
            if len(legacy_matches) != 1:
                raise RuntimeError(
                    f"legacy_engine_onnx_match_count:{ordinal}:{expected_legacy_name}:{len(legacy_matches)}"
                )
            legacy_index, legacy_row = legacy_matches[0]
        else:
            legacy_matches = [
                (index, row)
                for index, row in enumerate(legacy_engine)
                if all(str(node["node_name"]) in str(row.get("Name", "")) for node in legacy_functional)
            ]
            if len(legacy_matches) != 1:
                raise RuntimeError(f"legacy_functional_engine_group_match_count:{len(legacy_matches)}")
            legacy_index, legacy_row = legacy_matches[0]
        if legacy_index in used_legacy_rows:
            raise RuntimeError(f"legacy_engine_row_mapped_twice:{legacy_index}")
        used_legacy_rows.add(legacy_index)
        legacy_name = str(legacy_row.get("Name", ""))
        legacy_precision = precision_name(legacy_row)
        explicit_precision = precision_name(current_row)
        first_current = group["current_nodes"][0]
        first_legacy = group["legacy_nodes"][0]
        table.append(
            {
                "canonical_index": ordinal,
                "canonical_layer": str(entry.module_path),
                "canonical_node_name": canonical,
                "canonical_kind": kind,
                "mapping_status": str(getattr(entry, "mapping_status", "mapped")),
                "initializer": str(first_current.get("initializer", "")),
                "initializer_shape": first_current.get("initializer_shape", []),
                "initializer_dtype": str(first_current.get("initializer_dtype", "")),
                "initializer_sha256": str(first_current.get("initializer_sha256", "")),
                "legacy_initializer": str(first_legacy.get("initializer", "")),
                "legacy_initializer_sha256": str(first_legacy.get("initializer_sha256", "")),
                "legacy_precision": legacy_precision,
                "matched_profile_precision": legacy_precision,
                "e27_control_precision": explicit_precision,
                "e27_matches_legacy": explicit_precision == legacy_precision,
                "legacy_engine_layer_name": legacy_name,
                "e27_engine_layer_name": str(current_row.get("Name", "")),
                "legacy_onnx_nodes": [str(node["node_name"]) for node in group["legacy_nodes"]],
                "explicit_onnx_nodes": [str(node["node_name"]) for node in group["current_nodes"]],
            }
        )

    if len(used_current_rows) != 70 or len(used_legacy_rows) != 70:
        raise RuntimeError(
            f"engine_rows_not_fully_mapped:explicit={len(used_current_rows)} legacy={len(used_legacy_rows)}"
        )

    legacy_int8 = [row for row in table if row["legacy_precision"] == "int8"]
    legacy_fp16 = [row for row in table if row["legacy_precision"] == "fp16"]
    if len(legacy_int8) != 67 or len(legacy_fp16) != 3:
        raise RuntimeError(f"legacy_canonical_coverage_not_67_3:int8={len(legacy_int8)} fp16={len(legacy_fp16)}")
    if any(row["legacy_precision"] not in {"int8", "fp16"} for row in table):
        raise RuntimeError("legacy_canonical_coverage_contains_unknown_precision")
    e27_int8 = sum(row["e27_control_precision"] == "int8" for row in table)
    e27_fp16 = sum(row["e27_control_precision"] == "fp16" for row in table)
    if (e27_int8, e27_fp16) != (27, 43):
        raise RuntimeError(f"e27_canonical_coverage_not_27_43:{e27_int8}/{e27_fp16}")

    parameterized_int8_modules = [
        row["canonical_layer"]
        for row in table
        if row["canonical_kind"] == "parameterized_weighted" and row["legacy_precision"] == "int8"
    ]
    if len(parameterized_int8_modules) != 67:
        raise RuntimeError("legacy_int8_set_contains_unexpected_functional_compute")
    profile = {
        "profile_id": "canonical_legacy_single_engine_maxK_fixedK29696_int8_train200_67_3_v1",
        "canonical_weighted_count": 70,
        "canonical_compute_count": 70,
        "parameterized_weighted_count": 69,
        "parameter_free_functional_compute_count": 1,
        "unmapped_weighted_count": 0,
        "int8_count": 67,
        "fp16_count": 3,
        "int8_module_paths": parameterized_int8_modules,
        "fp16_entries": [row["canonical_layer"] for row in legacy_fp16],
        "functional_entry_policy": "mapped_but_protected_fp16_not_a_precision_gene",
        "layers": table,
        "provenance": {
            "legacy_onnx": str(args.legacy_onnx),
            "legacy_onnx_sha256": sha256_file(args.legacy_onnx),
            "legacy_inspector": str(args.legacy_inspector),
            "legacy_inspector_sha256": sha256_file(args.legacy_inspector),
            "legacy_calibration_cache": str(args.legacy_cache),
            "legacy_calibration_cache_sha256": sha256_file(args.legacy_cache),
            "explicit_base_onnx": str(current_base),
            "explicit_base_onnx_sha256": sha256_file(current_base),
            "explicit_inspector": str(current_inspector_path),
            "explicit_inspector_sha256": sha256_file(current_inspector_path),
            "origin_map_hash": origin.origin_map_hash,
        },
    }
    write_json(output / "canonical_70_layer_precision_profile.json", profile)

    fields = [
        "canonical_index",
        "canonical_layer",
        "canonical_node_name",
        "canonical_kind",
        "mapping_status",
        "legacy_precision",
        "matched_profile_precision",
        "e27_control_precision",
        "e27_matches_legacy",
        "initializer_shape",
        "initializer_dtype",
        "initializer_sha256",
        "legacy_initializer_sha256",
        "legacy_engine_layer_name",
        "e27_engine_layer_name",
    ]
    with (output / "legacy_vs_explicit_layer_set_diff.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in table:
            writer.writerow(row)

    current_tensor_info = tensor_inventory(current_model)
    legacy_tensor_info = tensor_inventory(legacy_model)
    functional_audit = {
        "status": "mapped_but_protected_fp16",
        "canonical_layer": functional_group.module_path,
        "canonical_node_name": functional_group.canonical_node_name,
        "canonical_kind": "parameter_free_functional_compute",
        "engine_compute_row_count": 1,
        "onnx_member_count": 3,
        "weight_initializer": None,
        "weight_shape": None,
        "weight_dtype": None,
        "weight_sha256": None,
        "weight_initializer_verdict": "not_applicable_both_inputs_are_runtime_tensors",
        "source_call": functional_group.source_call,
        "source_file": str(REPO / "quantization/export/heal_lidar_pyramid.py"),
        "source_semantics": "_weighted_fuse invokes _warp once per pyramid level; _warp uses torch.bmm to construct affine grids",
        "not_pillar_vfe_linear": True,
        "pillar_vfe_linear": {
            "module_path": "encoder_m1.pillar_vfe.pfn_layers.0.linear",
            "module_type": "nn.Linear",
            "weight_initializer_sha256": table[0]["initializer_sha256"],
            "legacy_precision": table[0]["legacy_precision"],
            "explicit_e27_precision": table[0]["e27_control_precision"],
            "mapping_status": "mapped_but_protected_fp16",
        },
        "legacy_engine": row_payload(legacy_engine[61]),
        "explicit_e27_engine": row_payload(current_engine[61]),
        "legacy_onnx_members": [
            {
                "graph_index": row["graph_index"],
                "name": row["node_name"],
                "inputs": [
                    {"name": name, **legacy_tensor_info.get(name, {})}
                    for name in row["inputs"]
                ],
                "outputs": [
                    {"name": name, **legacy_tensor_info.get(name, {})}
                    for name in row["outputs"]
                ],
                "input_initializer_flags": [False, False],
            }
            for row in legacy_functional
        ],
        "explicit_onnx_members": [
            {
                "graph_index": row["graph_index"],
                "name": row["node_name"],
                "inputs": [
                    {"name": name, **current_tensor_info.get(name, {})}
                    for name in row["inputs"]
                ],
                "outputs": [
                    {"name": name, **current_tensor_info.get(name, {})}
                    for name in row["outputs"]
                ],
                "input_initializer_flags": [False, False],
            }
            for row in current_functional
        ],
        "legacy_realized_precision": "fp16",
        "explicit_e27_realized_precision": "fp16",
        "matched_profile_precision": "fp16",
        "protection_reason": functional_group.protection_reason,
        "unmapped_weighted_count": 0,
    }
    write_json(output / "functional_matmul_mapping_audit.json", functional_audit)

    correction = {
        "status": "baseline_definition_corrected",
        "e27_definition": "accuracy_safe_mixed_precision_control",
        "e27_coverage": "27 INT8 / 43 FP16",
        "e27_quantization_equivalent_to_legacy": False,
        "legacy_definition": "L67_implicit_INT8",
        "legacy_coverage": "67 INT8 / 3 FP16",
        "coverage_equivalent": False,
        "scale_equivalent": "not_yet_verified_for_E67",
        "accuracy_equivalent": "not_yet_verified_for_E67",
        "latency_equivalent": "not_yet_verified_for_E67",
        "explicit_qdq_reproduces_legacy_int8": "not_yet_verified",
        "required_next": ["E67-LS_10_and_200", "E67-ENT_10_and_200"],
    }
    write_json(output / "baseline_definition_correction.json", correction)
    (output / "baseline_definition_correction.md").write_text(
        "# Baseline definition correction\n\n"
        "E27 is an accuracy-safe mixed-precision control with **27 INT8 / 43 FP16**. "
        "It is not coverage-equivalent or quantization-equivalent to Legacy L67.\n\n"
        "The canonical Legacy profile is **67 INT8 / 3 FP16**, with 70/70 engine "
        "compute rows mapped and zero unmapped rows. The three FP16 entries are the "
        "PillarVFE Linear, `pyramid_backbone.single_head_2`, and the parameter-free "
        "affine-grid functional MatMul fusion group.\n\n"
        "No explicit-Q/DQ reproduction claim is allowed until E67-LS and E67-ENT "
        "complete their 10-frame and 200-frame gates.\n",
        encoding="utf-8",
    )
    write_json(
        output / "audit_manifest.json",
        {
            "status": "complete",
            "canonical_weighted_count": 70,
            "engine_weighted_count_legacy": 70,
            "engine_weighted_count_explicit_e27": 70,
            "unmapped_weighted_count": 0,
            "legacy_coverage": {"int8": 67, "fp16": 3},
            "e27_coverage": {"int8": 27, "fp16": 43},
            "generated_files": sorted(path.name for path in output.iterdir()),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-artifact", type=Path, default=E27)
    parser.add_argument("--legacy-onnx", type=Path, default=LEGACY_ONNX)
    parser.add_argument("--legacy-inspector", type=Path, default=LEGACY_INSPECTOR)
    parser.add_argument("--legacy-cache", type=Path, default=LEGACY_CACHE)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
