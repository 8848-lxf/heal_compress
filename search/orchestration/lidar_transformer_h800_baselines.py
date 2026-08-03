"""Build fresh strongly-typed H800 Transformer FP32/FP16/F3 baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import onnx

from quantization.config import TensorRTBuildConfig
from quantization.precision.typed_graph import apply_strongly_typed_precision_contract
from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult, stable_json_hash
from search.model_families.transformer.canonical_roles import classify_weighted_module
from search.model_families.transformer.precision_contract import precision_profiles
from search.model_families.transformer.realized_precision import audit_realized_precision
from search.stage2.trt_modelopt import build_engine_modelopt


TRT_ROOT = Path("${TENSORRT_ROOT}")
TRTEXEC = TRT_ROOT / "targets/x86_64-linux-gnu/bin/trtexec"
PROFILES = ("B1_TRT_ATTN_FP32", "B2_TRT_STRICT_FP16", "B3_F3")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role_precision(profile: str, role: str) -> tuple[str, str, str]:
    contract = precision_profiles()[profile]
    if role in contract:
        row = contract[role]
        return row.operand_precision.lower(), row.output_precision.lower(), row.accumulator_precision
    # This audit deliberately keeps the established non-Transformer path in
    # FP16.  Attention sensitivity profiles only alter Transformer roles.
    return "fp16", "fp16", "unknown"


def _floating_contract_row(row: Mapping[str, Any]) -> bool:
    floating = {
        "FLOAT",
        "FLOAT16",
        "DOUBLE",
        "BFLOAT16",
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
    }
    input_dtype = str(row.get("input_dtype", "unknown")).upper()
    output_dtype = str(row.get("output_dtype", "unknown")).upper()
    return output_dtype in floating and (
        input_dtype in floating or str(row.get("onnx_op_type", "")) == "Where"
    )


def build_mapping(
    *,
    model_name: str,
    profile: str,
    origin: Mapping[str, Any],
    inventory: Mapping[str, Any],
    graph_nodes: Mapping[str, Any],
) -> tuple[CanonicalPrecisionMappingResult, list[dict[str, Any]]]:
    def tensors(node_name: str) -> list[str]:
        node = graph_nodes.get(str(node_name))
        return [*map(str, node.input), *map(str, node.output)] if node is not None else []

    inventory_by_node = {
        str(row["onnx_node"]): row
        for row in inventory["rows"]
        if str(row.get("onnx_node", ""))
    }
    entries: list[CanonicalPrecisionEntry] = []
    requested: list[dict[str, Any]] = []
    for source in origin["entries"]:
        path = str(source["module_path"])
        node_name = str(source["canonical_node_name"])
        role_row = inventory_by_node.get(node_name, {})
        role = str(role_row.get("canonical_role") or classify_weighted_module(model_name, path).canonical_role)
        compute, output, accumulator = _role_precision(profile, role)
        entry = CanonicalPrecisionEntry(
            module_path=path,
            canonical_node_name=node_name,
            precision_group=f"h800_transformer::{model_name}::{role}::{path}",
            requested_precision=compute,
            realized_request_precision=compute,
            realized_output_precision=output,
            original_node_name=str(source.get("original_node_name", "")),
            weight_initializer=str(source.get("weight_initializer", "")),
            onnx_op_type=str(source.get("onnx_op_type", "")),
            call_index=int(source.get("call_index", 0)),
        )
        entries.append(entry)
        requested.append(
            {
                "model": model_name,
                "profile": profile,
                "block": str(role_row.get("residual_scope", "")),
                "role": role,
                "module_path": path,
                "onnx_node": node_name,
                "requested_precision": compute.upper(),
                "requested_output_precision": output.upper(),
                "requested_accumulator": accumulator,
                "tensor_names": tensors(node_name),
            }
        )
    auxiliary: dict[str, str] = {}
    auxiliary_outputs: dict[str, str] = {}
    primitive_roles = {
        "qk_matmul",
        "qk_scale",
        "mask_relation_add",
        "softmax",
        "av_matmul",
        "layernorm",
        "residual_add",
        "split_attention_gate",
        "communication_fusion",
    }
    for row in inventory["rows"]:
        node_name = str(row.get("onnx_node", ""))
        role = str(row.get("canonical_role", ""))
        if not node_name or str(row.get("module_path", "")) or role not in primitive_roles:
            continue
        if not _floating_contract_row(row):
            # Dynamic-shape Add/Mul/Concat nodes can share an attention path
            # prefix.  They must retain their integer shape dtype and must never
            # inherit the attention compute precision.
            continue
        contract_role = role
        if role in {"qk_scale", "mask_relation_add"}:
            contract_role = "qk_matmul"
        if role in {"split_attention_gate", "communication_fusion"}:
            compute, output, accumulator = "fp16", "fp16", "unknown"
        else:
            compute, output, accumulator = _role_precision(profile, contract_role)
        auxiliary[node_name] = compute
        auxiliary_outputs[node_name] = output
        requested.append(
            {
                "model": model_name,
                "profile": profile,
                "block": str(row.get("residual_scope", "")),
                "role": role,
                "module_path": "",
                "onnx_node": node_name,
                "requested_precision": compute.upper(),
                "requested_output_precision": output.upper(),
                "requested_accumulator": accumulator,
                "tensor_names": tensors(node_name),
            }
        )
    for group in origin.get("functional_compute_groups", ()):
        node_name = str(group.get("canonical_node_name", ""))
        if not node_name:
            continue
        member_names = sorted(
            name
            for name in graph_nodes
            if name == node_name or name.startswith(f"{node_name}__member")
        )
        if not member_names:
            raise ValueError(f"functional_compute_group_members_unresolved:{node_name}")
        for member_name in member_names:
            auxiliary[member_name] = "fp16"
            auxiliary_outputs[member_name] = "fp16"
            requested.append(
                {
                    "model": model_name,
                    "profile": profile,
                    "block": "geometric_warp",
                    "role": "communication_fusion",
                    "module_path": str(group.get("module_path", "")),
                    "onnx_node": member_name,
                    "requested_precision": "FP16",
                    "requested_output_precision": "FP16",
                    "requested_accumulator": "unknown",
                    "tensor_names": tensors(member_name),
                }
            )
    mapping = CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id=profile,
        profile_hash=stable_json_hash({"profile": profile, "requested": requested}),
        origin_map_hash=str(origin["origin_map_hash"]),
        policy_version="h800-transformer-role-strongly-typed-v1",
        auxiliary_layer_precisions=auxiliary,
        auxiliary_layer_output_types=auxiliary_outputs,
    )
    return mapping, requested


def build_profile(
    *,
    output_root: Path,
    model_name: str,
    profile: str,
    physical_gpu: int,
    plugin_path: Path,
    destination_section: str = "baselines",
) -> dict[str, Any]:
    inventory_dir = output_root / "inventory" / model_name
    base = inventory_dir / "base_fp32_canonical.onnx"
    origin = json.loads((inventory_dir / "canonical_origin_map.json").read_text(encoding="utf-8"))
    inventory = json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))
    snapshot = json.loads((inventory_dir / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8"))
    base_graph = onnx.load(str(base), load_external_data=False)
    graph_nodes = {str(node.name): node for node in base_graph.graph.node}
    mapping, requested = build_mapping(
        model_name=model_name,
        profile=profile,
        origin=origin,
        inventory=inventory,
        graph_nodes=graph_nodes,
    )
    destination = output_root / destination_section / model_name / profile
    destination.mkdir(parents=True, exist_ok=True)
    mapping_path = destination / "canonical_precision_mapping.json"
    _write_json(mapping_path, mapping.to_dict())
    _write_json(destination / "requested_precision_contract.json", requested)
    typed_path = destination / "strongly_typed.onnx"
    typed = apply_strongly_typed_precision_contract(
        base, typed_path, mapping, plugin_boundary="fp16"
    )
    graph = onnx.load(str(typed_path), load_external_data=False)
    onnx.checker.check_model(graph)
    typed.update(
        {
            "checker_passed": True,
            "base_onnx_sha256": _sha256(base),
            "typed_onnx_sha256": _sha256(typed_path),
            "node_count": len(graph.graph.node),
        }
    )
    _write_json(destination / "typed_graph_report.json", typed)
    build_config = TensorRTBuildConfig(
        trtexec_path=TRTEXEC,
        plugin_path=plugin_path,
        workspace_mib=8192,
        timeout_seconds=7200,
        precision_constraints="none",
        enable_fp16=False,
        enable_int8=False,
        no_tf32=True,
        skip_inference=True,
        export_layer_info=True,
        strongly_typed=True,
        production_mode=True,
        plugin_boundary_dtype="fp16",
        policy_version="h800-transformer-trt10.9-strongly-typed-no-tf32-v1",
    )
    engine = destination / "engine.plan"
    build = build_engine_modelopt(
        qdq_onnx=typed_path,
        engine_path=engine,
        precision_mapping=mapping,
        build_config=build_config,
        physical_snapshot=snapshot,
        output_dir=destination / "engine_build",
        tensorrt_root=TRT_ROOT,
        conda_env="modelopt",
        gpu_id=physical_gpu,
    )
    _write_json(destination / "build_acceptance.json", build)
    layer_info = destination / "engine_build" / "engine_layer_info.json"
    realized_rows: list[dict[str, Any]] = []
    if layer_info.is_file():
        realized_rows = [
            row.to_dict()
            for row in audit_realized_precision(
                model=model_name,
                profile=profile,
                requested_rows=requested,
                layer_info_path=layer_info,
                typed_onnx_path=typed_path,
                strongly_typed=True,
            )
        ]
        _write_csv(destination / "requested_realized.csv", realized_rows)
    conflict_count = sum(bool(row["conflict"]) for row in realized_rows)
    build_status = str(build.get("status", "engine_build_failed"))
    acceptance_status = (
        "precision_conflict"
        if build_status == "ok" and conflict_count
        else build_status
    )
    result = {
        "model": model_name,
        "profile": profile,
        "status": acceptance_status,
        "engine_exists": engine.is_file() and engine.stat().st_size > 0,
        "engine_sha256": _sha256(engine) if engine.is_file() else "",
        "typed_onnx_sha256": typed["typed_onnx_sha256"],
        "strongly_typed": True,
        "no_tf32": True,
        "plugin_sha256": _sha256(plugin_path),
        "requested_realized_conflict_count": conflict_count,
        "requested_realized_record_count": len(realized_rows),
    }
    _write_json(destination / "baseline_result.json", result)
    return result


def reaudit_existing_profile(
    *, output_root: Path, destination_section: str, model_name: str, profile: str
) -> dict[str, Any]:
    """Refresh only EngineInspector parsing; never rebuild an accepted engine."""

    destination = output_root / destination_section / model_name / profile
    required = (
        destination / "baseline_result.json",
        destination / "requested_precision_contract.json",
        destination / "strongly_typed.onnx",
        destination / "engine_build" / "engine_layer_info.json",
        destination / "build_acceptance.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"existing_profile_reaudit_missing:{missing}")
    requested = json.loads(required[1].read_text(encoding="utf-8"))
    realized_rows = [
        row.to_dict()
        for row in audit_realized_precision(
            model=model_name,
            profile=profile,
            requested_rows=requested,
            layer_info_path=required[3],
            typed_onnx_path=required[2],
            strongly_typed=True,
        )
    ]
    _write_csv(destination / "requested_realized.csv", realized_rows)
    conflict_count = sum(bool(row["conflict"]) for row in realized_rows)
    result = json.loads(required[0].read_text(encoding="utf-8"))
    build = json.loads(required[4].read_text(encoding="utf-8"))
    build_status = str(build.get("status", "engine_build_failed"))
    result.update(
        {
            "status": "precision_conflict" if build_status == "ok" and conflict_count else build_status,
            "requested_realized_conflict_count": conflict_count,
            "requested_realized_record_count": len(realized_rows),
            "realized_parser_reaudited": True,
        }
    )
    _write_json(required[0], result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args(argv)
    profiles = tuple(value.strip() for value in args.profiles.split(",") if value.strip())
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown_transformer_baseline_profile:{unknown}")
    results = [
        build_profile(
            output_root=Path(args.output_root).resolve(),
            model_name=args.model,
            profile=profile,
            physical_gpu=args.physical_gpu,
            plugin_path=Path(args.plugin).resolve(),
        )
        for profile in profiles
    ]
    _write_json(Path(args.output_root).resolve() / "baselines" / args.model / "baseline_matrix.json", results)
    print(json.dumps(results, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
