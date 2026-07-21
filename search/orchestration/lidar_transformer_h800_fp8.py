"""ModelOpt FP8 E4M3 projection experiments on frozen H800 Transformer graphs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import onnx
import torch

from quantization.config import TensorRTBuildConfig
from quantization.precision.typed_graph import apply_strongly_typed_precision_contract
from quantization.types import CanonicalPrecisionMappingResult, stable_json_hash
from search.model_families.transformer.qdq_adjacency import (
    restore_projection_qdq_adjacency,
    validate_projection_qdq_adjacency,
)
from search.model_families.transformer.fp8_profiles import FP8_PROFILE_ROLES
from search.model_families.transformer.realized_precision import audit_realized_precision
from search.model_families.transformer.projection_rewrite import (
    split_cobevt_fused_qkv,
    split_v2xvit_fused_qkv,
)
from search.orchestration.lidar_transformer_h800_baselines import (
    TRTEXEC,
    TRT_ROOT,
    _write_csv,
    _write_json,
    build_mapping,
)
from search.orchestration.lidar_transformer_h800_inventory import (
    MODEL_SPECS,
    _load,
    _real_batch,
)
from search.orchestration.lidar_transformer_h800_smoothquant import (
    _calibrate_train200,
    _export_quantized,
    _qdq_inventory,
    _selected_module_paths,
    _sha256,
)
from search.stage2.trt_modelopt import build_engine_modelopt
from search.integration.runtime_environment import (
    configure_modelopt_inprocess,
    require_modelopt_cuda_extension,
    runtime_cuda_index_for_physical,
)


def _fp8_config(paths: tuple[str, ...]) -> dict[str, Any]:
    quant_cfg: dict[str, Any] = {"default": {"enable": False}}
    for path in paths:
        quant_cfg[f"*{path}*weight_quantizer"] = {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        }
        quant_cfg[f"*{path}*input_quantizer"] = {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        }
    return {"algorithm": "max", "quant_cfg": quant_cfg}


def _quantizer_state(model: torch.nn.Module, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows = []
    for path in paths:
        module = modules[path]
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        enabled = lambda value: bool(value() if callable(value) else value)
        row = {
            "module_path": path,
            "module_class": type(module).__name__,
            "fp8_format": "E4M3",
            "activation_granularity": "static_per_tensor",
            "weight_granularity": "per_tensor",
            "input_axis": getattr(input_quantizer, "axis", None),
            "weight_axis": getattr(weight_quantizer, "axis", None),
            "input_enabled": enabled(getattr(input_quantizer, "is_enabled", False)),
            "weight_enabled": enabled(getattr(weight_quantizer, "is_enabled", False)),
        }
        rows.append(row)
    if any(
        not row["input_enabled"]
        or not row["weight_enabled"]
        or row["input_axis"] is not None
        or row["weight_axis"] is not None
        for row in rows
    ):
        raise RuntimeError("fp8_quantizer_contract_incomplete")
    return rows


def build_profile(
    *, output_root: Path, model_name: str, profile_id: str,
    physical_gpu: int, plugin_path: Path
) -> dict[str, Any]:
    toolchain = configure_modelopt_inprocess(
        output_root=output_root,
        cache_namespace=f"{model_name}_{profile_id}_gpu{physical_gpu}",
    )
    cuda_extension = require_modelopt_cuda_extension("fp8")
    import modelopt.torch.quantization as mtq  # type: ignore

    if profile_id not in FP8_PROFILE_ROLES:
        raise ValueError(f"unknown_fp8_profile:{profile_id}")
    roles = FP8_PROFILE_ROLES[profile_id]
    destination = output_root / "fp8" / model_name / profile_id
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(
        destination / "modelopt_cuda_toolchain.json",
        {"toolchain": toolchain, "extension": cuda_extension},
    )
    inventory_dir = output_root / "inventory" / model_name
    inventory = json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))
    selected_paths = _selected_module_paths(inventory, roles)
    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle, _ = _load(model_name, device)
    rewrite = (split_cobevt_fused_qkv if model_name == "lidar_cobevt" else split_v2xvit_fused_qkv)(bundle.model)
    modules = dict(bundle.model.named_modules())
    invalid = [path for path in selected_paths if path not in modules or not isinstance(modules[path], torch.nn.Linear)]
    if invalid:
        raise RuntimeError(f"fp8_selected_non_linear:{invalid[:8]}")
    config = _fp8_config(selected_paths)
    calibration: dict[str, Any] = {}
    manifest = output_root / "evaluation" / "manifests" / model_name / "calibration200.json"

    def forward_loop(_model: torch.nn.Module) -> None:
        calibration.update(
            _calibrate_train200(
                bundle=bundle,
                config_path=str(MODEL_SPECS[model_name]["config"]),
                manifest_path=manifest,
                device=device,
            )
        )

    quantized = mtq.quantize(bundle.model, config, forward_loop=forward_loop)
    if quantized is not bundle.model:
        bundle.model = quantized
    quantizers = _quantizer_state(bundle.model, selected_paths)
    _write_json(destination / "calibration_report.json", calibration)
    _write_json(destination / "quantizer_inventory.json", quantizers)
    _write_json(destination / "modelopt_config.json", config)
    batch = _real_batch(bundle, str(MODEL_SPECS[model_name]["config"]), device)
    origin, qdq_source, export_report = _export_quantized(
        model_name=model_name, bundle=bundle, batch=batch, destination=destination
    )
    graph = onnx.load(str(qdq_source), load_external_data=False)
    mapping, requested = build_mapping(
        model_name=model_name,
        profile="B3_F3",
        origin=origin,
        inventory=inventory,
        graph_nodes={str(node.name): node for node in graph.graph.node},
    )
    mapped_paths = {
        str(entry.module_path)
        for entry in mapping.entries
        if str(entry.module_path) in selected_paths
    }
    if mapped_paths != set(selected_paths):
        raise RuntimeError("fp8_selected_mapping_incomplete")
    typed = destination / "typed_with_qdq_casts.onnx"
    typed_report = apply_strongly_typed_precision_contract(
        qdq_source, typed, mapping, plugin_boundary="fp16"
    )
    selected_nodes = {
        str(entry.canonical_node_name)
        for entry in mapping.entries
        if str(entry.module_path) in selected_paths
    }
    # The strongly typed graph already owns output/merge boundary Casts.
    # Q/DQ adjacency repair is intentionally input-only so that an INT8/FP8
    # projection cannot acquire a redundant FP16 -> FP32 Cast chain before a
    # bias, activation, residual, or attention boundary.
    fp16_outputs: dict[str, str] = {}
    final_qdq = destination / "strongly_typed_explicit_fp8_qdq.onnx"
    final_graph = onnx.load(str(typed), load_external_data=False)
    rewrite_rows = restore_projection_qdq_adjacency(
        final_graph, sorted(selected_nodes), output_cast_precisions=fp16_outputs
    )
    adjacency = validate_projection_qdq_adjacency(final_graph, sorted(selected_nodes))
    onnx.checker.check_model(final_graph)
    onnx.save(final_graph, str(final_qdq))
    deployment_entries = [
        replace(entry, requested_precision="fp8", realized_request_precision="fp8")
        if str(entry.module_path) in selected_paths
        else entry
        for entry in mapping.entries
    ]
    deployment_mapping = CanonicalPrecisionMappingResult(
        entries=deployment_entries,
        profile_id=profile_id,
        profile_hash=stable_json_hash(
            {"model": model_name, "profile": profile_id, "selected": selected_paths, "format": "E4M3"}
        ),
        origin_map_hash=mapping.origin_map_hash,
        policy_version="h800-transformer-modelopt-fp8-e4m3-explicit-qdq-v1",
        auxiliary_layer_precisions=dict(mapping.auxiliary_layer_precisions),
        auxiliary_layer_output_types=dict(mapping.auxiliary_layer_output_types),
    )
    for row in requested:
        if str(row.get("module_path", "")) in selected_paths:
            row["requested_precision"] = "FP8"
    qdq_rows = _qdq_inventory(final_qdq, requested)
    _write_csv(destination / "qdq_inventory.csv", qdq_rows)
    if not qdq_rows or any(not bool(row["scale_finite_nonzero"]) for row in qdq_rows):
        raise RuntimeError("fp8_qdq_scale_nonfinite_or_zero")
    _write_json(destination / "deployment_precision_mapping.json", deployment_mapping.to_dict())
    _write_json(destination / "requested_precision_contract.json", requested)
    _write_json(destination / "typed_graph_report.json", typed_report)
    _write_json(destination / "qdq_adjacency_rewrite.json", rewrite_rows)
    _write_json(destination / "qdq_adjacency_audit.json", adjacency)
    snapshot = json.loads((inventory_dir / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8"))
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
        policy_version="h800-transformer-trt10.9-strongly-typed-explicit-fp8-v1",
    )
    engine = destination / "engine.plan"
    build = build_engine_modelopt(
        qdq_onnx=final_qdq,
        engine_path=engine,
        precision_mapping=deployment_mapping,
        build_config=build_config,
        physical_snapshot=snapshot,
        output_dir=destination / "engine_build",
        tensorrt_root=TRT_ROOT,
        conda_env="modelopt",
        gpu_id=physical_gpu,
    )
    _write_json(destination / "build_acceptance.json", build)
    layer_info = destination / "engine_build" / "engine_layer_info.json"
    realized = []
    if layer_info.is_file():
        realized = [
            row.to_dict()
            for row in audit_realized_precision(
                model=model_name,
                profile=profile_id,
                requested_rows=requested,
                layer_info_path=layer_info,
                typed_onnx_path=final_qdq,
                strongly_typed=True,
            )
        ]
        _write_csv(destination / "requested_realized.csv", realized)
    conflicts = [row for row in realized if row["conflict"]]
    selected_realized = [
        row for row in realized
        if row["onnx_node"] in selected_nodes and row["realized_precision"] == "FP8"
    ]
    status = str(build.get("status", "engine_build_failed"))
    if status == "ok" and (conflicts or len(selected_realized) != len(selected_nodes)):
        status = "precision_fallback"
    result = {
        "status": status,
        "model": model_name,
        "profile": profile_id,
        "fp8_format": "E4M3",
        "selected_roles": list(roles),
        "selected_module_count": len(selected_paths),
        "selected_node_count": len(selected_nodes),
        "realized_fp8_selected_count": len(selected_realized),
        "requested_realized_conflict_count": len(conflicts),
        "engine_exists": engine.is_file() and engine.stat().st_size > 0,
        "engine_sha256": _sha256(engine) if engine.is_file() else "",
        "final_qdq_sha256": _sha256(final_qdq),
        "calibration_manifest_hash": calibration.get("manifest_hash", ""),
        "projection_rewrite": rewrite.to_dict(),
        "export_report": export_report,
        "physical_structure_frozen": True,
        "modelopt_cuda_extension": cuda_extension,
    }
    _write_json(destination / "baseline_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--profile", choices=tuple(FP8_PROFILE_ROLES), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args(argv)
    result = build_profile(
        output_root=Path(args.output_root).resolve(),
        model_name=args.model,
        profile_id=args.profile,
        physical_gpu=args.physical_gpu,
        plugin_path=Path(args.plugin).resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
