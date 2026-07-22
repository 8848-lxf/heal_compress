"""Fresh physical-model, ONNX and strongly-typed TensorRT d_h builds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import onnx
import torch
from torch import nn

from quantization.config import TensorRTBuildConfig
from quantization.precision.typed_graph import apply_strongly_typed_precision_contract
from quantization.types import CanonicalPrecisionMappingResult, stable_json_hash
from search.integration.runtime_environment import (
    configure_modelopt_inprocess,
    require_modelopt_cuda_extension,
    runtime_cuda_index_for_physical,
)
from search.model_families.transformer.dh_alignment_audit import (
    audit_engine_alignment,
    audit_onnx_head_dimension,
    parse_engine_memory_audit,
)
from search.model_families.transformer.dh_physical_rewrite import (
    discover_attention_families,
    materialize_family_head_dimension,
)
from search.model_families.transformer.dh_precision_profiles import require_profile
from search.model_families.transformer.dh_pruning_contract import masks_from_rankings, stable_hash
from search.model_families.transformer.model_inventory import build_model_inventory
from search.model_families.transformer.qdq_adjacency import (
    restore_projection_qdq_adjacency,
    validate_projection_qdq_adjacency,
)
from search.model_families.transformer.realized_precision import audit_realized_precision
from search.model_families.transformer.smoothquant_profiles import selective_smoothquant_config
from search.orchestration.lidar_transformer_h800_baselines import (
    TRTEXEC,
    TRT_ROOT,
    build_mapping,
)
from search.orchestration.lidar_transformer_h800_inventory import (
    MODEL_SPECS,
    _export_cobevt,
    _export_v2xvit,
    _load,
    _real_batch,
)
from search.orchestration.lidar_transformer_h800_smoothquant import (
    _calibrate_train200,
    _qdq_inventory,
    _quantizer_state,
    _selected_module_paths,
)
from search.stage2.trt_modelopt import build_engine_modelopt


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family_dir(root: Path, model: str, family_id: str, d_h: int) -> Path:
    return root / "structures" / model / family_id / f"dh_{int(d_h):03d}"


def _engine_dir(root: Path, model: str, family_id: str, d_h: int, profile: str) -> Path:
    return root / "engines" / model / family_id / f"dh_{int(d_h):03d}" / profile


def _load_and_materialize(
    *, output_root: Path, model_name: str, family_id: str, d_h: int, device: torch.device,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    bundle, _ = _load(model_name, device)
    families = {row.family_id: row for row in discover_attention_families(model_name, bundle.model)}
    if family_id not in families:
        raise RuntimeError(f"attention_family_missing:{model_name}:{family_id}")
    family = families[family_id]
    ranking_path = output_root / "importance" / f"{model_name.removeprefix('lidar_')}_ranking_manifest.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    masks = masks_from_rankings(family, ranking["rankings"], int(d_h))
    report = materialize_family_head_dimension(model_name, bundle.model, family, masks)
    if not report.passed:
        raise RuntimeError(f"physical_head_dimension_rewrite_failed:{report.issues}")
    return bundle, family, report, {path: value.to_dict() for path, value in masks.items()}


def _finite_outputs(output: Any) -> tuple[bool, dict[str, Any]]:
    if isinstance(output, Mapping):
        tensors = {str(key): value for key, value in output.items() if torch.is_tensor(value)}
    elif isinstance(output, (tuple, list)):
        tensors = {str(index): value for index, value in enumerate(output) if torch.is_tensor(value)}
    else:
        tensors = {"output": output} if torch.is_tensor(output) else {}
    rows = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all()),
            "absmax": float(value.detach().float().abs().max().item()),
        }
        for name, value in tensors.items()
    }
    return bool(rows) and all(row["finite"] for row in rows.values()), rows


def _runtime_costs(model: nn.Module, forward: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    handles = []

    def capture(name: str, module: nn.Module):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output if torch.is_tensor(output) else next((value for value in output if torch.is_tensor(value)), None) if isinstance(output, (tuple, list)) else None
            if tensor is None:
                return
            if isinstance(module, nn.Linear):
                instances = int(tensor.numel()) // max(int(module.out_features), 1)
                macs = instances * int(module.in_features) * int(module.out_features)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                instances = int(tensor.numel()) // max(int(module.out_channels), 1)
                kernel = int(np.prod(module.kernel_size))
                macs = instances * kernel * int(module.in_channels) * int(module.out_channels) // max(int(module.groups), 1)
            else:
                return
            rows.append({"module_path": name, "module_type": type(module).__name__, "macs": int(macs), "output_shape": list(tensor.shape)})
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(capture(name, module)))
        elif module.__class__.__name__ in {
            "PrunableCobevtAttention", "PrunedV2XWindowAttention", "HGTCavAttention"
        }:
            def attention_capture(path: str, attention: nn.Module):
                def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
                    tensor = inputs[0]
                    heads = int(attention.heads)
                    d_h = int(getattr(attention, "d_qk", round(float(attention.scale) ** -2)))
                    if attention.__class__.__name__ == "PrunableCobevtAttention":
                        batch, agents, grid_h, grid_w, win_h, win_w, _ = tensor.shape
                        sequence = int(agents * win_h * win_w)
                        instances = int(batch * grid_h * grid_w)
                        qk = instances * heads * sequence * sequence * d_h
                        av = qk
                    elif attention.__class__.__name__ == "PrunedV2XWindowAttention":
                        batch, agents, height, width, _ = tensor.shape
                        window = int(attention.window_size)
                        sequence = window * window
                        instances = int(batch * agents * (height // window) * (width // window))
                        qk = instances * heads * sequence * sequence * d_h
                        av = qk
                    else:
                        batch, agents, height, width, _ = tensor.shape
                        pairs = int(batch * heads * height * width * agents * agents)
                        # Relation-aware QK and message AV each contract one
                        # d_h x d_h relation matrix plus one d_h vector.
                        qk = pairs * (d_h * d_h + d_h)
                        av = qk
                    attention_rows.append({
                        "module_path": path,
                        "module_type": type(attention).__name__,
                        "heads": heads,
                        "d_h": d_h,
                        "qk_macs": int(qk),
                        "av_macs": int(av),
                        "input_shape": list(tensor.shape),
                    })
                return hook
            handles.append(module.register_forward_hook(attention_capture(name, module)))
    with torch.inference_mode():
        forward()
    for handle in handles:
        handle.remove()
    weighted_macs = sum(int(row["macs"]) for row in rows)
    qk_macs = sum(int(row["qk_macs"]) for row in attention_rows)
    av_macs = sum(int(row["av_macs"]) for row in attention_rows)
    return {
        "weighted_macs": weighted_macs,
        "qk_macs": qk_macs,
        "av_macs": av_macs,
        "total_macs_including_qk_av": weighted_macs + qk_macs + av_macs,
        "weighted_bops_fp32": weighted_macs * 32 * 32,
        "weighted_bops_fp16": weighted_macs * 16 * 16,
        "layers": rows,
        "attention_primitives": attention_rows,
        "note": "weighted layers include runtime invocation multiplicity; primitive QK/AV MAC-equivalents are reported separately and are included by the final precision-aware BOPS report",
    }


def prepare_structure(
    *, output_root: Path, model_name: str, family_id: str, d_h: int, physical_gpu: int,
) -> dict[str, Any]:
    destination = _family_dir(output_root, model_name, family_id, d_h)
    result_path = destination / "structure_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle, family, report, masks = _load_and_materialize(
        output_root=output_root, model_name=model_name, family_id=family_id, d_h=d_h, device=device,
    )
    batch = _real_batch(bundle, str(MODEL_SPECS[model_name]["config"]), device)
    with torch.inference_mode():
        output = bundle.adapter.forward_for_task(bundle.model, batch)
    finite, output_rows = _finite_outputs(output)
    if not finite:
        raise RuntimeError("physical_head_dimension_forward_nonfinite")
    costs = _runtime_costs(bundle.model, lambda: bundle.adapter.forward_for_task(bundle.model, batch))
    destination.mkdir(parents=True, exist_ok=True)
    origin, export = (
        _export_cobevt(bundle, batch, destination, int(MODEL_SPECS[model_name]["fixed_k"]))
        if model_name == "lidar_cobevt"
        else _export_v2xvit(bundle, batch, destination, int(MODEL_SPECS[model_name]["fixed_k"]))
    )
    export.pop("wrapper_output", None)
    origin_payload = origin if isinstance(origin, Mapping) else origin.to_dict()
    inventory = build_model_inventory(
        model_family=model_name,
        model=bundle.model,
        onnx_path=destination / "base_fp32_canonical.onnx",
        origin_map=origin,
    )
    from search.model_family.deployment import build_physical_structure_snapshot_v2

    snapshot = build_physical_structure_snapshot_v2(bundle.model, model_family=model_name)
    selected_paths = [path for path in inventory["rows"] if any(str(path.get("module_path", "")).startswith(parent) for parent in family.module_paths)]
    onnx_audit = audit_onnx_head_dimension(
        destination / "base_fp32_canonical.onnx",
        selected_module_paths=family.module_paths,
        origin_map=origin_payload,
        heads=family.heads,
        target_d_h=int(d_h),
    )
    # Some fused-source parent paths become four explicit child modules.  The
    # ONNX audit validates those child initializer shapes through prefix matching.
    if not onnx_audit["passed"]:
        raise RuntimeError(f"physical_onnx_head_dimension_audit_failed:{onnx_audit['issues']}")
    _write_json(destination / "canonical_origin_map.json", origin_payload)
    _write_json(destination / "inventory.json", inventory)
    _write_json(destination / "physical_structure_snapshot_v2.json", snapshot)
    _write_json(destination / "physical_rewrite.json", report.to_dict())
    _write_json(destination / "head_local_masks.json", masks)
    _write_json(destination / "physical_forward.json", {"finite": finite, "outputs": output_rows})
    _write_json(destination / "runtime_costs.json", costs)
    _write_json(destination / "onnx_head_dimension_audit.json", onnx_audit)
    result = {
        "status": "ok",
        "model": model_name,
        "family_id": family_id,
        "d_h": int(d_h),
        "heads": family.heads,
        "projection_width": family.heads * int(d_h),
        "original_parameter_count": report.original_parameter_count,
        "physical_parameter_count": report.physical_parameter_count,
        "parameter_reduction": 1.0 - report.physical_parameter_count / report.original_parameter_count,
        "structure_hash": report.structure_hash,
        "state_dict_shape_hash": report.state_dict_shape_hash,
        "onnx_sha256": _sha256(destination / "base_fp32_canonical.onnx"),
        "onnx_shape_hash": stable_hash(onnx_audit),
        "physical_forward_finite": finite,
        "onnx_checker_passed": True,
        "onnx_shape_inference_passed": True,
        "weighted_macs": costs["weighted_macs"],
        "qk_macs": costs["qk_macs"],
        "av_macs": costs["av_macs"],
        "total_macs_including_qk_av": costs["total_macs_including_qk_av"],
        "weighted_bops_fp32": costs["weighted_bops_fp32"],
    }
    _write_json(result_path, result)
    del bundle, batch, output
    torch.cuda.empty_cache()
    return result


def _build_config(plugin_path: Path) -> TensorRTBuildConfig:
    return TensorRTBuildConfig(
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
        policy_version="h800-transformer-dh-trt10.9-strongly-typed-v1",
    )


def _finalize_build(
    *, destination: Path, model_name: str, family_id: str, d_h: int, profile_id: str,
    typed_path: Path, mapping: CanonicalPrecisionMappingResult, requested: list[dict[str, Any]],
    snapshot: Mapping[str, Any], physical_gpu: int, plugin_path: Path,
) -> dict[str, Any]:
    engine = destination / "engine.plan"
    build = build_engine_modelopt(
        qdq_onnx=typed_path,
        engine_path=engine,
        precision_mapping=mapping,
        build_config=_build_config(plugin_path),
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
                typed_onnx_path=typed_path,
                strongly_typed=True,
            )
        ]
        _write_csv(destination / "requested_realized.csv", realized)
    conflicts = [row for row in realized if row["conflict"]]
    alignment = (
        audit_engine_alignment(layer_info, logical_d_h=int(d_h), projection_width=int(next(row["projection_width"] for row in [json.loads((_family_dir(destination.parents[4], model_name, family_id, d_h) / "structure_result.json").read_text(encoding="utf-8"))])))
        if layer_info.is_file() else {"padding_status": "UNKNOWN", "evidence_level": "C"}
    )
    _write_json(destination / "engine_alignment_audit.json", alignment)
    status = str(build.get("status", "engine_build_failed"))
    if status == "ok" and conflicts:
        status = "precision_conflict"
    if status == "ok" and alignment.get("fallback_hint"):
        status = "shape_or_tactic_fallback"
    if status == "ok" and alignment.get("padding_status") == "UNKNOWN":
        status = "alignment_evidence_unknown"
    result = {
        "status": status,
        "model": model_name,
        "family_id": family_id,
        "d_h": int(d_h),
        "profile": profile_id,
        "engine_exists": engine.is_file() and engine.stat().st_size > 0,
        "engine_sha256": _sha256(engine) if engine.is_file() else "",
        "typed_onnx_sha256": _sha256(typed_path),
        "strongly_typed": True,
        "no_tf32": True,
        "qk_contract": "F32A32O32",
        "plugin_sha256": _sha256(plugin_path),
        "builder_flags": ["--stronglyTyped", "--noTF32", "--skipInference", "--memPoolSize=workspace:8192"],
        "workspace_mib": 8192,
        "timing_cache_reused": False,
        "timing_cache_sha256": "not_applicable_fresh_build",
        "requested_realized_conflict_count": len(conflicts),
        "requested_realized_record_count": len(realized),
        "padding_status": alignment["padding_status"],
        "tensor_core_hint": alignment.get("tensor_core_hint"),
        "cast_count": alignment.get("cast_count"),
        "reformat_count": alignment.get("reformat_count"),
        "engine_memory": parse_engine_memory_audit(destination / "engine_build" / "engine_build.log"),
    }
    _write_json(destination / "baseline_result.json", result)
    return result


def build_float_profile(
    *, output_root: Path, model_name: str, family_id: str, d_h: int,
    profile_id: str, physical_gpu: int, plugin_path: Path,
) -> dict[str, Any]:
    profile = require_profile(profile_id)
    if profile.int8_roles:
        raise ValueError("build_float_profile_received_int8")
    structure = _family_dir(output_root, model_name, family_id, d_h)
    destination = _engine_dir(output_root, model_name, family_id, d_h, profile_id)
    base = structure / "base_fp32_canonical.onnx"
    origin = json.loads((structure / "canonical_origin_map.json").read_text(encoding="utf-8"))
    inventory = json.loads((structure / "inventory.json").read_text(encoding="utf-8"))
    snapshot = json.loads((structure / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8"))
    graph = onnx.load(str(base), load_external_data=False)
    graph_nodes = {str(node.name): node for node in graph.graph.node}
    mapping, requested = build_mapping(
        model_name=model_name, profile=profile.base_profile, origin=origin,
        inventory=inventory, graph_nodes=graph_nodes,
    )
    destination.mkdir(parents=True, exist_ok=True)
    typed = destination / "strongly_typed.onnx"
    report = apply_strongly_typed_precision_contract(base, typed, mapping, plugin_boundary="fp16")
    checked = onnx.load(str(typed), load_external_data=False)
    onnx.checker.check_model(checked)
    _write_json(destination / "canonical_precision_mapping.json", mapping.to_dict())
    _write_json(destination / "requested_precision_contract.json", requested)
    _write_json(destination / "typed_graph_report.json", {**report, "checker_passed": True})
    return _finalize_build(
        destination=destination, model_name=model_name, family_id=family_id, d_h=d_h,
        profile_id=profile_id, typed_path=typed, mapping=mapping, requested=requested,
        snapshot=snapshot, physical_gpu=physical_gpu, plugin_path=plugin_path,
    )


def build_int8_profile(
    *, output_root: Path, model_name: str, family_id: str, d_h: int,
    physical_gpu: int, plugin_path: Path,
) -> dict[str, Any]:
    profile = require_profile("P8")
    destination = _engine_dir(output_root, model_name, family_id, d_h, "P8")
    configure_modelopt_inprocess(
        output_root=output_root,
        cache_namespace=f"dh_{model_name}_{family_id}_{d_h}_P8_gpu{physical_gpu}",
    )
    require_modelopt_cuda_extension("int8")
    import modelopt.torch.quantization as mtq  # type: ignore

    runtime = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime}")
    torch.cuda.set_device(device)
    bundle, family, physical, masks = _load_and_materialize(
        output_root=output_root, model_name=model_name, family_id=family_id, d_h=d_h, device=device,
    )
    structure = _family_dir(output_root, model_name, family_id, d_h)
    inventory = json.loads((structure / "inventory.json").read_text(encoding="utf-8"))
    selected_paths = _selected_module_paths(inventory, profile.int8_roles)
    modules = dict(bundle.model.named_modules())
    missing = [path for path in selected_paths if path not in modules or not isinstance(modules[path], nn.Linear)]
    if missing:
        raise RuntimeError(f"dh_smoothquant_selected_module_invalid:{missing[:8]}")
    alpha = profile.alpha(model_name)
    if alpha is None:
        raise RuntimeError("dh_smoothquant_alpha_missing")
    quant_config = selective_smoothquant_config(selected_paths, alpha=float(alpha))
    calibration_manifest = output_root / "evaluation" / "manifests" / model_name / "calibration200.json"
    calibration_report: dict[str, Any] = {}

    def loop(_model: nn.Module) -> None:
        calibration_report.update(_calibrate_train200(
            bundle=bundle,
            config_path=str(MODEL_SPECS[model_name]["config"]),
            manifest_path=calibration_manifest,
            device=device,
        ))

    quantized = mtq.quantize(bundle.model, quant_config, forward_loop=loop)
    if quantized is not bundle.model:
        bundle.model = quantized
    quantizer = _quantizer_state(bundle.model, selected_paths)
    destination.mkdir(parents=True, exist_ok=True)
    batch = _real_batch(bundle, str(MODEL_SPECS[model_name]["config"]), device)
    export_dir = destination / "modelopt_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    origin, export = (
        _export_cobevt(bundle, batch, export_dir, int(MODEL_SPECS[model_name]["fixed_k"]))
        if model_name == "lidar_cobevt"
        else _export_v2xvit(bundle, batch, export_dir, int(MODEL_SPECS[model_name]["fixed_k"]))
    )
    export.pop("wrapper_output", None)
    qdq_source = destination / "modelopt_qdq_canonical.onnx"
    (export_dir / "base_fp32_canonical.onnx").replace(qdq_source)
    origin_payload = origin if isinstance(origin, Mapping) else origin.to_dict()
    graph = onnx.load(str(qdq_source), load_external_data=False)
    graph_nodes = {str(node.name): node for node in graph.graph.node}
    # Refresh inventory after ModelOpt wraps the physical Linears; canonical
    # origin entries remain the authority for selected path/node adjacency.
    typed_mapping, requested = build_mapping(
        model_name=model_name,
        profile=profile.base_profile,
        origin=origin_payload,
        inventory=inventory,
        graph_nodes=graph_nodes,
    )
    typed_with_casts = destination / "typed_with_qdq_casts.onnx"
    typed_report = apply_strongly_typed_precision_contract(
        qdq_source, typed_with_casts, typed_mapping, plugin_boundary="fp16"
    )
    selected_nodes = {
        str(entry.canonical_node_name)
        for entry in typed_mapping.entries
        if str(entry.module_path) in selected_paths
    }
    mapped = {str(entry.module_path) for entry in typed_mapping.entries if str(entry.module_path) in selected_paths}
    if mapped != set(selected_paths):
        raise RuntimeError(f"dh_smoothquant_mapping_incomplete:{len(mapped)}:{len(selected_paths)}")
    final = destination / "strongly_typed_explicit_qdq.onnx"
    final_graph = onnx.load(str(typed_with_casts), load_external_data=False)
    adjacency_rewrite = restore_projection_qdq_adjacency(final_graph, sorted(selected_nodes), output_cast_precisions={})
    adjacency = validate_projection_qdq_adjacency(final_graph, sorted(selected_nodes))
    onnx.checker.check_model(final_graph)
    onnx.save(final_graph, str(final))
    entries = [
        replace(entry, requested_precision="int8", realized_request_precision="int8")
        if str(entry.module_path) in selected_paths else entry
        for entry in typed_mapping.entries
    ]
    deployment = CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id="P8",
        profile_hash=stable_json_hash({
            "model": model_name, "family": family_id, "d_h": int(d_h),
            "alpha": alpha, "selected": selected_paths,
            "calibration_manifest_hash": calibration_report.get("manifest_hash"),
        }),
        origin_map_hash=typed_mapping.origin_map_hash,
        policy_version="h800-transformer-dh-sq1-fresh-width-v1",
        auxiliary_layer_precisions=dict(typed_mapping.auxiliary_layer_precisions),
        auxiliary_layer_output_types=dict(typed_mapping.auxiliary_layer_output_types),
    )
    for row in requested:
        if str(row.get("module_path", "")) in selected_paths:
            row["requested_precision"] = "INT8"
    qdq_rows = _qdq_inventory(final, requested)
    if any(not bool(row["scale_finite_nonzero"]) for row in qdq_rows):
        raise RuntimeError("dh_smoothquant_qdq_scale_invalid")
    _write_json(destination / "canonical_origin_map.json", origin_payload)
    _write_json(destination / "deployment_precision_mapping.json", deployment.to_dict())
    _write_json(destination / "requested_precision_contract.json", requested)
    _write_json(destination / "calibration_report.json", calibration_report)
    _write_json(destination / "quantizer_inventory.json", quantizer)
    _write_json(destination / "modelopt_config.json", quant_config)
    _write_json(destination / "typed_graph_report.json", typed_report)
    _write_json(destination / "qdq_adjacency_rewrite.json", adjacency_rewrite)
    _write_json(destination / "qdq_adjacency_audit.json", adjacency)
    _write_csv(destination / "qdq_inventory.csv", qdq_rows)
    snapshot = json.loads((structure / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8"))
    result = _finalize_build(
        destination=destination, model_name=model_name, family_id=family_id, d_h=d_h,
        profile_id="P8", typed_path=final, mapping=deployment, requested=requested,
        snapshot=snapshot, physical_gpu=physical_gpu, plugin_path=plugin_path,
    )
    result.update({
        "alpha": alpha,
        "calibration_manifest_hash": calibration_report.get("manifest_hash", ""),
        "calibration_sample_count": calibration_report.get("sample_count", 0),
        "scale_hash": stable_hash(qdq_rows),
        "qdq_count": len(qdq_rows),
        "fresh_width_calibration": True,
    })
    _write_json(destination / "baseline_result.json", result)
    del bundle, batch
    torch.cuda.empty_cache()
    return result


def build_candidate(
    *, output_root: Path, model_name: str, family_id: str, d_h: int,
    profiles: Iterable[str], physical_gpu: int, plugin_path: Path,
) -> dict[str, Any]:
    structure = prepare_structure(
        output_root=output_root, model_name=model_name, family_id=family_id,
        d_h=d_h, physical_gpu=physical_gpu,
    )
    builds = []
    for profile in profiles:
        destination = _engine_dir(output_root, model_name, family_id, d_h, profile)
        result_path = destination / "baseline_result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("status") == "ok":
                builds.append(existing)
                continue
        builds.append(
            build_int8_profile(
                output_root=output_root, model_name=model_name, family_id=family_id,
                d_h=d_h, physical_gpu=physical_gpu, plugin_path=plugin_path,
            ) if profile == "P8" else build_float_profile(
                output_root=output_root, model_name=model_name, family_id=family_id,
                d_h=d_h, profile_id=profile, physical_gpu=physical_gpu, plugin_path=plugin_path,
            )
        )
    return {"structure": structure, "builds": builds}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--dh", required=True, type=int)
    parser.add_argument("--profiles", default="P32,P16,P8")
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args(argv)
    profiles = tuple(value.strip() for value in args.profiles.split(",") if value.strip())
    result = build_candidate(
        output_root=Path(args.output_root).resolve(),
        model_name=args.model,
        family_id=args.family,
        d_h=args.dh,
        profiles=profiles,
        physical_gpu=args.physical_gpu,
        plugin_path=Path(args.plugin).resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in result["builds"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
