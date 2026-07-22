"""Phase-B joint-family d_h candidates selected from completed Phase-A sweeps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import onnx
import torch
from torch import nn

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
    materialize_joint_head_dimensions,
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
from search.orchestration.lidar_transformer_dh_build import (
    TRT_ROOT,
    _build_config,
    _finite_outputs,
    _runtime_costs,
    _sha256,
    _write_csv,
    _write_json,
)
from search.orchestration.lidar_transformer_dh_evaluate import FRAMES
from search.orchestration.lidar_transformer_h800_baselines import build_mapping
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


def parse_targets(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in (item.strip() for item in value.split(",") if item.strip()):
        if "=" not in entry:
            raise ValueError(f"joint_target_requires_family_equals_width:{entry}")
        family, width = entry.rsplit("=", 1)
        if family in result or int(width) <= 0:
            raise ValueError(f"joint_target_invalid:{entry}")
        result[family] = int(width)
    if len(result) < 2:
        raise ValueError("joint_target_requires_multiple_families")
    return result


def joint_id(targets: Mapping[str, int]) -> str:
    readable = "__".join(f"{name}-dh{int(width):03d}" for name, width in sorted(targets.items()))
    return f"joint__{readable}__{stable_hash(dict(sorted(targets.items())))[:12]}"


def _structure_dir(root: Path, model: str, targets: Mapping[str, int]) -> Path:
    return root / "structures" / model / "joint" / joint_id(targets)


def _engine_dir(root: Path, model: str, targets: Mapping[str, int], profile: str) -> Path:
    return root / "engines" / model / "joint" / joint_id(targets) / profile


def _load_joint(
    *, output_root: Path, model_name: str, targets: Mapping[str, int], device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    bundle, _ = _load(model_name, device)
    families = {family.family_id: family for family in discover_attention_families(model_name, bundle.model)}
    unknown = sorted(set(targets) - set(families))
    if unknown:
        raise RuntimeError(f"joint_target_family_missing:{unknown}")
    ranking = json.loads((output_root / "importance" / f"{model_name.removeprefix('lidar_')}_ranking_manifest.json").read_text(encoding="utf-8"))
    masks_by_family = {
        family_id: masks_from_rankings(families[family_id], ranking["rankings"], int(width))
        for family_id, width in targets.items()
    }
    report = materialize_joint_head_dimensions(model_name, bundle.model, masks_by_family)
    if not report.passed:
        raise RuntimeError(f"joint_physical_rewrite_failed:{report.issues}")
    masks = {
        family_id: {path: mask.to_dict() for path, mask in values.items()}
        for family_id, values in masks_by_family.items()
    }
    return bundle, {"report": report, "families": families}, masks


def prepare_joint_structure(
    *, output_root: Path, model_name: str, targets: Mapping[str, int], physical_gpu: int,
) -> dict[str, Any]:
    destination = _structure_dir(output_root, model_name, targets)
    result_path = destination / "structure_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle, context, masks = _load_joint(output_root=output_root, model_name=model_name, targets=targets, device=device)
    report = context["report"]
    families = context["families"]
    batch = _real_batch(bundle, str(MODEL_SPECS[model_name]["config"]), device)
    with torch.inference_mode():
        output = bundle.adapter.forward_for_task(bundle.model, batch)
    finite, output_rows = _finite_outputs(output)
    if not finite:
        raise RuntimeError("joint_physical_forward_nonfinite")
    costs = _runtime_costs(bundle.model, lambda: bundle.adapter.forward_for_task(bundle.model, batch))
    destination.mkdir(parents=True, exist_ok=True)
    origin, export = (
        _export_cobevt(bundle, batch, destination, int(MODEL_SPECS[model_name]["fixed_k"]))
        if model_name == "lidar_cobevt"
        else _export_v2xvit(bundle, batch, destination, int(MODEL_SPECS[model_name]["fixed_k"]))
    )
    export.pop("wrapper_output", None)
    origin_payload = origin if isinstance(origin, Mapping) else origin.to_dict()
    inventory = build_model_inventory(model_family=model_name, model=bundle.model, onnx_path=destination / "base_fp32_canonical.onnx", origin_map=origin)
    from search.model_family.deployment import build_physical_structure_snapshot_v2

    snapshot = build_physical_structure_snapshot_v2(bundle.model, model_family=model_name)
    audits: dict[str, Any] = {}
    for family_id, width in targets.items():
        family = families[family_id]
        audits[family_id] = audit_onnx_head_dimension(
            destination / "base_fp32_canonical.onnx",
            selected_module_paths=family.module_paths,
            origin_map=origin_payload,
            heads=family.heads,
            target_d_h=int(width),
        )
    if not all(value["passed"] for value in audits.values()):
        raise RuntimeError(f"joint_onnx_head_dimension_audit_failed:{audits}")
    _write_json(destination / "canonical_origin_map.json", origin_payload)
    _write_json(destination / "inventory.json", inventory)
    _write_json(destination / "physical_structure_snapshot_v2.json", snapshot)
    _write_json(destination / "physical_rewrite.json", report.to_dict())
    _write_json(destination / "head_local_masks.json", masks)
    _write_json(destination / "physical_forward.json", {"finite": finite, "outputs": output_rows})
    _write_json(destination / "runtime_costs.json", costs)
    _write_json(destination / "onnx_head_dimension_audit.json", audits)
    result = {
        "status": "ok",
        "model": model_name,
        "joint_id": joint_id(targets),
        "target_d_h_by_family": dict(targets),
        "projection_width_by_family": {name: families[name].heads * int(width) for name, width in targets.items()},
        "original_parameter_count": report.original_parameter_count,
        "physical_parameter_count": report.physical_parameter_count,
        "parameter_reduction": 1.0 - report.physical_parameter_count / report.original_parameter_count,
        "structure_hash": report.structure_hash,
        "state_dict_shape_hash": report.state_dict_shape_hash,
        "onnx_sha256": _sha256(destination / "base_fp32_canonical.onnx"),
        "onnx_shape_hash": stable_hash(audits),
        "physical_forward_finite": finite,
        "onnx_checker_passed": True,
        "onnx_shape_inference_passed": True,
        "weighted_macs": costs["weighted_macs"],
        "qk_macs": costs["qk_macs"],
        "av_macs": costs["av_macs"],
        "total_macs_including_qk_av": costs["total_macs_including_qk_av"],
    }
    _write_json(result_path, result)
    del bundle, batch, output
    torch.cuda.empty_cache()
    return result


def _finalize(
    *, output_root: Path, destination: Path, model_name: str, targets: Mapping[str, int],
    profile: str, typed_path: Path, mapping: CanonicalPrecisionMappingResult,
    requested: list[dict[str, Any]], snapshot: Mapping[str, Any], physical_gpu: int,
    plugin: Path,
) -> dict[str, Any]:
    engine = destination / "engine.plan"
    build = build_engine_modelopt(
        qdq_onnx=typed_path, engine_path=engine, precision_mapping=mapping,
        build_config=_build_config(plugin), physical_snapshot=snapshot,
        output_dir=destination / "engine_build", tensorrt_root=TRT_ROOT,
        conda_env="modelopt", gpu_id=physical_gpu,
    )
    _write_json(destination / "build_acceptance.json", build)
    layer_info = destination / "engine_build" / "engine_layer_info.json"
    realized = [
        row.to_dict()
        for row in audit_realized_precision(
            model=model_name, profile=profile, requested_rows=requested,
            layer_info_path=layer_info, typed_onnx_path=typed_path, strongly_typed=True,
        )
    ] if layer_info.is_file() else []
    _write_csv(destination / "requested_realized.csv", realized)
    conflicts = [row for row in realized if row["conflict"]]
    structure = json.loads((_structure_dir(output_root, model_name, targets) / "structure_result.json").read_text(encoding="utf-8"))
    alignments = {
        family: audit_engine_alignment(layer_info, logical_d_h=int(width), projection_width=int(structure["projection_width_by_family"][family]))
        for family, width in targets.items()
    } if layer_info.is_file() else {}
    _write_json(destination / "engine_alignment_audit_by_family.json", alignments)
    status = str(build.get("status", "engine_build_failed"))
    if status == "ok" and conflicts:
        status = "precision_conflict"
    if status == "ok" and any(value.get("fallback_hint") for value in alignments.values()):
        status = "shape_or_tactic_fallback"
    if status == "ok" and any(value.get("padding_status") == "UNKNOWN" for value in alignments.values()):
        status = "alignment_evidence_unknown"
    result = {
        "status": status,
        "model": model_name,
        "joint_id": joint_id(targets),
        "target_d_h_by_family": dict(targets),
        "profile": profile,
        "engine_exists": engine.is_file() and engine.stat().st_size > 0,
        "engine_sha256": _sha256(engine) if engine.is_file() else "",
        "typed_onnx_sha256": _sha256(typed_path),
        "strongly_typed": True,
        "no_tf32": True,
        "qk_contract": "F32A32O32",
        "plugin_sha256": _sha256(plugin),
        "builder_flags": ["--stronglyTyped", "--noTF32", "--skipInference", "--memPoolSize=workspace:8192"],
        "workspace_mib": 8192,
        "timing_cache_reused": False,
        "timing_cache_sha256": "not_applicable_fresh_build",
        "requested_realized_conflict_count": len(conflicts),
        "requested_realized_record_count": len(realized),
        "alignment_status_by_family": {name: value["padding_status"] for name, value in alignments.items()},
        "engine_memory": parse_engine_memory_audit(destination / "engine_build" / "engine_build.log"),
    }
    _write_json(destination / "baseline_result.json", result)
    return result


def build_joint_float(
    *, output_root: Path, model_name: str, targets: Mapping[str, int], profile_id: str,
    physical_gpu: int, plugin: Path,
) -> dict[str, Any]:
    profile = require_profile(profile_id)
    if profile.int8_roles:
        raise ValueError("joint_float_received_int8_profile")
    structure = _structure_dir(output_root, model_name, targets)
    destination = _engine_dir(output_root, model_name, targets, profile_id)
    origin = json.loads((structure / "canonical_origin_map.json").read_text(encoding="utf-8"))
    inventory = json.loads((structure / "inventory.json").read_text(encoding="utf-8"))
    snapshot = json.loads((structure / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8"))
    base = structure / "base_fp32_canonical.onnx"
    graph = onnx.load(str(base), load_external_data=False)
    mapping, requested = build_mapping(model_name=model_name, profile=profile.base_profile, origin=origin, inventory=inventory, graph_nodes={str(node.name): node for node in graph.graph.node})
    destination.mkdir(parents=True, exist_ok=True)
    typed = destination / "strongly_typed.onnx"
    report = apply_strongly_typed_precision_contract(base, typed, mapping, plugin_boundary="fp16")
    onnx.checker.check_model(onnx.load(str(typed), load_external_data=False))
    _write_json(destination / "canonical_precision_mapping.json", mapping.to_dict())
    _write_json(destination / "requested_precision_contract.json", requested)
    _write_json(destination / "typed_graph_report.json", {**report, "checker_passed": True})
    return _finalize(output_root=output_root, destination=destination, model_name=model_name, targets=targets, profile=profile_id, typed_path=typed, mapping=mapping, requested=requested, snapshot=snapshot, physical_gpu=physical_gpu, plugin=plugin)


def build_joint_int8(
    *, output_root: Path, model_name: str, targets: Mapping[str, int], physical_gpu: int,
    plugin: Path,
) -> dict[str, Any]:
    profile = require_profile("P8")
    identifier = joint_id(targets)
    destination = _engine_dir(output_root, model_name, targets, "P8")
    configure_modelopt_inprocess(output_root=output_root, cache_namespace=f"dh_joint_{model_name}_{stable_hash(dict(targets))[:16]}_P8_gpu{physical_gpu}")
    require_modelopt_cuda_extension("int8")
    import modelopt.torch.quantization as mtq  # type: ignore

    runtime = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime}")
    torch.cuda.set_device(device)
    bundle, _, _ = _load_joint(output_root=output_root, model_name=model_name, targets=targets, device=device)
    structure = _structure_dir(output_root, model_name, targets)
    inventory = json.loads((structure / "inventory.json").read_text(encoding="utf-8"))
    selected_paths = _selected_module_paths(inventory, profile.int8_roles)
    modules = dict(bundle.model.named_modules())
    missing = [path for path in selected_paths if path not in modules or not isinstance(modules[path], nn.Linear)]
    if missing:
        raise RuntimeError(f"joint_smoothquant_selected_module_invalid:{missing[:8]}")
    alpha = profile.alpha(model_name)
    if alpha is None:
        raise RuntimeError("joint_smoothquant_alpha_missing")
    quant_config = selective_smoothquant_config(selected_paths, alpha=float(alpha))
    calibration_manifest = output_root / "evaluation" / "manifests" / model_name / "calibration200.json"
    calibration_report: dict[str, Any] = {}

    def loop(_model: nn.Module) -> None:
        calibration_report.update(_calibrate_train200(bundle=bundle, config_path=str(MODEL_SPECS[model_name]["config"]), manifest_path=calibration_manifest, device=device))

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
    mapping, requested = build_mapping(model_name=model_name, profile=profile.base_profile, origin=origin_payload, inventory=inventory, graph_nodes={str(node.name): node for node in graph.graph.node})
    typed_with_casts = destination / "typed_with_qdq_casts.onnx"
    typed_report = apply_strongly_typed_precision_contract(qdq_source, typed_with_casts, mapping, plugin_boundary="fp16")
    selected_nodes = {str(entry.canonical_node_name) for entry in mapping.entries if str(entry.module_path) in selected_paths}
    if {str(entry.module_path) for entry in mapping.entries if str(entry.module_path) in selected_paths} != set(selected_paths):
        raise RuntimeError("joint_smoothquant_mapping_incomplete")
    final = destination / "strongly_typed_explicit_qdq.onnx"
    final_graph = onnx.load(str(typed_with_casts), load_external_data=False)
    adjacency_rewrite = restore_projection_qdq_adjacency(final_graph, sorted(selected_nodes), output_cast_precisions={})
    adjacency = validate_projection_qdq_adjacency(final_graph, sorted(selected_nodes))
    onnx.checker.check_model(final_graph)
    onnx.save(final_graph, str(final))
    entries = [replace(entry, requested_precision="int8", realized_request_precision="int8") if str(entry.module_path) in selected_paths else entry for entry in mapping.entries]
    deployment = CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id="P8",
        profile_hash=stable_json_hash({"model": model_name, "joint_id": identifier, "targets": dict(targets), "alpha": alpha, "selected": selected_paths, "calibration_manifest_hash": calibration_report.get("manifest_hash")}),
        origin_map_hash=mapping.origin_map_hash,
        policy_version="h800-transformer-dh-joint-sq1-fresh-v1",
        auxiliary_layer_precisions=dict(mapping.auxiliary_layer_precisions),
        auxiliary_layer_output_types=dict(mapping.auxiliary_layer_output_types),
    )
    for row in requested:
        if str(row.get("module_path", "")) in selected_paths:
            row["requested_precision"] = "INT8"
    qdq_rows = _qdq_inventory(final, requested)
    if any(not bool(row["scale_finite_nonzero"]) for row in qdq_rows):
        raise RuntimeError("joint_smoothquant_qdq_scale_invalid")
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
    result = _finalize(output_root=output_root, destination=destination, model_name=model_name, targets=targets, profile="P8", typed_path=final, mapping=deployment, requested=requested, snapshot=snapshot, physical_gpu=physical_gpu, plugin=plugin)
    result.update({"alpha": alpha, "calibration_manifest_hash": calibration_report.get("manifest_hash", ""), "calibration_sample_count": calibration_report.get("sample_count", 0), "scale_hash": stable_hash(qdq_rows), "qdq_count": len(qdq_rows), "fresh_joint_calibration": True})
    _write_json(destination / "baseline_result.json", result)
    del bundle, batch
    torch.cuda.empty_cache()
    return result


def evaluate_joint(
    *, output_root: Path, model_name: str, targets: Mapping[str, int], profile: str,
    protocol: str, physical_gpu: int, plugin: Path,
) -> dict[str, Any]:
    from search.orchestration.lidar_transformer_dh_evaluate import HEAL_ROOT

    directory = _engine_dir(output_root, model_name, targets, profile)
    build = json.loads((directory / "baseline_result.json").read_text(encoding="utf-8"))
    if build.get("status") != "ok" or int(build.get("requested_realized_conflict_count", -1)) != 0:
        raise RuntimeError("joint_evaluation_build_not_exact")
    manifest = output_root / "evaluation" / "manifests" / model_name / f"{protocol}.json"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    destination = directory / "evaluation" / protocol
    spec = MODEL_SPECS[model_name]
    if model_name == "lidar_cobevt":
        from search.integration.lidar_cobevt_evaluation_provider import evaluate_cobevt_engine_modelopt

        raw = evaluate_cobevt_engine_modelopt(engine_path=directory / "engine.plan", checkpoint=spec["checkpoint"], model_config=spec["config"], heal_root=HEAL_ROOT, device=f"cuda:{physical_gpu}", output_dir=destination, tensorrt_root=TRT_ROOT, plugin_path=plugin, fixed_k=int(spec["fixed_k"]), num_frames=FRAMES[protocol], warmup_frames=len(manifest_payload["warmup_frame_ids"]), eval_manifest_path=manifest, num_workers=8, ap_iou_backend="gpu", latency_rounds=1)
    else:
        from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt

        raw = evaluate_v2xvit_engine_modelopt(engine_path=directory / "engine.plan", model_config=spec["config"], heal_root=HEAL_ROOT, output_dir=destination, tensorrt_root=TRT_ROOT, plugin_path=plugin, eval_manifest_path=manifest, physical_gpu_id=physical_gpu, fixed_k=int(spec["fixed_k"]), max_agents=2, num_frames=FRAMES[protocol], warmup_frames=len(manifest_payload["warmup_frame_ids"]), latency_rounds=1, dataloader_num_workers=8)
    accepted = raw.get("status") == "ok" and int(raw.get("num_evaluated_frames", -1)) == FRAMES[protocol] and int(raw.get("num_skipped_frames", -1)) == 0 and bool(raw.get("reset_after_warmup", False))
    structure = json.loads((_structure_dir(output_root, model_name, targets) / "structure_result.json").read_text(encoding="utf-8"))
    result = {
        "status": "ok" if accepted else "evaluation_failed",
        "model": model_name,
        "joint_id": joint_id(targets),
        "target_d_h_by_family": dict(targets),
        "profile": profile,
        "protocol": protocol,
        "evaluated": int(raw.get("num_evaluated_frames", 0)),
        "skipped": int(raw.get("num_skipped_frames", 0)),
        "AP@0.3": raw.get("AP@0.3"), "AP@0.5": raw.get("AP@0.5"), "AP@0.7": raw.get("AP@0.7"), "mAP": raw.get("mAP"),
        "engine_sha256": _sha256(directory / "engine.plan"),
        "structure_hash": structure["structure_hash"],
        "scale_hash": build.get("scale_hash", "not_quantized"),
        "manifest_hash": manifest_payload["manifest_hash"],
        "workers": 8, "ap_iou_backend": "gpu", "reset_after_warmup": raw.get("reset_after_warmup"),
    }
    _write_json(destination / "evaluation_acceptance.json", result)
    return result


def run_joint(
    *, output_root: Path, model_name: str, targets: Mapping[str, int], physical_gpu: int,
    plugin: Path, profiles: tuple[str, ...], protocols: tuple[str, ...],
) -> dict[str, Any]:
    structure = prepare_joint_structure(output_root=output_root, model_name=model_name, targets=targets, physical_gpu=physical_gpu)
    builds = []
    evaluations = []
    for profile in profiles:
        result_path = _engine_dir(output_root, model_name, targets, profile) / "baseline_result.json"
        if result_path.is_file() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "ok":
            build = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            build = build_joint_int8(output_root=output_root, model_name=model_name, targets=targets, physical_gpu=physical_gpu, plugin=plugin) if profile == "P8" else build_joint_float(output_root=output_root, model_name=model_name, targets=targets, profile_id=profile, physical_gpu=physical_gpu, plugin=plugin)
        builds.append(build)
        if build["status"] != "ok":
            continue
        for protocol in protocols:
            evaluation_path = _engine_dir(output_root, model_name, targets, profile) / "evaluation" / protocol / "evaluation_acceptance.json"
            if evaluation_path.is_file() and json.loads(evaluation_path.read_text(encoding="utf-8")).get("status") == "ok":
                evaluations.append(json.loads(evaluation_path.read_text(encoding="utf-8")))
            else:
                evaluations.append(evaluate_joint(output_root=output_root, model_name=model_name, targets=targets, profile=profile, protocol=protocol, physical_gpu=physical_gpu, plugin=plugin))
    result = {"structure": structure, "builds": builds, "evaluations": evaluations}
    _write_json(output_root / "reports" / f"{joint_id(targets)}_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--profiles", default="P32,P16,P8")
    parser.add_argument("--protocols", default="smoke10,fixed50,fixed500")
    args = parser.parse_args(argv)
    result = run_joint(output_root=Path(args.output_root).resolve(), model_name=args.model, targets=parse_targets(args.targets), physical_gpu=args.physical_gpu, plugin=Path(args.plugin).resolve(), profiles=tuple(value for value in args.profiles.split(",") if value), protocols=tuple(value for value in args.protocols.split(",") if value))
    print(json.dumps({"builds": len(result["builds"]), "evaluations": len(result["evaluations"])}, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in (*result["builds"], *result["evaluations"])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
