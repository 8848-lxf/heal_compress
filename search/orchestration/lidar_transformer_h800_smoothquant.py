"""Fresh dual-model SmoothQuant explicit-Q/DQ build on the frozen H800 graph."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from quantization.config import TensorRTBuildConfig
from quantization.precision.typed_graph import apply_strongly_typed_precision_contract
from quantization.types import CanonicalPrecisionMappingResult, stable_json_hash
from search.model_families.transformer.qdq_adjacency import (
    restore_projection_qdq_adjacency,
    validate_projection_qdq_adjacency,
)
from search.model_families.transformer.realized_precision import audit_realized_precision
from search.model_families.transformer.smoothquant_profiles import (
    selective_smoothquant_config,
    smoothquant_profiles,
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
    _export_cobevt,
    _export_v2xvit,
    _load,
    _real_batch,
)
from search.stage2.trt_modelopt import build_engine_modelopt
from search.integration.runtime_environment import (
    configure_modelopt_inprocess,
    require_modelopt_cuda_extension,
    runtime_cuda_index_for_physical,
)


HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(profile_id: str) -> Any:
    try:
        return next(row for row in smoothquant_profiles() if row.profile_id == profile_id)
    except StopIteration as exc:
        raise ValueError(f"unknown_smoothquant_profile:{profile_id}") from exc


def _selected_module_paths(
    inventory: Mapping[str, Any], roles: Iterable[str]
) -> tuple[str, ...]:
    wanted = {str(role) for role in roles}
    paths = sorted(
        {
            str(row["module_path"])
            for row in inventory["rows"]
            if str(row.get("canonical_role", "")) in wanted
            and str(row.get("module_path", ""))
            and str(row.get("onnx_node", ""))
            and str(row.get("onnx_op_type", "")) in {"MatMul", "Gemm"}
        }
    )
    realized_roles = {
        str(row.get("canonical_role", ""))
        for row in inventory["rows"]
        if str(row.get("module_path", "")) in paths
    }
    missing = sorted(wanted - realized_roles)
    if missing:
        raise RuntimeError(f"smoothquant_role_has_no_realized_linear:{missing}")
    return tuple(paths)


def _calibrate_manifest(
    *, bundle: Any, config_path: str, manifest_path: Path, device: torch.device,
    expected_samples: int,
) -> dict[str, Any]:
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from search.integration.data_provider import move_batch_to_device

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("split") != "train"
        or int(manifest.get("sample_count", 0)) != int(expected_samples)
    ):
        raise RuntimeError(
            f"smoothquant_requires_frozen_train_manifest:{expected_samples}"
        )
    hypes = yaml_utils.load_yaml(config_path)
    hypes = bundle.adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=False, train=True)
    calibrated: list[str] = []
    bundle.model.eval()
    with torch.inference_mode():
        for row in manifest["samples"]:
            index = int(row["dataset_index"])
            seed = int(row["sample_seed"])
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            item = dataset[index]
            batch = dataset.collate_batch_train([item])
            if batch is None:
                raise RuntimeError(f"smoothquant_empty_calibration_batch:{index}")
            bundle.adapter.forward_for_task(
                bundle.model, move_batch_to_device(batch, device)
            )
            calibrated.append(str(row["frame_id"]))
    expected = [str(row["frame_id"]) for row in manifest["samples"]]
    if calibrated != expected:
        raise RuntimeError("smoothquant_calibration_order_mismatch")
    return {
        "manifest_path": str(manifest_path),
        "manifest_hash": str(manifest["manifest_hash"]),
        "sample_count": len(calibrated),
        "frame_ids": calibrated,
        "selection_policy": str(manifest["selection_policy"]),
    }


def _calibrate_train200(
    *, bundle: Any, config_path: str, manifest_path: Path, device: torch.device
) -> dict[str, Any]:
    return _calibrate_manifest(
        bundle=bundle,
        config_path=config_path,
        manifest_path=manifest_path,
        device=device,
        expected_samples=200,
    )


def _quantizer_state(model: torch.nn.Module, selected: Iterable[str]) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows = []
    for path in selected:
        module = modules.get(str(path))
        if module is None:
            raise RuntimeError(f"smoothquant_selected_module_missing_after_transform:{path}")
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        pre_scale = getattr(input_quantizer, "pre_quant_scale", None)
        input_amax = getattr(input_quantizer, "_amax", None)
        weight_amax = getattr(weight_quantizer, "_amax", None)
        enabled = lambda value: bool(value() if callable(value) else value)
        rows.append(
            {
                "module_path": str(path),
                "module_class": type(module).__name__,
                "input_quantizer_enabled": enabled(getattr(input_quantizer, "is_enabled", False)),
                "weight_quantizer_enabled": enabled(getattr(weight_quantizer, "is_enabled", False)),
                "input_axis": getattr(input_quantizer, "axis", None),
                "weight_axis": getattr(weight_quantizer, "axis", None),
                "pre_quant_scale_shape": list(pre_scale.shape) if torch.is_tensor(pre_scale) else [],
                "pre_quant_scale_finite": bool(torch.isfinite(pre_scale).all()) if torch.is_tensor(pre_scale) else False,
                "input_amax": np.asarray(input_amax.detach().cpu()).tolist() if torch.is_tensor(input_amax) else None,
                "weight_amax": np.asarray(weight_amax.detach().cpu()).tolist() if torch.is_tensor(weight_amax) else None,
            }
        )
    if any(
        not row["input_quantizer_enabled"]
        or not row["weight_quantizer_enabled"]
        or row["weight_axis"] != 0
        or row["input_axis"] is not None
        or not row["pre_quant_scale_finite"]
        for row in rows
    ):
        raise RuntimeError("smoothquant_quantizer_contract_incomplete")
    return rows


def _export_quantized(
    *, model_name: str, bundle: Any, batch: Mapping[str, Any], destination: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    export_dir = destination / "modelopt_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    fixed_k = int(MODEL_SPECS[model_name]["fixed_k"])
    origin, export = (
        _export_cobevt(bundle, batch, export_dir, fixed_k)
        if model_name == "lidar_cobevt"
        else _export_v2xvit(bundle, batch, export_dir, fixed_k)
    )
    generated = export_dir / "base_fp32_canonical.onnx"
    qdq = destination / "modelopt_qdq_canonical.onnx"
    generated.replace(qdq)
    origin_payload = origin if isinstance(origin, Mapping) else origin.to_dict()
    _write_json(destination / "canonical_origin_map.json", origin_payload)
    return dict(origin_payload), qdq, export["report"]


def _qdq_inventory(
    model_path: Path,
    requested_contract: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    graph = onnx.load(str(model_path), load_external_data=False)
    initializers = {str(value.name): value for value in graph.graph.initializer}
    constant_tensors = {
        str(output): attribute.t
        for node in graph.graph.node
        if str(node.op_type) == "Constant"
        for output in node.output
        for attribute in node.attribute
        if str(attribute.name) == "value" and attribute.HasField("t")
    }

    def constant_array(name: str) -> np.ndarray:
        value = initializers.get(str(name), constant_tensors.get(str(name)))
        return numpy_helper.to_array(value) if value is not None else np.asarray([])

    consumers: dict[str, list[str]] = {}
    producer: dict[str, str] = {}
    nodes_by_name = {str(node.name): node for node in graph.graph.node}
    for node in graph.graph.node:
        for value in node.output:
            producer[str(value)] = str(node.name)
        for value in node.input:
            consumers.setdefault(str(value), []).append(str(node.name))
    requested_by_node = {
        str(row.get("onnx_node", "")): dict(row)
        for row in requested_contract
        if str(row.get("onnx_node", ""))
    }

    def weighted_consumer(output: str) -> dict[str, Any]:
        frontier = [str(output)]
        visited: set[str] = set()
        for _ in range(5):
            next_frontier = []
            for tensor in frontier:
                for node_name in consumers.get(tensor, ()):
                    if node_name in requested_by_node:
                        return requested_by_node[node_name]
                    if node_name in visited:
                        continue
                    visited.add(node_name)
                    node = nodes_by_name.get(node_name)
                    if node is not None:
                        next_frontier.extend(str(value) for value in node.output)
            frontier = next_frontier
        return {}
    rows = []
    for node in graph.graph.node:
        if str(node.op_type) not in {
            "QuantizeLinear", "DequantizeLinear",
            "TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear",
        }:
            continue
        scale_values = constant_array(str(node.input[1])) if len(node.input) > 1 else np.asarray([])
        zero_values = constant_array(str(node.input[2])) if len(node.input) > 2 else np.asarray([0], dtype=np.int8)
        axis = next((int(attr.i) for attr in node.attribute if str(attr.name) == "axis"), None)
        owner = weighted_consumer(str(node.output[0]))
        quant_role = (
            "weight" if "weight_quantizer" in str(node.name)
            else "input_activation" if "input_quantizer" in str(node.name)
            else "activation"
        )
        granularity = "per_channel" if scale_values.size > 1 else "per_tensor"
        symmetric = bool(zero_values.size and np.all(zero_values == 0))
        rows.append(
            {
                "node": str(node.name),
                "op_type": str(node.op_type),
                "input_tensor": str(node.input[0]),
                "output_tensor": str(node.output[0]),
                "producer": producer.get(str(node.input[0]), "graph_input_or_initializer"),
                "consumers": consumers.get(str(node.output[0]), []),
                "canonical_layer": str(owner.get("onnx_node", "")),
                "module_path": str(owner.get("module_path", "")),
                "canonical_role": str(owner.get("role", "")),
                "requested_precision": str(owner.get("requested_precision", "")),
                "quantization_role": quant_role,
                "granularity": granularity,
                "symmetric": symmetric,
                "scale_owner": (
                    f"{owner.get('module_path', '')}:{quant_role}"
                    if owner else "unresolved"
                ),
                "scale_initializer": str(node.input[1]) if len(node.input) > 1 else "",
                "scale_shape": list(scale_values.shape),
                "scale_min": float(scale_values.min()) if scale_values.size else None,
                "scale_max": float(scale_values.max()) if scale_values.size else None,
                "scale_finite_nonzero": bool(np.isfinite(scale_values).all() and np.all(scale_values > 0)) if scale_values.size else False,
                "zero_point_shape": list(zero_values.shape),
                "zero_point_values": np.unique(zero_values).tolist() if zero_values.size else [],
                "axis": axis,
            }
        )
    return rows


def build_profile(
    *, output_root: Path, model_name: str, profile_id: str, alpha: float,
    physical_gpu: int, plugin_path: Path, destination_section: str = "smoothquant"
) -> dict[str, Any]:
    toolchain = configure_modelopt_inprocess(
        output_root=output_root,
        cache_namespace=f"{model_name}_{profile_id}_gpu{physical_gpu}",
    )
    cuda_extension = require_modelopt_cuda_extension("int8")
    import modelopt.torch.quantization as mtq  # type: ignore
    from search.model_families.transformer.projection_rewrite import (
        split_cobevt_fused_qkv,
        split_v2xvit_fused_qkv,
    )

    profile = _profile(profile_id)
    if not profile.int8_roles:
        raise ValueError("SQ0_reuses_B3_and_is_not_rebuilt")
    destination = output_root / destination_section / model_name / profile_id
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(
        destination / "modelopt_cuda_toolchain.json",
        {"toolchain": toolchain, "extension": cuda_extension},
    )
    inventory_dir = output_root / "inventory" / model_name
    inventory = json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))
    selected_paths = _selected_module_paths(inventory, profile.int8_roles)
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
        raise RuntimeError(f"smoothquant_selected_non_linear:{invalid[:8]}")
    quant_config = selective_smoothquant_config(selected_paths, alpha=alpha)
    calibration_manifest = output_root / "evaluation" / "manifests" / model_name / "calibration200.json"
    calibration_report: dict[str, Any] = {}

    def forward_loop(_model: torch.nn.Module) -> None:
        calibration_report.update(
            _calibrate_train200(
                bundle=bundle,
                config_path=str(MODEL_SPECS[model_name]["config"]),
                manifest_path=calibration_manifest,
                device=device,
            )
        )

    quantized = mtq.quantize(bundle.model, quant_config, forward_loop=forward_loop)
    if quantized is not bundle.model:
        bundle.model = quantized
    quantizer_rows = _quantizer_state(bundle.model, selected_paths)
    # Persist provenance before export/graph legalization.  A later ONNX or
    # mapping failure must not erase proof of which frozen train200 samples
    # created the embedded Q/DQ scales.
    _write_json(destination / "calibration_report.json", calibration_report)
    _write_json(destination / "quantizer_inventory.json", quantizer_rows)
    _write_json(destination / "modelopt_config.json", quant_config)
    batch = _real_batch(bundle, str(MODEL_SPECS[model_name]["config"]), device)
    origin, qdq_source, export_report = _export_quantized(
        model_name=model_name, bundle=bundle, batch=batch, destination=destination
    )
    graph = onnx.load(str(qdq_source), load_external_data=False)
    graph_nodes = {str(node.name): node for node in graph.graph.node}
    typed_mapping, requested = build_mapping(
        model_name=model_name,
        profile="B3_F3",
        origin=origin,
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
    mapped_paths = {
        str(entry.module_path)
        for entry in typed_mapping.entries
        if str(entry.module_path) in selected_paths
    }
    if mapped_paths != set(selected_paths):
        raise RuntimeError(
            "smoothquant_selected_mapping_incomplete:"
            f"mapped_paths={len(mapped_paths)}:selected_paths={len(selected_paths)}"
        )
    # ``apply_strongly_typed_precision_contract`` has already placed the
    # consumer-boundary Cast required by the semantic graph.  Adding another
    # FP16 Cast here is not merely redundant for FFN projections: the bias and
    # GELU boundary may immediately require FP32, yielding FP16 -> FP32 Casts
    # after an INT8 MatMul and triggering TensorRT 10.9's compiler backend.
    # Q/DQ adjacency restoration therefore owns only the two MatMul inputs;
    # output/merge precision remains owned by the typed semantic contract.
    output_fp16: dict[str, str] = {}
    final_qdq = destination / "strongly_typed_explicit_qdq.onnx"
    final_graph = onnx.load(str(typed_with_casts), load_external_data=False)
    adjacency_rewrite = restore_projection_qdq_adjacency(
        final_graph,
        sorted(selected_nodes),
        output_cast_precisions=output_fp16,
    )
    adjacency = validate_projection_qdq_adjacency(final_graph, sorted(selected_nodes))
    onnx.checker.check_model(final_graph)
    onnx.save(final_graph, str(final_qdq))

    deployment_entries = [
        replace(entry, requested_precision="int8", realized_request_precision="int8")
        if str(entry.module_path) in selected_paths
        else entry
        for entry in typed_mapping.entries
    ]
    deployment_mapping = CanonicalPrecisionMappingResult(
        entries=deployment_entries,
        profile_id=profile_id,
        profile_hash=stable_json_hash(
            {"model": model_name, "profile": profile_id, "alpha": alpha, "selected": selected_paths}
        ),
        origin_map_hash=typed_mapping.origin_map_hash,
        policy_version="h800-transformer-modelopt-smoothquant-explicit-qdq-v1",
        auxiliary_layer_precisions=dict(typed_mapping.auxiliary_layer_precisions),
        auxiliary_layer_output_types=dict(typed_mapping.auxiliary_layer_output_types),
    )
    for row in requested:
        if str(row.get("module_path", "")) in selected_paths:
            row["requested_precision"] = "INT8"
    _write_json(destination / "deployment_precision_mapping.json", deployment_mapping.to_dict())
    _write_json(destination / "requested_precision_contract.json", requested)
    _write_json(destination / "calibration_report.json", calibration_report)
    _write_json(destination / "quantizer_inventory.json", quantizer_rows)
    _write_json(destination / "modelopt_config.json", quant_config)
    _write_json(destination / "typed_graph_report.json", typed_report)
    _write_json(destination / "qdq_adjacency_rewrite.json", adjacency_rewrite)
    _write_json(destination / "qdq_adjacency_audit.json", adjacency)
    qdq_rows = _qdq_inventory(final_qdq, requested)
    _write_csv(destination / "qdq_inventory.csv", qdq_rows)
    if any(not bool(row["scale_finite_nonzero"]) for row in qdq_rows):
        raise RuntimeError("smoothquant_qdq_scale_nonfinite_or_zero")
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
        policy_version="h800-transformer-trt10.9-strongly-typed-explicit-qdq-v1",
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
    realized_rows = []
    if layer_info.is_file():
        realized_rows = [
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
        _write_csv(destination / "requested_realized.csv", realized_rows)
    conflicts = [row for row in realized_rows if row["conflict"]]
    selected_realized = [
        row for row in realized_rows
        if row["onnx_node"] in selected_nodes and row["realized_precision"] == "INT8"
    ]
    status = str(build.get("status", "engine_build_failed"))
    if status == "ok" and (conflicts or len(selected_realized) != len(selected_nodes)):
        status = "precision_conflict"
    result = {
        "status": status,
        "model": model_name,
        "profile": profile_id,
        "alpha": float(alpha),
        "selected_roles": list(profile.int8_roles),
        "selected_module_count": len(selected_paths),
        "selected_node_count": len(selected_nodes),
        "realized_int8_selected_count": len(selected_realized),
        "requested_realized_conflict_count": len(conflicts),
        "engine_exists": engine.is_file() and engine.stat().st_size > 0,
        "engine_sha256": _sha256(engine) if engine.is_file() else "",
        "final_qdq_sha256": _sha256(final_qdq),
        "source_qdq_sha256": _sha256(qdq_source),
        "calibration_manifest_hash": calibration_report.get("manifest_hash", ""),
        "projection_rewrite": rewrite.to_dict(),
        "export_report": export_report,
        "physical_structure_frozen": True,
        "modelopt_cuda_extension": cuda_extension,
    }
    _write_json(destination / "baseline_result.json", result)
    return result


def finalize_exported_profile(
    *, output_root: Path, model_name: str, profile_id: str, alpha: float,
    physical_gpu: int, plugin_path: Path, destination_section: str = "smoothquant"
) -> dict[str, Any]:
    """Finalize an already fresh ModelOpt export without repeating train200."""

    destination = output_root / destination_section / model_name / profile_id
    inventory_dir = output_root / "inventory" / model_name
    qdq_source = destination / "modelopt_qdq_canonical.onnx"
    origin_path = destination / "canonical_origin_map.json"
    calibration_path = destination / "calibration_report.json"
    quantizer_path = destination / "quantizer_inventory.json"
    required = (qdq_source, origin_path, calibration_path, quantizer_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"smoothquant_finalize_export_missing:{missing}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if int(calibration.get("sample_count", 0)) != 200:
        raise RuntimeError("smoothquant_finalize_requires_fresh_train200")
    inventory = json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))
    origin = json.loads(origin_path.read_text(encoding="utf-8"))
    profile = _profile(profile_id)
    selected_paths = _selected_module_paths(inventory, profile.int8_roles)
    graph = onnx.load(str(qdq_source), load_external_data=False)
    graph_nodes = {str(node.name): node for node in graph.graph.node}
    typed_mapping, requested = build_mapping(
        model_name=model_name,
        profile="B3_F3",
        origin=origin,
        inventory=inventory,
        graph_nodes=graph_nodes,
    )
    mapped_paths = {
        str(entry.module_path)
        for entry in typed_mapping.entries
        if str(entry.module_path) in selected_paths
    }
    if mapped_paths != set(selected_paths):
        raise RuntimeError(
            "smoothquant_selected_mapping_incomplete:"
            f"mapped_paths={len(mapped_paths)}:selected_paths={len(selected_paths)}"
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
    # Output boundary Casts are already present in ``typed_with_casts`` and
    # must not be duplicated by the Q/DQ adjacency repair (see build path).
    output_fp16: dict[str, str] = {}
    final_qdq = destination / "strongly_typed_explicit_qdq.onnx"
    final_graph = onnx.load(str(typed_with_casts), load_external_data=False)
    adjacency_rewrite = restore_projection_qdq_adjacency(
        final_graph, sorted(selected_nodes), output_cast_precisions=output_fp16
    )
    adjacency = validate_projection_qdq_adjacency(final_graph, sorted(selected_nodes))
    onnx.checker.check_model(final_graph)
    onnx.save(final_graph, str(final_qdq))
    deployment_entries = [
        replace(entry, requested_precision="int8", realized_request_precision="int8")
        if str(entry.module_path) in selected_paths
        else entry
        for entry in typed_mapping.entries
    ]
    deployment_mapping = CanonicalPrecisionMappingResult(
        entries=deployment_entries,
        profile_id=profile_id,
        profile_hash=stable_json_hash(
            {"model": model_name, "profile": profile_id, "alpha": alpha, "selected": selected_paths}
        ),
        origin_map_hash=typed_mapping.origin_map_hash,
        policy_version="h800-transformer-modelopt-smoothquant-explicit-qdq-v1",
        auxiliary_layer_precisions=dict(typed_mapping.auxiliary_layer_precisions),
        auxiliary_layer_output_types=dict(typed_mapping.auxiliary_layer_output_types),
    )
    for row in requested:
        if str(row.get("module_path", "")) in selected_paths:
            row["requested_precision"] = "INT8"
    _write_json(destination / "deployment_precision_mapping.json", deployment_mapping.to_dict())
    _write_json(destination / "requested_precision_contract.json", requested)
    _write_json(destination / "typed_graph_report.json", typed_report)
    _write_json(destination / "qdq_adjacency_rewrite.json", adjacency_rewrite)
    _write_json(destination / "qdq_adjacency_audit.json", adjacency)
    return resume_after_export(
        output_root=output_root,
        model_name=model_name,
        profile_id=profile_id,
        alpha=alpha,
        physical_gpu=physical_gpu,
        plugin_path=plugin_path,
        destination_section=destination_section,
    )


def resume_after_export(
    *, output_root: Path, model_name: str, profile_id: str, alpha: float,
    physical_gpu: int, plugin_path: Path, destination_section: str = "smoothquant"
) -> dict[str, Any]:
    """Resume a fresh run after calibration/export without recalibrating it."""

    destination = output_root / destination_section / model_name / profile_id
    final_qdq = destination / "strongly_typed_explicit_qdq.onnx"
    mapping_path = destination / "deployment_precision_mapping.json"
    requested_path = destination / "requested_precision_contract.json"
    calibration_path = destination / "calibration_report.json"
    required = (final_qdq, mapping_path, requested_path, calibration_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"smoothquant_resume_artifact_missing:{missing}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if int(calibration.get("sample_count", 0)) != 200:
        raise RuntimeError("smoothquant_resume_calibration_not_train200")
    requested = json.loads(requested_path.read_text(encoding="utf-8"))
    selected_nodes = {
        str(row["onnx_node"])
        for row in requested
        if str(row.get("requested_precision", "")).upper() == "INT8"
    }
    graph = onnx.load(str(final_qdq), load_external_data=False)
    adjacency = validate_projection_qdq_adjacency(graph, sorted(selected_nodes))
    qdq_rows = _qdq_inventory(final_qdq, requested)
    _write_csv(destination / "qdq_inventory.csv", qdq_rows)
    if not qdq_rows or any(not bool(row["scale_finite_nonzero"]) for row in qdq_rows):
        raise RuntimeError("smoothquant_qdq_scale_nonfinite_or_zero")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    inventory_dir = output_root / "inventory" / model_name
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
        policy_version="h800-transformer-trt10.9-strongly-typed-explicit-qdq-v1",
    )
    engine = destination / "engine.plan"
    build = build_engine_modelopt(
        qdq_onnx=final_qdq,
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
    realized_rows = []
    if layer_info.is_file():
        realized_rows = [
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
        _write_csv(destination / "requested_realized.csv", realized_rows)
    conflicts = [row for row in realized_rows if row["conflict"]]
    selected_realized = [
        row for row in realized_rows
        if row["onnx_node"] in selected_nodes and row["realized_precision"] == "INT8"
    ]
    status = str(build.get("status", "engine_build_failed"))
    if status == "ok" and (conflicts or len(selected_realized) != len(selected_nodes)):
        status = "precision_conflict"
    result = {
        "status": status,
        "model": model_name,
        "profile": profile_id,
        "alpha": float(alpha),
        "selected_roles": list(_profile(profile_id).int8_roles),
        "selected_node_count": len(selected_nodes),
        "realized_int8_selected_count": len(selected_realized),
        "requested_realized_conflict_count": len(conflicts),
        "engine_exists": engine.is_file() and engine.stat().st_size > 0,
        "engine_sha256": _sha256(engine) if engine.is_file() else "",
        "final_qdq_sha256": _sha256(final_qdq),
        "calibration_manifest_hash": calibration.get("manifest_hash", ""),
        "physical_structure_frozen": True,
        "resumed_after_fresh_export": True,
        "qdq_adjacency_count": len(adjacency),
    }
    _write_json(destination / "baseline_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--profile", choices=tuple(row.profile_id for row in smoothquant_profiles()), required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument(
        "--destination-section",
        choices=("smoothquant", "smoothquant_sm90"),
        default="smoothquant",
    )
    parser.add_argument("--resume-after-export", action="store_true")
    parser.add_argument("--finalize-exported", action="store_true")
    args = parser.parse_args(argv)
    if args.resume_after_export and args.finalize_exported:
        raise ValueError("smoothquant_resume_modes_are_mutually_exclusive")
    function = (
        finalize_exported_profile
        if args.finalize_exported
        else resume_after_export
        if args.resume_after_export
        else build_profile
    )
    result = function(
        output_root=Path(args.output_root).resolve(), model_name=args.model,
        profile_id=args.profile, alpha=args.alpha, physical_gpu=args.physical_gpu,
        plugin_path=Path(args.plugin).resolve(), destination_section=args.destination_section,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
