"""Fresh physical V2X-ViT export, calibration and TensorRT deployment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import onnx
import torch

from quantization.config import QDQConfig, TensorRTBuildConfig
from quantization.export.signal_maxk import capture_weighted_module_calls
from quantization.precision.qdq_inserter import insert_explicit_qdq
from quantization.precision.typed_graph import apply_strongly_typed_precision_contract
from quantization.types import stable_json_hash
from search.model_families.transformer.model_inventory import build_model_inventory
from search.model_families.transformer.realized_precision import audit_realized_precision
from search.model_family.deployment import (
    build_physical_structure_snapshot_v2,
    canonicalize_v2xvit_onnx,
    collect_v2xvit_train200_entropy_scales,
    file_sha256,
)
from search.model_family.export.heal_v2xvit import (
    HealV2XViTExportPolicy,
    build_heal_v2xvit_export_module,
    prepare_v2xvit_fixed_k_inputs,
)
from search.integration.runtime_environment import ensure_modelopt_source_available
from search.stage2.trt_modelopt import build_engine_modelopt
from search.stage2.v2xvit_deployment_closed import (
    audit_deployment_closed_profile,
    bind_train200_calibration_identity,
    build_deployment_closed_precision_mapping,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row}) or ["status"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_build_candidate(
    *,
    candidate_dir: str | Path,
    model: torch.nn.Module,
    adapter: Any,
    hypes: Mapping[str, Any],
    real_batch: Mapping[str, Any],
    module_precision_profile: Mapping[str, str],
    candidate_identity: Mapping[str, Any],
    train200_manifest: Mapping[str, Any],
    train200_manifest_path: str | Path,
    checkpoint_path: str | Path,
    physical_report: Mapping[str, Any],
    fixed_k: int,
    physical_gpu: int,
    tensorrt_root: str | Path,
    plugin_path: str | Path,
    build_engine: bool = True,
    functional_contract: str = "F3",
) -> dict[str, Any]:
    """Build one immutable candidate; no calibration or engine cache is reused."""

    output = Path(candidate_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    modelopt_source = ensure_modelopt_source_available(tensorrt_root)
    modelopt_init = modelopt_source / "modelopt" / "__init__.py"
    _write_json(
        output / "modelopt_source_manifest.json",
        {
            "source_root": str(modelopt_source),
            "package_init": str(modelopt_init),
            "package_init_sha256": file_sha256(modelopt_init),
            "source_kind": "vendored_nvidia_modelopt_0_29",
            "sys_path_explicit": True,
        },
    )
    device = next(model.parameters()).device
    model.eval()
    policy = HealV2XViTExportPolicy(fixed_k=int(fixed_k), max_agents=2)
    wrapper = build_heal_v2xvit_export_module(model, policy=policy).eval()
    prepared = prepare_v2xvit_fixed_k_inputs(real_batch["ego"], policy=policy)
    input_names = tuple(prepared)
    inputs = tuple(prepared[name] for name in input_names)
    with torch.inference_mode():
        reference = adapter.forward_for_task(model, real_batch)
        observed = wrapper(*inputs)
    parity_rows = {}
    for name, actual in zip(policy.output_names, observed):
        expected = reference[name]
        delta = (expected.float() - actual.float()).abs()
        parity_rows[name] = {
            "shape_equal": list(expected.shape) == list(actual.shape),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "allclose": bool(torch.allclose(expected.float(), actual.float(), atol=5.0e-4, rtol=1.0e-4)),
        }
    if not all(row["allclose"] for row in parity_rows.values()):
        raise RuntimeError(f"v2xvit_candidate_wrapper_parity_failed:{parity_rows}")
    _write_json(output / "wrapper_parity.json", parity_rows)

    base = output / "physical_fp32_canonical.onnx"
    with capture_weighted_module_calls(wrapper) as calls:
        torch.onnx.export(
            wrapper,
            inputs,
            str(base),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(input_names),
            output_names=list(policy.output_names),
            custom_opsets={"trt": 1},
        )
    origin = canonicalize_v2xvit_onnx(base, calls, output_path=base)
    graph = onnx.load(str(base), load_external_data=False)
    onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=False)
    onnx.checker.check_model(inferred)
    inventory = build_model_inventory(
        model_family="lidar_v2xvit",
        model=model,
        onnx_path=base,
        origin_map=origin,
    )
    graph_nodes = {str(node.name): node for node in graph.graph.node}
    origin_paths = {str(row.module_path) for row in origin.entries}
    profile_paths = {str(path) for path in module_precision_profile}
    fixed_fp32_paths = sorted(
        path
        for path in origin_paths
        if not any(
            path == candidate
            or path.endswith(f".{candidate}")
            or candidate.endswith(f".{path}")
            for candidate in profile_paths
        )
    )
    modules = dict(model.named_modules())
    unexported_contracts = []
    matched_profile_paths = {
        candidate
        for candidate in profile_paths
        if any(
            path == candidate
            or path.endswith(f".{candidate}")
            or candidate.endswith(f".{path}")
            for path in origin_paths
        )
    }
    f3 = str(functional_contract).upper() == "F3"
    expected_functional = {
        "::__qk_matmul__": "FP32",
        "::__softmax_output__": "FP16" if f3 else "FP32",
        "::__av_matmul__": "FP16" if f3 else "FP32",
        "::__attention_residual_add__": "FP16" if f3 else "FP32",
    }
    for path in sorted(profile_paths - matched_profile_paths):
        precision = str(module_precision_profile[path]).upper()
        reason = "dormant_weighted_branch_not_present_in_fixed_export"
        expected = None
        for suffix, value in expected_functional.items():
            if path.endswith(suffix):
                expected = value
                reason = "functional_precision_owned_by_canonical_onnx_role"
                break
        module = modules.get(path)
        if module is not None and not any(True for _ in module.parameters(recurse=False)):
            reason = "parameter_free_activation_precision_owned_by_onnx_role"
            expected = "FP16" if f3 else "FP32"
        if expected is not None and precision != expected:
            if str(functional_contract).upper() == "P32":
                reason = "explicit_P32_functional_contract_override"
            else:
                raise RuntimeError(
                    f"v2xvit_unexported_functional_precision_conflict:{path}:{precision}!={expected}"
                )
        unexported_contracts.append(
            {
                "module_path": path,
                "requested_precision": precision,
                "reason": reason,
                "engine_realization_scope": "not_a_weighted_onnx_call",
            }
        )
    mapping, requested = build_deployment_closed_precision_mapping(
        origin=origin,
        inventory=inventory,
        graph_nodes=graph_nodes,
        module_precision_profile=module_precision_profile,
        profile_id=str(candidate_identity["profile_id"]),
        fixed_fp32_module_paths=fixed_fp32_paths,
        allowed_unexported_profile_paths=sorted(profile_paths - matched_profile_paths),
        functional_contract=functional_contract,
    )
    snapshot = build_physical_structure_snapshot_v2(model, model_family="heal_lidar_v2xvit")
    _write_json(output / "candidate_identity.json", dict(candidate_identity))
    _write_json(output / "physical_report.json", dict(physical_report))
    _write_json(output / "physical_structure_snapshot_v2.json", snapshot)
    _write_json(output / "canonical_origin_map.json", origin.to_dict())
    _write_json(output / "model_inventory.json", inventory)
    _write_json(output / "canonical_precision_mapping.json", mapping.to_dict())
    _write_json(output / "requested_precision_contract.json", requested)
    _write_json(
        output / "fixed_fp32_not_search_gene.json",
        {"module_paths": fixed_fp32_paths, "count": len(fixed_fp32_paths)},
    )
    _write_json(
        output / "unexported_precision_contracts.json",
        {"rows": unexported_contracts, "count": len(unexported_contracts)},
    )

    int8_entries = [entry for entry in mapping.entries if entry.realized_request_precision == "int8"]
    source_for_typed = base
    calibration_identity: dict[str, Any]
    if int8_entries:
        bundle = SimpleNamespace(
            model=model,
            adapter=adapter,
            config_path=Path(str(candidate_identity["config_path"])),
        )
        scales, calibration_metadata = collect_v2xvit_train200_entropy_scales(
            bundle=bundle,
            manifest=train200_manifest,
            mapping=mapping,
            canonical_onnx_path=base,
            device=device,
        )
        calibration_config = {
            "algorithm": "EntropyCalibration2",
            "histogram_bins": 2048,
            "quantized_bins": 128,
            "activation": "static_per_tensor_symmetric_int8",
            "weight": "per_output_channel_symmetric_int8",
        }
        calibration_identity = bind_train200_calibration_identity(
            metadata=calibration_metadata,
            scales=scales,
            train200_manifest=train200_manifest_path,
            checkpoint=checkpoint_path,
            physical_structure_hash=str(physical_report["structure_hash"]),
            state_dict_shape_hash=str(physical_report["state_dict_shape_hash"]),
            precision_map_hash=str(mapping.profile_hash),
            onnx_path=base,
            calibration_config=calibration_config,
        )
        _write_json(output / "train200_calibration_identity.json", calibration_identity)
        _write_json(
            output / "calibration_scales.json",
            {"scales": scales, "scale_hash": calibration_identity["scale_hash"]},
        )
        qdq_path = output / "physical_explicit_qdq.onnx"
        qdq = insert_explicit_qdq(
            base,
            qdq_path,
            mapping,
            scales=scales,
            config=QDQConfig(
                allowed_precisions=("fp32", "fp16", "int8"),
                require_calibration_scales=True,
                insert_activation_input_qdq=True,
                insert_weight_qdq=True,
                insert_activation_output_qdq=False,
                weight_granularity="per_channel",
                merge_policy="fp16_merge",
                explicit_fp16_compute_casts=True,
                explicit_fp32_compute_casts=True,
                policy_version="v2xvit-train200-deployment-closed-v1",
            ),
            calibration_metadata=calibration_identity,
        )
        _write_json(output / "qdq_insertion_report.json", qdq.to_dict())
        source_for_typed = qdq_path
    else:
        calibration_identity = {
            "algorithm": "not_required_no_int8",
            "requested_frames": 0,
            "processed_frames": 0,
            "skipped_frames": 0,
            "physical_hash": str(physical_report["structure_hash"]),
            "state_dict_shape_hash": str(physical_report["state_dict_shape_hash"]),
            "precision_map_hash": str(mapping.profile_hash),
            "onnx_hash": file_sha256(base),
            "fresh_for_exact_candidate": True,
        }
        calibration_identity["cache_hash"] = stable_json_hash(calibration_identity)
        _write_json(output / "train200_calibration_identity.json", calibration_identity)

    typed = output / "physical_strongly_typed.onnx"
    typed_report = apply_strongly_typed_precision_contract(
        source_for_typed,
        typed,
        mapping,
        plugin_boundary="fp16",
    )
    typed_graph = onnx.load(str(typed), load_external_data=False)
    onnx.checker.check_model(typed_graph)
    _write_json(
        output / "typed_graph_report.json",
        {
            **typed_report,
            "checker_passed": True,
            "base_onnx_sha256": file_sha256(base),
            "typed_onnx_sha256": file_sha256(typed),
            "node_count": len(typed_graph.graph.node),
        },
    )
    result = {
        "status": "exported",
        "candidate_dir": str(output),
        "profile_id": str(candidate_identity["profile_id"]),
        "physical_structure_hash": str(physical_report["structure_hash"]),
        "state_dict_shape_hash": str(physical_report["state_dict_shape_hash"]),
        "state_dict_value_hash": _state_hash(model),
        "precision_map_hash": str(mapping.profile_hash),
        "base_onnx_sha256": file_sha256(base),
        "typed_onnx_sha256": file_sha256(typed),
        "calibration_cache_hash": calibration_identity["cache_hash"],
        "int8_weighted_call_count": len(int8_entries),
        "engine_build_attempted": bool(build_engine),
    }
    if not build_engine:
        _write_json(output / "candidate_result.json", result)
        return result

    trt_root = Path(tensorrt_root).resolve()
    trtexec = trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    if not trtexec.is_file():
        trtexec = trt_root / "bin/trtexec"
    plugin = Path(plugin_path).resolve()
    if not plugin.is_file() or not trtexec.is_file():
        raise RuntimeError(f"v2xvit_deployment_dependency_missing:{trtexec}:{plugin}")
    engine = output / "candidate.plan"
    build = build_engine_modelopt(
        qdq_onnx=typed,
        engine_path=engine,
        precision_mapping=mapping,
        build_config=TensorRTBuildConfig(
            trtexec_path=trtexec,
            plugin_path=plugin,
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
            policy_version="v2xvit-deployment-closed-trt10.9-sm89-v1",
        ),
        physical_snapshot=snapshot,
        output_dir=output / "engine_build",
        tensorrt_root=trt_root,
        conda_env="modelopt",
        gpu_id=int(physical_gpu),
    )
    _write_json(output / "engine_build_acceptance.json", build)
    if str(build.get("status")) != "ok" or not engine.is_file():
        result.update(
            {
                "status": "engine_build_failed",
                "engine_build": build,
                "failure_reason": str(build.get("failure_reason", "unknown")),
            }
        )
        _write_json(output / "candidate_result.json", result)
        return result
    layer_info = output / "engine_build/engine_layer_info.json"
    realized = [
        row.to_dict()
        for row in audit_realized_precision(
            model="lidar_v2xvit",
            profile=str(candidate_identity["profile_id"]),
            requested_rows=requested,
            layer_info_path=layer_info,
            typed_onnx_path=typed,
            strongly_typed=True,
        )
    ]
    _write_csv(output / "requested_realized_precision.csv", realized)
    deployment_audit = audit_deployment_closed_profile(
        requested_rows=requested,
        realized_rows=realized,
    )
    _write_json(output / "deployment_closed_precision_audit.json", deployment_audit)
    result.update(
        {
            "status": "ok" if deployment_audit["requested_realized_exact"] else "precision_conflict",
            "engine_path": str(engine.resolve()),
            "engine_sha256": file_sha256(engine),
            "engine_size_bytes": engine.stat().st_size,
            "requested_realized_exact": deployment_audit["requested_realized_exact"],
            "precision_conflict_count": deployment_audit["conflict_count"],
            "engine_build": build,
        }
    )
    _write_json(output / "candidate_result.json", result)
    return result


__all__ = ["export_build_candidate"]
