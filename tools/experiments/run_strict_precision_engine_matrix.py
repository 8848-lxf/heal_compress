"""Thin resumable orchestrator for the formal strict TensorRT engine matrix."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path
from typing import Any

from quantization.api import (
    build_canonical_precision_mapping,
    build_trt_command,
    build_trt_engine,
    generate_precision_profile,
    insert_explicit_qdq,
    load_trt_engine,
    prepare_signal_maxk_inputs,
    run_engine_smoke,
    validate_engine_provenance,
    validate_engine_structure,
    validate_precision_realization,
    validate_qdq_against_physical_snapshot,
)
from quantization.artifacts.io import atomic_write_json, file_sha256, load_json
from quantization.config import PrecisionProfileConfig, QDQConfig, TensorRTBuildConfig
from quantization.types import (
    CanonicalMappingEntry,
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
    OnnxOriginMapResult,
)
from tools.experiments.run_lidar_pyramid_formal_pruner_validation import _append_process, _batch, _load_dataset


MODEL_IDS = ("original", "prune_0.1", "prune_0.2", "prune_0.3", "prune_0.4", "prune_0.5", "prune_0.6", "prune_0.7")
PRECISION_MODES = ("strict_fp32", "strict_fp16", "strict_int8")


def _origin_map(payload: dict[str, Any]) -> OnnxOriginMapResult:
    rows = []
    for value in payload["entries"]:
        row = dict(value)
        row["weight_shape"] = tuple(row.get("weight_shape", ()))
        row["root_trace"] = tuple(row.get("root_trace", ()))
        rows.append(CanonicalMappingEntry(**row))
    return OnnxOriginMapResult(
        entries=rows,
        source_onnx=payload.get("source_onnx", ""),
        unresolved_weighted_nodes=list(payload.get("unresolved_weighted_nodes", [])),
        functional_matmul_nodes=list(payload.get("functional_matmul_nodes", [])),
        naming_policy_version=payload.get("naming_policy_version", "canonical-v2-trt-safe-max68-sha256"),
        schema_version=payload.get("schema_version", "onnx-origin-map-v1"),
        origin_map_hash=payload.get("origin_map_hash", ""),
    )


def _mapping(payload: dict[str, Any]) -> CanonicalPrecisionMappingResult:
    return CanonicalPrecisionMappingResult(
        entries=[CanonicalPrecisionEntry(**row) for row in payload["entries"]],
        profile_id=payload.get("profile_id", ""),
        profile_hash=payload.get("profile_hash", ""),
        origin_map_hash=payload.get("origin_map_hash", ""),
        policy_version=payload.get("policy_version", "canonical-precision-mapping-v1"),
        mapping_hash=payload.get("mapping_hash", ""),
        schema_version=payload.get("schema_version", "canonical-precision-mapping-v1"),
    )


def _snapshot_paths(output: Path, model_id: str) -> tuple[Path, Path]:
    if model_id == "original":
        return output / "original_physical_structure_snapshot_v2.json", output / "original_physical_hash_v2.json"
    ratio = model_id.removeprefix("prune_0.")
    root = output / "pruning_sweep" / f"ratio_0{ratio}"
    return root / "physical_structure_snapshot_v2.json", root / "physical_hash_v2.json"


def _shapes() -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        "voxel_features": {kind: (29696, 32, 4) for kind in ("min", "opt", "max")},
        "voxel_coords": {kind: (29696, 4) for kind in ("min", "opt", "max")},
        "voxel_num_points": {kind: (29696,) for kind in ("min", "opt", "max")},
        "pairwise_t_matrix": {"min": (1, 1, 1, 4, 4), "opt": (1, 2, 2, 4, 4), "max": (1, 2, 2, 4, 4)},
        "valid_voxel_mask": {kind: (29696,) for kind in ("min", "opt", "max")},
    }


def _parser_preflight(onnx_path: Path, plugin_path: Path) -> dict[str, Any]:
    import tensorrt as trt

    ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    passed = bool(parser.parse(onnx_path.read_bytes()))
    return {
        "passed": passed,
        "errors": [str(parser.get_error(index)) for index in range(parser.num_errors)],
        "network_layer_count": int(network.num_layers),
        "input_count": int(network.num_inputs),
        "output_count": int(network.num_outputs),
    }


def _target_ratio(model_id: str) -> float:
    return 0.0 if model_id == "original" else float(model_id.removeprefix("prune_"))


def _run_one(
    *,
    output: Path,
    model_id: str,
    mode: str,
    trtexec: Path,
    plugin: Path,
    trt_version: str,
    smoke_inputs: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    model_root = output / "engines" / model_id
    mode_root = model_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    base_onnx = model_root / "base.onnx"
    snapshot_path, hashes_path = _snapshot_paths(output, model_id)
    snapshot = load_json(snapshot_path)
    physical_hashes = load_json(hashes_path)
    export_report = load_json(model_root / "onnx_export_report.json")
    origin = _origin_map(export_report["origin_map"])
    row: dict[str, Any] = {
        "model_id": model_id,
        "target_prune_ratio": _target_ratio(model_id),
        "actual_prune_ratio": 1.0 - float(snapshot["parameter_count"]) / float(load_json(output / "original_physical_structure_snapshot_v2.json")["parameter_count"]),
        "physical_structure_hash": physical_hashes["structure_hash_v2"],
        "precision_mode": mode,
        "requested_weighted_layer_count": len(origin.entries),
        "preflight_pass": False,
        "build_success": False,
        "engine_structure_pass": False,
        "precision_realization_pass": False,
        "provenance_pass": False,
        "smoke_pass": False,
        "failure_stage": "",
        "failure_reason": "",
    }
    try:
        descriptors = [entry.to_dict() for entry in origin.entries]
        profile = generate_precision_profile(descriptors, profile_id=mode, config=PrecisionProfileConfig())
        mapping = build_canonical_precision_mapping(origin, profile, config=QDQConfig())
        atomic_write_json(mode_root / "precision_profile.json", profile.to_dict())
        atomic_write_json(mode_root / "canonical_precision_mapping.json", mapping.to_dict())
        row["requested_int8_count"] = int(profile.requested_int8_count)
        row["canonical_mapping_hash"] = mapping.mapping_hash
        row["precision_profile_hash"] = profile.profile_hash
        row["explicit_exemption_count"] = sum(entry.requested_precision != "int8" for entry in mapping.entries) if mode == "strict_int8" else 0
        row["mapping_fallback_count"] = sum(entry.requested_precision != entry.realized_request_precision for entry in mapping.entries)
        source_onnx = base_onnx
        qdq_validation_passed = True
        qdq_coverage_passed = True
        if mode == "strict_int8":
            calibration = load_json(model_root / "calibration_scales.json")
            scales = {
                record["module_path"]: {
                    key: record[key]
                    for key in ("activation_input_scale", "weight_scale", "activation_output_scale")
                }
                for record in calibration["records"]
            }
            source_onnx = mode_root / "model.qdq.onnx"
            insertion = insert_explicit_qdq(
                base_onnx,
                source_onnx,
                mapping,
                scales=scales,
                config=QDQConfig(),
                calibration_metadata={
                    "split": calibration["split"],
                    "frame_count": calibration["frame_count"],
                    "frame_list_hash": calibration["frame_list_hash"],
                    "scale_method": calibration["scale_method"],
                    "zero_point_policy": "symmetric_zero_point_0",
                },
            )
            atomic_write_json(mode_root / "qdq_insertion_report.json", insertion.to_dict())
            qdq_validation = validate_qdq_against_physical_snapshot(source_onnx, snapshot, origin)
            atomic_write_json(mode_root / "qdq_physical_validation.json", qdq_validation.to_dict())
            qdq_validation_passed = bool(qdq_validation.passed)
            qdq_coverage_passed = insertion.inserted_layer_count == profile.requested_int8_count
            row["qdq_inserted_layer_count"] = int(insertion.inserted_layer_count)
        parser = _parser_preflight(source_onnx, plugin)
        enable_fp16 = mode in {"strict_fp16", "strict_int8"}
        enable_int8 = mode == "strict_int8"
        policy_version = {
            "strict_fp32": "strict-fp32-obey-v1",
            "strict_fp16": "strict-fp16-obey-v1",
            "strict_int8": "strict-int8-explicit-qdq-obey-v1",
        }[mode]
        config = TensorRTBuildConfig(
            trtexec_path=trtexec,
            plugin_path=plugin,
            workspace_mib=4096,
            shape_profiles=_shapes(),
            timeout_seconds=1800,
            enable_fp16=enable_fp16,
            enable_int8=enable_int8,
            no_tf32=True,
            skip_inference=True,
            export_layer_info=True,
            policy_version=policy_version,
        )
        engine_path = mode_root / "model.engine"
        layer_info_path = mode_root / "layer_info.json"
        command = build_trt_command(source_onnx, engine_path, mapping, config=config, layer_info_path=layer_info_path)
        atomic_write_json(mode_root / "build_command.json", command.to_dict())
        preflight = {
            "passed": bool(
                export_report["validation"]["passed"]
                and parser["passed"]
                and qdq_validation_passed
                and qdq_coverage_passed
                and row["mapping_fallback_count"] == 0
            ),
            "live_model_state_snapshot_validation_source": str(snapshot_path),
            "base_onnx_physical_validation_passed": bool(export_report["validation"]["passed"]),
            "qdq_physical_validation_passed": qdq_validation_passed,
            "qdq_coverage_passed": qdq_coverage_passed,
            "parser": parser,
            "build_command": command.command,
            "schema_version": "strict-engine-preflight-v1",
        }
        atomic_write_json(mode_root / "preflight.json", preflight)
        row["preflight_pass"] = preflight["passed"]
        if not preflight["passed"]:
            row["failure_stage"] = "preflight"
            row["failure_reason"] = json.dumps(preflight, sort_keys=True)
            return row
        existing = load_json(mode_root / "build_report.json") if resume and (mode_root / "build_report.json").is_file() else {}
        if existing.get("success") and engine_path.is_file() and layer_info_path.is_file():
            build_payload = existing
        else:
            build = build_trt_engine(
                source_onnx,
                engine_path,
                mapping,
                config=config,
                layer_info_path=layer_info_path,
                log_path=mode_root / "build.log",
                raise_on_failure=False,
            )
            build_payload = build.to_dict()
            atomic_write_json(mode_root / "build_report.json", build_payload)
        row["build_success"] = bool(build_payload.get("success"))
        row["build_time_seconds"] = float(build_payload.get("elapsed_seconds", 0.0))
        if not row["build_success"]:
            row["failure_stage"] = "build"
            row["failure_reason"] = str(build_payload.get("failure_reason", "build failed"))
            return row
        structure = validate_engine_structure(layer_info_path, mapping, physical_snapshot=snapshot)
        precision = validate_precision_realization(layer_info_path, mapping)
        provenance_payload = {
            "physical_structure_hash": physical_hashes["structure_hash_v2"],
            "precision_profile_hash": profile.profile_hash,
            "canonical_mapping_hash": mapping.mapping_hash,
            "base_onnx_hash": file_sha256(base_onnx),
            "qdq_onnx_hash": file_sha256(source_onnx),
            "engine_hash": file_sha256(engine_path),
            "plugin_hash": file_sha256(plugin),
            "tensorrt_version": trt_version,
            "build_policy_version": policy_version,
        }
        provenance = validate_engine_provenance(provenance_payload)
        atomic_write_json(mode_root / "engine_structure_validation.json", structure.to_dict())
        atomic_write_json(mode_root / "precision_realization_validation.json", precision.to_dict())
        atomic_write_json(mode_root / "provenance.json", provenance_payload)
        atomic_write_json(mode_root / "provenance_validation.json", provenance.to_dict())
        handle = load_trt_engine(engine_path, plugin_path=plugin)
        smoke = run_engine_smoke(handle, smoke_inputs, plugin_path=plugin, device="cuda:0")
        atomic_write_json(mode_root / "smoke.json", smoke.to_dict())
        row.update(
            {
                "base_onnx_hash": provenance_payload["base_onnx_hash"],
                "qdq_onnx_hash": provenance_payload["qdq_onnx_hash"],
                "engine_hash": provenance_payload["engine_hash"],
                "realized_int8_count": int(precision.realized_int8_count),
                "realized_fp16_count": int(precision.realized_fp16_count),
                "precision_coverage": (len(mapping.entries) - len(precision.mismatches)) / len(mapping.entries),
                "engine_structure_pass": bool(structure.passed),
                "precision_realization_pass": bool(precision.passed),
                "provenance_pass": bool(provenance.passed),
                "smoke_pass": bool(smoke.success),
                "engine_size_bytes": engine_path.stat().st_size,
            }
        )
        if not structure.passed:
            row["failure_stage"] = "postbuild_structure"
            row["failure_reason"] = json.dumps(structure.to_dict(), sort_keys=True)
        elif not precision.passed:
            row["failure_stage"] = "postbuild_precision"
            row["failure_reason"] = json.dumps(precision.to_dict(), sort_keys=True)
        elif not provenance.passed:
            row["failure_stage"] = "postbuild_provenance"
            row["failure_reason"] = json.dumps(provenance.to_dict(), sort_keys=True)
        elif not smoke.success:
            row["failure_stage"] = "smoke"
            row["failure_reason"] = smoke.failure_reason
        row["engine_validation_passed"] = not bool(row["failure_stage"])
        atomic_write_json(mode_root / "postbuild_summary.json", row)
        return row
    except Exception as exc:
        if not row["failure_stage"]:
            row["failure_stage"] = "orchestration_exception"
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(mode_root / "failure.json", row)
        return row
    finally:
        row["orchestration_wall_seconds"] = time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heal-root", required=True)
    parser.add_argument("--trtexec", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--trt-version", required=True)
    parser.add_argument("--models", default=",".join(MODEL_IDS))
    parser.add_argument("--modes", default=",".join(PRECISION_MODES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    plugin = Path(args.plugin).resolve()
    selected_models = [value for value in args.models.split(",") if value]
    selected_modes = [value for value in args.modes.split(",") if value]
    invalid_models = sorted(set(selected_models) - set(MODEL_IDS))
    invalid_modes = sorted(set(selected_modes) - set(PRECISION_MODES))
    if invalid_models or invalid_modes:
        raise SystemExit(f"invalid model/mode selection: models={invalid_models}, modes={invalid_modes}")
    dataset, _ = _load_dataset(Path(args.config), Path(args.heal_root), train=False)
    smoke_inputs = prepare_signal_maxk_inputs(_batch(dataset, 0, "cuda:0", train=False)["ego"])
    started = time.time()
    rows = []
    for model_id in selected_models:
        for mode in selected_modes:
            row = _run_one(
                output=output,
                model_id=model_id,
                mode=mode,
                trtexec=Path(args.trtexec).resolve(),
                plugin=plugin,
                trt_version=args.trt_version,
                smoke_inputs=smoke_inputs,
                resume=bool(args.resume),
            )
            rows.append(row)
            print(json.dumps({key: row.get(key) for key in ("model_id", "precision_mode", "build_success", "engine_validation_passed", "failure_stage", "failure_reason")}), flush=True)
    existing_path = output / "engines" / "engine_build_matrix.json"
    existing = load_json(existing_path).get("engines", []) if existing_path.is_file() else []
    keyed = {(row["model_id"], row["precision_mode"]): row for row in existing}
    keyed.update({(row["model_id"], row["precision_mode"]): row for row in rows})
    matrix = {
        "engines": [keyed[key] for key in sorted(keyed)],
        "target_engine_count": 24,
        "completed_engine_count": len(keyed),
        "strict_valid_engine_count": sum(bool(row.get("engine_validation_passed")) for row in keyed.values()),
        "schema_version": "strict-precision-engine-matrix-v1",
    }
    atomic_write_json(existing_path, matrix)
    _append_process(
        output,
        {
            "stage": "strict_engine_matrix",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "started_unix": started,
            "ended_unix": time.time(),
            "status": "completed",
            "background": False,
            "models": selected_models,
            "modes": selected_modes,
        },
    )
    return 0 if all(not row.get("failure_stage") for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
