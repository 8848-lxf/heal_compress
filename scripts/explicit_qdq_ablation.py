#!/usr/bin/env python3
"""Prepare, build, and evaluate controlled explicit-Q/DQ ablations.

The command is split into environment-specific actions. ``prepare`` must run
in univ2x-opt. TensorRT ``build``, ``evaluate``, and ``tensor-parity`` must run
after explicitly activating modelopt with the project CUDA/GCC toolchain.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO.parent, Path("../../HEAL")):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

TRT_ROOT = Path("${TENSORRT_ROOT}")
MODEL_OPT = Path("${CONDA_BASE}/envs/modelopt")
PLUGIN = REPO / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
DEFAULT_SOURCE = REPO / "outputs/int8_baseline_equivalence_audit_20260713_021108"
WEIGHTED_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_env(name: str) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV", "") != name:
        raise RuntimeError(f"action_requires_conda_environment:{name}")


def require_modelopt() -> dict[str, Any]:
    require_env("modelopt")
    expected = MODEL_OPT.resolve()
    checks = {
        "python": Path(sys.executable).resolve(),
        "CONDA_PREFIX": Path(os.environ.get("CONDA_PREFIX", ".")).resolve(),
        "CUDA_HOME": Path(os.environ.get("CUDA_HOME", ".")).resolve(),
        "CC": Path(os.environ.get("CC", ".")).resolve(),
        "CXX": Path(os.environ.get("CXX", ".")).resolve(),
        "CUDACXX": Path(os.environ.get("CUDACXX", ".")).resolve(),
    }
    expected_values = {
        "python": (expected / "bin/python").resolve(),
        "CONDA_PREFIX": expected,
        "CUDA_HOME": expected,
        "CC": (expected / "bin/gcc").resolve(),
        "CXX": (expected / "bin/g++").resolve(),
        "CUDACXX": (expected / "bin/nvcc").resolve(),
    }
    failures = {key: (str(value), str(expected_values[key])) for key, value in checks.items() if value != expected_values[key]}
    if failures:
        raise RuntimeError(f"modelopt_toolchain_isolation_failed:{failures}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "6":
        raise RuntimeError("CUDA_VISIBLE_DEVICES_must_be_fixed_to_6")
    if str(TRT_ROOT / "targets/x86_64-linux-gnu/lib") not in os.environ.get("LD_LIBRARY_PATH", ""):
        raise RuntimeError("TensorRT_root_library_path_not_in_LD_LIBRARY_PATH")

    def version(command: list[str]) -> str:
        return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()

    import tensorrt as trt

    return {
        **{key: str(value) for key, value in checks.items()},
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "TensorRT_root": str(TRT_ROOT),
        "TensorRT_version": trt.__version__,
        "nvcc_version": version([str(expected / "bin/nvcc"), "--version"]),
        "gcc_version": version([str(expected / "bin/gcc"), "--version"]).splitlines()[0],
        "gxx_version": version([str(expected / "bin/g++"), "--version"]).splitlines()[0],
        "plugin": str(PLUGIN),
        "plugin_sha256": sha256_file(PLUGIN),
    }


def mapping_from_dict(payload: dict[str, Any]) -> Any:
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    data = dict(payload)
    data["entries"] = [CanonicalPrecisionEntry(**dict(row)) for row in data.get("entries", [])]
    return CanonicalPrecisionMappingResult(**data)


def coverage_mapping(source_mapping: dict[str, Any], variant: str) -> Any:
    from dataclasses import replace
    from quantization.types import CanonicalPrecisionMappingResult

    base = mapping_from_dict(source_mapping)
    coverage_variants = {
        "LEGACY67_A2_W2",
        "COV_DEBLOCK0_PC",
        "COV_DEBLOCK0_PT",
        "COV_CONV_ONLY",
        "COV_LEGACY67_DEBLOCK_PT",
        "COV_BACKBONE",
        "COV_PYR0",
        "COV_PYR1",
        "COV_PYR2",
        "COV_HEADS",
        "COV_DEBLOCK_ALL",
        "COV_SINGLE_HEAD0",
        "COV_SINGLE_HEAD1",
        "COV_CLS_HEAD",
        "COV_REG_HEAD",
        "COV_DIR_HEAD",
        "COV_SINGLE_HEAD0_A_MERGE",
        "COV_SINGLE_HEAD1_A_MERGE",
        "COV_HEADS_A_MERGE",
        "COV_HEADS_ENTROPY_A_MERGE",
        "COV_LEGACY67_A_MERGE",
        "A3_LEGACY67_ENTROPY_A_MERGE",
    }
    if variant not in coverage_variants:
        return base
    protected = {"encoder_m1.pillar_vfe.pfn_layers.0.linear", "pyramid_backbone.single_head_2"}
    current_int8 = {row.module_path for row in base.entries if row.realized_request_precision == "int8"}
    if variant in {"COV_DEBLOCK0_PC", "COV_DEBLOCK0_PT"}:
        target_int8 = current_int8 | {"pyramid_backbone.deblocks.0.0"}
    elif variant == "COV_BACKBONE":
        target_int8 = current_int8 | {
            row.module_path for row in base.entries if row.module_path.startswith("backbone_m1.")
        }
    elif variant in {"COV_PYR0", "COV_PYR1", "COV_PYR2"}:
        stage = {"COV_PYR0": "layer0", "COV_PYR1": "layer1", "COV_PYR2": "layer2"}[variant]
        target_int8 = current_int8 | {
            row.module_path
            for row in base.entries
            if row.module_path.startswith(f"pyramid_backbone.resnet.{stage}.")
        }
    elif variant == "COV_HEADS":
        target_int8 = current_int8 | {
            row.module_path
            for row in base.entries
            if "head" in row.module_path.lower() and row.module_path not in protected
        }
    elif variant in {
        "COV_SINGLE_HEAD0", "COV_SINGLE_HEAD1", "COV_CLS_HEAD", "COV_REG_HEAD", "COV_DIR_HEAD",
        "COV_SINGLE_HEAD0_A_MERGE", "COV_SINGLE_HEAD1_A_MERGE",
    }:
        module = {
            "COV_SINGLE_HEAD0": "pyramid_backbone.single_head_0",
            "COV_SINGLE_HEAD1": "pyramid_backbone.single_head_1",
            "COV_CLS_HEAD": "cls_head",
            "COV_REG_HEAD": "reg_head",
            "COV_DIR_HEAD": "dir_head",
            "COV_SINGLE_HEAD0_A_MERGE": "pyramid_backbone.single_head_0",
            "COV_SINGLE_HEAD1_A_MERGE": "pyramid_backbone.single_head_1",
        }[variant]
        target_int8 = current_int8 | {module}
    elif variant in {"COV_HEADS_A_MERGE", "COV_HEADS_ENTROPY_A_MERGE"}:
        target_int8 = current_int8 | {
            row.module_path
            for row in base.entries
            if "head" in row.module_path.lower() and row.module_path not in protected
        }
    elif variant == "COV_DEBLOCK_ALL":
        target_int8 = current_int8 | {
            row.module_path for row in base.entries if row.onnx_op_type == "ConvTranspose"
        }
    elif variant == "COV_CONV_ONLY":
        target_int8 = {
            row.module_path
            for row in base.entries
            if row.module_path not in protected and row.onnx_op_type != "ConvTranspose"
        }
    else:
        target_int8 = {row.module_path for row in base.entries if row.module_path not in protected}
    entries = []
    for row in base.entries:
        precision = "int8" if row.module_path in target_int8 else "fp16"
        fp16_merge_boundary = variant.endswith("_A_MERGE") and row.module_path in {
            "pyramid_backbone.single_head_0", "pyramid_backbone.single_head_1"
        }
        entries.append(
            replace(
                row,
                requested_precision=precision,
                realized_request_precision=precision,
                fallback_reason="legacy_coverage_control_protected" if precision == "fp16" else "",
                protected_precision="fp16" if precision == "fp16" else "",
                realized_output_precision="fp16" if fp16_merge_boundary else "",
            )
        )
    return CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id=f"all_keep_coverage_control_{variant.lower()}",
        profile_hash="",
        origin_map_hash=base.origin_map_hash,
        policy_version=f"explicit-qdq-{variant.lower()}-control-v1",
    )


def parse_legacy_cache(path: Path) -> dict[str, float]:
    import struct

    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        if ": " not in line:
            continue
        name, encoded = line.rsplit(": ", 1)
        try:
            value = float(struct.unpack("!f", bytes.fromhex(encoded))[0])
        except (ValueError, struct.error):
            continue
        if np.isfinite(value) and value > 0:
            values[name] = value
    return values


def weight_axis(node: Any) -> int:
    if str(node.op_type) == "Conv":
        return 0
    if str(node.op_type) == "ConvTranspose":
        return 1
    if str(node.op_type) == "MatMul":
        return 1
    if str(node.op_type) == "Gemm":
        trans_b = next((int(attr.i) for attr in node.attribute if str(attr.name) == "transB"), 0)
        return 0 if trans_b else 1
    raise RuntimeError(f"unsupported_per_channel_weight_op:{node.op_type}")


def per_channel_weight_scale(weight: np.ndarray, axis: int) -> list[float]:
    reduce_axes = tuple(index for index in range(weight.ndim) if index != axis)
    amax = np.max(np.abs(weight), axis=reduce_axes)
    if not np.all(np.isfinite(amax)) or np.any(amax <= 0):
        raise RuntimeError("invalid_per_channel_weight_amax")
    return (amax / 127.0).astype(np.float32).tolist()


def make_scales(
    source: Path,
    audit_root: Path,
    mapping: Any,
    variant: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import onnx
    from onnx import numpy_helper

    artifacts = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    base_scales = read_json(artifacts / "calibration_scales.json", {})
    model = onnx.load(str(artifacts / "pruned_fp32.onnx"))
    nodes = {str(node.name): node for node in model.graph.node}
    initializers = {str(row.name): numpy_helper.to_array(row) for row in model.graph.initializer}
    legacy_cache_path = REPO / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK29696_int8_train_calib200.cache"
    legacy = parse_legacy_cache(legacy_cache_path)
    entropy_variant = variant in {
        "A1_W1",
        "COV_HEADS_ENTROPY_A_MERGE",
        "A3_LEGACY67_ENTROPY_A_MERGE",
    }
    entropy_path = audit_root / (
        "entropy_calibration_legacy67_200/entropy_calibration_scales.json"
        if variant in {"COV_HEADS_ENTROPY_A_MERGE", "A3_LEGACY67_ENTROPY_A_MERGE"}
        else "entropy_calibration_200_v3/entropy_calibration_scales.json"
    )
    entropy_payload = read_json(entropy_path, {}) if entropy_variant else {}
    entropy_scales = entropy_payload.get("scales", {})
    result: dict[str, dict[str, Any]] = {}
    exact_activation_matches = 0
    for row in mapping.entries:
        if row.realized_request_precision != "int8":
            continue
        node = nodes[str(row.canonical_node_name)]
        weight = initializers[str(row.weight_initializer)]
        current = base_scales.get(row.module_path, {})
        use_legacy_activation = variant in {
            "A2_W1",
            "LEGACY67_A2_W2",
            "COV_DEBLOCK0_PC",
            "COV_DEBLOCK0_PT",
            "COV_CONV_ONLY",
            "COV_LEGACY67_DEBLOCK_PT",
            "COV_BACKBONE",
            "COV_PYR0",
            "COV_PYR1",
            "COV_PYR2",
            "COV_HEADS",
            "COV_DEBLOCK_ALL",
            "COV_SINGLE_HEAD0",
            "COV_SINGLE_HEAD1",
            "COV_CLS_HEAD",
            "COV_REG_HEAD",
            "COV_DIR_HEAD",
            "COV_SINGLE_HEAD0_A_MERGE",
            "COV_SINGLE_HEAD1_A_MERGE",
            "COV_HEADS_A_MERGE",
            "COV_LEGACY67_A_MERGE",
        }
        if entropy_variant:
            entropy = entropy_scales.get(row.module_path, {})
            if not entropy:
                raise RuntimeError(f"entropy_activation_scale_missing:{row.module_path}:{entropy_path}")
            if str(entropy.get("activation_input_tensor", "")) != str(node.input[0]):
                raise RuntimeError(f"entropy_input_tensor_mismatch:{row.module_path}")
            if str(entropy.get("activation_output_tensor", "")) != str(node.output[0]):
                raise RuntimeError(f"entropy_output_tensor_mismatch:{row.module_path}")
            activation_input = float(entropy["activation_input_scale"])
            activation_output = float(entropy["activation_output_scale"])
        elif use_legacy_activation:
            input_name = str(node.input[0])
            output_name = str(node.output[0])
            if input_name not in legacy or output_name not in legacy:
                raise RuntimeError(f"legacy_activation_scale_exact_match_missing:{row.module_path}:{input_name}:{output_name}")
            activation_input = legacy[input_name]
            activation_output = legacy[output_name]
            exact_activation_matches += 2
        else:
            if not current:
                raise RuntimeError(f"current_activation_scale_missing:{row.module_path}")
            activation_input = float(current["activation_input_scale"])
            activation_output = float(current["activation_output_scale"])
        force_per_tensor = variant == "W0" or (variant in {"COV_DEBLOCK0_PT", "COV_LEGACY67_DEBLOCK_PT"} and str(node.op_type) == "ConvTranspose")
        if force_per_tensor:
            weight_value: float | list[float] = float(np.max(np.abs(weight))) / 127.0
            axis: int | None = None
            granularity = "per_tensor"
        else:
            axis = weight_axis(node)
            weight_value = per_channel_weight_scale(weight, axis)
            granularity = "per_channel"
        result[row.module_path] = {
            "activation_input_scale": activation_input,
            "weight_scale": weight_value,
            "weight_axis": axis,
            "weight_granularity": granularity,
            "weight_scale_shape": [len(weight_value)] if isinstance(weight_value, list) else [],
            "activation_output_scale": activation_output,
            "activation_input_tensor": str(node.input[0]),
            "activation_output_tensor": str(node.output[0]),
            "activation_scale_source": (
                "signal_maxK_wrapper_entropy_KL_2048_to_128"
                if entropy_variant
                else "legacy_cache_exact_tensor_match"
                if use_legacy_activation
                else "current_absmax_task_model_hook"
            ),
            "weight_scale_source": "final_folded_onnx_initializer",
            "insert_activation_output_qdq": not (
                variant.endswith("_A_MERGE")
                and row.module_path in {"pyramid_backbone.single_head_0", "pyramid_backbone.single_head_1"}
            ),
        }
    metadata = {
        "variant": variant,
        "int8_layer_count": len(result),
        "activation_scale_source": (
            "signal_maxK_wrapper_entropy_KL_2048_to_128"
            if entropy_variant
            else "legacy_cache_exact_tensor_match"
            if any(row.get("activation_scale_source") == "legacy_cache_exact_tensor_match" for row in result.values())
            else "current_absmax_task_model_hook"
        ),
        "exact_activation_scale_match_count": exact_activation_matches,
        "weight_granularity": "per_tensor" if variant == "W0" else ("mixed_per_channel_with_convtranspose_per_tensor" if variant in {"COV_DEBLOCK0_PT", "COV_LEGACY67_DEBLOCK_PT"} else "per_channel"),
        "weight_axis_policy": {"Conv": 0, "ConvTranspose": 1, "MatMul": 1, "Gemm": "0 if transB else 1"},
        "legacy_cache": str(legacy_cache_path),
        "legacy_cache_sha256": sha256_file(legacy_cache_path),
        "entropy_calibration": str(entropy_path) if entropy_variant else "",
        "entropy_calibration_sha256": sha256_file(entropy_path) if entropy_variant else "",
    }
    return result, metadata


def prepare_variant(source: Path, root: Path, variant: str) -> None:
    require_env("univ2x-opt")
    import onnx
    from quantization.api import insert_explicit_qdq
    from quantization.config import QDQConfig
    from search.integration.trt_compatible_export import make_pointpillar_domain_compatible

    destination = root / "ablations" / variant
    if destination.exists():
        raise RuntimeError(f"force_rebuild_destination_exists:{destination}")
    destination.mkdir(parents=True)
    artifacts = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    mapping_payload = read_json(artifacts / "canonical_layer_map.json", {})
    mapping = coverage_mapping(mapping_payload, variant)
    scales, scale_metadata = make_scales(source, root, mapping, variant)
    write_json(destination / "precision_mapping.json", mapping.to_dict())
    write_json(destination / "calibration_scales.json", scales)
    qdq = insert_explicit_qdq(
        artifacts / "pruned_fp32.onnx",
        destination / "qdq.onnx",
        mapping,
        scales=scales,
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            weight_granularity=scale_metadata["weight_granularity"],
            merge_policy="fp16_merge",
            grouped_conv_int8_allowed_channels_per_group=(4, 8, 16, 32, 64, 128, 256, 512),
        ),
        calibration_metadata=scale_metadata,
    )
    compatibility = make_pointpillar_domain_compatible(destination / "qdq.onnx", destination / "qdq_trt_compatible.onnx")
    onnx.checker.check_model(onnx.load(str(destination / "qdq.onnx")))
    write_json(destination / "qdq_report.json", qdq.to_dict())
    write_json(destination / "onnx_domain_compatibility_report.json", compatibility)
    write_json(
        destination / "variant_manifest.json",
        {
            **scale_metadata,
            "coverage_control": (
                "legacy_67_int8_3_fp16"
                if variant in {
                    "LEGACY67_A2_W2",
                    "COV_LEGACY67_DEBLOCK_PT",
                    "COV_LEGACY67_A_MERGE",
                    "A3_LEGACY67_ENTROPY_A_MERGE",
                }
                else "incremental_coverage_diagnostic"
                if variant.startswith("COV_")
                else "current_22_int8_48_fp16"
            ),
            "merge_policy": "A_fp16_merge",
            "base_onnx": str(artifacts / "pruned_fp32.onnx"),
            "base_onnx_sha256": sha256_file(artifacts / "pruned_fp32.onnx"),
            "qdq_onnx_sha256": sha256_file(destination / "qdq.onnx"),
            "trt_compatible_onnx_sha256": sha256_file(destination / "qdq_trt_compatible.onnx"),
            "mapping_hash": mapping.mapping_hash,
            "source_calibration_scales_sha256": sha256_file(artifacts / "calibration_scales.json"),
        },
    )


def build_variant(source: Path, root: Path, variant: str) -> None:
    toolchain = require_modelopt()
    from quantization.api import build_trt_engine, validate_precision_realization
    from quantization.config import TensorRTBuildConfig

    destination = root / "ablations" / variant
    engine = destination / "engine.plan"
    layer_info = destination / "engine_layer_info.json"
    if engine.exists() or layer_info.exists():
        raise RuntimeError(f"force_rebuild_engine_destination_exists:{destination}")
    mapping = mapping_from_dict(read_json(destination / "precision_mapping.json", {}))
    old_request = read_json(source / "search_maximal_legal_int8_force_rebuild/artifacts/trt_build_request.json", {})
    config_payload = dict(old_request["build_config"])
    config_payload["policy_version"] = f"explicit-qdq-ablation-{variant.lower()}-v1"
    config_payload["plugin_path"] = str(PLUGIN)
    config = TensorRTBuildConfig.from_dict(config_payload)
    build = build_trt_engine(
        destination / "qdq_trt_compatible.onnx",
        engine,
        mapping,
        config=config,
        layer_info_path=layer_info,
        log_path=destination / "engine_build.log",
        raise_on_failure=False,
    )
    result: dict[str, Any] = {"build": build.to_dict(), "toolchain": toolchain, "status": "ok" if build.success else "engine_build_failed"}
    if build.success:
        precision = validate_precision_realization(layer_info, mapping)
        result["precision_realization"] = precision.to_dict()
        result["status"] = "ok" if precision.passed else "precision_realization_failed"
        result["engine_sha256"] = sha256_file(engine)
        result["layer_info_sha256"] = sha256_file(layer_info)
    write_json(destination / "engine_build_result.json", result)
    if result["status"] != "ok":
        raise RuntimeError(f"engine_build_or_precision_validation_failed:{result['status']}")


def evaluate_variant(source: Path, root: Path, variant: str, num_frames: int) -> None:
    require_modelopt()
    destination = root / "ablations" / variant
    output = destination / f"evaluation_{num_frames}_fixed_manifest.json"
    if output.exists():
        raise RuntimeError(f"force_rebuild_evaluation_exists:{output}")
    template = read_json(source / "search_maximal_legal_int8_force_rebuild/artifacts/evaluation_request.json", {})
    full_manifest = read_json(source / "baseline/eval_manifest.json", {})
    frame_ids = [str(value) for value in full_manifest.get("evaluation_frame_ids", [])[: int(num_frames)]]
    if len(frame_ids) != int(num_frames):
        raise RuntimeError(f"insufficient_source_eval_manifest_frames:{len(frame_ids)}<{num_frames}")
    manifest_path = root / "baseline" / f"eval_manifest_{num_frames}.json"
    payload = {"split": "val", "frame_ids": frame_ids}
    manifest_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    small_manifest = {
        **payload,
        "warmup_frame_ids": [],
        "evaluation_frame_ids": frame_ids,
        "warmup_frames": 0,
        "num_frames": int(num_frames),
        "manifest_hash": manifest_hash,
        "source_full_manifest": str(source / "baseline/eval_manifest.json"),
        "source_full_manifest_hash": full_manifest.get("manifest_hash", ""),
    }
    if manifest_path.is_file():
        if read_json(manifest_path, {}) != small_manifest:
            raise RuntimeError(f"small_eval_manifest_conflict:{manifest_path}")
    else:
        write_json(manifest_path, small_manifest)
    engine_path = destination / "engine.plan"
    reuse_proof: dict[str, Any] | None = None
    if not engine_path.is_file() and variant == "W0":
        source_qdq = source / "search_maximal_legal_int8_force_rebuild/artifacts/qdq.onnx"
        source_engine = source / "search_maximal_legal_int8_force_rebuild/artifacts/engine.plan"
        if sha256_file(destination / "qdq.onnx") != sha256_file(source_qdq):
            raise RuntimeError("W0_source_engine_reuse_qdq_hash_mismatch")
        engine_path = source_engine
        reuse_proof = {
            "reason": "freshly regenerated W0 Q/DQ is byte-identical to the prior force-rebuilt control Q/DQ",
            "regenerated_qdq_sha256": sha256_file(destination / "qdq.onnx"),
            "source_qdq_sha256": sha256_file(source_qdq),
            "source_engine": str(source_engine),
            "source_engine_sha256": sha256_file(source_engine),
        }
    elif not engine_path.is_file() and variant == "W2":
        w1 = root / "ablations/W1"
        if sha256_file(destination / "qdq.onnx") != sha256_file(w1 / "qdq.onnx"):
            raise RuntimeError("W2_W1_engine_reuse_qdq_hash_mismatch")
        if read_json(destination / "precision_mapping.json", {}) != read_json(w1 / "precision_mapping.json", {}):
            raise RuntimeError("W2_W1_engine_reuse_mapping_mismatch")
        engine_path = w1 / "engine.plan"
        reuse_proof = {
            "reason": "W1 and W2 are byte-identical because current 22-layer topology contains only Conv weights",
            "W1_qdq_sha256": sha256_file(w1 / "qdq.onnx"),
            "W2_qdq_sha256": sha256_file(destination / "qdq.onnx"),
            "W1_engine": str(engine_path),
            "W1_engine_sha256": sha256_file(engine_path),
        }
    if not engine_path.is_file():
        raise RuntimeError(f"variant_engine_missing:{variant}:{engine_path}")
    request = {
        **template,
        "engine_path": str(engine_path),
        "plugin_path": str(PLUGIN),
        "num_frames": int(num_frames),
        "warmup_frames": 0,
        "latency_rounds": 1,
        "device": "cuda:0",
        "physical_device": "cuda:6",
        "eval_manifest_path": str(manifest_path),
        "output_path": str(output),
    }
    request_path = destination / f"evaluation_request_{num_frames}_fixed_manifest.json"
    write_json(request_path, request)
    if reuse_proof is not None:
        write_json(destination / f"evaluation_engine_reuse_proof_{num_frames}.json", reuse_proof)
    from search.integration.evaluation_worker import main as evaluation_main

    returncode = evaluation_main(["--request", str(request_path)])
    if returncode != 0 or not output.is_file() or read_json(output, {}).get("status") != "ok":
        raise RuntimeError(f"evaluation_failed:{variant}:{num_frames}:{returncode}")


def tensor_thresholds(qdq_path: Path, target_map: dict[str, str]) -> dict[str, float | None]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(qdq_path))
    initializers = {str(row.name): numpy_helper.to_array(row) for row in model.graph.initializer}
    result: dict[str, float | None] = {key: None for key in target_map}
    for alias, target in target_map.items():
        normalized = target.replace("__before_output_qdq", "")
        node = next(
            (
                row
                for row in model.graph.node
                if str(row.op_type) == "QuantizeLinear"
                and str(row.input[0]).replace("__before_output_qdq", "") == normalized
            ),
            None,
        )
        if node is not None:
            scale = np.asarray(initializers[str(node.input[1])])
            if scale.size == 1:
                result[alias] = float(scale.reshape(-1)[0]) * 127.0
    return result


def tensor_parity(source: Path, root: Path, variant: str) -> None:
    require_modelopt()
    import torch
    from quantization.api import build_trt_engine
    from quantization.config import TensorRTBuildConfig
    from scripts.int8_equivalence_tensor_parity import Accumulator, augment_outputs, ort_session
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    destination = root / "ablations" / variant
    debug = destination / "tensor_parity_10"
    if debug.exists():
        raise RuntimeError(f"force_rebuild_tensor_parity_exists:{debug}")
    debug.mkdir()
    source_work = source / "tensor_parity_work_v2"
    shutil.copyfile(source_work / "ten_frame_inputs.json", debug / "ten_frame_inputs.json")
    augmented = debug / "augmented.onnx"
    target_map = augment_outputs(destination / "qdq_trt_compatible.onnx", augmented)
    mapping = mapping_from_dict(read_json(destination / "precision_mapping.json", {}))
    old_request = read_json(source / "search_maximal_legal_int8_force_rebuild/artifacts/trt_build_request.json", {})
    config_payload = dict(old_request["build_config"])
    config_payload["policy_version"] = f"explicit-qdq-ablation-debug-{variant.lower()}-v1"
    config_payload["plugin_path"] = str(PLUGIN)
    build = build_trt_engine(
        augmented,
        debug / "augmented.engine",
        mapping,
        config=TensorRTBuildConfig.from_dict(config_payload),
        layer_info_path=debug / "augmented_layer_info.json",
        log_path=debug / "augmented_build.log",
        raise_on_failure=False,
    )
    if not build.success:
        write_json(debug / "result.json", {"status": "build_failed", "build": build.to_dict()})
        raise RuntimeError(f"debug_engine_build_failed:{variant}")
    ctypes.CDLL(str(PLUGIN.resolve()), mode=ctypes.RTLD_GLOBAL)
    session = ort_session(source_work / "ort_fp32_augmented.onnx")
    ort_map = read_json(source_work / "target_map.json", {})["ort_fp32"]
    runner = TensorRTEngineRunner(debug / "augmented.engine", torch.device("cuda:0"))
    thresholds = tensor_thresholds(destination / "qdq.onnx", target_map)
    accumulators = {alias: Accumulator() for alias in target_map if alias in ort_map}
    per_frame = []
    for frame in read_json(debug / "ten_frame_inputs.json", {}).get("frames", []):
        values = np.load(frame["path"])
        feeds = {name: values[name] for name in values.files}
        ort_values = session.run(None, feeds)
        ort_outputs = {row.name: value for row, value in zip(session.get_outputs(), ort_values)}
        outputs = runner.run({name: torch.as_tensor(value, device="cuda:0") for name, value in feeds.items()})
        frame_status = {}
        for alias, engine_name in target_map.items():
            reference_name = ort_map.get(alias)
            if not reference_name or reference_name not in ort_outputs or engine_name not in outputs:
                frame_status[alias] = {"missing": True, "reference": reference_name, "candidate": engine_name}
                continue
            reference = np.asarray(ort_outputs[reference_name])
            candidate = outputs[engine_name].detach().float().cpu().numpy()
            accumulators[alias].update(reference, candidate, thresholds.get(alias))
            frame_status[alias] = {"reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
        per_frame.append({"frame_id": frame["frame_id"], "tensors": frame_status})
    write_json(
        debug / "result.json",
        {
            "status": "ok",
            "variant": variant,
            "diagnostic_engine_warning": "extra outputs change fusion; official AP uses the unmodified engine",
            "frame_count": len(per_frame),
            "engine_sha256": sha256_file(debug / "augmented.engine"),
            "target_map": target_map,
            "saturation_thresholds": thresholds,
            "metrics": {alias: value.result() for alias, value in accumulators.items()},
            "frames": per_frame,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--action", choices=("prepare", "build", "evaluate", "tensor-parity"), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "W0", "W1", "W2", "A1_W1", "A2_W1", "LEGACY67_A2_W2",
            "COV_DEBLOCK0_PC", "COV_DEBLOCK0_PT", "COV_CONV_ONLY", "COV_LEGACY67_DEBLOCK_PT",
            "COV_BACKBONE", "COV_PYR0", "COV_PYR1", "COV_PYR2", "COV_HEADS", "COV_DEBLOCK_ALL",
            "COV_SINGLE_HEAD0", "COV_SINGLE_HEAD1", "COV_CLS_HEAD", "COV_REG_HEAD", "COV_DIR_HEAD",
            "COV_SINGLE_HEAD0_A_MERGE", "COV_SINGLE_HEAD1_A_MERGE", "COV_HEADS_A_MERGE", "COV_LEGACY67_A_MERGE",
            "COV_HEADS_ENTROPY_A_MERGE",
            "A3_LEGACY67_ENTROPY_A_MERGE",
        ),
        required=True,
    )
    parser.add_argument("--num-frames", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_audit_root.resolve()
    root = args.audit_root.resolve()
    if args.action == "prepare":
        prepare_variant(source, root, args.variant)
    elif args.action == "build":
        build_variant(source, root, args.variant)
    elif args.action == "evaluate":
        evaluate_variant(source, root, args.variant, args.num_frames)
    else:
        tensor_parity(source, root, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
