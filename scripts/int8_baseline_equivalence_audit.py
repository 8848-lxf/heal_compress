#!/usr/bin/env python3
"""Reproducible all-keep legacy/search INT8 baseline equivalence audit.

The driver deliberately writes only below a caller-provided audit directory.
Legacy artifacts are always treated as read-only inputs.  Heavy work is split
into phases so the FP16 equivalence gate can stop the INT8 build early.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CHECKPOINT = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
CONFIG = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
MODEL_OPT = Path("/home/lixingfeng/miniconda3/envs/modelopt")
CURRENT_PLUGIN = REPO / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"

LEGACY_ROOT = REPO / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare"
LEGACY_BASE_ONNX = LEGACY_ROOT / "artifacts/onnx/fixedK29696/dynamic_agent_single_engine_maxK/lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
LEGACY_FP16_ENGINE = LEGACY_ROOT / "artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp16/lidar_pyramid_dynamic_agent_single_engine_maxK_fp16.engine"
LEGACY_INT8_ENGINE = LEGACY_ROOT / "artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/int8_train_calib200/lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib200.engine"
LEGACY_FP16_LAYERINFO = LEGACY_ROOT / "artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp16/layerinfo_dynamic_agent_single_engine_maxK_fp16.json"
LEGACY_INT8_LAYERINFO = LEGACY_ROOT / "artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/int8_train_calib200/layerinfo_dynamic_agent_single_engine_maxK_int8_train_calib200.json"
LEGACY_CALIBRATION_CACHE = LEGACY_ROOT / "artifacts/calibration/lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK29696_int8_train_calib200.cache"
LEGACY_CALIBRATION_MANIFEST = LEGACY_ROOT / "artifacts/calibration/train_calib_single_engine_maxK29696_200/manifest.json"
LEGACY_PLUGIN = LEGACY_ROOT / "artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so"
LEGACY_FP16_HISTORICAL_EVAL = LEGACY_ROOT / "evaluation/full_val_fixedK29696_trainCalib_new_server/single_engine_maxK_fp16_full_val.json"
LEGACY_INT8_HISTORICAL_EVAL = LEGACY_ROOT / "evaluation/full_val_fixedK29696_trainCalib_new_server/single_engine_maxK_int8_train_calib200_full_val.json"
LEGACY_BUILD_REPORT = LEGACY_ROOT / "benchmark/dynamic_single_engine_maxK_build_report.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_json(path: str | Path, default: Any = None) -> Any:
    candidate = Path(path)
    if not candidate.is_file():
        return default
    return json.loads(candidate.read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def artifact(path: str | Path, *, role: str, dependencies: Iterable[str | Path] = ()) -> dict[str, Any]:
    candidate = Path(path)
    exists = candidate.is_file()
    dependency_rows = []
    for dependency in dependencies:
        dep = Path(dependency)
        dependency_rows.append(
            {
                "path": str(dep),
                "exists": dep.is_file(),
                "sha256": sha256_file(dep) if dep.is_file() else "",
            }
        )
    return {
        "role": role,
        "path": str(candidate),
        "exists": exists,
        "size": candidate.stat().st_size if exists else 0,
        "sha256": sha256_file(candidate) if exists else "",
        "read_only_source": str(candidate).startswith(str(LEGACY_ROOT)),
        "dependencies": dependency_rows,
        "dependency_hash": json_hash(dependency_rows),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compare_state_dicts(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    rows = []
    all_keys = sorted(set(reference) | set(candidate))
    for key in all_keys:
        left = reference.get(key)
        right = candidate.get(key)
        row: dict[str, Any] = {"key": key, "reference_present": left is not None, "physical_present": right is not None}
        if left is not None:
            row.update(reference_shape=list(left.shape), reference_dtype=str(left.dtype), reference_sha256=tensor_sha256(left))
        if right is not None:
            row.update(physical_shape=list(right.shape), physical_dtype=str(right.dtype), physical_sha256=tensor_sha256(right))
        row["shape_equal"] = left is not None and right is not None and tuple(left.shape) == tuple(right.shape)
        row["dtype_equal"] = left is not None and right is not None and left.dtype == right.dtype
        row["exact_equal"] = bool(left is not None and right is not None and row["shape_equal"] and row["dtype_equal"] and torch.equal(left.detach().cpu(), right.detach().cpu()))
        rows.append(row)
    mismatches = [row for row in rows if not row["exact_equal"]]
    return {
        "reference_key_count": len(reference),
        "physical_key_count": len(candidate),
        "all_key_count": len(all_keys),
        "missing_from_physical": sorted(set(reference) - set(candidate)),
        "unexpected_in_physical": sorted(set(candidate) - set(reference)),
        "mismatch_count": len(mismatches),
        "all_exact_equal": not mismatches,
        "rows": rows,
    }


def _value_info(value: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    shape = []
    for dim in tensor.shape.dim:
        shape.append(int(dim.dim_value) if dim.dim_value else str(dim.dim_param))
    return {"name": value.name, "dtype": int(tensor.elem_type), "shape": shape}


def compare_onnx_graphs(legacy_path: Path, search_path: Path, output_dir: Path) -> dict[str, Any]:
    import onnx
    from onnx import numpy_helper

    left = onnx.load(str(legacy_path))
    right = onnx.load(str(search_path))

    def node_rows(model: Any) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "name": node.name,
                "domain": node.domain,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for index, node in enumerate(model.graph.node)
        ]

    left_nodes = node_rows(left)
    right_nodes = node_rows(right)
    left_init = {row.name: row for row in left.graph.initializer}
    right_init = {row.name: row for row in right.graph.initializer}
    def initializer_content_counter(values: dict[str, Any]) -> Counter[tuple[tuple[int, ...], str, str]]:
        counter: Counter[tuple[tuple[int, ...], str, str]] = Counter()
        for tensor in values.values():
            array = numpy_helper.to_array(tensor)
            digest = hashlib.sha256(
                str(array.dtype).encode()
                + np.asarray(array.shape, dtype=np.int64).tobytes()
                + np.ascontiguousarray(array).tobytes()
            ).hexdigest()
            counter[(tuple(int(value) for value in array.shape), str(array.dtype), digest)] += 1
        return counter

    left_content = initializer_content_counter(left_init)
    right_content = initializer_content_counter(right_init)
    init_rows = []
    for name in sorted(set(left_init) | set(right_init)):
        a = left_init.get(name)
        b = right_init.get(name)
        a_array = numpy_helper.to_array(a) if a is not None else None
        b_array = numpy_helper.to_array(b) if b is not None else None
        a_hash = hashlib.sha256(np.ascontiguousarray(a_array).tobytes()).hexdigest() if a_array is not None else ""
        b_hash = hashlib.sha256(np.ascontiguousarray(b_array).tobytes()).hexdigest() if b_array is not None else ""
        init_rows.append(
            {
                "initializer": name,
                "legacy_present": a is not None,
                "search_present": b is not None,
                "legacy_shape": list(a_array.shape) if a_array is not None else "",
                "search_shape": list(b_array.shape) if b_array is not None else "",
                "legacy_dtype": str(a_array.dtype) if a_array is not None else "",
                "search_dtype": str(b_array.dtype) if b_array is not None else "",
                "legacy_sha256": a_hash,
                "search_sha256": b_hash,
                "shape_equal": bool(a_array is not None and b_array is not None and a_array.shape == b_array.shape),
                "dtype_equal": bool(a_array is not None and b_array is not None and a_array.dtype == b_array.dtype),
                "exact_equal": bool(a_array is not None and b_array is not None and a_hash == b_hash),
            }
        )
    with (output_dir / "initializer_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(init_rows[0]) if init_rows else ["initializer"])
        writer.writeheader()
        writer.writerows(init_rows)
    left_ops = Counter((row["domain"], row["op_type"]) for row in left_nodes)
    right_ops = Counter((row["domain"], row["op_type"]) for row in right_nodes)
    positional = []
    for index in range(max(len(left_nodes), len(right_nodes))):
        a = left_nodes[index] if index < len(left_nodes) else None
        b = right_nodes[index] if index < len(right_nodes) else None
        if a != b:
            positional.append({"index": index, "legacy": a, "search": b})
    result = {
        "legacy_path": str(legacy_path),
        "search_path": str(search_path),
        "legacy_sha256": sha256_file(legacy_path),
        "search_sha256": sha256_file(search_path),
        "serialized_equal": sha256_file(legacy_path) == sha256_file(search_path),
        "legacy_node_count": len(left_nodes),
        "search_node_count": len(right_nodes),
        "legacy_initializer_count": len(left_init),
        "search_initializer_count": len(right_init),
        "legacy_inputs": [_value_info(row) for row in left.graph.input],
        "search_inputs": [_value_info(row) for row in right.graph.input],
        "legacy_outputs": [_value_info(row) for row in left.graph.output],
        "search_outputs": [_value_info(row) for row in right.graph.output],
        "fixed_k_legacy": next((row["shape"][0] for row in map(_value_info, left.graph.input) if row["name"] == "voxel_features"), None),
        "fixed_k_search": next((row["shape"][0] for row in map(_value_info, right.graph.input) if row["name"] == "voxel_features"), None),
        "legacy_op_inventory": {f"{domain or 'ai.onnx'}::{op}": count for (domain, op), count in sorted(left_ops.items())},
        "search_op_inventory": {f"{domain or 'ai.onnx'}::{op}": count for (domain, op), count in sorted(right_ops.items())},
        "plugin_nodes_legacy": [row for row in left_nodes if row["op_type"] == "PointPillarScatterTRT"],
        "plugin_nodes_search": [row for row in right_nodes if row["op_type"] == "PointPillarScatterTRT"],
        "positionally_different_node_count": len(positional),
        "positionally_different_nodes": positional,
        "initializer_exact_match_count": sum(bool(row["exact_equal"]) for row in init_rows),
        "initializer_difference_count": sum(not bool(row["exact_equal"]) for row in init_rows),
        "initializer_content_multiset_equal_ignoring_names": left_content == right_content,
        "initializer_content_match_count_ignoring_names": sum((left_content & right_content).values()),
        "legacy_only_initializer_content_count": sum((left_content - right_content).values()),
        "search_only_initializer_content_count": sum((right_content - left_content).values()),
        "binding_name_order_equal": [row.name for row in left.graph.output] == [row.name for row in right.graph.output],
    }
    write_json(output_dir / "onnx_graph_diff.json", result)
    return result


def parse_legacy_calibration_cache(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        if ": " not in line:
            continue
        name, encoded = line.rsplit(": ", 1)
        try:
            rows[name] = float(struct.unpack("!f", bytes.fromhex(encoded.strip()))[0])
        except (ValueError, struct.error):
            continue
    return rows


def legacy_provenance() -> dict[str, Any]:
    legacy_build = read_json(LEGACY_BUILD_REPORT, {})
    int8_build = next((row for row in legacy_build.get("builds", []) if row.get("precision") == "int8"), {})
    return {
        "label": "legacy_single_engine_maxK_fixedK29696_INT8_train200",
        "recipe": "TensorRT_EntropyCalibration2_implicit_INT8_no_explicit_QDQ",
        "fixed_K": 29696,
        "artifacts": {
            "checkpoint": artifact(CHECKPOINT, role="original_checkpoint"),
            "config": artifact(CONFIG, role="model_config"),
            "physical_checkpoint": {"role": "physical_checkpoint", "path": "", "exists": False, "reason": "legacy_all_keep_used_original_checkpoint_directly"},
            "base_onnx": artifact(LEGACY_BASE_ONNX, role="legacy_base_fp32_onnx", dependencies=(CHECKPOINT, CONFIG)),
            "qdq_onnx": {"role": "qdq_onnx", "path": "", "exists": False, "reason": "legacy_recipe_has_no_explicit_QDQ_ONNX"},
            "calibration_manifest": artifact(LEGACY_CALIBRATION_MANIFEST, role="legacy_train200_calibration_manifest", dependencies=(CHECKPOINT, CONFIG)),
            "calibration_scales": artifact(LEGACY_CALIBRATION_CACHE, role="TensorRT_EntropyCalibration2_cache", dependencies=(LEGACY_BASE_ONNX, LEGACY_CALIBRATION_MANIFEST)),
            "plugin": artifact(LEGACY_PLUGIN, role="legacy_PointPillarScatterTRT_plugin"),
            "fp16_engine": artifact(LEGACY_FP16_ENGINE, role="legacy_FP16_engine", dependencies=(LEGACY_BASE_ONNX, LEGACY_PLUGIN)),
            "int8_engine": artifact(LEGACY_INT8_ENGINE, role="legacy_INT8_train200_engine", dependencies=(LEGACY_BASE_ONNX, LEGACY_CALIBRATION_CACHE, LEGACY_PLUGIN)),
            "engine_build_report": artifact(LEGACY_BUILD_REPORT, role="legacy_engine_builder_flags_and_profiles", dependencies=(LEGACY_BASE_ONNX, LEGACY_PLUGIN)),
            "fp16_eval": artifact(LEGACY_FP16_HISTORICAL_EVAL, role="legacy_historical_FP16_full_val", dependencies=(LEGACY_FP16_ENGINE,)),
            "int8_eval": artifact(LEGACY_INT8_HISTORICAL_EVAL, role="legacy_historical_INT8_full_val", dependencies=(LEGACY_INT8_ENGINE,)),
        },
        "calibration_scale_count": len(parse_legacy_calibration_cache(LEGACY_CALIBRATION_CACHE)),
        "builder": {
            "command": int8_build.get("command", []),
            "builder_flags": int8_build.get("builder_flags", []),
            "strict_types": int8_build.get("strict_types"),
            "prefer_precision_constraints": int8_build.get("prefer_precision_constraints"),
            "obey_precision_constraints": int8_build.get("obey_precision_constraints"),
            "TensorRT_version": legacy_build.get("TensorRT version", ""),
            "GPU_name": legacy_build.get("GPU name", ""),
        },
        "deployment_hash": json_hash([sha256_file(path) for path in (LEGACY_BASE_ONNX, LEGACY_CALIBRATION_CACHE, LEGACY_PLUGIN, LEGACY_INT8_ENGINE) if path.is_file()]),
        "eval_hash": sha256_file(LEGACY_INT8_HISTORICAL_EVAL) if LEGACY_INT8_HISTORICAL_EVAL.is_file() else "",
    }


def modelopt_manifest(gpu: int) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{MODEL_OPT / 'bin'}:{env.get('PATH', '')}",
            "CUDA_HOME": str(MODEL_OPT),
            "CC": str(MODEL_OPT / "bin/gcc"),
            "CXX": str(MODEL_OPT / "bin/g++"),
            "CUDACXX": str(MODEL_OPT / "bin/nvcc"),
            "LD_LIBRARY_PATH": f"{TRT_ROOT / 'lib'}:{MODEL_OPT / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
    )
    commands = {
        "python": [str(MODEL_OPT / "bin/python"), "-c", "import sys,tensorrt as trt; print(sys.executable); print(trt.__version__)"],
        "nvcc": [str(MODEL_OPT / "bin/nvcc"), "--version"],
        "gcc": [str(MODEL_OPT / "bin/gcc"), "--version"],
        "g++": [str(MODEL_OPT / "bin/g++"), "--version"],
    }
    outputs = {}
    for name, command in commands.items():
        completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        outputs[name] = {"command": command, "returncode": completed.returncode, "output": completed.stdout.strip()}
    return {
        "conda_env": "modelopt",
        "conda_prefix": str(MODEL_OPT),
        "python_path": str(MODEL_OPT / "bin/python"),
        "nvcc_path": str(MODEL_OPT / "bin/nvcc"),
        "gcc_path": str(MODEL_OPT / "bin/gcc"),
        "gxx_path": str(MODEL_OPT / "bin/g++"),
        "CUDA_HOME": str(MODEL_OPT),
        "TensorRT_root": str(TRT_ROOT),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "system_paths_rejected": ["/usr/bin/gcc", "/usr/bin/g++", "/usr/local/cuda/bin/nvcc"],
        "tool_outputs": outputs,
    }


def build_context(output: Path, gpu: int):
    from search.integration.lidar_pyramid_context import build_lidar_pyramid_context

    return build_lidar_pyramid_context(
        checkpoint_path=CHECKPOINT,
        model_config_path=CONFIG,
        heal_root=HEAL_ROOT,
        tensorrt_root=TRT_ROOT,
        plugin_path=CURRENT_PLUGIN,
        output_dir=output,
        gpu_id=str(gpu),
        tensorrt_env="modelopt",
        fisher_calibration_batches=0,
        quant_calibration_batches=200,
        num_frames=1789,
        warmup_frames=0,
        default_precision="FP16",
        max_pruning_units=96,
    )


def run_fresh_search_baseline(context: Any, output: Path, kind: str) -> dict[str, Any]:
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator

    run_dir = output / f"search_{kind}_force_rebuild"
    artifact_dir = run_dir / "artifacts"
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"force_rebuild_destination_not_empty:{artifact_dir}")
    evaluator = LidarPyramidRealEvaluator(
        context=context,
        run_dir=run_dir,
        num_frames=1789,
        warmup_frames=0,
        latency_rounds=1,
    )
    phenotype = evaluator._baseline_precision_phenotype(kind)
    if phenotype.pruned_unit_ids:
        raise RuntimeError(f"all_keep_contract_failed:{phenotype.pruned_unit_ids[:8]}")
    write_json(artifact_dir / "phenotype.json", phenotype.to_dict())
    result = evaluator._deploy_and_evaluate(
        phenotype=phenotype,
        output_dir=artifact_dir,
        candidate_label=f"audit_all_keep_{kind}",
        pruned_unit_ids=[],
        baseline_precision=kind,
    )
    write_json(run_dir / "result.json", result)
    if result.get("status") != "ok":
        raise RuntimeError(f"fresh_search_{kind}_failed:{result.get('failure_reason', result)}")
    return result


def run_fresh_legacy_evaluation(context: Any, output: Path, *, precision: str, engine: Path, plugin: Path) -> dict[str, Any]:
    from search.integration.evaluation_provider import evaluate_engine_modelopt

    destination = output / f"legacy_{precision}_fresh_evaluation"
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"fresh_evaluation_destination_not_empty:{destination}")
    result = evaluate_engine_modelopt(
        engine_path=engine,
        checkpoint=CHECKPOINT,
        model_config=CONFIG,
        heal_root=HEAL_ROOT,
        device=context.runtime_device,
        output_dir=destination,
        tensorrt_root=TRT_ROOT,
        plugin_path=plugin,
        num_frames=1789,
        warmup_frames=0,
        fixed_k=29696,
        latency_rounds=1,
        conda_env="modelopt",
        eval_manifest_path=context.eval_manifest_path,
    )
    write_json(destination / "result.json", result)
    if result.get("status") != "ok":
        raise RuntimeError(f"fresh_legacy_{precision}_failed:{result.get('failure_reason', result)}")
    return result


def verify_physical_identity(context: Any, artifact_dir: Path, output: Path) -> dict[str, Any]:
    phenotype = read_json(artifact_dir / "phenotype.json", {})
    checkpoint = torch.load(artifact_dir / "pruned_checkpoint.pth", map_location="cpu")
    physical_state = checkpoint.get("model", checkpoint)
    comparison = compare_state_dicts(context.model.state_dict(), physical_state)
    physical_hash = read_json(artifact_dir / "physical_hash.json", {})
    request = read_json(artifact_dir / "sampling_pruning_request.json", {})
    plan = read_json(artifact_dir / "physical_plan.json", {})
    result = {
        "candidate_pruned_unit_count": len(phenotype.get("pruned_unit_ids", [])),
        "candidate_is_all_keep": len(phenotype.get("pruned_unit_ids", [])) == 0,
        "request_selected_atomic_unit_count": len(request.get("selected_atomic_unit_ids", []) or []),
        "physical_plan_entry_count": len(plan.get("entries", []) or []),
        "parameter_count_base": physical_hash.get("parameter_count_base"),
        "parameter_count_physical": physical_hash.get("parameter_count_pruned"),
        "parameter_count_equal": physical_hash.get("parameter_count_base") == physical_hash.get("parameter_count_pruned"),
        "state_dict": comparison,
        "passed": bool(
            not phenotype.get("pruned_unit_ids")
            and not request.get("selected_atomic_unit_ids")
            and not plan.get("entries")
            and physical_hash.get("parameter_count_base") == physical_hash.get("parameter_count_pruned")
            and comparison["all_exact_equal"]
        ),
    }
    write_json(output / "physical_model_identity.json", result)
    return result


def _map_value(result: dict[str, Any]) -> float:
    return float(result.get("mAP", result.get("map", 0.0)) or 0.0)


def qdq_inventory(qdq_path: Path, mapping_path: Path, output: Path) -> dict[str, Any]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(qdq_path))
    initializers = {row.name: numpy_helper.to_array(row) for row in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    mapping = read_json(mapping_path, {})
    calibration_scales = read_json(qdq_path.parent / "calibration_scales.json", {})
    canonical_to_module = {
        str(row.get("canonical_node_name", "")): str(row.get("module_path", ""))
        for row in mapping.get("entries", [])
    }
    rows = []
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        tensor_name = str(node.input[0])
        scale_name = str(node.input[1]) if len(node.input) > 1 else ""
        zero_name = str(node.input[2]) if len(node.input) > 2 else ""
        scale = initializers.get(scale_name)
        zero = initializers.get(zero_name)
        role = "unknown"
        for candidate in ("activation_input", "activation_output", "weight"):
            if f"__{candidate}" in node.name:
                role = candidate
                break
        canonical = node.name.split(f"__{role}", 1)[0] if role != "unknown" else node.name
        quantized = str(node.output[0]) if node.output else ""
        dq_nodes = [row for row in consumers.get(quantized, []) if row.op_type == "DequantizeLinear"]
        dq = dq_nodes[0] if dq_nodes else None
        dequantized = str(dq.output[0]) if dq is not None and dq.output else ""
        following = consumers.get(dequantized, []) if dequantized else []
        producer = producers.get(tensor_name)
        axis = next((int(onnx.helper.get_attribute_value(attr)) for attr in node.attribute if attr.name == "axis"), None)
        scale_values = np.asarray(scale).reshape(-1).astype(np.float64) if scale is not None else np.asarray([], dtype=np.float64)
        zero_values = np.asarray(zero).reshape(-1).astype(np.int64) if zero is not None else np.asarray([], dtype=np.int64)
        module_path = canonical_to_module.get(canonical, "")
        calibration_key = {
            "activation_input": "activation_input_scale",
            "activation_output": "activation_output_scale",
            "weight": "weight_scale",
        }.get(role, "")
        expected_scale = calibration_scales.get(module_path, {}).get(calibration_key) if calibration_key else None
        expected_scale_values = (
            np.asarray(expected_scale, dtype=np.float64).reshape(-1)
            if expected_scale is not None
            else np.asarray([], dtype=np.float64)
        )
        weight_values = np.asarray(initializers[tensor_name]).astype(np.float64) if role == "weight" and tensor_name in initializers else None
        initializer_absmax_div127 = float(np.max(np.abs(weight_values)) / 127.0) if weight_values is not None and weight_values.size else None
        initializer_expected_scale = np.asarray([], dtype=np.float64)
        if weight_values is not None and weight_values.size:
            if scale_values.size == 1 or axis is None:
                initializer_expected_scale = np.asarray([initializer_absmax_div127], dtype=np.float64)
            else:
                normalized_axis = int(axis) % weight_values.ndim
                reduce_axes = tuple(index for index in range(weight_values.ndim) if index != normalized_axis)
                initializer_expected_scale = np.max(np.abs(weight_values), axis=reduce_axes).reshape(-1) / 127.0
        scalar_scale = float(scale_values[0]) if scale_values.size == 1 else None
        row = {
            "tensor": tensor_name,
            "producer": producer.name if producer is not None else "graph_input_or_initializer",
            "producer_op": producer.op_type if producer is not None else "",
            "consumer": ";".join(item.name for item in following),
            "consumer_op": ";".join(item.op_type for item in following),
            "canonical_layer": module_path,
            "canonical_node": canonical,
            "quantize_node": node.name,
            "dequantize_node": dq.name if dq is not None else "",
            "scale": ";".join(f"{value:.17g}" for value in scale_values),
            "scale_min": float(scale_values.min()) if scale_values.size else "",
            "scale_max": float(scale_values.max()) if scale_values.size else "",
            "scale_shape": list(np.asarray(scale).shape) if scale is not None else "",
            "axis": axis if axis is not None else "",
            "zero_point": ";".join(str(int(value)) for value in zero_values),
            "zero_point_shape": list(np.asarray(zero).shape) if zero is not None else "",
            "zero_point_dtype": str(np.asarray(zero).dtype) if zero is not None else "",
            "granularity": "per_tensor" if scale_values.size == 1 else "per_channel",
            "symmetry": "symmetric" if zero_values.size and np.all(zero_values == 0) else "asymmetric_or_missing",
            "quant_role": role,
            "nonfinite_scale": bool(scale_values.size and not np.isfinite(scale_values).all()),
            "zero_scale": bool(scale_values.size and np.any(scale_values == 0)),
            "extremely_small_scale": bool(scale_values.size and np.any(np.abs(scale_values) < 1.0e-8)),
            "extremely_large_scale": bool(scale_values.size and np.any(np.abs(scale_values) > 1.0e4)),
            "fixed_one_scale": bool(scale_values.size and np.all(scale_values == 1.0)),
            "weight_axis_is_conv_output_axis": bool(role == "weight" and axis == 0),
            "calibration_expected_scale": ";".join(f"{value:.17g}" for value in expected_scale_values),
            "scale_matches_calibration": bool(
                expected_scale_values.size == scale_values.size
                and expected_scale_values.size > 0
                and np.allclose(scale_values, expected_scale_values, rtol=1.0e-7, atol=1.0e-12)
            ),
            "weight_initializer_present": bool(weight_values is not None),
            "weight_initializer_absmax_div127": "" if initializer_absmax_div127 is None else initializer_absmax_div127,
            "weight_scale_matches_final_folded_initializer": bool(
                role == "weight"
                and initializer_expected_scale.size == scale_values.size
                and initializer_expected_scale.size > 0
                and np.allclose(scale_values, initializer_expected_scale, rtol=1.0e-6, atol=1.0e-12)
            ),
        }
        rows.append(row)
    fields = list(rows[0]) if rows else ["tensor"]
    with (output / "qdq_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "qdq_onnx": str(qdq_path),
        "qdq_onnx_sha256": sha256_file(qdq_path),
        "QuantizeLinear_count": sum(node.op_type == "QuantizeLinear" for node in model.graph.node),
        "DequantizeLinear_count": sum(node.op_type == "DequantizeLinear" for node in model.graph.node),
        "inventory_row_count": len(rows),
        "role_counts": dict(Counter(str(row["quant_role"]) for row in rows)),
        "nonfinite_count": sum(bool(row["nonfinite_scale"]) for row in rows),
        "zero_count": sum(bool(row["zero_scale"]) for row in rows),
        "extremely_small_count": sum(bool(row["extremely_small_scale"]) for row in rows),
        "extremely_large_count": sum(bool(row["extremely_large_scale"]) for row in rows),
        "fixed_one_count": sum(bool(row["fixed_one_scale"]) for row in rows),
        "per_tensor_count": sum(row["granularity"] == "per_tensor" for row in rows),
        "per_channel_count": sum(row["granularity"] == "per_channel" for row in rows),
        "weight_axis_0_count": sum(bool(row["weight_axis_is_conv_output_axis"]) for row in rows),
        "weight_row_count": sum(row["quant_role"] == "weight" for row in rows),
        "weight_scale_final_initializer_match_count": sum(bool(row["weight_scale_matches_final_folded_initializer"]) for row in rows),
        "calibration_scale_match_count": sum(bool(row["scale_matches_calibration"]) for row in rows),
        "rows": rows,
    }
    write_json(output / "qdq_inventory.json", summary)
    return summary


def write_scale_diff(inventory: dict[str, Any], output: Path) -> dict[str, Any]:
    legacy = parse_legacy_calibration_cache(LEGACY_CALIBRATION_CACHE)
    search_rows = list(inventory.get("rows", []))
    compared = []
    matched_legacy: set[str] = set()
    for row in search_rows:
        tensor = str(row.get("tensor", ""))
        normalized = tensor.replace("__before_output_qdq", "")
        legacy_scale = legacy.get(normalized)
        if legacy_scale is not None:
            matched_legacy.add(normalized)
        search_scale = row.get("scale_min")
        search_value = float(search_scale) if search_scale != "" else None
        compared.append(
            {
                "tensor": normalized,
                "canonical_layer": row.get("canonical_layer", ""),
                "quant_role": row.get("quant_role", ""),
                "legacy_boundary_present": legacy_scale is not None,
                "search_boundary_present": True,
                "legacy_scale": "" if legacy_scale is None else legacy_scale,
                "search_scale_min": row.get("scale_min", ""),
                "search_scale_max": row.get("scale_max", ""),
                "scale_ratio_search_over_legacy": (search_value / legacy_scale) if search_value is not None and legacy_scale not in (None, 0.0) else "",
                "legacy_scale_shape": "scalar_activation_dynamic_range_cache" if legacy_scale is not None else "",
                "search_scale_shape": row.get("scale_shape", ""),
                "legacy_axis": "",
                "search_axis": row.get("axis", ""),
                "legacy_zero_point": "implicit_TensorRT_not_serialized",
                "search_zero_point": row.get("zero_point", ""),
                "legacy_granularity": "activation_per_tensor" if legacy_scale is not None else "",
                "search_granularity": row.get("granularity", ""),
                "quantization_boundary_equal": bool(legacy_scale is not None and row.get("quant_role") != "weight"),
            }
        )
    for tensor, legacy_scale in sorted(legacy.items()):
        if tensor in matched_legacy:
            continue
        compared.append(
            {
                "tensor": tensor,
                "canonical_layer": "",
                "quant_role": "legacy_activation_only",
                "legacy_boundary_present": True,
                "search_boundary_present": False,
                "legacy_scale": legacy_scale,
                "search_scale_min": "",
                "search_scale_max": "",
                "scale_ratio_search_over_legacy": "",
                "legacy_scale_shape": "scalar_activation_dynamic_range_cache",
                "search_scale_shape": "",
                "legacy_axis": "",
                "search_axis": "",
                "legacy_zero_point": "implicit_TensorRT_not_serialized",
                "search_zero_point": "",
                "legacy_granularity": "activation_per_tensor",
                "search_granularity": "",
                "quantization_boundary_equal": False,
            }
        )
    with (output / "scale_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compared[0]) if compared else ["tensor"])
        writer.writeheader()
        writer.writerows(compared)
    result = {
        "legacy_scale_count": len(legacy),
        "search_q_boundary_count": len(search_rows),
        "exact_name_boundary_overlap_count": len(matched_legacy),
        "legacy_only_boundary_count": len(set(legacy) - matched_legacy),
        "search_only_or_weight_boundary_count": sum(not bool(row["legacy_boundary_present"]) for row in compared if row.get("search_boundary_present")),
        "recipe_boundary_sets_equal": set(legacy) == {str(row.get("tensor", "")).replace("__before_output_qdq", "") for row in search_rows},
    }
    ratios = [
        float(row["scale_ratio_search_over_legacy"])
        for row in compared
        if row.get("scale_ratio_search_over_legacy") not in ("", None)
    ]
    result["matched_activation_scale_ratio"] = {
        "count": len(ratios),
        "min": min(ratios) if ratios else None,
        "median": float(np.median(np.asarray(ratios))) if ratios else None,
        "max": max(ratios) if ratios else None,
    }
    write_json(output / "scale_diff_summary.json", result)
    return result


def export_engine_layer_csv(layer_info_path: Path, destination: Path) -> dict[str, Any]:
    from quantization.tensorrt.layer_info import is_weighted_compute_layer, precision_name

    payload = read_json(layer_info_path, {})
    layers = payload.get("Layers", payload if isinstance(payload, list) else [])
    rows = []
    for index, layer in enumerate(layers):
        inputs = list(layer.get("Inputs", []) or [])
        outputs = list(layer.get("Outputs", []) or [])
        text = json.dumps(layer, sort_keys=True)
        precision = precision_name(layer)
        rows.append(
            {
                "index": index,
                "name": layer.get("Name", ""),
                "layer_type": layer.get("LayerType", ""),
                "precision": precision,
                "weighted_compute": bool(is_weighted_compute_layer(layer)),
                "input_names": ";".join(str(item.get("Name", "")) for item in inputs),
                "input_formats": ";".join(str(item.get("Format/Datatype", "")) for item in inputs),
                "output_names": ";".join(str(item.get("Name", "")) for item in outputs),
                "output_formats": ";".join(str(item.get("Format/Datatype", "")) for item in outputs),
                "tactic": layer.get("TacticValue", ""),
                "stream_id": layer.get("StreamId", ""),
                "metadata": layer.get("Metadata", ""),
                "reformat_layer": "reformat" in str(layer.get("Name", "")).lower() or str(layer.get("LayerType", "")).lower() == "noop",
                "plugin_boundary": "plugin" in text.lower() or "pointpillarscatter" in text.lower(),
                "fused_layer": " + " in str(layer.get("Name", "")) or "||" in str(layer.get("Name", "")),
            }
        )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(str(row["precision"]) for row in rows)
    summary = {
        "source": str(layer_info_path),
        "source_sha256": sha256_file(layer_info_path),
        "layer_count": len(rows),
        "precision_counts": dict(counts),
        "weighted_precision_counts": dict(Counter(str(row["precision"]) for row in rows if row["weighted_compute"])),
        "reformat_layer_count": sum(bool(row["reformat_layer"]) for row in rows),
        "plugin_boundary_count": sum(bool(row["plugin_boundary"]) for row in rows),
        "fused_layer_count": sum(bool(row["fused_layer"]) for row in rows),
    }
    return summary


def search_provenance(output: Path) -> dict[str, Any]:
    base = output / "search_maximal_legal_int8_force_rebuild/artifacts"
    fp16 = output / "search_strict_fp16_force_rebuild/artifacts"
    dependencies = (CHECKPOINT, CONFIG, CURRENT_PLUGIN)
    result = {
        "label": "search_all_keep_BN_fold_aware_explicit_QDQ_INT8_train200",
        "recipe": "explicit_QDQ_ONNX_BN_fold_aware_v1_maximal_legal_INT8",
        "fixed_K": 29696,
        "pruned_unit_count": len(read_json(base / "phenotype.json", {}).get("pruned_unit_ids", [])),
        "artifacts": {
            "checkpoint": artifact(CHECKPOINT, role="original_checkpoint"),
            "config": artifact(CONFIG, role="model_config"),
            "physical_checkpoint": artifact(base / "pruned_checkpoint.pth", role="all_keep_physical_checkpoint", dependencies=(CHECKPOINT,)),
            "base_onnx": artifact(base / "pruned_fp32.onnx", role="fresh_search_base_fp32_onnx", dependencies=dependencies),
            "qdq_onnx": artifact(base / "qdq.onnx", role="fresh_search_explicit_qdq_onnx", dependencies=(base / "pruned_fp32.onnx", base / "calibration_scales.json")),
            "calibration_manifest": artifact(base / "calibration_manifest.json", role="fresh_search_train200_calibration_manifest", dependencies=(base / "pruned_fp32.onnx",)),
            "calibration_scales": artifact(base / "calibration_scales.json", role="fresh_search_BN_fold_aware_scales", dependencies=(base / "pruned_fp32.onnx", CHECKPOINT)),
            "plugin": artifact(CURRENT_PLUGIN, role="current_search_PointPillarScatterTRT_plugin"),
            "fp16_engine": artifact(fp16 / "engine.plan", role="fresh_search_FP16_engine", dependencies=(fp16 / "qdq_trt_compatible.onnx", CURRENT_PLUGIN)),
            "int8_engine": artifact(base / "engine.plan", role="fresh_search_INT8_engine", dependencies=(base / "qdq_trt_compatible.onnx", CURRENT_PLUGIN)),
            "engine_build_request": artifact(base / "trt_build_request.json", role="fresh_search_builder_flags_and_profiles", dependencies=(base / "qdq_trt_compatible.onnx", CURRENT_PLUGIN)),
            "engine_build_manifest": artifact(base / "engine_manifest.json", role="fresh_search_engine_build_result", dependencies=(base / "trt_build_request.json",)),
            "fp16_eval": artifact(fp16 / "evaluation.json", role="fresh_search_FP16_full_val", dependencies=(fp16 / "engine.plan", output / "baseline/eval_manifest.json")),
            "int8_eval": artifact(base / "evaluation.json", role="fresh_search_INT8_full_val", dependencies=(base / "engine.plan", output / "baseline/eval_manifest.json")),
        },
    }
    deployment = read_json(base / "deployment_manifest.json", {})
    result["deployment_hash"] = deployment.get("deployment_hash", "")
    result["eval_hash"] = deployment.get("eval_hash", "")
    result["engine_hash"] = deployment.get("engine_hash", "")
    request = read_json(base / "trt_build_request.json", {})
    manifest = read_json(base / "engine_manifest.json", {})
    result["builder"] = {
        "build_config": request.get("build_config", {}),
        "command": manifest.get("build", {}).get("command", []),
        "worker_invocation": manifest.get("worker_invocation", []),
    }
    return result


def write_artifact_hash_diff(output: Path, legacy: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    rows = []
    all_roles = sorted(set(legacy.get("artifacts", {})) | set(search.get("artifacts", {})))
    for role in all_roles:
        left = legacy.get("artifacts", {}).get(role, {})
        right = search.get("artifacts", {}).get(role, {})
        rows.append(
            {
                "role": role,
                "legacy_path": left.get("path", ""),
                "search_path": right.get("path", ""),
                "legacy_exists": bool(left.get("exists", False)),
                "search_exists": bool(right.get("exists", False)),
                "legacy_sha256": left.get("sha256", ""),
                "search_sha256": right.get("sha256", ""),
                "same_sha256": bool(left.get("sha256") and left.get("sha256") == right.get("sha256")),
                "legacy_dependency_hash": left.get("dependency_hash", ""),
                "search_dependency_hash": right.get("dependency_hash", ""),
            }
        )
    result = {"rows": rows, "same_hash_roles": [row["role"] for row in rows if row["same_sha256"]], "different_hash_roles": [row["role"] for row in rows if not row["same_sha256"]]}
    write_json(output / "artifact_hash_diff.json", result)
    return result


def phase_base(output: Path, gpu: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "toolchain_manifest.json", modelopt_manifest(gpu))
    write_json(output / "provenance_legacy.json", legacy_provenance())
    context = build_context(output, gpu)
    write_json(
        output / "audit_contract.json",
        {
            "candidate": "all_keep_maximal_legal_INT8",
            "pruned_unit_count": 0,
            "fixed_K": 29696,
            "quant_calibration_batches": 200,
            "full_validation_frames": 1789,
            "warmup_frames": 0,
            "latency_rounds": 1,
            "gpu": gpu,
            "force_rebuild": True,
            "legacy_read_only": True,
        },
    )
    search = run_fresh_search_baseline(context, output, "strict_fp16")
    search_artifacts = output / "search_strict_fp16_force_rebuild/artifacts"
    identity = verify_physical_identity(context, search_artifacts, output)
    if not identity["passed"]:
        raise RuntimeError("all_keep_physical_model_not_identical")
    legacy = run_fresh_legacy_evaluation(context, output, precision="fp16", engine=LEGACY_FP16_ENGINE, plugin=LEGACY_PLUGIN)
    difference = abs(_map_value(legacy) - _map_value(search["evaluation"]))
    gate = {
        "threshold": 0.001,
        "legacy_mAP": _map_value(legacy),
        "search_mAP": _map_value(search["evaluation"]),
        "absolute_difference": difference,
        "passed": difference <= 0.001,
        "same_manifest": legacy.get("eval_manifest_hash") == search["evaluation"].get("eval_manifest_hash"),
        "legacy_evaluated_frames": legacy.get("num_evaluated_frames"),
        "search_evaluated_frames": search["evaluation"].get("num_evaluated_frames"),
        "legacy_skipped_frames": legacy.get("num_skipped_frames"),
        "search_skipped_frames": search["evaluation"].get("num_skipped_frames"),
    }
    write_json(output / "base_fp16_equivalence_gate.json", gate)
    compare_onnx_graphs(LEGACY_BASE_ONNX, search_artifacts / "pruned_fp32.onnx", output)
    if not gate["passed"]:
        write_json(output / "STOP_AFTER_FP16_GATE.json", {"reason": "base_fp16_map_difference_exceeds_0.001", **gate})
        raise RuntimeError("base_fp16_equivalence_gate_failed")


def phase_int8(output: Path, gpu: int) -> None:
    gate = read_json(output / "base_fp16_equivalence_gate.json", {})
    if not gate.get("passed", False):
        raise RuntimeError("int8_phase_forbidden_without_passing_fp16_gate")
    context_root = output / "int8_context_force_rebuild"
    context = build_context(context_root, gpu)
    canonical_manifest = output / "baseline/eval_manifest.json"
    generated_manifest = read_json(context.eval_manifest_path, {})
    expected_manifest = read_json(canonical_manifest, {})
    if generated_manifest != expected_manifest:
        raise RuntimeError("int8_context_manifest_differs_from_fp16_manifest")
    context.eval_manifest_path = canonical_manifest
    context.eval_manifest_hash = str(expected_manifest.get("manifest_hash", context.eval_manifest_hash))
    search = run_fresh_search_baseline(context, output, "maximal_legal_int8")
    artifacts = output / "search_maximal_legal_int8_force_rebuild/artifacts"
    identity = verify_physical_identity(context, artifacts, output / "int8_physical_identity")
    if not identity["passed"]:
        raise RuntimeError("int8_all_keep_physical_model_not_identical")
    legacy = run_fresh_legacy_evaluation(context, output, precision="int8", engine=LEGACY_INT8_ENGINE, plugin=LEGACY_PLUGIN)
    inventory = qdq_inventory(artifacts / "qdq.onnx", artifacts / "canonical_layer_map.json", output)
    scale_summary = write_scale_diff(inventory, output)
    legacy_layer_summary = export_engine_layer_csv(LEGACY_INT8_LAYERINFO, output / "engine_layer_precision_legacy.csv")
    search_layer_summary = export_engine_layer_csv(artifacts / "engine_layer_info.json", output / "engine_layer_precision_search.csv")
    write_json(output / "engine_layer_precision_summary.json", {"legacy": legacy_layer_summary, "search": search_layer_summary})
    legacy_prov = legacy_provenance()
    search_prov = search_provenance(output)
    write_json(output / "provenance_legacy.json", legacy_prov)
    write_json(output / "provenance_search.json", search_prov)
    write_artifact_hash_diff(output, legacy_prov, search_prov)
    comparison = {
        "same_eval_manifest": legacy.get("eval_manifest_hash") == search["evaluation"].get("eval_manifest_hash"),
        "legacy_mAP": _map_value(legacy),
        "search_mAP": _map_value(search["evaluation"]),
        "absolute_mAP_difference": abs(_map_value(legacy) - _map_value(search["evaluation"])),
        "deployment_result_threshold": 0.005,
        "deployment_result_within_threshold": abs(_map_value(legacy) - _map_value(search["evaluation"])) <= 0.005,
        "legacy_evaluated_frames": legacy.get("num_evaluated_frames"),
        "search_evaluated_frames": search["evaluation"].get("num_evaluated_frames"),
        "legacy_skipped_frame_ids": legacy.get("skipped_frame_ids", []),
        "search_skipped_frame_ids": search["evaluation"].get("skipped_frame_ids", []),
        "legacy_AP@0.30": legacy.get("AP@0.3"),
        "legacy_AP@0.50": legacy.get("AP@0.5"),
        "legacy_AP@0.70": legacy.get("AP@0.7"),
        "search_AP@0.30": search["evaluation"].get("AP@0.3"),
        "search_AP@0.50": search["evaluation"].get("AP@0.5"),
        "search_AP@0.70": search["evaluation"].get("AP@0.7"),
        "legacy_p50_ms": legacy.get("forward_p50_ms"),
        "search_p50_ms": search["evaluation"].get("forward_p50_ms"),
        "legacy_FPS": 1000.0 / float(legacy.get("forward_p50_ms")) if legacy.get("forward_p50_ms") else None,
        "search_FPS": 1000.0 / float(search["evaluation"].get("forward_p50_ms")) if search["evaluation"].get("forward_p50_ms") else None,
        "quantization_recipe_equal": bool(scale_summary.get("recipe_boundary_sets_equal", False)),
        "legacy_recipe": "implicit_TensorRT_EntropyCalibration2",
        "search_recipe": "explicit_QDQ_BN_fold_aware",
        "legacy_plugin_sha256": sha256_file(LEGACY_PLUGIN),
        "search_plugin_sha256": sha256_file(CURRENT_PLUGIN),
        "plugin_binary_equal": sha256_file(LEGACY_PLUGIN) == sha256_file(CURRENT_PLUGIN),
    }
    write_json(output / "int8_ab_comparison.json", comparison)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--phase", choices=("base", "int8"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO)
    if args.phase == "base":
        phase_base(args.output.resolve(), args.gpu)
    elif args.phase == "int8":
        phase_int8(args.output.resolve(), args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
