#!/usr/bin/env python3
"""Fresh production E67/E27 matched-profile builds with 10/200-frame gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHECKPOINT = Path(
    "${MODEL_ROOT}/"
    "lidar_pyramid/net_epoch_bestval_at17.pth"
)
CONFIG = Path(
    "${MODEL_ROOT}/"
    "lidar_pyramid/config.yaml"
)
HEAL_ROOT = Path("../../HEAL")
TRT_ROOT = Path("${TENSORRT_ROOT}")
MODEL_OPT = Path("${CONDA_BASE}/envs/modelopt")
PLUGIN = REPO / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
TRAIN200_NPZ = (
    REPO
    / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration"
    / "train_calib_single_engine_maxK29696_200/manifest.json"
)
LEGACY_CACHE = (
    REPO
    / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration"
    / "lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK29696_int8_train_calib200.cache"
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "output": (completed.stdout or "").strip(),
    }


def toolchain_manifest(gpu: int) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CONDA_DEFAULT_ENV": "modelopt",
            "CONDA_PREFIX": str(MODEL_OPT),
            "PATH": f"{MODEL_OPT / 'bin'}:{env.get('PATH', '')}",
            "CUDA_HOME": str(MODEL_OPT),
            "CC": str(MODEL_OPT / "bin/gcc"),
            "CXX": str(MODEL_OPT / "bin/g++"),
            "CUDACXX": str(MODEL_OPT / "bin/nvcc"),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "LD_LIBRARY_PATH": ":".join(
                [
                    str(TRT_ROOT / "targets/x86_64-linux-gnu/lib"),
                    str(TRT_ROOT / "lib"),
                    str(MODEL_OPT / "lib"),
                    env.get("LD_LIBRARY_PATH", ""),
                ]
            ),
        }
    )
    tools = {
        "python_tensorrt": command_output(
            [str(MODEL_OPT / "bin/python"), "-c", "import sys,tensorrt as trt; print(sys.executable); print(trt.__version__)"],
            env,
        ),
        "nvcc": command_output([str(MODEL_OPT / "bin/nvcc"), "--version"], env),
        "gcc": command_output([str(MODEL_OPT / "bin/gcc"), "--version"], env),
        "gxx": command_output([str(MODEL_OPT / "bin/g++"), "--version"], env),
    }
    if any(row["returncode"] != 0 for row in tools.values()):
        raise RuntimeError(f"modelopt_toolchain_verification_failed:{tools}")
    return {
        "conda_env": "modelopt",
        "conda_prefix": str(MODEL_OPT),
        "python": str(MODEL_OPT / "bin/python"),
        "nvcc": str(MODEL_OPT / "bin/nvcc"),
        "gcc": str(MODEL_OPT / "bin/gcc"),
        "gxx": str(MODEL_OPT / "bin/g++"),
        "CUDA_HOME": str(MODEL_OPT),
        "TensorRT_root": str(TRT_ROOT),
        "plugin": str(PLUGIN),
        "plugin_sha256": sha256_file(PLUGIN),
        "GPU": int(gpu),
        "system_toolchain_used": False,
        "tools": tools,
    }


def build_context(output: Path, gpu: int, variant: str) -> Any:
    from search.integration.lidar_pyramid_context import build_lidar_pyramid_context

    backend = (
        "external_tensorrt_entropy_cache_exact_match"
        if variant.endswith("-LS")
        else "tensorrt_entropy_calibration2"
    )
    return build_lidar_pyramid_context(
        checkpoint_path=CHECKPOINT,
        model_config_path=CONFIG,
        heal_root=HEAL_ROOT,
        tensorrt_root=TRT_ROOT,
        plugin_path=PLUGIN,
        output_dir=output,
        gpu_id=str(gpu),
        tensorrt_env="modelopt",
        fisher_calibration_batches=0,
        quant_calibration_batches=200,
        quant_calibration_npz_manifest=TRAIN200_NPZ,
        quant_activation_calibration_backend=backend,
        quant_activation_calibration_cache_path=LEGACY_CACHE if variant.endswith("-LS") else None,
        quant_calibration_force_rebuild=True,
        num_frames=10,
        warmup_frames=10,
        reset_after_warmup=True,
        default_precision="FP16",
        max_pruning_units=96,
    )


def validate_profile(context: Any, phenotype: Any, profile_path: Path, variant: str) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if variant.startswith("ST69-"):
        expected_int8 = {
            str(row["canonical_layer"])
            for row in profile["layers"]
            if row["canonical_kind"] == "parameterized_weighted"
        }
        expected_coverage = {"int8": 69, "fp16": 1}
    elif variant.startswith("E67-"):
        expected_int8 = set(str(value) for value in profile["int8_module_paths"])
        expected_coverage = {"int8": 67, "fp16": 3}
    else:
        from search.baselines.original_engines import TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES

        expected_int8 = set(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)
        expected_coverage = {"int8": 27, "fp16": 43}
    actual_int8 = {
        str(module)
        for module, decision in phenotype.precision_profile.items()
        if str(decision.requested_precision).upper() == "INT8"
    }
    parameterized = {
        str(row["canonical_layer"])
        for row in profile["layers"]
        if row["canonical_kind"] == "parameterized_weighted"
    }
    expected_fp16_parameterized = parameterized - expected_int8
    actual_fp16 = {
        str(module)
        for module, decision in phenotype.precision_profile.items()
        if str(decision.requested_precision).upper() == "FP16"
    }
    result = {
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "expected_int8_count": len(expected_int8),
        "actual_int8_count": len(actual_int8),
        "expected_parameterized_fp16_count": len(expected_fp16_parameterized),
        "actual_parameterized_fp16_count": len(actual_fp16),
        "missing_int8": sorted(expected_int8 - actual_int8),
        "unexpected_int8": sorted(actual_int8 - expected_int8),
        "missing_fp16": sorted(expected_fp16_parameterized - actual_fp16),
        "unexpected_fp16": sorted(actual_fp16 - expected_fp16_parameterized),
        "functional_fp16_entry": "pyramid_backbone.functional_affine_grid_matmul",
        "canonical_expected_coverage": expected_coverage,
    }
    result["passed"] = not any(
        result[key]
        for key in ("missing_int8", "unexpected_int8", "missing_fp16", "unexpected_fp16")
    )
    if not result["passed"]:
        raise RuntimeError(f"matched_legacy_layer_set_mismatch:{result}")
    return result


def evaluation_gate(result: dict[str, Any], frames: int) -> dict[str, Any]:
    gate = {
        "status": str(result.get("status", "")),
        "mAP": float(result.get("mAP", 0.0) or 0.0),
        "AP@0.3": float(result.get("AP@0.3", 0.0) or 0.0),
        "AP@0.5": float(result.get("AP@0.5", 0.0) or 0.0),
        "AP@0.7": float(result.get("AP@0.7", 0.0) or 0.0),
        "forward_p50_ms": float(result.get("forward_p50_ms", 0.0) or 0.0),
        "evaluated": int(result.get("num_evaluated_frames", 0) or 0),
        "skipped": int(result.get("num_skipped_frames", 0) or 0),
        "fixed_manifest_enforced": bool(result.get("fixed_manifest_enforced", False)),
        "reset_after_warmup": bool(result.get("reset_after_warmup", False)),
    }
    gate["passed"] = (
        gate["status"] == "ok"
        and gate["evaluated"] == int(frames)
        and gate["skipped"] == 0
        and gate["fixed_manifest_enforced"]
        and gate["reset_after_warmup"]
    )
    return gate


def main(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for path in (CHECKPOINT, CONFIG, TRT_ROOT, PLUGIN, TRAIN200_NPZ, LEGACY_CACHE, args.profile):
        if not path.exists():
            raise FileNotFoundError(path)
    write_json(output / "toolchain_manifest.json", toolchain_manifest(args.gpu))
    context = build_context(output / "context", args.gpu, args.variant)

    from search.integration.data_provider import load_split_frame_ids, write_eval_manifest
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator

    evaluator = LidarPyramidRealEvaluator(
        context=context,
        run_dir=output / "production",
        num_frames=10,
        warmup_frames=10,
        latency_rounds=1,
    )
    baseline_precision = (
        "maximal_legal_int8"
        if args.variant.startswith("ST69-")
        else "matched_legacy_int8"
        if args.variant.startswith("E67-")
        else "trusted_explicit_qdq_int8"
    )
    phenotype = evaluator._baseline_precision_phenotype(baseline_precision)
    if phenotype.pruned_unit_ids:
        raise RuntimeError(f"all_keep_contract_failed:{phenotype.pruned_unit_ids}")
    profile_check = validate_profile(context, phenotype, args.profile, args.variant)
    write_json(output / "matched_profile_validation.json", profile_check)
    artifacts = output / "artifacts"
    write_json(artifacts / "phenotype.json", phenotype.to_dict())
    result = evaluator._deploy_and_evaluate(
        phenotype=phenotype,
        output_dir=artifacts,
        candidate_label=f"{args.variant}_production_matched_coverage",
        pruned_unit_ids=[],
        baseline_precision=baseline_precision,
    )
    write_json(output / "production_result_10.json", result)
    if str(result.get("status", "")) != "ok":
        raise RuntimeError(f"production_build_or_10_frame_failed:{result.get('status')}:{result.get('failure_reason', '')}")
    gate10 = evaluation_gate(dict(result["evaluation"]), 10)
    gate10.update(
        {
            "precision_realization_passed": bool(
                json.loads((artifacts / "precision_realization_validation.json").read_text(encoding="utf-8"))["passed"]
            ),
            "merge_realization_passed": bool(
                json.loads((artifacts / "merge_precision_realization.json").read_text(encoding="utf-8"))["passed"]
            ),
            "boundary_audit_passed": bool(
                json.loads((artifacts / "production_qdq_boundary_audit.json").read_text(encoding="utf-8"))["passed"]
            ),
        }
    )
    gate10["passed"] = bool(
        gate10["passed"]
        and gate10["precision_realization_passed"]
        and gate10["merge_realization_passed"]
        and gate10["boundary_audit_passed"]
    )
    write_json(output / "gate_10.json", gate10)
    if not gate10["passed"]:
        raise RuntimeError(f"ten_frame_gate_failed:{gate10}")

    available = load_split_frame_ids(context.model_bundle.adapter, context.model_config, split="val")
    manifest200 = write_eval_manifest(
        output / "manifests/eval_200.json",
        num_frames=200,
        warmup_frames=10,
        available_frame_ids=available,
        reset_after_warmup=True,
    )
    context.eval_manifest_path = manifest200.path
    context.eval_manifest_hash = manifest200.manifest_hash
    evaluator200 = LidarPyramidRealEvaluator(
        context=context,
        run_dir=output / "production_eval_200",
        num_frames=200,
        warmup_frames=10,
        latency_rounds=1,
    )
    evaluation200 = evaluator200._evaluate_engine(result["engine_path"], output / "validation_200")
    gate200 = evaluation_gate(evaluation200, 200)
    gate200["accuracy_gate_mAP_ge_0_58"] = gate200["mAP"] >= 0.58
    write_json(output / "gate_200.json", gate200)
    write_json(
        output / "experiment_summary.json",
        {
            "variant": args.variant,
            "status": "complete",
            "all_keep": True,
            "pruned_unit_count": 0,
            "coverage": (
                "69 INT8 / 1 FP16"
                if args.variant.startswith("ST69-")
                else "67 INT8 / 3 FP16"
                if args.variant.startswith("E67-")
                else "27 INT8 / 43 FP16"
            ),
            "canonical_weighted_count": 70,
            "unmapped_weighted_count": 0,
            "activation_scale_source": (
                "legacy_exact_matched_TensorRT_entropy_cache"
                if args.variant.endswith("-LS")
                else "fresh_production_TensorRT_entropy_calibration2_train200"
            ),
            "gate_10": gate10,
            "gate_200": gate200,
            "full_validation_allowed": bool(gate200["passed"] and gate200["accuracy_gate_mAP_ge_0_58"]),
            "engine_path": str(result["engine_path"]),
            "engine_sha256": sha256_file(Path(result["engine_path"])),
            "qdq_onnx": str(artifacts / "qdq.onnx"),
            "qdq_sha256": sha256_file(artifacts / "qdq.onnx"),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("ST69-ENT", "E67-LS", "E67-ENT", "E27-LS", "E27-ENT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
