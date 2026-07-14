#!/usr/bin/env python3
"""Build a diagnostic augmented engine for one production matched-coverage run."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.explicit_qdq_ablation import tensor_thresholds  # noqa: E402
from scripts.int8_equivalence_tensor_parity import (  # noqa: E402
    Accumulator,
    CURRENT_PLUGIN,
    augment_outputs,
    official_trtexec_command,
    ort_session,
    read_json,
    require_modelopt,
    run_build,
    sha256_file,
    write_json,
)
from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner  # noqa: E402


def main(args: argparse.Namespace) -> None:
    environment = require_modelopt()
    production = args.production_artifacts.resolve()
    reference = args.reference_work.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    required = [
        production / "qdq.onnx",
        production / "qdq_trt_compatible.onnx",
        production / "engine_build.log",
        production / "precision_realization_validation.json",
        production / "merge_precision_realization.json",
        production / "production_qdq_boundary_audit.json",
        reference / "ten_frame_inputs.json",
        reference / "ort_fp32_augmented.onnx",
        reference / "target_map.json",
        reference / "route_pytorch_fp32.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"production_tensor_parity_inputs_missing:{missing}")

    shutil.copyfile(reference / "ten_frame_inputs.json", output / "ten_frame_inputs.json")
    augmented = output / "augmented.onnx"
    target_map = augment_outputs(production / "qdq_trt_compatible.onnx", augmented)
    ort_map = read_json(reference / "target_map.json", {})["ort_fp32"]
    missing_targets = sorted(set(ort_map) - set(target_map))
    if missing_targets:
        raise RuntimeError(f"production_tensor_parity_targets_missing:{missing_targets}")
    command = official_trtexec_command(
        production / "engine_build.log",
        augmented,
        output / "augmented.engine",
        output / "augmented_layer_info.json",
    )
    build = run_build(command, output / "augmented_build.log")
    build["engine_exists"] = (output / "augmented.engine").is_file()
    if build["engine_exists"]:
        build["engine_sha256"] = sha256_file(output / "augmented.engine")
    write_json(output / "debug_build_result.json", build)
    if build["returncode"] != 0 or not build["engine_exists"]:
        raise RuntimeError(f"production_augmented_engine_build_failed:{build}")

    ctypes.CDLL(str(CURRENT_PLUGIN.resolve()), mode=ctypes.RTLD_GLOBAL)
    session = ort_session(reference / "ort_fp32_augmented.onnx")
    runner = TensorRTEngineRunner(output / "augmented.engine", torch.device("cuda:0"))
    thresholds = tensor_thresholds(production / "qdq.onnx", target_map)
    accumulators = {alias: Accumulator() for alias in target_map if alias in ort_map}
    per_frame: list[dict[str, Any]] = []
    for frame in read_json(output / "ten_frame_inputs.json", {}).get("frames", []):
        values = np.load(frame["path"])
        feeds = {name: values[name] for name in values.files}
        ort_values = session.run(None, feeds)
        ort_outputs = {row.name: value for row, value in zip(session.get_outputs(), ort_values)}
        outputs = runner.run(
            {name: torch.as_tensor(value, device="cuda:0") for name, value in feeds.items()}
        )
        frame_status: dict[str, Any] = {}
        for alias, engine_name in target_map.items():
            reference_name = ort_map.get(alias)
            if not reference_name or reference_name not in ort_outputs or engine_name not in outputs:
                frame_status[alias] = {
                    "missing": True,
                    "reference": reference_name,
                    "candidate": engine_name,
                }
                continue
            reference_value = np.asarray(ort_outputs[reference_name])
            candidate_value = outputs[engine_name].detach().float().cpu().numpy()
            accumulators[alias].update(reference_value, candidate_value, thresholds.get(alias))
            frame_status[alias] = {
                "reference_shape": list(reference_value.shape),
                "candidate_shape": list(candidate_value.shape),
            }
        per_frame.append({"frame_id": frame["frame_id"], "tensors": frame_status})

    metrics = {alias: accumulator.result() for alias, accumulator in accumulators.items()}
    order = list(target_map)
    first_abnormal = next(
        (
            alias
            for alias in order
            if float(metrics[alias].get("cosine", 1.0)) < 0.99
            or float(metrics[alias].get("SQNR_dB") or 999.0) < 20.0
        ),
        None,
    )
    result = {
        "status": "ok",
        "label": args.label,
        "diagnostic_engine_warning": (
            "Extra outputs alter TensorRT fusion; official AP/latency uses the unmodified production engine."
        ),
        "frame_count": len(per_frame),
        "reference": "ORT FP32 with standard scatter reconstruction",
        "reference_work": str(reference),
        "reference_onnx_sha256": sha256_file(reference / "ort_fp32_augmented.onnx"),
        "pytorch_fp32_reference": read_json(reference / "route_pytorch_fp32.json", {}),
        "production_artifacts": str(production),
        "production_qdq_sha256": sha256_file(production / "qdq.onnx"),
        "production_engine_sha256": sha256_file(production / "engine.plan"),
        "precision_realization": read_json(production / "precision_realization_validation.json", {}),
        "merge_realization": read_json(production / "merge_precision_realization.json", {}),
        "boundary_audit": read_json(production / "production_qdq_boundary_audit.json", {}),
        "target_map": target_map,
        "saturation_thresholds": thresholds,
        "metrics": metrics,
        "first_abnormal_tensor": first_abnormal,
        "frames": per_frame,
        "environment": environment,
        "build": build,
    }
    write_json(output / "tensor_parity.json", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-artifacts", type=Path, required=True)
    parser.add_argument("--reference-work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
