#!/usr/bin/env python3
"""Sequentially build and fully evaluate strict-FP32 TRT baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly"
)
DEFAULT_TRT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
DEFAULT_PLUGIN = REPO / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--fixed-k-audit", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TRT)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=1789)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "univ2x-opt":
        raise RuntimeError("trt_fp32_baseline_orchestrator_requires_univ2x_opt")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixed_k_payload = json.loads(args.fixed_k_audit.read_text(encoding="utf-8"))
    fixed_k = {row["model_name"]: int(row["fixed_k_required"]) for row in fixed_k_payload["models"]}
    selected = set(str(value) for value in args.models)
    model_dirs = sorted(path for path in args.models_root.iterdir() if path.is_dir())
    if selected:
        model_dirs = [path for path in model_dirs if path.name in selected]
    summary = []
    for directory in model_dirs:
        checkpoints = list(directory.glob("net_epoch_bestval_at*.pth"))
        if len(checkpoints) != 1 or not (directory / "config.yaml").is_file():
            raise RuntimeError(f"strict_fp32_model_inputs_not_unique:{directory}")
        if directory.name not in fixed_k:
            raise RuntimeError(f"strict_fp32_fixed_k_missing:{directory.name}")
        destination = output / directory.name
        destination.mkdir(parents=True, exist_ok=False)
        request = {
            "model_name": directory.name,
            "config_path": str((directory / "config.yaml").resolve()),
            "checkpoint_path": str(checkpoints[0].resolve()),
            "fixed_k": fixed_k[directory.name],
            "eval_manifest_path": str(args.eval_manifest.resolve()),
            "output_dir": str(destination.resolve()),
            "physical_gpu": int(args.physical_gpu),
            "heal_root": str(args.heal_root.resolve()),
            "tensorrt_root": str(args.tensorrt_root.resolve()),
            "plugin_path": str(args.plugin.resolve()),
            "warmup_frames": int(args.warmup_frames),
            "num_frames": int(args.num_frames),
            "dataloader_num_workers": int(args.workers),
            "latency_rounds": 1,
        }
        request_path = destination / "request.json"
        _write_json(request_path, request)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
        completed = subprocess.run(
            [sys.executable, "-m", "search.model_family.fp32_deployment_worker", "--request", str(request_path)],
            cwd=REPO,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (destination / "worker.log").write_text(completed.stdout or "", encoding="utf-8")
        result_path = destination / "deployment_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {
            "status": "failed", "failure_reason": f"worker_no_output_rc_{completed.returncode}"
        }
        evaluation = result.get("evaluation", {})
        summary.append(
            {
                "model_name": directory.name,
                "status": result.get("status"),
                "failure_reason": result.get("failure_reason", ""),
                "AP@0.3": evaluation.get("AP@0.3"),
                "AP@0.5": evaluation.get("AP@0.5"),
                "AP@0.7": evaluation.get("AP@0.7"),
                "mAP": evaluation.get("mAP"),
                "forward_p50_ms": evaluation.get("forward_p50_ms"),
                "forward_p90_ms": evaluation.get("forward_p90_ms"),
                "forward_p99_ms": evaluation.get("forward_p99_ms"),
                "result_path": str(result_path),
            }
        )
        _write_json(output / "strict_fp32_trt_summary.json", {"models": summary})
        if result.get("status") != "ok":
            raise RuntimeError(f"strict_fp32_trt_model_failed:{directory.name}:{result.get('failure_reason', '')}")
    _write_json(output / "strict_fp32_trt_summary.json", {"models": summary, "all_passed": True})
    print(json.dumps({"status": "ok", "models": len(summary), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
