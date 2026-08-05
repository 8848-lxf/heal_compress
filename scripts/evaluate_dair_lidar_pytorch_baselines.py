#!/usr/bin/env python3
"""Evaluate every DAIR LiDAR best checkpoint with one immutable FP32 protocol."""

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
for path in (REPO.parent, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_MODELS = Path(
    "${MODEL_ROOT}"
)
DEFAULT_MANIFEST = (
    REPO
    / "outputs/H800_explicit_qdq_acceptance_20260714_023005/"
    "protocol_manifests_v2/eval_1789_warmup200_reset.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _discover(root: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        config = directory / "config.yaml"
        checkpoints = sorted(directory.glob("net_epoch_bestval_at*.pth"))
        if not config.is_file() or len(checkpoints) != 1:
            raise RuntimeError(
                f"baseline_model_inputs_not_unique:{directory}:config={config.is_file()}:"
                f"checkpoints={len(checkpoints)}"
            )
        checkpoint = checkpoints[0]
        rows.append(
            {
                "model_name": directory.name,
                "model_dir": str(directory.resolve()),
                "config_path": str(config.resolve()),
                "config_sha256": _sha256(config),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
            }
        )
    if not rows:
        raise RuntimeError(f"baseline_model_root_empty:{root}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--heal-root", type=Path, default=Path("../../HEAL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=1789)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--voxelization-backend", choices=("gpu", "cpu"), default="gpu"
    )
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--latency-isolation", default="co_resident_processes_recorded")
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "univ2x-opt":
        raise RuntimeError(
            f"dair_lidar_pytorch_baselines_require_univ2x_opt:{os.environ.get('CONDA_DEFAULT_ENV', '')}"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    models = _discover(args.models_root.resolve())
    selected = set(str(value) for value in args.models)
    if selected:
        models = [row for row in models if row["model_name"] in selected]
        missing = sorted(selected - {row["model_name"] for row in models})
        if missing:
            raise RuntimeError(f"requested_baseline_models_missing:{missing}")
    audit = {
        "schema_version": "heal-dair-lidar-baseline-model-audit-v1",
        "models_root": str(args.models_root.resolve()),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "eval_manifest_sha256": _sha256(args.eval_manifest.resolve()),
        "physical_gpu": int(args.physical_gpu),
        "voxelization_backend": str(args.voxelization_backend),
        "evaluation_seed": int(args.evaluation_seed),
        "models": models,
    }
    _write_json(output / "model_input_audit.json", audit)
    summary = []
    for index, model in enumerate(models):
        destination = output / "pytorch_fp32" / model["model_name"]
        destination.mkdir(parents=True, exist_ok=False)
        request = {
            **model,
            "output_path": str((destination / "evaluation.json").resolve()),
            "eval_manifest_path": str(args.eval_manifest.resolve()),
            "heal_root": str(args.heal_root.resolve()),
            "device": "cuda:0",
            "physical_gpu": int(args.physical_gpu),
            "warmup_frames": int(args.warmup_frames),
            "num_frames": int(args.num_frames),
            "dataloader_num_workers": int(args.workers),
            "voxelization_backend": str(args.voxelization_backend),
            "evaluation_seed": int(args.evaluation_seed),
            "torch_num_threads": 4,
            "conda_env": "univ2x-opt",
            "latency_isolation": str(args.latency_isolation),
            "sequence_index": index,
            "sequence_count": len(models),
        }
        request_path = destination / "request.json"
        _write_json(request_path, request)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
        command = [
            sys.executable,
            "-m",
            "search.model_family.pytorch_evaluation_worker",
            "--request",
            str(request_path),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (destination / "worker.log").write_text(completed.stdout or "", encoding="utf-8")
        result_path = destination / "evaluation.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {
            "status": "evaluation_failed",
            "failure_reason": f"worker_no_output_rc_{completed.returncode}",
        }
        summary.append(
            {
                "model_name": model["model_name"],
                "worker_returncode": completed.returncode,
                "result_path": str(result_path),
                "status": result.get("status"),
                "AP@0.3": result.get("AP@0.3"),
                "AP@0.5": result.get("AP@0.5"),
                "AP@0.7": result.get("AP@0.7"),
                "mAP": result.get("mAP"),
                "forward_p50_ms": result.get("forward_p50_ms"),
                "forward_p90_ms": result.get("forward_p90_ms"),
                "forward_p99_ms": result.get("forward_p99_ms"),
                "failure_reason": result.get("failure_reason", ""),
            }
        )
        _write_json(output / "pytorch_fp32_summary.json", {"models": summary})
        if result.get("status") != "ok":
            raise RuntimeError(
                f"pytorch_baseline_failed:{model['model_name']}:{result.get('failure_reason', '')}"
            )
    _write_json(output / "pytorch_fp32_summary.json", {"models": summary, "all_passed": True})
    print(json.dumps({"status": "ok", "output_dir": str(output), "models": len(summary)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
