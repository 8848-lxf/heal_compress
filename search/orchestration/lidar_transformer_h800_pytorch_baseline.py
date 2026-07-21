"""Run strict-checkpoint PyTorch FP32 Transformer baselines on fixed manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from search.orchestration.lidar_transformer_h800_inventory import MODEL_SPECS


UNIV2X_PYTHON = Path("/home/lixingfeng/miniconda3/envs/univ2x-opt/bin/python")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
PROTOCOL_FRAMES = {"smoke10": 10, "fixed50": 50, "fixed500": 500, "full1789": 1789}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def evaluate(
    *, output_root: Path, model_name: str, protocol: str, physical_gpu: int
) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    manifest = output_root / "evaluation" / "manifests" / model_name / f"{protocol}.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    destination = output_root / "baselines" / model_name / "B0_PYTORCH_FP32" / "evaluation" / protocol
    result_path = destination / "result.json"
    request = {
        "schema_version": "h800-transformer-pytorch-baseline-request-v1",
        "model_name": model_name,
        "conda_env": "univ2x-opt",
        "config_path": str(spec["config"]),
        "checkpoint_path": str(spec["checkpoint"]),
        "heal_root": str(HEAL_ROOT),
        "device": "cuda:0",
        "eval_manifest_path": str(manifest),
        "warmup_frames": len(manifest_data["warmup_frame_ids"]),
        "num_frames": PROTOCOL_FRAMES[protocol],
        "dataloader_num_workers": 8,
        "torch_num_threads": 4,
        "latency_isolation": "accuracy_protocol_not_formal_latency",
        "output_path": str(result_path),
    }
    request_path = destination / "request.json"
    _write_json(request_path, request)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(HEAL_ROOT.parent), str(HEAL_ROOT), env.get("PYTHONPATH", "")]
    )
    command = [
        str(UNIV2X_PYTHON),
        "-m",
        "search.model_family.pytorch_evaluation_worker",
        "--request",
        str(request_path),
    ]
    completed = subprocess.run(command, cwd=repo_root, env=env, text=True, capture_output=True, check=False)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (destination / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {
        "status": "evaluation_failed",
        "failure_reason": f"worker_missing_result:returncode={completed.returncode}",
    }
    accepted = (
        completed.returncode == 0
        and result.get("status") == "ok"
        and int(result.get("num_evaluated_frames", -1)) == PROTOCOL_FRAMES[protocol]
        and int(result.get("num_skipped_frames", -1)) == 0
        and bool(result.get("reset_after_warmup", False))
    )
    summary = {
        "model": model_name,
        "profile": "B0_PYTORCH_FP32",
        "protocol": protocol,
        "physical_gpu": physical_gpu,
        "status": "ok" if accepted else "evaluation_failed",
        "evaluated": result.get("num_evaluated_frames", 0),
        "skipped": result.get("num_skipped_frames", 0),
        "AP@0.3": result.get("AP@0.3"),
        "AP@0.5": result.get("AP@0.5"),
        "AP@0.7": result.get("AP@0.7"),
        "mAP": result.get("mAP"),
        "forward_p50_ms": result.get("forward_p50_ms"),
        "worker_returncode": completed.returncode,
        "failure_reason": result.get("failure_reason", ""),
    }
    _write_json(destination / "evaluation_acceptance.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", default="lidar_cobevt,lidar_v2xvit")
    parser.add_argument("--protocol", choices=tuple(PROTOCOL_FRAMES), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unknown = sorted(set(models) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"unknown_transformer_model:{unknown}")
    rows = [
        evaluate(
            output_root=Path(args.output_root).resolve(),
            model_name=model,
            protocol=args.protocol,
            physical_gpu=args.physical_gpu,
        )
        for model in models
    ]
    print(json.dumps(rows, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
