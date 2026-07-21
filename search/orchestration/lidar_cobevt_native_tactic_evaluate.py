#!/usr/bin/env python3
"""Run comparable CoBEVT engine evaluations on isolated physical GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from search.integration.lidar_cobevt_evaluation_provider import (
    evaluate_cobevt_engine_modelopt,
)


def _run_job(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(args.output_dir).resolve() / str(job["profile"]) / args.phase
    result = evaluate_cobevt_engine_modelopt(
        engine_path=Path(str(job["engine"])).resolve(),
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        heal_root=args.heal_root,
        device=f"cuda:{int(job['gpu'])}",
        output_dir=destination,
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin_path,
        fixed_k=29696,
        num_frames=int(args.frames),
        warmup_frames=20,
        eval_manifest_path=args.manifest,
        num_workers=8,
        ap_iou_backend="gpu",
        strict_gpu_ap_iou=True,
        latency_rounds=1,
        warmup_latency_rounds=1,
        conda_env="modelopt",
    )
    return {"profile": job["profile"], "gpu": int(job["gpu"]), **result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="JSON list of profile/engine/gpu objects")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--heal-root", required=True)
    parser.add_argument("--tensorrt-root", required=True)
    parser.add_argument("--plugin-path", required=True)
    args = parser.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("native_tactic_evaluation_jobs_required")
    physical = [int(job["gpu"]) for job in jobs]
    if len(set(physical)) != len(physical):
        raise ValueError("native_tactic_evaluation_requires_distinct_gpus")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(lambda job: _run_job(job, args), jobs))
    summary = Path(args.output_dir).resolve() / f"{args.phase}_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps([{key: row.get(key) for key in ("profile", "gpu", "status", "AP@0.3", "AP@0.5", "AP@0.7", "mAP", "num_evaluated_frames", "num_skipped_frames")} for row in results], indent=2))
    return 0 if all(str(row.get("status")) == "ok" for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
