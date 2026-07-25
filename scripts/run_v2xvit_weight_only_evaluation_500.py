#!/usr/bin/env python3
"""Run identical fixed500 AP evaluation for B0, S32 and JMIX-FRESH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.fixed500_manifest.read_text(encoding="utf-8"))
    if len(manifest.get("evaluation_frame_ids", [])) < int(args.num_frames):
        raise RuntimeError("evaluation_manifest_has_too_few_frames")
    request_source = args.request_source.resolve()
    request_path = args.request_json or (request_source / "evaluation/v2xvit_greedy005_final_fixed50/evaluation_request.json")
    if not request_path.is_file():
        request_path = request_source / "evaluation_500/B0/evaluation_request.json"
    inherited = json.loads(request_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for control in args.controls:
        engine = args.output_root / "engines" / control / "candidate.plan"
        if not engine.is_file():
            raise RuntimeError(f"engine_missing:{control}:{engine}")
        destination = args.output_root / args.output_dir_name / control
        result = evaluate_v2xvit_engine_modelopt(
            engine_path=engine,
            model_config=inherited["model_config"],
            heal_root=inherited["heal_root"],
            output_dir=destination,
            tensorrt_root=args.tensorrt_root,
            plugin_path=inherited["plugin_path"],
            eval_manifest_path=args.fixed500_manifest,
            physical_gpu_id=int(args.physical_gpu),
            fixed_k=int(inherited["fixed_k"]),
            max_agents=int(inherited["max_agents"]),
            num_frames=int(args.num_frames),
            warmup_frames=int(args.warmup_frames),
            latency_rounds=1,
            dataloader_num_workers=8,
        )
        result.update({"control": control, "backend": "TensorRT", "diagnostic_control": False, "manifest_hash": manifest.get("manifest_hash")})
        if result.get("status") != "ok" or int(result.get("num_evaluated_frames", -1)) != int(args.num_frames) or int(result.get("num_skipped_frames", -1)) != 0:
            raise RuntimeError(f"fixed_evaluation_failed:{control}:{result.get('failure_reason','')}")
        write(destination / "evaluation_result.json", result)
        results[control] = result
    write(args.output_root / "reports" / args.report_name, {"manifest": str(args.fixed500_manifest), "manifest_hash": manifest.get("manifest_hash"), "frames": int(args.num_frames), "controls": results})
    print(json.dumps({control: row.get("mAP") for control, row in results.items()}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-source", type=Path, required=True)
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--controls", nargs="+", choices=("B0", "S32", "JMIX-FRESH"), default=("B0", "S32", "JMIX-FRESH"))
    parser.add_argument("--report-name", default="evaluation_500_metrics.json")
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--output-dir-name", default="evaluation_500")
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
