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
    if len(manifest.get("evaluation_frame_ids", [])) < 500:
        raise RuntimeError("fixed500_manifest_has_fewer_than_500_frames")
    request_source = args.request_source.resolve()
    inherited = json.loads((request_source / "evaluation/v2xvit_greedy005_final_fixed50/evaluation_request.json").read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for control in args.controls:
        engine = args.output_root / "engines" / control / "candidate.plan"
        if not engine.is_file():
            raise RuntimeError(f"engine_missing:{control}:{engine}")
        destination = args.output_root / "evaluation_500" / control
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
            num_frames=500,
            warmup_frames=200,
            latency_rounds=1,
            dataloader_num_workers=8,
        )
        result.update({"control": control, "backend": "TensorRT", "diagnostic_control": False, "manifest_hash": manifest.get("manifest_hash")})
        if result.get("status") != "ok" or int(result.get("num_evaluated_frames", -1)) != 500 or int(result.get("num_skipped_frames", -1)) != 0:
            raise RuntimeError(f"fixed500_evaluation_failed:{control}:{result.get('failure_reason','')}")
        write(destination / "evaluation_result.json", result)
        results[control] = result
    write(args.output_root / "reports" / args.report_name, {"manifest": str(args.fixed500_manifest), "manifest_hash": manifest.get("manifest_hash"), "frames": 500, "controls": results})
    print(json.dumps({control: row.get("mAP") for control, row in results.items()}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-source", type=Path, required=True)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--controls", nargs="+", choices=("B0", "S32", "JMIX-FRESH"), default=("B0", "S32", "JMIX-FRESH"))
    parser.add_argument("--report-name", default="evaluation_500_metrics.json")
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
