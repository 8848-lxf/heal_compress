#!/usr/bin/env python3
"""Evaluate diagnostic TensorRT controls on the inherited immutable manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = args.output_root.resolve()
    source = args.source_root.resolve()
    inherited = json.loads((source / "evaluation/v2xvit_greedy005_final_fixed50/evaluation_request.json").read_text(encoding="utf-8"))
    results = {}
    for control in args.controls:
        engine = out / "tensorrt" / control.replace("-", "_") / "candidate.plan"
        if not engine.is_file():
            raise RuntimeError(f"diagnostic_engine_missing:{control}:{engine}")
        per_control = {}
        for label, frames in (("smoke10", 10), ("fixed50", 50)):
            destination = out / "tensorrt" / "evaluation" / control.replace("-", "_") / label
            result = evaluate_v2xvit_engine_modelopt(
                engine_path=engine,
                model_config=inherited["model_config"],
                heal_root=inherited["heal_root"],
                output_dir=destination,
                tensorrt_root=args.tensorrt_root,
                plugin_path=inherited["plugin_path"],
                eval_manifest_path=inherited["eval_manifest_path"],
                physical_gpu_id=5,
                fixed_k=27904,
                max_agents=2,
                num_frames=frames,
                warmup_frames=5,
                latency_rounds=1,
                dataloader_num_workers=8,
            )
            result["diagnostic_control"] = True
            result["control_name"] = control
            result["backend"] = "TensorRT"
            if result.get("status") != "ok" or int(result.get("num_skipped_frames", -1)) != 0:
                raise RuntimeError(f"diagnostic_trt_evaluation_failed:{control}:{label}:{result.get('failure_reason','')}")
            per_control[label] = result
        results[control] = per_control
    write(out / "reports/phase2_trt_evaluations.json", {"schema_version": "v2xvit-greedy005-trt-evaluations-v1", "controls": results, "formal_latency": False, "diagnostic_control": True})
    print(json.dumps({"status": "ok", "controls": list(results), "formal_latency": False}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controls", nargs="+", default=["S32", "S16", "JMIX-FRESH"])
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
