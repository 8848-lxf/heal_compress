#!/usr/bin/env python3
"""Evaluate B0 and six exact Greedy S32/JMIX engines on one fixed500."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


LABELS = ("030", "025", "020", "015", "010", "005")


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    inherited = json.loads(args.request_json.read_text(encoding="utf-8"))
    manifest = json.loads(args.fixed500_manifest.read_text(encoding="utf-8"))
    jobs = [("B0", root / "engines/greedy_exact_winners/B0/candidate.plan")]
    for label in LABELS:
        for control in ("S32", "JMIX-FRESH"):
            jobs.append(
                (
                    f"budget_{label}/{control}",
                    root
                    / f"engines/greedy_exact_winners/budget_{label}/{control}/candidate.plan",
                )
            )
    results: dict[str, Any] = {}
    summary_path = root / "reports/six_budget_fixed500.json"
    if summary_path.is_file():
        results.update(json.loads(summary_path.read_text()).get("controls", {}))
    for name, engine in jobs:
        if name in results and results[name].get("status") == "ok":
            continue
        if not engine.is_file():
            raise RuntimeError(f"fixed500_engine_missing:{name}:{engine}")
        destination = root / "evaluation_fixed500" / name
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
        if (
            result.get("status") != "ok"
            or int(result.get("num_evaluated_frames", -1)) != 500
            or int(result.get("num_skipped_frames", -1)) != 0
        ):
            result["fixed500_gate_passed"] = False
            results[name] = result
            atomic_write(
                summary_path,
                {
                    "manifest": str(args.fixed500_manifest),
                    "manifest_hash": manifest.get("manifest_hash"),
                    "controls": results,
                },
            )
            raise RuntimeError(
                f"fixed500_failed:{name}:{result.get('failure_reason', '')}"
            )
        result.update(
            {
                "control": name,
                "fixed500_gate_passed": True,
                "manifest_hash": manifest.get("manifest_hash"),
            }
        )
        results[name] = result
        atomic_write(
            destination / "evaluation_result.json",
            result,
        )
        atomic_write(
            summary_path,
            {
                "manifest": str(args.fixed500_manifest),
                "manifest_hash": manifest.get("manifest_hash"),
                "controls": results,
            },
        )
        print(
            json.dumps(
                {
                    "control": name,
                    "mAP": result.get("mAP"),
                    "AP30": result.get("AP@0.3"),
                    "AP50": result.get("AP@0.5"),
                    "AP70": result.get("AP@0.7"),
                    "evaluated": result.get("num_evaluated_frames"),
                    "skipped": result.get("num_skipped_frames"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=6)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
