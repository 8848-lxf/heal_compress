#!/usr/bin/env python3
"""Wait for one formal budget and evaluate its frozen Greedy or GA winner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    label = str(args.label).zfill(3)
    summary = root / f"ga/budget_{label}/seed_0/budget_summary.json"
    failure = root / f"ga/budget_{label}/failure.json"
    status_path = root / f"reports/final_fixed500_watcher_{label}_{args.control}.json"
    while not summary.is_file():
        if failure.is_file():
            payload = {
                "status": "formal_budget_failed",
                "budget_label": label,
                "control": args.control,
                "failure": json.loads(failure.read_text(encoding="utf-8")),
            }
            _write(status_path, payload)
            return 2
        time.sleep(float(args.poll_seconds))

    budget = json.loads(summary.read_text(encoding="utf-8"))
    key = "final_winner" if args.control == "ga" else "greedy_anchor"
    candidate = budget[key]
    candidate_hash = str(candidate["complete_phenotype_hash"])
    engine = Path(str(candidate["metadata"]["engine_path"]))
    destination = (
        root
        / f"{args.control}_final_fixed500_partial"
        / f"budget_{label}"
        / candidate_hash
    )
    result_path = destination / "evaluation.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reused = True
    else:
        result = evaluate_v2xvit_engine_modelopt(
            engine_path=engine,
            model_config=args.model_config,
            heal_root=args.heal_root,
            output_dir=destination,
            tensorrt_root=args.tensorrt_root,
            plugin_path=args.plugin,
            eval_manifest_path=args.fixed500_manifest,
            physical_gpu_id=int(args.physical_gpu),
            fixed_k=int(args.fixed_k),
            max_agents=2,
            num_frames=500,
            warmup_frames=200,
            latency_rounds=1,
            dataloader_num_workers=8,
        )
        reused = False
    passed = bool(
        result.get("status") == "ok"
        and int(result.get("num_evaluated_frames", -1)) == 500
        and int(result.get("num_skipped_frames", -1)) == 0
    )
    _write(
        status_path,
        {
            "status": "ok" if passed else "fixed500_failed",
            "budget_label": label,
            "control": args.control,
            "candidate_hash": candidate_hash,
            "engine": str(engine),
            "physical_gpu": int(args.physical_gpu),
            "result": result,
            "reused_existing_result": reused,
        },
    )
    return 0 if passed else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", choices=("030", "025", "020", "015", "010", "005"), required=True)
    parser.add_argument("--control", choices=("greedy", "ga"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
