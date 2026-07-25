#!/usr/bin/env python3
"""Evaluate the rebuilt R=0.10 Greedy B0/S32/JMIX controls on fixed50.

The resulting request is also frozen as the common fixed500/GA deployment
request.  This script never substitutes another budget-band candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


CONTROLS = {
    "B0": "engines/greedy_exact_winners/B0/candidate.plan",
    "Greedy-S32": "engines/greedy_exact_winners/budget_010/S32/candidate.plan",
    "Greedy-JMIX": "engines/greedy_exact_winners/budget_010/JMIX-FRESH/candidate.plan",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    request = json.loads(args.request_json.read_text())
    manifest = json.loads(args.fixed50_manifest.read_text())
    results: dict[str, Any] = {}
    for control, relative in CONTROLS.items():
        engine = root / relative
        if not engine.is_file():
            raise RuntimeError(f"greedy_r010_fixed50_engine_missing:{control}:{engine}")
        destination = root / "evaluation_fixed50" / control
        result = evaluate_v2xvit_engine_modelopt(
            engine_path=engine,
            model_config=request["model_config"],
            heal_root=request["heal_root"],
            output_dir=destination,
            tensorrt_root=args.tensorrt_root,
            plugin_path=request["plugin_path"],
            eval_manifest_path=args.fixed50_manifest,
            physical_gpu_id=args.physical_gpu,
            fixed_k=int(request["fixed_k"]),
            max_agents=int(request["max_agents"]),
            num_frames=50,
            warmup_frames=20,
            latency_rounds=1,
            dataloader_num_workers=8,
        )
        valid = bool(
            result.get("status") == "ok"
            and int(result.get("num_evaluated_frames", -1)) == 50
            and int(result.get("num_skipped_frames", -1)) == 0
        )
        results[control] = {**result, "fixed50_gate_passed": valid}
        write_json(root / "reports/greedy_r010_fixed50.json", {
            "manifest": str(args.fixed50_manifest),
            "manifest_hash": manifest.get("manifest_hash"),
            "controls": results,
        })
        if not valid:
            raise RuntimeError(f"greedy_r010_fixed50_failed:{control}")

    fixed500_request = {
        **request,
        "source": "rebuilt_r010_greedy_controls",
        "fixed50_manifest_hash": manifest.get("manifest_hash"),
    }
    write_json(root / "evaluation_fixed500/B0/evaluation_request.json", fixed500_request)
    write_json(root / "reports/ga_budget_admission.json", {
        "ga_admissible_budgets": [0.10],
        "basis": "new_contract_greedy_anchor_fixed50_and_deployment_closure",
        "greedy_anchor_hash": json.loads(
            (root / "greedy/budget_010/exact_winner.json").read_text()
        )["candidate_hash"],
        "fixed50": {
            key: {"mAP": value["mAP"], "evaluated": 50, "skipped": 0}
            for key, value in results.items()
        },
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--fixed50-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument(
        "--tensorrt-root", type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
