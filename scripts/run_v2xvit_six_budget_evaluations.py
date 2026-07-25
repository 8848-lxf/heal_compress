#!/usr/bin/env python3
"""Evaluate B0 and selected six-budget S32/JMIX engines with fixed500."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/lixingfeng/anaconda3/envs/modelopt/bin/python")
BUDGETS = ("030", "025", "020", "015", "010", "005")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _env(gpu: int) -> dict[str, str]:
    prefix = "/home/lixingfeng/anaconda3/envs/modelopt"
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CONDA_PREFIX": prefix,
            "PATH": f"{prefix}/bin:{env.get('PATH', '')}",
            "PYTHONPATH": ":".join(
                (
                    "/home/lixingfeng/UniAD_examine/HEAL",
                    "/home/lixingfeng/UniAD_examine/HEAL/prune_model/Model-Optimizer-0.29.0",
                    str(REPO),
                    env.get("PYTHONPATH", ""),
                )
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _evaluate(engine_dir: Path, output_dir: Path, manifest: Path, gpu: int) -> dict[str, Any]:
    engine = engine_dir / "candidate.plan"
    if not engine.is_file():
        return {"status": "engine_missing", "engine": str(engine), "physical_gpu": gpu}
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(PYTHON),
        str(REPO / "scripts/evaluate_v2xvit_deployment_closed_engine.py"),
        "--engine",
        str(engine),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--frames",
        "500",
        "--physical-gpu",
        str(gpu),
        "--fixed-k",
        "27904",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=_env(gpu),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (output_dir / "orchestration.log").write_text(completed.stdout, encoding="utf-8")
    evaluation = output_dir / "evaluation.json"
    payload = json.loads(evaluation.read_text(encoding="utf-8")) if evaluation.is_file() else {}
    return {
        "status": payload.get("status", "failed_no_evaluation"),
        "returncode": completed.returncode,
        "engine": str(engine),
        "physical_gpu": gpu,
        "num_evaluated_frames": payload.get("num_evaluated_frames", 0),
        "num_skipped_frames": payload.get("num_skipped_frames", 0),
        "AP30": payload.get("AP@0.3"),
        "AP50": payload.get("AP@0.5"),
        "AP70": payload.get("AP@0.7"),
        "mAP": payload.get("mAP"),
        "manifest_hash": payload.get("eval_manifest_hash"),
        "workers": payload.get("dataloader_num_workers"),
        "cuda_postprocess_passed": payload.get("cuda_postprocess_audit", {}).get("passed"),
        "success": completed.returncode == 0
        and payload.get("status") == "ok"
        and payload.get("num_evaluated_frames") == 500
        and payload.get("num_skipped_frames") == 0,
    }


def _budget_pair(root: Path, budget: str, manifest: Path, gpu: int) -> list[dict[str, Any]]:
    audit = json.loads((root / "budgets" / budget / "build_audit.json").read_text(encoding="utf-8"))
    rows = []
    for profile, key in (("S32", "selected_s32_dir"), ("JMIX", "selected_jmix_dir")):
        engine_dir = Path(str(audit.get(key, "")))
        row = _evaluate(
            engine_dir,
            root / "budgets" / budget / "fixed500" / profile,
            manifest,
            gpu,
        )
        rows.append({"budget": budget, "profile": profile, **row})
        if not row["success"]:
            break
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    manifest = args.manifest.resolve()
    b0 = _evaluate(root / "engines/B0", root / "evaluation_fixed500/B0", manifest, args.physical_gpus[0])
    if not b0["success"]:
        raise RuntimeError(f"six_budget_b0_fixed500_failed:{b0}")
    rows = [{"budget": "B0", "profile": "B0", **b0}]
    with ThreadPoolExecutor(max_workers=len(args.physical_gpus)) as executor:
        futures = {
            executor.submit(
                _budget_pair,
                root,
                budget,
                manifest,
                args.physical_gpus[index % len(args.physical_gpus)],
            ): budget
            for index, budget in enumerate(BUDGETS)
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    result = {
        "schema_version": "v2xvit-six-budget-fixed500-v1",
        "manifest": str(manifest),
        "B0": b0,
        "rows": sorted(rows, key=lambda row: (str(row["budget"]), str(row["profile"]))),
        "all_success": len(rows) == 13 and all(row["success"] for row in rows),
        "workers_per_evaluation": 8,
        "cuda_postprocess_required": True,
    }
    _write(root / "reports/six_budget_fixed500_raw.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    result = run(parser.parse_args())
    return 0 if result["all_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
