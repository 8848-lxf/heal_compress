#!/usr/bin/env python3
"""Run CoBEVT six-budget, single-seed, ten-generation strict formal GA."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.ga.cnn_stage12_v3 import (
    create_real_evaluator,
    greedy_anchors,
    run_budget,
    write_csv,
    write_json,
)
from search.ga.transformer_stage12_v3 import prepare_cobevt_search
from search.integration.runtime_environment import query_gpus


TARGETS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)


def _gpu_uuid(index: int) -> str:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = {
        int(parts[0].strip()): parts[1].strip()
        for line in text.splitlines()
        if len(parts := line.split(",", 1)) == 2
    }
    return rows[int(index)]


def run(args: argparse.Namespace) -> int:
    if int(args.seed) != 0:
        raise RuntimeError("cobevt_formal_ga_requires_seed_zero")
    if int(args.generations) != 10:
        raise RuntimeError("cobevt_formal_ga_requires_exactly_ten_generations")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", str(args.physical_gpu)):
        raise RuntimeError("cobevt_formal_ga_cuda_visible_devices_mismatch")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise RuntimeError(f"cobevt_formal_ga_output_root_not_empty:{root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "provenance", "proxy", "greedy", "ga", "stage2_runtime", "reports",
        "logs", "process_snapshots", "evaluation_fixed500", "latency",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    write_json(
        root / "provenance/start.json",
        {
            "model": "cobevt",
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=REPO, text=True
            ).strip(),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "physical_gpu": int(args.physical_gpu),
            "gpu_uuid": _gpu_uuid(args.physical_gpu),
            "all_gpus": query_gpus(),
            "framework": "StrictStage12V3Runner",
            "seed_count": 1,
            "executed_seeds": [0],
            "generations": 10,
            "population_size": 64,
            "offspring_size": 64,
            "stage2_quota": 5,
            "targets": list(TARGETS),
            "full1789": False,
        },
    )
    prepared = prepare_cobevt_search(
        output_root=root,
        physical_gpu=args.physical_gpu,
        plugin=args.plugin.resolve(),
        tensorrt_root=args.tensorrt_root.resolve(),
        taylor_samples=32,
    )
    anchors = greedy_anchors(prepared, targets=TARGETS, output_root=root)
    real = create_real_evaluator(prepared, output_root=root)
    results = {}
    failures = []
    for target in TARGETS:
        label = f"{int(round(target * 100)):03d}"
        anchor = anchors.get(target)
        if anchor is None:
            failures.append({
                "budget": target,
                "status": "budget_unreachable_by_selected_greedy_trajectory",
            })
            continue
        try:
            results[label] = run_budget(
                prepared,
                target=target,
                anchor_genotype=anchor,
                output_root=root,
                seed=0,
                generations=10,
                real_evaluator=real,
            )
        except Exception as exc:
            failure = {
                "budget": target,
                "status": "formal_budget_failed",
                "failure": f"{type(exc).__name__}:{exc}",
            }
            failures.append(failure)
            write_json(root / f"ga/budget_{label}/failure.json", failure)
            print(json.dumps(failure, sort_keys=True), flush=True)
    rows = [
        {
            "budget": row["target_bops"],
            "completed_generations": row["completed_evolution_generations"],
            "stage2_real_evaluation_count": row["stage2_real_evaluation_count"],
            "greedy_hash": row["greedy_anchor"]["complete_phenotype_hash"],
            "greedy_map_fixed50": row["greedy_anchor"]["mAP"],
            "greedy_p50_ms": row["greedy_anchor"]["p50_ms"],
            "ga_hash": row["final_winner"]["complete_phenotype_hash"],
            "ga_map_fixed50": row["final_winner"]["mAP"],
            "ga_p50_ms": row["final_winner"]["p50_ms"],
            "ga_improved_greedy": row["ga_improved_greedy"],
        }
        for row in results.values()
    ]
    write_csv(root / "reports/formal_ga_budget_summary.csv", rows)
    write_json(
        root / "reports/formal_ga_results.json",
        {
            "model": "cobevt",
            "framework": "StrictStage12V3Runner",
            "seed_count": 1,
            "executed_seeds": [0],
            "formal_generations": 10,
            "generation_zero_counted": False,
            "population_size": 64,
            "offspring_size": 64,
            "stage2_new_candidate_quota": 5,
            "targets": list(TARGETS),
            "results": results,
            "failures": failures,
            "full1789_executed": False,
        },
    )
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
