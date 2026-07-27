#!/usr/bin/env python3
"""Run HEAL CNN-family models with strict single-seed Stage-1/Stage-2/V1--V3 GA."""

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

from search.ga.cnn_stage12_v3 import (  # noqa: E402
    GENERATION_WINNER_FRAMES,
    GENERATION_WINNER_WARMUP_FRAMES,
    MODEL_SPECS,
    STAGE2_SCREENING_FRAMES,
    STAGE2_SCREENING_WARMUP_FRAMES,
    create_real_evaluator,
    greedy_anchors,
    prepare_search,
    run_budget,
    write_csv,
    write_json,
)
from search.integration.runtime_environment import query_gpus  # noqa: E402


DEFAULT_TARGETS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True
    ).strip()


def gpu_uuid(physical_gpu: int) -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = {
        int(parts[0].strip()): parts[1].strip()
        for line in output.splitlines()
        if len(parts := line.split(",", 1)) == 2
    }
    if int(physical_gpu) not in rows:
        raise RuntimeError(f"cnn_formal_ga_gpu_uuid_missing:{physical_gpu}")
    return rows[int(physical_gpu)]


def run(args: argparse.Namespace) -> int:
    if int(args.generations) != 10:
        raise RuntimeError(f"cnn_formal_ga_requires_exactly_10_generations:{args.generations}")
    if int(args.seed) != 0:
        raise RuntimeError(f"cnn_formal_ga_single_seed_zero_required:{args.seed}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", str(args.physical_gpu)):
        raise RuntimeError(
            "cnn_formal_ga_cuda_visible_devices_mismatch:"
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}!={args.physical_gpu}"
        )
    targets = tuple(float(value) for value in args.targets.split(",") if value)
    if targets != DEFAULT_TARGETS:
        raise RuntimeError(f"cnn_formal_ga_budget_contract_mismatch:{targets}")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise RuntimeError(f"cnn_formal_ga_output_root_not_empty:{root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "provenance", "proxy", "greedy", "ga", "stage2_runtime", "reports",
        "logs", "process_snapshots", "evaluation_fixed500", "latency",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[args.model]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpu_rows = query_gpus()
    selected = next(
        (row for row in gpu_rows if int(row["index"]) == int(args.physical_gpu)), None
    )
    if selected is None:
        raise RuntimeError(f"cnn_formal_ga_physical_gpu_missing:{args.physical_gpu}")
    selected_uuid = gpu_uuid(args.physical_gpu)
    write_json(root / "provenance/start.json", {
        "model": args.model,
        "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "physical_gpu": int(args.physical_gpu),
        "gpu_uuid": selected_uuid,
        "gpu": selected,
        "all_gpus": gpu_rows,
        "framework": "StrictStage12V3Runner",
        "generation_contract": "formal_gen10",
        "generations": 10,
        "population_size": 64,
        "offspring_size": 64,
        "stage2_quota": 5,
        "stage2_top5_screening_frames": STAGE2_SCREENING_FRAMES,
        "stage2_top5_screening_warmup_frames": STAGE2_SCREENING_WARMUP_FRAMES,
        "generation_winner_validation_frames": GENERATION_WINNER_FRAMES,
        "generation_winner_validation_warmup_frames": GENERATION_WINNER_WARMUP_FRAMES,
        "seed": 0,
        "targets": list(targets),
        "old_framework_started": False,
        "full1789": False,
    })
    print(json.dumps({
        "event": "cnn_formal_ga_start",
        "model": args.model,
        "gpu": args.physical_gpu,
        "gpu_uuid": selected_uuid,
        "framework": "StrictStage12V3Runner",
        "generations": 10,
        "targets": targets,
        "output_root": str(root),
    }, sort_keys=True), flush=True)
    prepared = prepare_search(
        spec,
        output_root=root,
        physical_gpu=args.physical_gpu,
        plugin=args.plugin.resolve(),
        tensorrt_root=args.tensorrt_root.resolve(),
        taylor_samples=args.taylor_samples,
    )
    anchors = greedy_anchors(prepared, targets=targets, output_root=root)
    real_evaluator = create_real_evaluator(
        prepared,
        output_root=root,
        num_frames=STAGE2_SCREENING_FRAMES,
        warmup_frames=STAGE2_SCREENING_WARMUP_FRAMES,
        run_dir_name="stage2_screening_runtime",
    )
    validation_evaluator = create_real_evaluator(
        prepared,
        output_root=root,
        num_frames=GENERATION_WINNER_FRAMES,
        warmup_frames=GENERATION_WINNER_WARMUP_FRAMES,
        run_dir_name="generation_winner_validation_runtime",
    )
    results: dict[str, dict] = {}
    failures: list[dict] = []
    for target in targets:
        label = f"{int(round(target * 100)):03d}"
        anchor = anchors.get(target)
        if anchor is None:
            failures.append({
                "target": target,
                "status": "budget_unreachable_by_selected_greedy_trajectory",
            })
            continue
        try:
            results[label] = run_budget(
                prepared,
                target=target,
                anchor_genotype=anchor,
                output_root=root,
                seed=args.seed,
                generations=args.generations,
                real_evaluator=real_evaluator,
                validation_evaluator=validation_evaluator,
            )
        except Exception as exc:  # preserve other completed budgets, fail closed per budget
            failure = {
                "target": target,
                "status": "formal_budget_failed",
                "failure": f"{type(exc).__name__}:{exc}",
            }
            failures.append(failure)
            write_json(root / f"ga/budget_{label}/failure.json", failure)
            print(json.dumps({"event": "cnn_formal_ga_budget_failed", **failure},
                             sort_keys=True), flush=True)
    summary_rows = []
    for label, row in results.items():
        greedy = row["greedy_anchor"]
        final = row["final_winner"]
        summary_rows.append({
            "model": args.model,
            "budget": row["target_bops"],
            "completed_generations": row["completed_evolution_generations"],
            "stage2_real_evaluation_count": row["stage2_real_evaluation_count"],
            "greedy_hash": greedy["complete_phenotype_hash"],
            "greedy_map_fixed500": greedy["mAP"],
            "greedy_p50_ms": greedy["p50_ms"],
            "ga_hash": final["complete_phenotype_hash"],
            "ga_map_fixed500": final["mAP"],
            "ga_p50_ms": final["p50_ms"],
            "ga_improved_greedy": row["ga_improved_greedy"],
        })
    write_csv(root / "reports/formal_ga_budget_summary.csv", summary_rows)
    write_json(root / "reports/formal_ga_results.json", {
        "model": args.model,
        "framework": "StrictStage12V3Runner",
        "old_two_stage_search_used": False,
        "generation_zero_counted": False,
        "formal_generations": 10,
        "formal_generation_ids": list(range(1, 11)),
        "seed_count": 1,
        "executed_seeds": [0],
        "population_size": 64,
        "offspring_size": 64,
        "survivor_size": 64,
        "stage2_new_candidate_quota": 5,
        "targets": list(targets),
        "results": results,
        "failures": failures,
        "full1789_executed": False,
    })
    write_json(root / "reports/final_acceptance.json", {
        "model": args.model,
        "new_ga_framework_used": True,
        "runner": "StrictStage12V3Runner",
        "old_ga_framework_used": False,
        "generations_requested": 10,
        "generation_ids": list(range(1, 11)),
        "generation_zero_counted": False,
        "seed_count": 1,
        "population_size": 64,
        "offspring_size": 64,
        "stage2_quota": 5,
        "stage1_proxy": "J_struct_gate + J_WQ + J_AQ",
        "repair_enabled": False,
        "budgets_requested": list(targets),
        "budgets_completed": [float(row["target_bops"]) for row in results.values()],
        "failures": failures,
        "full1789_executed": False,
    })
    gpu_end = query_gpus()
    write_json(root / "provenance/end.json", {
        "gpu": next(
            row for row in gpu_end if int(row["index"]) == int(args.physical_gpu)
        ),
        "gpu_uuid": gpu_uuid(args.physical_gpu),
        "all_gpus": gpu_end,
    })
    print(json.dumps({
        "event": "cnn_formal_ga_complete",
        "model": args.model,
        "completed_budgets": list(results),
        "failed_budget_count": len(failures),
        "output_root": str(root),
    }, sort_keys=True), flush=True)
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--taylor-samples", type=int, default=8)
    parser.add_argument(
        "--targets", default=",".join(str(value) for value in DEFAULT_TARGETS)
    )
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
