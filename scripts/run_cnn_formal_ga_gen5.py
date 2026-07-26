#!/usr/bin/env python3
"""Run Pyramid/DiscoNet/F-Cooper with the strict Stage-1/Stage-2/V1--V3 GA."""

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
    CNNRealStage2Evaluator,
    MODEL_SPECS,
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
    if int(args.generations) != 5:
        raise RuntimeError(f"cnn_formal_ga_requires_exactly_5_generations:{args.generations}")
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
        "generation_contract": "formal_gen5",
        "generations": 5,
        "population_size": 64,
        "offspring_size": 64,
        "stage2_quota": 5,
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
        "generations": 5,
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
    real_evaluator = create_real_evaluator(prepared, output_root=root)
    results: dict[str, dict] = {}
    failures: list[dict] = []
    greedy_gate_rows: list[dict] = []
    greedy_gate_results: dict[str, dict] = {}

    # Finish and validate every exact Greedy anchor before starting even one
    # formal GA budget.  run_budget() reads the same immutable Stage-2 cache,
    # so this ordering adds no duplicate engine build when GA is subsequently
    # allowed.
    for target in targets:
        label = f"{int(round(target * 100)):03d}"
        anchor = anchors.get(target)
        if anchor is None:
            failures.append({
                "target": target,
                "status": "budget_unreachable_after_legal_frontier_and_beam_recovery",
            })
            continue
        greedy_stage2 = CNNRealStage2Evaluator(
            prepared=prepared,
            output_root=root,
            budget_label=label,
            real_evaluator=real_evaluator,
        )(anchor, 0)
        payload = stage2_payload(greedy_stage2)
        greedy_gate_results[label] = payload
        greedy_gate_rows.append({
            "model": args.model,
            "budget": target,
            "candidate_hash": greedy_stage2.complete_phenotype_hash,
            "status": greedy_stage2.status,
            "deployable": greedy_stage2.deployable,
            "mAP_fixed50": greedy_stage2.map,
            "p50_ms": greedy_stage2.p50_ms,
            "requested_realized_exact": greedy_stage2.requested_realized_exact,
            "evaluated": greedy_stage2.evaluated,
            "skipped": greedy_stage2.skipped,
        })
        if not greedy_stage2.deployable:
            failures.append({
                "target": target,
                "status": "greedy_exact_anchor_deployment_gate_failed",
                "candidate_hash": greedy_stage2.complete_phenotype_hash,
                "stage2": payload,
            })

    write_csv(root / "reports/greedy_anchor_deployment_validation.csv", greedy_gate_rows)
    write_json(root / "reports/greedy_anchor_deployment_validation.json", {
        "model": args.model,
        "all_budgets_checked_before_formal_ga": True,
        "formal_ga_started": False,
        "targets": list(targets),
        "anchors": greedy_gate_results,
        "failures": failures,
        "gate_passed": not failures and len(greedy_gate_results) == len(targets),
    })
    if failures or len(greedy_gate_results) != len(targets):
        write_json(root / "reports/pre_ga_greedy_anchor_gate.json", {
            "formal_ga_allowed": False,
            "reason": "all_six_greedy_exact_anchors_must_exist_and_be_deployable",
            "failures": failures,
        })
        print(json.dumps({
            "event": "cnn_formal_ga_blocked_by_greedy_anchor_gate",
            "model": args.model,
            "failed_budget_count": len(failures),
            "output_root": str(root),
        }, sort_keys=True), flush=True)
        return 2
    write_json(root / "reports/pre_ga_greedy_anchor_gate.json", {
        "formal_ga_allowed": not args.greedy_only,
        "all_six_greedy_exact_anchors_in_band": True,
        "all_six_greedy_exact_anchors_deployable": True,
        "greedy_only_requested": bool(args.greedy_only),
        "frontier_and_recovery_used": True,
        "budget_projection_used": False,
        "repair_used": False,
    })
    if args.greedy_only:
        write_json(root / "reports/final_acceptance.json", {
            "model": args.model,
            "mode": "six_budget_greedy_anchor_gate_only",
            "new_ga_framework_used": True,
            "formal_ga_executed": False,
            "all_six_greedy_exact_anchors_in_band": True,
            "all_six_greedy_exact_anchors_deployable": True,
            "greedy_frontier_recovery_enabled": True,
            "budget_projection_used": False,
            "repair_enabled": False,
            "budgets_requested": list(targets),
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
            "event": "cnn_six_budget_greedy_gate_complete",
            "model": args.model,
            "formal_ga_started": False,
            "output_root": str(root),
        }, sort_keys=True), flush=True)
        return 0

    # Only now, after all budgets passed the exact-anchor gate, may formal GA
    # begin.  The Greedy anchor is loaded from the Stage-2 cache by run_budget.
    for target in targets:
        label = f"{int(round(target * 100)):03d}"
        anchor = anchors[target]
        try:
            results[label] = run_budget(
                prepared,
                target=target,
                anchor_genotype=anchor,
                output_root=root,
                seed=args.seed,
                generations=args.generations,
                real_evaluator=real_evaluator,
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
            "greedy_map_fixed50": greedy["mAP"],
            "greedy_p50_ms": greedy["p50_ms"],
            "ga_hash": final["complete_phenotype_hash"],
            "ga_map_fixed50": final["mAP"],
            "ga_p50_ms": final["p50_ms"],
            "ga_improved_greedy": row["ga_improved_greedy"],
        })
    write_csv(root / "reports/formal_ga_budget_summary.csv", summary_rows)
    write_json(root / "reports/formal_ga_results.json", {
        "model": args.model,
        "framework": "StrictStage12V3Runner",
        "old_two_stage_search_used": False,
        "generation_zero_counted": False,
        "formal_generations": 5,
        "formal_generation_ids": [1, 2, 3, 4, 5],
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
        "generations_requested": 5,
        "generation_ids": [1, 2, 3, 4, 5],
        "generation_zero_counted": False,
        "seed_count": 1,
        "population_size": 64,
        "offspring_size": 64,
        "stage2_quota": 5,
        "greedy_anchor_gate_completed_before_ga": True,
        "greedy_frontier_recovery_enabled": True,
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
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--taylor-samples", type=int, default=8)
    parser.add_argument(
        "--targets", default=",".join(str(value) for value in DEFAULT_TARGETS)
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--greedy-only",
        action="store_true",
        help=(
            "build and deploy-validate all six exact Greedy anchors, then stop "
            "before formal GA"
        ),
    )
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
