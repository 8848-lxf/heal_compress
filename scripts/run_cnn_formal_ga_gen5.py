#!/usr/bin/env python3
"""Run Pyramid/DiscoNet/F-Cooper with the strict Stage-1/Stage-2/V1--V3 GA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
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
    GENERATION_WINNER_FRAMES,
    GENERATION_WINNER_PROTOCOL,
    GENERATION_WINNER_WARMUP_FRAMES,
    MODEL_SPECS,
    STAGE2_SCREENING_FRAMES,
    STAGE2_SCREENING_PROTOCOL,
    STAGE2_SCREENING_WARMUP_FRAMES,
    create_real_evaluator,
    greedy_anchors,
    load_greedy_anchors,
    prepare_search,
    run_budget,
    stage2_payload,
    write_csv,
    write_json,
)
from search.integration.runtime_environment import query_gpus  # noqa: E402


DEFAULT_TARGETS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
SUPPORTED_GENERATIONS = (5, 10)
CONTROLLED_JAQ0_MODELS = frozenset(("pyramid", "disco", "fcooper"))


def is_controlled_jaq0_experiment(
    model: str,
    activation_taylor_fitness_weight: float,
    targets: str,
) -> bool:
    """Return whether this is the single-budget CNN activation-Taylor ablation."""

    return bool(
        model in CONTROLLED_JAQ0_MODELS
        and float(activation_taylor_fitness_weight) == 0.0
        and targets == "0.05"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_gen5_continuation_state(
    root: Path,
    *,
    labels: tuple[str, ...] | None = None,
    expected_activation_taylor_weight: float | None = None,
) -> dict[str, object]:
    """Preserve the completed five-generation audit before deterministic replay.

    The original runner did not serialize all 64 survivors or ``random.Random``
    state.  A strict continuation therefore reconstructs generations 1--5 from
    the same seed and immutable Stage-2 cache, verifies their summaries byte for
    byte, then proceeds with generations 6--10.  This small snapshot keeps the
    original reports and generation summaries without duplicating engine/ONNX
    artifacts.
    """

    destination = root / "reports" / "gen5_frozen_snapshot"
    manifest_path = destination / "snapshot_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if labels is not None and set(payload.get("budget_labels", [])) != set(labels):
            raise RuntimeError("cnn_gen5_snapshot_budget_contract_mismatch")
        if (
            expected_activation_taylor_weight is not None
            and float(payload.get("activation_taylor_fitness_weight", -1.0))
            != float(expected_activation_taylor_weight)
        ):
            raise RuntimeError("cnn_gen5_snapshot_activation_weight_mismatch")
        for row in payload.get("files", []):
            snapshot = destination / str(row["snapshot_relative_path"])
            if not snapshot.is_file() or sha256_file(snapshot) != str(row["sha256"]):
                raise RuntimeError(
                    f"cnn_gen5_snapshot_integrity_failed:{snapshot}"
                )
        return payload

    report = root / "reports" / "formal_ga_results.json"
    if not report.is_file():
        raise RuntimeError("cnn_gen10_continuation_missing_gen5_results")
    results = json.loads(report.read_text(encoding="utf-8"))
    if int(results.get("formal_generations", -1)) != 5:
        raise RuntimeError("cnn_gen10_continuation_source_not_gen5")
    completed = dict(results.get("results") or {})
    expected_labels = set(
        labels
        or tuple(f"{int(round(value * 100)):03d}" for value in DEFAULT_TARGETS)
    )
    if (
        expected_activation_taylor_weight is not None
        and float(results.get("activation_taylor_fitness_weight", -1.0))
        != float(expected_activation_taylor_weight)
    ):
        raise RuntimeError("cnn_gen10_continuation_activation_weight_mismatch")
    if set(completed) != expected_labels or any(
        int(row.get("completed_evolution_generations", -1)) != 5
        for row in completed.values()
    ):
        raise RuntimeError("cnn_gen10_continuation_source_incomplete")

    sources = [
        root / "provenance" / "start.json",
        root / "reports" / "formal_ga_results.json",
        root / "reports" / "formal_ga_budget_summary.csv",
        root / "reports" / "final_acceptance.json",
    ]
    for label in sorted(expected_labels):
        sources.append(root / f"ga/budget_{label}/seed_0/budget_summary.json")
        for generation in range(0, 6):
            sources.append(
                root
                / f"ga/budget_{label}/seed_0/generation_{generation:02d}"
                / "generation_summary.json"
            )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"cnn_gen10_continuation_snapshot_missing:{missing}")

    rows: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(root)
        snapshot = destination / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        rows.append(
            {
                "source_relative_path": str(relative),
                "snapshot_relative_path": str(relative),
                "size_bytes": int(snapshot.stat().st_size),
                "sha256": sha256_file(snapshot),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "pyramid-gen5-deterministic-replay-snapshot-v1",
        "source_formal_generations": 5,
        "continuation_target_generation": 10,
        "replay_generations": [1, 2, 3, 4, 5],
        "new_generations": [6, 7, 8, 9, 10],
        "budget_labels": sorted(expected_labels),
        "activation_taylor_fitness_weight": float(
            results.get("activation_taylor_fitness_weight", 1.0)
        ),
        "stage2_artifacts_copied": False,
        "stage2_cache_reused_by_complete_phenotype_hash": True,
        "files": rows,
    }
    write_json(manifest_path, payload)
    return payload


def verify_gen5_replay_prefix(root: Path, labels: tuple[str, ...]) -> dict[str, object]:
    snapshot = root / "reports" / "gen5_frozen_snapshot"
    comparisons: list[dict[str, object]] = []
    for label in labels:
        for generation in range(0, 6):
            relative = Path(
                f"ga/budget_{label}/seed_0/generation_{generation:02d}/"
                "generation_summary.json"
            )
            expected = snapshot / relative
            actual = root / relative
            expected_hash = sha256_file(expected)
            actual_hash = sha256_file(actual)
            comparisons.append(
                {
                    "budget_label": label,
                    "generation": generation,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "exact": expected_hash == actual_hash,
                }
            )
    exact = all(bool(row["exact"]) for row in comparisons)
    result: dict[str, object] = {
        "schema_version": "pyramid-gen5-deterministic-replay-verification-v1",
        "all_generation_00_to_05_summaries_exact": exact,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }
    write_json(root / "reports" / "gen5_replay_verification.json", result)
    if not exact:
        raise RuntimeError("cnn_gen10_continuation_replay_prefix_mismatch")
    return result


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
    generations = int(args.generations)
    activation_weight = float(args.activation_taylor_fitness_weight)
    if activation_weight not in (0.0, 1.0):
        raise RuntimeError(
            "cnn_formal_ga_activation_taylor_weight_must_be_zero_or_one"
        )
    if generations not in SUPPORTED_GENERATIONS:
        raise RuntimeError(
            f"cnn_formal_ga_requires_5_or_10_generations:{args.generations}"
        )
    continuation_mode = bool(generations == 10 and args.resume)
    controlled_jaq0_experiment = is_controlled_jaq0_experiment(
        args.model,
        activation_weight,
        args.targets,
    )
    direct_jaq0_experiment = bool(
        controlled_jaq0_experiment and not args.resume
    )
    if activation_weight == 0.0 and not controlled_jaq0_experiment:
        raise RuntimeError(
            "cnn_controlled_jaq0_requires_supported_model_and_single_budget_005"
        )
    if generations == 10 and not (
        (args.model == "pyramid" and continuation_mode)
        or direct_jaq0_experiment
    ):
        raise RuntimeError(
            "cnn_formal_ga_gen10_requires_pyramid_resume_or_direct_jaq0_ablation"
        )
    if int(args.seed) != 0:
        raise RuntimeError(f"cnn_formal_ga_single_seed_zero_required:{args.seed}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", str(args.physical_gpu)):
        raise RuntimeError(
            "cnn_formal_ga_cuda_visible_devices_mismatch:"
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}!={args.physical_gpu}"
        )
    targets = tuple(float(value) for value in args.targets.split(",") if value)
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(target not in DEFAULT_TARGETS for target in targets)
        or targets != tuple(
            target for target in DEFAULT_TARGETS if target in set(targets)
        )
    ):
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
    continuation_snapshot = None
    if continuation_mode:
        continuation_snapshot = freeze_gen5_continuation_state(
            root,
            labels=tuple(f"{int(round(target * 100)):03d}" for target in targets),
            expected_activation_taylor_weight=activation_weight,
        )
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
    provenance_name = "continuation_start.json" if continuation_mode else "start.json"
    write_json(root / "provenance" / provenance_name, {
        "model": args.model,
        "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "physical_gpu": int(args.physical_gpu),
        "gpu_uuid": selected_uuid,
        "gpu": selected,
        "all_gpus": gpu_rows,
        "framework": "StrictStage12V3Runner",
        "generation_contract": f"formal_gen{generations}",
        "generations": generations,
        "deterministic_replay_continuation": continuation_mode,
        "controlled_jaq0_ablation": controlled_jaq0_experiment,
        "activation_taylor_fitness_weight": activation_weight,
        "activation_taylor_used_for_fitness": bool(activation_weight),
        "continuation_snapshot": continuation_snapshot,
        "population_size": 64,
        "offspring_size": 64,
        "stage2_quota": 5,
        "stage2_top5_screening_frames": STAGE2_SCREENING_FRAMES,
        "stage2_top5_screening_warmup_frames": (
            STAGE2_SCREENING_WARMUP_FRAMES
        ),
        "generation_winner_validation_frames": GENERATION_WINNER_FRAMES,
        "generation_winner_validation_warmup_frames": (
            GENERATION_WINNER_WARMUP_FRAMES
        ),
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
        "generations": generations,
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
        activation_taylor_fitness_weight=activation_weight,
    )
    if args.resume and (root / "reports/greedy_exact_winners.json").is_file():
        anchors = load_greedy_anchors(
            prepared, targets=targets, output_root=root
        )
        print(json.dumps({
            "event": "cnn_exact_greedy_anchors_resumed",
            "model": args.model,
            "anchor_count": len(anchors),
            "validation": "schema+phenotype_hash+exact_current_bops_hard_gate",
        }, sort_keys=True), flush=True)
    else:
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
    greedy_gate_payload = {
        "model": args.model,
        "all_budgets_checked_before_formal_ga": True,
        "formal_ga_started": False,
        "targets": list(targets),
        "anchors": greedy_gate_results,
        "failures": failures,
        "gate_passed": not failures and len(greedy_gate_results) == len(targets),
    }
    write_json(
        root / "reports/greedy_anchor_deployment_validation.json",
        greedy_gate_payload,
    )
    if failures or len(greedy_gate_results) != len(targets):
        write_json(root / "reports/pre_ga_greedy_anchor_gate.json", {
            "formal_ga_allowed": False,
            "reason": "all_requested_greedy_exact_anchors_must_exist_and_be_deployable",
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
        "all_requested_greedy_exact_anchors_in_band": True,
        "all_requested_greedy_exact_anchors_deployable": True,
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
            "all_requested_greedy_exact_anchors_in_band": True,
            "all_requested_greedy_exact_anchors_deployable": True,
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
    greedy_gate_payload["formal_ga_started"] = True
    greedy_gate_payload["formal_ga_start_authorized_by_gate"] = True
    write_json(
        root / "reports/greedy_anchor_deployment_validation.json",
        greedy_gate_payload,
    )
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
    replay_verification = None
    if continuation_mode and not failures:
        labels = tuple(f"{int(round(target * 100)):03d}" for target in targets)
        replay_verification = verify_gen5_replay_prefix(root, labels)
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
        "formal_generations": generations,
        "formal_generation_ids": list(range(1, generations + 1)),
        "deterministic_replay_continuation": continuation_mode,
        "controlled_jaq0_ablation": controlled_jaq0_experiment,
        "activation_taylor_fitness_weight": activation_weight,
        "activation_taylor_used_for_fitness": bool(activation_weight),
        "gen5_replay_verification": replay_verification,
        "seed_count": 1,
        "executed_seeds": [0],
        "population_size": 64,
        "offspring_size": 64,
        "survivor_size": 64,
        "stage2_new_candidate_quota": 5,
        "stage2_top5_screening_protocol": STAGE2_SCREENING_PROTOCOL,
        "stage2_top5_screening_frames": STAGE2_SCREENING_FRAMES,
        "stage2_top5_screening_warmup_frames": (
            STAGE2_SCREENING_WARMUP_FRAMES
        ),
        "generation_winner_validation_protocol": GENERATION_WINNER_PROTOCOL,
        "generation_winner_validation_frames": GENERATION_WINNER_FRAMES,
        "generation_winner_validation_warmup_frames": (
            GENERATION_WINNER_WARMUP_FRAMES
        ),
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
        "generations_requested": generations,
        "generation_ids": list(range(1, generations + 1)),
        "deterministic_replay_continuation": continuation_mode,
        "controlled_jaq0_ablation": controlled_jaq0_experiment,
        "activation_taylor_fitness_weight": activation_weight,
        "activation_taylor_used_for_fitness": bool(activation_weight),
        "gen5_replay_prefix_exact": (
            replay_verification is not None
            and bool(
                replay_verification[
                    "all_generation_00_to_05_summaries_exact"
                ]
            )
        ) if continuation_mode else None,
        "generation_zero_counted": False,
        "seed_count": 1,
        "population_size": 64,
        "offspring_size": 64,
        "stage2_quota": 5,
        "stage2_top5_screening_frames": STAGE2_SCREENING_FRAMES,
        "stage2_top5_screening_warmup_frames": (
            STAGE2_SCREENING_WARMUP_FRAMES
        ),
        "generation_winner_validation_frames": GENERATION_WINNER_FRAMES,
        "generation_winner_validation_warmup_frames": (
            GENERATION_WINNER_WARMUP_FRAMES
        ),
        "greedy_anchor_gate_completed_before_ga": True,
        "greedy_frontier_recovery_enabled": True,
        "stage1_proxy": (
            "J_struct_gate + J_WQ + J_AQ"
            if activation_weight == 1.0
            else "J_struct_gate + J_WQ + 0 * J_AQ"
        ),
        "repair_enabled": False,
        "budgets_requested": list(targets),
        "budgets_completed": [float(row["target_bops"]) for row in results.values()],
        "failures": failures,
        "full1789_executed": False,
    })
    if generations == 5 and not failures:
        continuation_files = []
        for target in targets:
            label = f"{int(round(target * 100)):03d}"
            for generation in range(0, 6):
                path = (
                    root
                    / f"ga/budget_{label}/seed_0/generation_{generation:02d}"
                    / "generation_summary.json"
                )
                if not path.is_file():
                    raise RuntimeError(
                        f"cnn_gen5_continuation_generation_missing:{path}"
                    )
                continuation_files.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "sha256": sha256_file(path),
                    }
                )
        write_json(root / "reports/continuation_ready.json", {
            "schema_version": "cnn-formal-ga-gen5-continuation-ready-v1",
            "ready": True,
            "source_generations": 5,
            "supported_continuation_target_generations": [10],
            "continuation_mode": (
                "deterministic_replay_generations_1_to_5_with_stage2_cache_reuse_"
                "and_exact_prefix_verification_then_run_6_to_10"
            ),
            "model": args.model,
            "seed": int(args.seed),
            "targets": list(targets),
            "activation_taylor_fitness_weight": activation_weight,
            "activation_taylor_used_for_fitness": bool(activation_weight),
            "resume_arguments": [
                "--resume", "--generations", "10",
                "--activation-taylor-fitness-weight", str(activation_weight),
                "--targets", args.targets,
            ],
            "generation_summary_files": continuation_files,
            "stage2_cache_reused_by_complete_phenotype_hash": True,
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
        "--activation-taylor-fitness-weight",
        type=float,
        choices=(0.0, 1.0),
        default=1.0,
        help=(
            "Controlled fitness ablation; activation quantization remains "
            "enabled in physical deployment."
        ),
    )
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
