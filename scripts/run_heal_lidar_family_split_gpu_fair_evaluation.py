#!/usr/bin/env python3
"""Five-repeat fair evaluation for F-Cooper or DiscoNet P/Q ablations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.integration.heal_lidar_family_fair_evaluation import (  # noqa: E402
    ablation_contribution_rows,
    aggregate_contribution_rows,
    aggregate_five_repeat_results,
    build_family_evaluation_inventories,
    canonical_json_hash,
    compact_result,
    evaluate_existing_family_engine,
    gpu_snapshot,
    load_resumable_family_evaluation,
    read_json,
    sha256_file,
    summarize_repeat_results,
    write_csv,
    write_json,
)


DEFAULT_HEAL_ROOT = Path("../../HEAL")
DEFAULT_TRT_ROOT = Path("${TENSORRT_ROOT}")
DEFAULT_PLUGIN = (
    REPO_ROOT
    / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair five-repeat evaluation of existing HEAL family ablation engines."
    )
    parser.add_argument(
        "--family-id",
        required=True,
        choices=("heal_lidar_fcooper", "heal_lidar_disco"),
    )
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--baseline-engine", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--heal-root", type=Path, default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TRT_ROOT)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Resume an identity-matched run directory; partial item directories fail closed.",
    )
    parser.add_argument("--ga-gpu", type=int, default=0)
    parser.add_argument("--greedy-gpu", type=int, default=3)
    parser.add_argument(
        "--ga-extra-gpu",
        action="append",
        type=int,
        default=[],
        help="Additional idle GPU for whole-repeat GA sharding.",
    )
    parser.add_argument(
        "--greedy-extra-gpu",
        action="append",
        type=int,
        default=[],
        help="Additional idle GPU for whole-repeat Greedy sharding.",
    )
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=1789)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--latency-rounds", type=int, default=3)
    parser.add_argument("--fixed-k", type=int, default=29696)
    parser.add_argument("--max-agents", type=int, default=2)
    parser.add_argument("--idle-max-used-mib", type=int, default=256)
    parser.add_argument("--idle-max-utilization", type=int, default=5)
    parser.add_argument(
        "--seed-method-root",
        action="append",
        default=[],
        metavar="METHOD=RUN_DIR",
        help=(
            "Seed one already-complete method from another fair-evaluation run. "
            "Every item is validated fail-closed before hard-linking; the source "
            "method must use the same physical GPU in the new assignment."
        ),
    )
    parser.add_argument(
        "--seed-method-prefix-root",
        action="append",
        default=[],
        metavar="METHOD=RUN_DIR",
        help=(
            "Seed only the contiguous accepted prefix of a method from an "
            "interrupted run. The first missing or rejected item and everything "
            "after it are rebuilt in the new run."
        ),
    )
    return parser.parse_args(argv)


def _parse_method_roots(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in list(values or []):
        method, separator, value = str(raw).partition("=")
        method = method.strip().lower()
        if not separator or method not in {"ga", "greedy"} or not value.strip():
            raise ValueError(f"invalid_{label}:{raw}")
        if method in result:
            raise ValueError(f"duplicate_{label}:{method}")
        result[method] = Path(value.strip()).expanduser().resolve()
    return result


def _seed_method_roots(args: argparse.Namespace) -> dict[str, Path]:
    return _parse_method_roots(
        list(getattr(args, "seed_method_root", []) or []),
        label="seed_method_root",
    )


def _seed_method_prefix_roots(args: argparse.Namespace) -> dict[str, Path]:
    return _parse_method_roots(
        list(getattr(args, "seed_method_prefix_root", []) or []),
        label="seed_method_prefix_root",
    )


def _gpu_pools(args: argparse.Namespace) -> dict[str, list[int]]:
    pools = {
        "ga": [int(args.ga_gpu), *[int(value) for value in args.ga_extra_gpu]],
        "greedy": [
            int(args.greedy_gpu),
            *[int(value) for value in args.greedy_extra_gpu],
        ],
    }
    fully_seeded_methods = set(_seed_method_roots(args))
    flattened = [
        gpu
        for method, values in pools.items()
        if method not in fully_seeded_methods
        for gpu in values
    ]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError(f"family_fair_evaluation_requires_distinct_gpus:{pools}")
    return pools


def _new_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        destination = args.run_dir.expanduser().resolve()
        if destination.exists() and not destination.is_dir():
            raise RuntimeError(f"family_resume_run_path_not_directory:{destination}")
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()) and not (
            destination / "family_fair_evaluation_manifest.json"
        ).is_file():
            raise RuntimeError(
                f"family_resume_nonempty_run_without_manifest:{destination}"
            )
        return destination
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = {
        "heal_lidar_fcooper": "fcooper",
        "heal_lidar_disco": "disconet",
    }[args.family_id]
    destination = (
        args.output_root
        / f"h800_{short}_pq_fair_repeat{int(args.repeat_count)}_{stamp}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    return destination.resolve()


def _preflight(
    args: argparse.Namespace, run_dir: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    required = (
        args.build_root,
        args.baseline_engine,
        args.model_config,
        args.eval_manifest,
        args.heal_root,
        args.tensorrt_root,
        args.plugin,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"family_fair_evaluation_preflight_missing:{missing}")
    if int(args.repeat_count) <= 0:
        raise ValueError("repeat_count_must_be_positive")
    gpu_assignment = {"ga": int(args.ga_gpu), "greedy": int(args.greedy_gpu)}
    gpu_pools = _gpu_pools(args)
    fully_seeded_methods = set(_seed_method_roots(args))
    snapshots = {
        f"{method}:{gpu_id}": gpu_snapshot(gpu_id)
        for method, gpu_ids in gpu_pools.items()
        if method not in fully_seeded_methods
        for gpu_id in gpu_ids
    }
    occupied = {
        method: snapshot
        for method, snapshot in snapshots.items()
        if int(snapshot["memory_used_mib"]) > int(args.idle_max_used_mib)
        or int(snapshot["utilization_percent"]) > int(args.idle_max_utilization)
    }
    if occupied:
        raise RuntimeError(f"family_fair_evaluation_assigned_gpu_not_idle:{occupied}")
    manifest = read_json(args.eval_manifest)
    if not bool(manifest.get("reset_after_warmup", False)):
        raise RuntimeError("family_fair_evaluation_manifest_not_reset")
    if len(list(manifest.get("warmup_frame_ids") or [])) < int(args.warmup_frames):
        raise RuntimeError("family_fair_evaluation_manifest_warmup_insufficient")
    if len(list(manifest.get("evaluation_frame_ids") or [])) < int(args.num_frames):
        raise RuntimeError("family_fair_evaluation_manifest_evaluation_insufficient")
    inventories = build_family_evaluation_inventories(
        build_root=args.build_root,
        baseline_engine_path=args.baseline_engine,
        family_id=args.family_id,
    )
    if any(len(inventory) != 19 for inventory in inventories.values()):
        raise RuntimeError(
            f"family_fair_evaluation_inventory_count:{ {k: len(v) for k, v in inventories.items()} }"
        )
    seed_roots = _seed_method_roots(args)
    prefix_seed_roots = _seed_method_prefix_roots(args)
    overlap = sorted(set(seed_roots) & set(prefix_seed_roots))
    if overlap:
        raise RuntimeError(f"family_seed_method_mode_overlap:{overlap}")
    run_identity = {
        "family_id": args.family_id,
        "build_root": str(args.build_root.resolve()),
        "baseline_engine_path": str(args.baseline_engine.resolve()),
        "baseline_engine_sha256": inventories["ga"][0]["engine_sha256"],
        "gpu_assignment": gpu_assignment,
        "gpu_pools": gpu_pools,
        "protocol": {
            "num_frames": int(args.num_frames),
            "warmup_frames": int(args.warmup_frames),
            "latency_rounds": int(args.latency_rounds),
            "dataloader_num_workers": 8,
            "cuda_postprocess": True,
            "fixed_k": int(args.fixed_k),
            "max_agents": int(args.max_agents),
            "eval_manifest_path": str(args.eval_manifest.resolve()),
            "eval_manifest_hash": manifest.get("manifest_hash"),
            "eval_manifest_file_sha256": sha256_file(args.eval_manifest),
        },
        "engine_inventory": {
            method: [
                {
                    "sequence_index": row["sequence_index"],
                    "item_id": row["item_id"],
                    "engine_path": row["engine_path"],
                    "engine_sha256": row["engine_sha256"],
                    "variant": row["variant"],
                    "budget": row.get("budget"),
                }
                for row in inventory
            ]
            for method, inventory in inventories.items()
        },
        "repeat_count": int(args.repeat_count),
        "seed_method_roots": {
            method: str(path) for method, path in sorted(seed_roots.items())
        },
        "seed_method_prefix_roots": {
            method: str(path)
            for method, path in sorted(prefix_seed_roots.items())
        },
    }
    run_identity_hash = canonical_json_hash(run_identity)
    run_manifest = {
        "schema_version": "heal-lidar-family-split-gpu-fair-evaluation-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "family_id": args.family_id,
        "build_root": str(args.build_root.resolve()),
        "baseline_engine": str(args.baseline_engine.resolve()),
        "gpu_assignment": gpu_assignment,
        "gpu_pools": gpu_pools,
        "gpu_preflight": snapshots,
        "execution_policy": {
            "cross_method_parallel": True,
            "same_gpu_serial": True,
            "whole_repeat_gpu_sharding": True,
            "complete_seed_methods_skip_gpu_idle_requirement": sorted(
                fully_seeded_methods
            ),
            "fresh_modelopt_worker_per_evaluation": True,
            "engine_build_allowed": False,
            "cuda_cache_cleanup_after_each_evaluation": True,
            "engine_hash_size_mtime_locked": True,
            "repeat_count": int(args.repeat_count),
        },
        "protocol": {
            "num_frames": int(args.num_frames),
            "warmup_frames": int(args.warmup_frames),
            "latency_rounds": int(args.latency_rounds),
            "dataloader_num_workers": 8,
            "cuda_postprocess": True,
            "fixed_k": int(args.fixed_k),
            "max_agents": int(args.max_agents),
            "eval_manifest": str(args.eval_manifest.resolve()),
            "eval_manifest_hash": manifest.get("manifest_hash"),
            "eval_manifest_file_sha256": sha256_file(args.eval_manifest),
        },
        "inventories": inventories,
        "run_identity": run_identity,
        "run_identity_hash": run_identity_hash,
    }
    run_manifest_path = run_dir / "family_fair_evaluation_manifest.json"
    if run_manifest_path.is_file():
        existing = read_json(run_manifest_path)
        if str(existing.get("run_identity_hash", "")) != run_identity_hash:
            raise RuntimeError(
                "family_resume_run_identity_mismatch:"
                f"{existing.get('run_identity_hash')}:{run_identity_hash}"
            )
        write_json(
            run_dir / "resume_preflight_snapshot.json",
            {
                "resumed_at": datetime.now().astimezone().isoformat(),
                "run_identity_hash": run_identity_hash,
                "gpu_preflight": snapshots,
            },
        )
    else:
        write_json(run_manifest_path, run_manifest)
    return inventories, gpu_assignment


def _seed_method_results(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    inventories: Mapping[str, list[dict[str, Any]]],
    gpu_assignment: Mapping[str, int],
) -> dict[str, Any]:
    """Validate and hard-link complete streams or accepted contiguous prefixes."""

    seeded: dict[str, Any] = {}
    complete_roots = _seed_method_roots(args)
    prefix_roots = _seed_method_prefix_roots(args)
    overlap = sorted(set(complete_roots) & set(prefix_roots))
    if overlap:
        raise RuntimeError(f"family_seed_method_mode_overlap:{overlap}")
    seed_specs = [
        (method, source_root, "complete")
        for method, source_root in sorted(complete_roots.items())
    ] + [
        (method, source_root, "accepted_prefix")
        for method, source_root in sorted(prefix_roots.items())
    ]
    for method, source_root, seed_mode in seed_specs:
        source_manifest_path = source_root / "family_fair_evaluation_manifest.json"
        if not source_manifest_path.is_file():
            raise RuntimeError(
                f"family_seed_manifest_missing:{method}:{source_manifest_path}"
            )
        source_manifest = read_json(source_manifest_path)
        if str(source_manifest.get("family_id", "")) != str(args.family_id):
            raise RuntimeError(
                f"family_seed_family_mismatch:{method}:"
                f"{source_manifest.get('family_id')}:{args.family_id}"
            )
        source_protocol = dict(source_manifest.get("protocol") or {})
        expected_protocol = {
            "num_frames": int(args.num_frames),
            "warmup_frames": int(args.warmup_frames),
            "latency_rounds": int(args.latency_rounds),
            "dataloader_num_workers": 8,
            "cuda_postprocess": True,
            "fixed_k": int(args.fixed_k),
            "max_agents": int(args.max_agents),
            "eval_manifest_hash": read_json(args.eval_manifest).get("manifest_hash"),
            "eval_manifest_file_sha256": sha256_file(args.eval_manifest),
        }
        mismatches = {
            key: {"expected": value, "actual": source_protocol.get(key)}
            for key, value in expected_protocol.items()
            if source_protocol.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"family_seed_protocol_mismatch:{method}:{mismatches}"
            )
        source_assignment = dict(source_manifest.get("gpu_assignment") or {})
        expected_gpu = int(gpu_assignment[method])
        if int(source_assignment.get(method, -1)) != expected_gpu:
            raise RuntimeError(
                f"family_seed_gpu_mismatch:{method}:"
                f"{source_assignment.get(method)}:{expected_gpu}"
            )
        source_inventory = list(
            dict(source_manifest.get("inventories") or {}).get(method) or []
        )
        expected_inventory = list(inventories[method])
        source_identity = [
            (
                int(row["sequence_index"]),
                str(row["item_id"]),
                str(row["engine_sha256"]),
            )
            for row in source_inventory
        ]
        expected_identity = [
            (
                int(row["sequence_index"]),
                str(row["item_id"]),
                str(row["engine_sha256"]),
            )
            for row in expected_inventory
        ]
        if source_identity != expected_identity:
            raise RuntimeError(f"family_seed_inventory_mismatch:{method}")

        work_items = []
        for repeat_index in range(int(args.repeat_count)):
            for source in expected_inventory:
                relative = Path(f"repeat_{repeat_index:02d}") / (
                    f"gpu_{expected_gpu}_{method}"
                ) / f"{int(source['sequence_index']):02d}_{source['item_id']}"
                work_items.append((repeat_index, source, relative))

        copied: list[dict[str, Any]] = []
        first_unaccepted: dict[str, Any] | None = None
        stop_index = len(work_items)
        for work_index, (repeat_index, source, relative) in enumerate(work_items):
            source_dir = source_root / relative
            destination = run_dir / relative
            try:
                load_resumable_family_evaluation(
                    source=source,
                    gpu_id=expected_gpu,
                    repeat_index=repeat_index,
                    output_dir=source_dir,
                    eval_manifest_path=args.eval_manifest,
                    num_frames=args.num_frames,
                    warmup_frames=args.warmup_frames,
                    latency_rounds=args.latency_rounds,
                    fixed_k=args.fixed_k,
                )
            except RuntimeError as exc:
                if seed_mode == "complete":
                    raise
                first_unaccepted = {
                    "work_index": work_index,
                    "repeat_index": repeat_index,
                    "item_id": source["item_id"],
                    "source_dir": str(source_dir),
                    "reason": str(exc),
                }
                stop_index = work_index
                break
            if destination.exists():
                raise RuntimeError(
                    f"family_seed_destination_exists:{method}:{destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(source_dir, destination, copy_function=os.link)
                copy_mode = "hardlink"
            except OSError:
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source_dir, destination, copy_function=shutil.copy2)
                copy_mode = "copy"
            # Re-validate the destination instead of trusting the copy operation.
            load_resumable_family_evaluation(
                source=source,
                gpu_id=expected_gpu,
                repeat_index=repeat_index,
                output_dir=destination,
                eval_manifest_path=args.eval_manifest,
                num_frames=args.num_frames,
                warmup_frames=args.warmup_frames,
                latency_rounds=args.latency_rounds,
                fixed_k=args.fixed_k,
            )
            copied.append(
                {
                    "repeat_index": repeat_index,
                    "item_id": source["item_id"],
                    "source_dir": str(source_dir),
                    "destination": str(destination),
                    "copy_mode": copy_mode,
                }
            )

        if seed_mode == "accepted_prefix" and stop_index < len(work_items):
            for repeat_index, source, relative in work_items[stop_index + 1 :]:
                later_dir = source_root / relative
                if not later_dir.exists():
                    continue
                try:
                    load_resumable_family_evaluation(
                        source=source,
                        gpu_id=expected_gpu,
                        repeat_index=repeat_index,
                        output_dir=later_dir,
                        eval_manifest_path=args.eval_manifest,
                        num_frames=args.num_frames,
                        warmup_frames=args.warmup_frames,
                        latency_rounds=args.latency_rounds,
                        fixed_k=args.fixed_k,
                    )
                except RuntimeError:
                    continue
                raise RuntimeError(
                    f"family_seed_prefix_noncontiguous:{method}:{later_dir}"
                )
        seeded[method] = {
            "source_root": str(source_root),
            "gpu_id": expected_gpu,
            "seed_mode": seed_mode,
            "validated_and_seeded_count": len(copied),
            "first_unaccepted": first_unaccepted,
            "items": copied,
        }
    if seeded:
        write_json(
            run_dir / "seeded_method_resume.json",
            {
                "schema_version": "heal-lidar-family-seeded-method-resume-v1",
                "created_at": datetime.now().astimezone().isoformat(),
                "methods": seeded,
            },
        )
    return seeded


def _run_method(
    *,
    method: str,
    gpu_id: int,
    inventory: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    repeat_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    assigned_repeats = (
        list(range(int(args.repeat_count)))
        if repeat_indices is None
        else [int(value) for value in repeat_indices]
    )
    for repeat_index in assigned_repeats:
        for source in inventory:
            destination = (
                run_dir
                / f"repeat_{repeat_index:02d}"
                / f"gpu_{gpu_id}_{method}"
                / f"{int(source['sequence_index']):02d}_{source['item_id']}"
            )
            print(
                json.dumps(
                    {
                        "event": "family_evaluation_start",
                        "family": args.family_id,
                        "method": method,
                        "gpu": gpu_id,
                        "repeat": repeat_index,
                        "item": source["item_id"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if destination.exists():
                result = load_resumable_family_evaluation(
                    source=source,
                    gpu_id=gpu_id,
                    repeat_index=repeat_index,
                    output_dir=destination,
                    eval_manifest_path=args.eval_manifest,
                    num_frames=args.num_frames,
                    warmup_frames=args.warmup_frames,
                    latency_rounds=args.latency_rounds,
                    fixed_k=args.fixed_k,
                )
                print(
                    json.dumps(
                        {
                            "event": "family_evaluation_resume_hit",
                            "method": method,
                            "gpu": gpu_id,
                            "repeat": repeat_index,
                            "item": source["item_id"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                result = evaluate_existing_family_engine(
                    source=source,
                    gpu_id=gpu_id,
                    repeat_index=repeat_index,
                    output_dir=destination,
                    model_config=args.model_config,
                    heal_root=args.heal_root,
                    tensorrt_root=args.tensorrt_root,
                    plugin_path=args.plugin,
                    eval_manifest_path=args.eval_manifest,
                    num_frames=args.num_frames,
                    warmup_frames=args.warmup_frames,
                    latency_rounds=args.latency_rounds,
                    fixed_k=args.fixed_k,
                    max_agents=args.max_agents,
                )
            results.append(compact_result(result))
            write_json(
                run_dir / f"gpu_{gpu_id}_{method}_progress.json",
                {"status": "running", "completed": results},
            )
            print(
                json.dumps(
                    {
                        "event": "family_evaluation_complete",
                        "family": args.family_id,
                        "method": method,
                        "gpu": gpu_id,
                        "repeat": repeat_index,
                        "item": source["item_id"],
                        "mAP": result["mAP"],
                        "forward_mean_ms": result["forward_mean_ms"],
                        "forward_p50_ms": result["forward_p50_ms"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    write_json(
        run_dir / f"gpu_{gpu_id}_{method}_completed.json",
        {
            "status": "complete",
            "repeat_indices": assigned_repeats,
            "completed": results,
        },
    )
    return results


def _combine_per_frame_csv(run_dir: Path, *, expected_rows: int) -> dict[str, Any]:
    sources = sorted(run_dir.glob("repeat_*/*/*/per_frame_latency.csv"))
    destination = run_dir / "per_frame_latency_all.csv"
    temporary = run_dir / ".per_frame_latency_all.csv.tmp"
    fields: list[str] | None = None
    row_count = 0
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer: csv.DictWriter[str] | None = None
        for source in sources:
            with source.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                current = list(reader.fieldnames or [])
                if fields is None:
                    fields = current
                    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                elif current != fields:
                    raise RuntimeError(f"family_per_frame_csv_schema_mismatch:{source}")
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    if row_count != int(expected_rows):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"family_per_frame_combined_count:{row_count}:{expected_rows}"
        )
    temporary.replace(destination)
    return {
        "path": str(destination),
        "source_csv_count": len(sources),
        "row_count": row_count,
        "sha256": sha256_file(destination),
        "warmup_rows_included": False,
    }


def _report_markdown(
    *,
    family_id: str,
    aggregate: list[Mapping[str, Any]],
    contribution: list[Mapping[str, Any]],
    repeat_count: int,
) -> str:
    lines = [
        f"# {family_id} P/Q fair full-validation",
        "",
        f"All values below are arithmetic means across {repeat_count} complete runs. "
        "Each method uses its own same-GPU strict-FP32 baseline.",
        "",
        "| method | budget | variant | mAP mean±std | forward p50 mean±std (ms) | speedup |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in aggregate:
        budget = "-" if row["budget"] is None else f"{float(row['budget']):.2f}"
        lines.append(
            "| {method} | {budget} | {variant} | {map_mean:.6f}±{map_std:.6f} | "
            "{p50_mean:.3f}±{p50_std:.3f} | {speedup:.3f}x |".format(
                method=row["assigned_method"],
                budget=budget,
                variant=row["variant"],
                map_mean=float(row["mAP_across_runs_mean"]),
                map_std=float(row["mAP_across_runs_std"]),
                p50_mean=float(row["forward_p50_ms_across_runs_mean"]),
                p50_std=float(row["forward_p50_ms_across_runs_std"]),
                speedup=float(row["speedup_vs_same_gpu_fp32_across_runs_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decomposed contribution",
            "",
            "| method | budget | ΔmAP P+Q | ΔmAP P-only | ΔmAP Q-only | interaction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contribution:
        lines.append(
            "| {method} | {budget:.2f} | {pq:.6f} | {p:.6f} | {q:.6f} | {interaction:.6f} |".format(
                method=row["assigned_method"],
                budget=float(row["budget"]),
                pq=float(row["prune_quant_delta_vs_fp32_across_runs_mean"]),
                p=float(row["prune_only_delta_vs_fp32_across_runs_mean"]),
                q=float(row["quant_only_delta_vs_fp32_across_runs_mean"]),
                interaction=float(row["pq_interaction_mAP_across_runs_mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = _new_run_dir(args)
    inventories, assignment = _preflight(args, run_dir)
    _seed_method_results(
        args=args,
        run_dir=run_dir,
        inventories=inventories,
        gpu_assignment=assignment,
    )
    pools = _gpu_pools(args)
    by_method: dict[str, list[dict[str, Any]]] = {"ga": [], "greedy": []}
    shard_specs: list[tuple[str, int, list[int]]] = []
    all_repeats = list(range(int(args.repeat_count)))
    for method in ("ga", "greedy"):
        for shard_index, gpu_id in enumerate(pools[method]):
            repeats = all_repeats[shard_index :: len(pools[method])]
            if repeats:
                shard_specs.append((method, gpu_id, repeats))
    with ThreadPoolExecutor(max_workers=len(shard_specs)) as executor:
        futures = {
            executor.submit(
                _run_method,
                method=method,
                gpu_id=gpu_id,
                inventory=inventories[method],
                args=args,
                run_dir=run_dir,
                repeat_indices=repeats,
            ): (method, gpu_id, repeats)
            for method, gpu_id, repeats in shard_specs
        }
        for future in as_completed(futures):
            method, _, _ = futures[future]
            by_method[method].extend(future.result())
    for method in by_method:
        by_method[method].sort(
            key=lambda row: (int(row["repeat_index"]), int(row["sequence_index"]))
        )
    raw = by_method["ga"] + by_method["greedy"]
    repeat_rows = summarize_repeat_results(raw)
    aggregate = aggregate_five_repeat_results(
        repeat_rows, repeat_count=int(args.repeat_count)
    )
    contribution_rows = ablation_contribution_rows(repeat_rows)
    contribution_aggregate = aggregate_contribution_rows(
        contribution_rows, repeat_count=int(args.repeat_count)
    )
    expected_per_frame = (
        2 * 19 * int(args.repeat_count) * int(args.num_frames)
    )
    per_frame = _combine_per_frame_csv(
        run_dir, expected_rows=expected_per_frame
    )
    write_csv(run_dir / "repeat_results.csv", repeat_rows)
    write_csv(run_dir / "five_repeat_mean_std.csv", aggregate)
    write_csv(run_dir / "ablation_contribution_by_repeat.csv", contribution_rows)
    write_csv(
        run_dir / "ablation_contribution_five_repeat_mean_std.csv",
        contribution_aggregate,
    )
    report = {
        "status": "complete",
        "family_id": args.family_id,
        "repeat_count": int(args.repeat_count),
        "repeat_results": repeat_rows,
        "five_repeat_mean_std": aggregate,
        "contribution_by_repeat": contribution_rows,
        "contribution_five_repeat_mean_std": contribution_aggregate,
        "per_frame_manifest": per_frame,
    }
    write_json(run_dir / "family_fair_evaluation_report.json", report)
    (run_dir / "family_fair_evaluation_report.md").write_text(
        _report_markdown(
            family_id=args.family_id,
            aggregate=aggregate,
            contribution=contribution_aggregate,
            repeat_count=int(args.repeat_count),
        ),
        encoding="utf-8",
    )
    write_json(
        run_dir / "run_complete.json",
        {
            "status": "complete",
            "run_dir": str(run_dir),
            "report_json": str(run_dir / "family_fair_evaluation_report.json"),
            "report_markdown": str(run_dir / "family_fair_evaluation_report.md"),
        },
    )
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
