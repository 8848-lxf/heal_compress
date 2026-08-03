#!/usr/bin/env python3
"""Evaluate GA ablations on GPU0 and greedy ablations on GPU1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.integration.dual_gpu_fair_evaluation import (  # noqa: E402
    ablation_contribution_rows,
    aggregate_repeated_results,
    build_method_ablation_inventory,
    canonical_json_hash,
    compact_evaluation_result,
    evaluation_frame_latency_rows,
    frame_order_hash,
    gpu_snapshot,
    run_one_evaluation,
    sha256_file,
    summarize_results,
    verify_existing_engine,
    write_json,
    write_summary_csv,
)


DEFAULT_ABLATION = (
    REPO_ROOT / "outputs/h800_lidar_pyramid_prune_quant_ablation_20260718_124729"
)
DEFAULT_CHECKPOINT = Path(
    "../../Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
)
DEFAULT_CONFIG = Path(
    "../../Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/config.yaml"
)
DEFAULT_HEAL_ROOT = Path("../../HEAL")
DEFAULT_TRT_ROOT = Path("${TENSORRT_ROOT}")
DEFAULT_PLUGIN = (
    REPO_ROOT
    / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
)
DEFAULT_EVAL_MANIFEST = DEFAULT_ABLATION / "baseline/eval_manifest.json"
GPU_ASSIGNMENT = {"ga": 0, "greedy": 1}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", type=Path, default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TRT_ROOT)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--num-frames", type=int, default=1789)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--latency-rounds", type=int, default=3)
    parser.add_argument("--fixed-k", type=int, default=29696)
    parser.add_argument("--repeat-count", type=int, default=1)
    return parser.parse_args()


def _new_run_dir(output_root: Path, repeat_count: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_root / (
        f"h800_lidar_pyramid_split_gpu_ablation_repeat{int(repeat_count)}_{stamp}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    return destination.resolve()


def _preflight(
    args: argparse.Namespace, run_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    required = (
        args.checkpoint,
        args.model_config,
        args.heal_root,
        args.tensorrt_root,
        args.plugin,
        args.eval_manifest,
        args.ablation_root / "ablation_results.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    trtexec = (
        args.tensorrt_root / "bin/trtexec",
        args.tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec",
    )
    if not any(path.is_file() for path in trtexec):
        missing.append("trtexec:" + ":".join(str(path) for path in trtexec))
    if missing:
        raise RuntimeError(f"split_gpu_ablation_preflight_missing:{missing}")
    gpu_preflight = {method: gpu_snapshot(gpu) for method, gpu in GPU_ASSIGNMENT.items()}
    occupied = {
        method: row
        for method, row in gpu_preflight.items()
        if row["memory_used_mib"] > 256 or row["utilization_percent"] > 5
    }
    if occupied:
        raise RuntimeError(f"assigned_gpu_not_idle:{occupied}")
    inventories = {
        method: [
            verify_existing_engine(row)
            for row in build_method_ablation_inventory(
                ablation_root=args.ablation_root, method=method
            )
        ]
        for method in GPU_ASSIGNMENT
    }
    if any(len(rows) != 19 for rows in inventories.values()):
        raise RuntimeError(
            f"split_gpu_inventory_count_invalid:{ {k: len(v) for k, v in inventories.items()} }"
        )
    manifest = json.loads(args.eval_manifest.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "h800-split-gpu-ablation-evaluation-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "gpu_assignment": GPU_ASSIGNMENT,
        "execution_policy": {
            "cross_gpu_parallel": True,
            "same_gpu_serial": True,
            "engine_build_allowed": False,
            "fresh_worker_process_per_evaluation": True,
            "cuda_cache_cleanup_after_each_evaluation": True,
            "variants_per_budget": ["prune_quant", "prune_only", "quant_only"],
            "full_evaluation_repeat_count": int(args.repeat_count),
        },
        "protocol": {
            "num_frames": args.num_frames,
            "warmup_frames": args.warmup_frames,
            "reset_after_warmup": True,
            "latency_rounds": args.latency_rounds,
            "dataloader_num_workers": 8,
            "cuda_postprocess": True,
            "fixed_k": args.fixed_k,
            "eval_manifest_path": str(args.eval_manifest.resolve()),
            "eval_manifest_hash": manifest.get("manifest_hash"),
            "eval_manifest_file_sha256": sha256_file(args.eval_manifest),
        },
        "gpu_preflight": gpu_preflight,
        "inventories": inventories,
    }
    payload["inventory_hash"] = canonical_json_hash(inventories)
    write_json(run_dir / "split_gpu_evaluation_manifest.json", payload)
    return inventories


def _run_method(
    *,
    method: str,
    gpu_id: int,
    sources: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for repeat_index in range(int(args.repeat_count)):
        for source in sources:
            source_for_repeat = {**source, "repeat_index": repeat_index}
            item_id = str(source["item_id"])
            destination = (
                run_dir
                / f"repeat_{repeat_index:02d}"
                / f"gpu_{gpu_id}_{method}"
                / f"{int(source['sequence_index']):02d}_{item_id}"
            )
            print(
                json.dumps(
                    {
                        "event": "evaluation_start",
                        "gpu": gpu_id,
                        "method": method,
                        "repeat": repeat_index,
                        "item": item_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            result = run_one_evaluation(
                source=source_for_repeat,
                gpu_id=gpu_id,
                output_dir=destination,
                checkpoint=args.checkpoint,
                model_config=args.model_config,
                heal_root=args.heal_root,
                tensorrt_root=args.tensorrt_root,
                plugin_path=args.plugin,
                eval_manifest_path=args.eval_manifest,
                num_frames=args.num_frames,
                warmup_frames=args.warmup_frames,
                latency_rounds=args.latency_rounds,
                fixed_k=args.fixed_k,
            )
            per_frame = evaluation_frame_latency_rows(
                result,
                metadata={
                    "repeat_index": repeat_index,
                    "gpu_id": gpu_id,
                    "assigned_method": method,
                    "item_id": item_id,
                    "budget": source.get("budget"),
                    "actual_bops": source.get("actual_bops"),
                    "variant": source.get("variant"),
                    "engine_sha256": source.get("engine_sha256"),
                },
            )
            write_summary_csv(destination / "per_frame_latency.csv", per_frame)
            compact = compact_evaluation_result(result)
            results.append(compact)
            write_json(
                run_dir / f"gpu_{gpu_id}_{method}_completed.json",
                {"results": results},
            )
            print(
                json.dumps(
                    {
                        "event": "evaluation_complete",
                        "gpu": gpu_id,
                        "method": method,
                        "repeat": repeat_index,
                        "item": item_id,
                        "mAP": result["mAP"],
                        "forward_mean_ms": result["forward_mean_ms"],
                        "p50_ms": result["forward_p50_ms"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return results


def _combine_per_frame_csv(run_dir: Path, *, expected_rows: int) -> dict[str, Any]:
    sources = sorted(run_dir.glob("repeat_*/*/*/per_frame_latency.csv"))
    destination = run_dir / "per_frame_latency.csv"
    row_count = 0
    fieldnames: list[str] | None = None
    with destination.open("w", encoding="utf-8", newline="") as output_handle:
        writer: csv.DictWriter[str] | None = None
        for source in sources:
            with source.open("r", encoding="utf-8", newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    writer = csv.DictWriter(
                        output_handle, fieldnames=fieldnames, lineterminator="\n"
                    )
                    writer.writeheader()
                elif list(reader.fieldnames or []) != fieldnames:
                    raise RuntimeError(f"per_frame_csv_schema_mismatch:{source}")
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    if row_count != int(expected_rows):
        raise RuntimeError(
            f"combined_per_frame_latency_count_mismatch:{row_count}:{expected_rows}"
        )
    return {
        "path": str(destination),
        "source_csv_count": len(sources),
        "row_count": row_count,
        "sha256": sha256_file(destination),
        "warmup_rows_included": False,
        "columns": fieldnames or [],
    }


def _mean_contribution_rows(
    aggregated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    translated = []
    for row in aggregated:
        translated.append(
            {
                "assigned_method": row["assigned_method"],
                "gpu_id": row["gpu_id"],
                "variant": row["variant"],
                "budget": row["budget"],
                "actual_bops": row["actual_bops"],
                "mAP": row["mAP_across_runs_mean"],
                "forward_p50_ms": row["forward_p50_ms_across_runs_mean"],
                "speedup_vs_same_gpu_fp32": row[
                    "speedup_vs_same_gpu_fp32_across_runs_mean"
                ],
            }
        )
    return ablation_contribution_rows(translated)


def _markdown(
    rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> str:
    lines = [
        "# H800 split-GPU P/Q ablation re-evaluation",
        "",
        "GPU 0 is assigned exclusively to GA and GPU 1 exclusively to greedy.",
        "Each GPU is serial internally; the two independent sequences run in parallel.",
        "No engine was built.",
        "",
    ]
    for method, gpu_id in GPU_ASSIGNMENT.items():
        lines.extend(
            [
                f"## {method.upper()} on GPU {gpu_id}",
                "",
                "| Budget | Variant | AP30 | AP50 | AP70 | mAP | p50 | p90 | p99 | Speedup | INT8/FP16/FP32 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if str(row["assigned_method"]) != method:
                continue
            budget = "-" if row["budget"] is None else f"{float(row['budget']):.2f}"
            lines.append(
                "| {budget} | {variant} | {ap30:.6f} | {ap50:.6f} | {ap70:.6f} | "
                "{map:.6f} | {p50:.4f} | {p90:.4f} | {p99:.4f} | {speedup:.4f}x | "
                "{i8}/{f16}/{f32} |".format(
                    budget=budget,
                    variant=row["variant"],
                    ap30=float(row["AP@0.3"]),
                    ap50=float(row["AP@0.5"]),
                    ap70=float(row["AP@0.7"]),
                    map=float(row["mAP"]),
                    p50=float(row["forward_p50_ms"]),
                    p90=float(row["forward_p90_ms"]),
                    p99=float(row["forward_p99_ms"]),
                    speedup=float(row["speedup_vs_same_gpu_fp32"]),
                    i8=int(row["int8_count"]),
                    f16=int(row["fp16_count"]),
                    f32=int(row["fp32_count"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Contribution summary",
            "",
            "| Method | Budget | P+Q mAP | P-only mAP | Q-only mAP | Interaction | P+Q/P/Q speedup |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {method} | {budget:.2f} | {pq:.6f} | {p:.6f} | {q:.6f} | "
            "{interaction:.6f} | {pqx:.4f}/{px:.4f}/{qx:.4f}x |".format(
                method=row["assigned_method"],
                budget=float(row["budget"]),
                pq=float(row["prune_quant_mAP"]),
                p=float(row["prune_only_mAP"]),
                q=float(row["quant_only_mAP"]),
                interaction=float(row["pq_interaction_mAP"]),
                pqx=float(row["prune_quant_speedup"]),
                px=float(row["prune_only_speedup"]),
                qx=float(row["quant_only_speedup"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _repeated_markdown(
    aggregated: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    *,
    repeat_count: int,
) -> str:
    lines = [
        f"# H800 split-GPU P/Q ablation: {repeat_count}-repeat means",
        "",
        "Each value is the arithmetic mean of independent full 1789-frame runs.",
        "Latency `mean` is the mean over evaluated frames in each run, then averaged",
        "over the repeated runs. The per-frame CSV excludes warmup rows.",
        "",
    ]
    for method, gpu_id in GPU_ASSIGNMENT.items():
        lines.extend(
            [
                f"## {method.upper()} on GPU {gpu_id}",
                "",
                "| Budget | Variant | AP30/50/70 mean | mAP mean±std | Forward mean/p50/p90/p99 ms | Post mean ms | Total mean ms | Speedup mean |",
                "|---:|---|---|---:|---|---:|---:|---:|",
            ]
        )
        for row in aggregated:
            if str(row["assigned_method"]) != method:
                continue
            budget = "-" if row["budget"] is None else f"{float(row['budget']):.2f}"
            lines.append(
                "| {budget} | {variant} | {ap30:.6f}/{ap50:.6f}/{ap70:.6f} | "
                "{map:.6f}±{map_std:.6f} | {fmean:.4f}/{p50:.4f}/{p90:.4f}/{p99:.4f} | "
                "{post:.4f} | {total:.4f} | {speedup:.4f}x |".format(
                    budget=budget,
                    variant=row["variant"],
                    ap30=float(row["AP@0.3_across_runs_mean"]),
                    ap50=float(row["AP@0.5_across_runs_mean"]),
                    ap70=float(row["AP@0.7_across_runs_mean"]),
                    map=float(row["mAP_across_runs_mean"]),
                    map_std=float(row["mAP_across_runs_std"]),
                    fmean=float(row["forward_mean_ms_across_runs_mean"]),
                    p50=float(row["forward_p50_ms_across_runs_mean"]),
                    p90=float(row["forward_p90_ms_across_runs_mean"]),
                    p99=float(row["forward_p99_ms_across_runs_mean"]),
                    post=float(row["postprocess_mean_ms_across_runs_mean"]),
                    total=float(row["total_mean_ms_across_runs_mean"]),
                    speedup=float(row["speedup_vs_same_gpu_fp32_across_runs_mean"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Mean contribution summary",
            "",
            "| Method | Budget | P+Q/P-only/Q-only mAP | P+Q/P-only/Q-only p50 ms | P+Q/P-only/Q-only speedup |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {method} | {budget:.2f} | {pq:.6f}/{p:.6f}/{q:.6f} | "
            "{pqt:.4f}/{pt:.4f}/{qt:.4f} | {pqx:.4f}/{px:.4f}/{qx:.4f}x |".format(
                method=row["assigned_method"],
                budget=float(row["budget"]),
                pq=float(row["prune_quant_mAP"]),
                p=float(row["prune_only_mAP"]),
                q=float(row["quant_only_mAP"]),
                pqt=float(row["prune_quant_p50_ms"]),
                pt=float(row["prune_only_p50_ms"]),
                qt=float(row["quant_only_p50_ms"]),
                pqx=float(row["prune_quant_speedup"]),
                px=float(row["prune_only_speedup"]),
                qx=float(row["quant_only_speedup"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    if int(args.repeat_count) <= 0:
        raise ValueError("repeat_count_must_be_positive")
    run_dir = _new_run_dir(args.output_root, args.repeat_count)
    inventories = _preflight(args, run_dir)
    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _run_method,
                method=method,
                gpu_id=gpu_id,
                sources=inventories[method],
                args=args,
                run_dir=run_dir,
            ): method
            for method, gpu_id in GPU_ASSIGNMENT.items()
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
    summary = summarize_results(all_results)
    summary.sort(
        key=lambda row: (
            int(row["gpu_id"]),
            int(row["repeat_index"]),
            int(row["sequence_index"]),
        )
    )
    expected_assignments = {
        (str(row["assigned_method"]), int(row["gpu_id"])) for row in summary
    }
    if expected_assignments != {("ga", 0), ("greedy", 1)}:
        raise RuntimeError(f"split_gpu_assignment_violation:{expected_assignments}")
    frame_hashes = {frame_order_hash(row) for row in all_results}
    if len(frame_hashes) != 1:
        raise RuntimeError(f"evaluated_frame_order_mismatch:{sorted(frame_hashes)}")
    if not all(bool(row["cache_cleanup_passed"]) for row in summary):
        raise RuntimeError("not_all_gpu_cache_cleanup_audits_passed")
    aggregated = aggregate_repeated_results(
        summary, repeat_count=int(args.repeat_count)
    )
    comparisons = _mean_contribution_rows(aggregated)
    per_frame_manifest = _combine_per_frame_csv(
        run_dir,
        expected_rows=(
            len(summary) * int(args.num_frames)
        ),
    )
    write_json(run_dir / "per_frame_latency_manifest.json", per_frame_manifest)
    write_summary_csv(run_dir / "split_gpu_repeat_results.csv", summary)
    write_summary_csv(run_dir / "split_gpu_five_repeat_mean_results.csv", aggregated)
    write_summary_csv(
        run_dir / "split_gpu_five_repeat_mean_comparisons.csv", comparisons
    )
    report = {
        "schema_version": "h800-split-gpu-ablation-report-v1",
        "passed": True,
        "gpu_assignment": GPU_ASSIGNMENT,
        "engine_build_count": 0,
        "evaluation_count": len(summary),
        "repeat_count": int(args.repeat_count),
        "same_frame_order": True,
        "frame_order_hash": next(iter(frame_hashes)),
        "all_cache_cleanup_passed": True,
        "per_frame_latency": per_frame_manifest,
        "five_repeat_mean_results": aggregated,
        "five_repeat_mean_comparisons": comparisons,
    }
    write_json(run_dir / "split_gpu_five_repeat_report.json", report)
    (run_dir / "split_gpu_five_repeat_report.md").write_text(
        _repeated_markdown(
            aggregated, comparisons, repeat_count=int(args.repeat_count)
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
