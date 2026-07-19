#!/usr/bin/env python3
"""Evaluate GA ablations on GPU0 and greedy ablations on GPU1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    build_method_ablation_inventory,
    canonical_json_hash,
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
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
)
DEFAULT_CONFIG = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/config.yaml"
)
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
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
    return parser.parse_args()


def _new_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_root / f"h800_lidar_pyramid_split_gpu_ablation_{stamp}"
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
    for source in sources:
        item_id = str(source["item_id"])
        destination = (
            run_dir
            / f"gpu_{gpu_id}_{method}"
            / f"{int(source['sequence_index']):02d}_{item_id}"
        )
        print(
            json.dumps(
                {"event": "evaluation_start", "gpu": gpu_id, "method": method, "item": item_id},
                sort_keys=True,
            ),
            flush=True,
        )
        result = run_one_evaluation(
            source=source,
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
        results.append(result)
        write_json(
            run_dir / f"gpu_{gpu_id}_{method}_completed.json", {"results": results}
        )
        print(
            json.dumps(
                {
                    "event": "evaluation_complete",
                    "gpu": gpu_id,
                    "method": method,
                    "item": item_id,
                    "mAP": result["mAP"],
                    "p50_ms": result["forward_p50_ms"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


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


def main() -> int:
    args = _parse_args()
    run_dir = _new_run_dir(args.output_root)
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
    summary.sort(key=lambda row: (int(row["gpu_id"]), int(row["sequence_index"])))
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
    comparisons = ablation_contribution_rows(summary)
    write_summary_csv(run_dir / "split_gpu_full_results.csv", summary)
    write_summary_csv(run_dir / "split_gpu_ablation_comparisons.csv", comparisons)
    report = {
        "schema_version": "h800-split-gpu-ablation-report-v1",
        "passed": True,
        "gpu_assignment": GPU_ASSIGNMENT,
        "engine_build_count": 0,
        "evaluation_count": len(summary),
        "same_frame_order": True,
        "frame_order_hash": next(iter(frame_hashes)),
        "all_cache_cleanup_passed": True,
        "results": summary,
        "comparisons": comparisons,
    }
    write_json(run_dir / "split_gpu_ablation_report.json", report)
    (run_dir / "split_gpu_ablation_report.md").write_text(
        _markdown(summary, comparisons), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
