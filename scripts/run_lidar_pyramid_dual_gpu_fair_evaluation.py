#!/usr/bin/env python3
"""Evaluate existing lidar_pyramid engines fairly on H800 GPU 0 and 1."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.integration.dual_gpu_fair_evaluation import (  # noqa: E402
    build_evaluation_inventory,
    canonical_json_hash,
    cross_gpu_differences,
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
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num-frames", type=int, default=1789)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--latency-rounds", type=int, default=3)
    parser.add_argument("--fixed-k", type=int, default=29696)
    return parser.parse_args()


def _run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_root / f"h800_lidar_pyramid_dual_gpu_eval_only_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination.resolve()


def _preflight(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    trtexec_candidates = (
        args.tensorrt_root / "bin/trtexec",
        args.tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec",
    )
    required = (
        args.checkpoint,
        args.model_config,
        args.heal_root,
        args.tensorrt_root,
        args.plugin,
        args.eval_manifest,
    )
    missing = [str(path) for path in required if not path.exists()]
    if not any(path.is_file() for path in trtexec_candidates):
        missing.append("trtexec:" + ":".join(str(path) for path in trtexec_candidates))
    if missing:
        raise RuntimeError(f"fair_evaluation_preflight_missing:{missing}")
    if list(args.gpu_ids) != [0, 1]:
        raise RuntimeError(f"fair_evaluation_requires_gpu_0_then_1:{args.gpu_ids}")
    gpu_preflight = [gpu_snapshot(gpu_id) for gpu_id in args.gpu_ids]
    occupied = [
        row
        for row in gpu_preflight
        if row["memory_used_mib"] > 256 or row["utilization_percent"] > 5
    ]
    if occupied:
        raise RuntimeError(f"requested_gpu_not_idle:{occupied}")
    sources = [
        verify_existing_engine(row)
        for row in build_evaluation_inventory(ablation_root=args.ablation_root)
    ]
    manifest = json.loads(args.eval_manifest.read_text(encoding="utf-8"))
    preflight = {
        "schema_version": "h800-dual-gpu-evaluation-only-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "execution_policy": {
            "gpu_order": list(args.gpu_ids),
            "global_serial": True,
            "per_gpu_serial": True,
            "engine_build_allowed": False,
            "fresh_worker_process_per_evaluation": True,
            "cuda_cache_cleanup_after_each_evaluation": True,
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
        "source_count": len(sources),
        "sources": sources,
    }
    preflight["inventory_hash"] = canonical_json_hash(sources)
    write_json(run_dir / "evaluation_only_manifest.json", preflight)
    return sources


def _markdown(
    rows: list[dict[str, Any]], cross_gpu: list[dict[str, Any]]
) -> str:
    lines = [
        "# H800 lidar_pyramid dual-GPU evaluation-only fairness audit",
        "",
        "No engine was built. GPU 0 and GPU 1 were evaluated globally serially; each",
        "engine used a fresh worker process followed by a CUDA cache cleanup audit.",
        "",
    ]
    for gpu_id in (0, 1):
        lines.extend(
            [
                f"## GPU {gpu_id}",
                "",
                "| Method | Budget | Actual BOPS | AP30 | AP50 | AP70 | mAP | p50 ms | p90 ms | p99 ms | Speedup | INT8/FP16/FP32 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if int(row["gpu_id"]) != gpu_id:
                continue
            budget = "-" if row["budget"] is None else f"{float(row['budget']):.2f}"
            lines.append(
                "| {method} | {budget} | {bops:.6f} | {ap30:.6f} | {ap50:.6f} | "
                "{ap70:.6f} | {map:.6f} | {p50:.4f} | {p90:.4f} | {p99:.4f} | "
                "{speedup:.4f}x | {i8}/{f16}/{f32} |".format(
                    method=row["method"],
                    budget=budget,
                    bops=float(row["actual_bops"]),
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
            "## Cross-GPU consistency",
            "",
            "| Engine | abs ΔmAP | abs Δp50 ms | GPU1/GPU0 p50 | Same hash |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in cross_gpu:
        lines.append(
            "| {item} | {map_delta:.6f} | {p50_delta:.4f} | {ratio:.4f} | {same} |".format(
                item=row["item_id"],
                map_delta=float(row["mAP_abs_delta"]),
                p50_delta=float(row["p50_abs_delta_ms"]),
                ratio=float(row["p50_ratio_gpu1_vs_gpu0"]),
                same=str(bool(row["same_engine_hash"])).lower(),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    run_dir = _run_dir(args.output_root)
    sources = _preflight(args, run_dir)
    results: list[dict[str, Any]] = []
    for gpu_id in args.gpu_ids:
        for source in sources:
            item = str(source["item_id"])
            destination = (
                run_dir
                / f"gpu_{gpu_id}"
                / f"{int(source['sequence_index']):02d}_{item}"
            )
            print(
                json.dumps(
                    {"event": "evaluation_start", "gpu": gpu_id, "item": item},
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
            write_json(run_dir / "completed_results.json", {"results": results})
            print(
                json.dumps(
                    {
                        "event": "evaluation_complete",
                        "gpu": gpu_id,
                        "item": item,
                        "mAP": result["mAP"],
                        "p50_ms": result["forward_p50_ms"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = summarize_results(results)
    cross_gpu = cross_gpu_differences(summary)
    frame_hashes = {str(row["frame_order_hash"]) for row in summary}
    if len(frame_hashes) != 1:
        raise RuntimeError(f"evaluated_frame_order_mismatch:{sorted(frame_hashes)}")
    if not all(bool(row["cache_cleanup_passed"]) for row in summary):
        raise RuntimeError("not_all_gpu_cache_cleanup_audits_passed")
    write_summary_csv(run_dir / "fair_evaluation_results.csv", summary)
    write_summary_csv(run_dir / "cross_gpu_consistency.csv", cross_gpu)
    write_json(
        run_dir / "fair_evaluation_report.json",
        {
            "schema_version": "h800-dual-gpu-fair-evaluation-report-v1",
            "passed": True,
            "engine_build_count": 0,
            "evaluation_count": len(summary),
            "gpu_count": len(args.gpu_ids),
            "same_frame_order": True,
            "frame_order_hash": next(iter(frame_hashes)),
            "all_cache_cleanup_passed": True,
            "max_cross_gpu_map_abs_delta": max(
                float(row["mAP_abs_delta"]) for row in cross_gpu
            ),
            "max_cross_gpu_p50_abs_delta_ms": max(
                float(row["p50_abs_delta_ms"]) for row in cross_gpu
            ),
            "cross_gpu_consistency": cross_gpu,
            "results": summary,
        },
    )
    (run_dir / "fair_evaluation_report.md").write_text(
        _markdown(summary, cross_gpu), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
