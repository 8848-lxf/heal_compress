from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_mode_fixed_k_plugin_ablation import FIXED_K_BUCKETS, _plugin_so, evaluate_agent_mode
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, DEFAULT_TRT_ROOT, ensure_quant_deploy_run_dirs, read_json, save_json


MODE = "dynamic_agent_dim_fixed_k_plugin"


def _precision_tag(calibration_frames: int, mixed_heads_fp16: bool = False) -> str:
    suffix = "_mixed_heads_fp16" if mixed_heads_fp16 else ""
    return f"int8_calib{int(calibration_frames)}{suffix}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamic fixed-K PointPillarScatterTRT INT8 AP/latency.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calibration_frames", type=int, required=True)
    parser.add_argument("--eval_frames", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--calibration_cache_path", required=True)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--ap_iou_backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--mixed_heads_fp16", action="store_true")
    return parser.parse_args(argv)


def _baseline(dirs: dict[str, Path], precision: str, frames: int) -> dict[str, Any]:
    return read_json(dirs["evaluation"] / f"trt_{precision}_ap_report_{MODE}_{frames}.json", default={}) or {}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _latency_stats(report: dict[str, Any], field: str) -> dict[str, Any]:
    value = report.get(field)
    if isinstance(value, dict):
        stats = dict(value)
    elif field == "forward_ms":
        stats = {
            "p50": report.get("forward_p50_ms"),
            "p90": report.get("forward_p90_ms"),
            "p95": report.get("forward_p95_ms"),
            "p99": report.get("forward_p99_ms"),
            "mean": report.get("forward_mean_ms"),
            "max": report.get("forward_max_ms"),
        }
    else:
        stats = {}
    rows = report.get("frames") or []
    samples = [float(row[field]) for row in rows if row.get(field) is not None]
    if samples:
        stats.setdefault("p50", _percentile(samples, 50))
        stats.setdefault("p90", _percentile(samples, 90))
        stats.setdefault("p95", _percentile(samples, 95))
        stats.setdefault("p99", _percentile(samples, 99))
        stats.setdefault("mean", float(sum(samples) / len(samples)))
        stats.setdefault("max", max(samples))
    if "p99" not in stats:
        stats["p99"] = None
    if "max" not in stats:
        stats["max"] = None
    return stats


def _augment(report: dict[str, Any], *, dirs: dict[str, Path], args: argparse.Namespace, frames: int) -> dict[str, Any]:
    fp32 = _baseline(dirs, "fp32", frames)
    fp16 = _baseline(dirs, "fp16", frames)
    report = dict(report)
    report.update(
        {
            "precision": "int8",
            "precision_strategy": "mixed_heads_fp16" if args.mixed_heads_fp16 else "native",
            "calibration_frames": int(args.calibration_frames),
            "eval_frames": int(frames),
            "mAP drop vs FP32 same eval set": round(float(fp32.get("mAP", fp32.get("map", 0.0))) - float(report.get("mAP", report.get("map", 0.0))), 4) if fp32 else None,
            "mAP drop vs FP16 same eval set": round(float(fp16.get("mAP", fp16.get("map", 0.0))) - float(report.get("mAP", report.get("map", 0.0))), 4) if fp16 else None,
            "AP@0.70 drop vs FP32": round(float(fp32.get("AP@0.70", fp32.get("ap_0_7", 0.0))) - float(report.get("AP@0.70", report.get("ap_0_7", 0.0))), 4) if fp32 else None,
            "AP@0.70 drop vs FP16": round(float(fp16.get("AP@0.70", fp16.get("ap_0_7", 0.0))) - float(report.get("AP@0.70", report.get("ap_0_7", 0.0))), 4) if fp16 else None,
            "execute_ms p50/p90/p95/p99/mean/max": _latency_stats(report, "execute_ms"),
            "forward_ms p50/p90/p95/p99/mean/max": _latency_stats(report, "forward_ms"),
            "total_runner_ms p50/p90/p95/p99/mean/max": _latency_stats(report, "total_runner_ms"),
            "FPS": report.get("FPS") or report.get("fps"),
            "engine_path": "bucket_router_multiple_int8_engines",
            "engine_precision_tag": _precision_tag(int(args.calibration_frames), bool(args.mixed_heads_fp16)),
            "onnx_path": str(dirs["onnx_fp32"] / "dynamic_agent_dim_N1_N2_fixed_k_scatter_plugin"),
            "plugin_path": str(args.plugin_path),
            "calibration_cache_path": str(args.calibration_cache_path),
            "valid_voxel_mask_enabled": True,
            "pointpillar_scatter_plugin_enabled": True,
            "agent_export_mode": "dynamic_agent_dim",
            "fixed_k_buckets": FIXED_K_BUCKETS,
        }
    )
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    plugin = Path(args.plugin_path)
    reports: dict[str, Any] = {}
    precision_tag = _precision_tag(int(args.calibration_frames), bool(args.mixed_heads_fp16))
    for frames in [int(v) for v in args.eval_frames]:
        try:
            report = evaluate_agent_mode(args, dirs, FIXED_K_BUCKETS, plugin, mode=MODE, precision=precision_tag, frames=frames)
            report = _augment(report, dirs=dirs, args=args, frames=frames)
        except Exception as exc:
            report = {
                "success": False,
                "precision": "int8",
                "precision_strategy": "mixed_heads_fp16" if args.mixed_heads_fp16 else "native",
                "calibration_frames": int(args.calibration_frames),
                "eval_frames": frames,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "plugin_path": str(plugin),
                "calibration_cache_path": str(args.calibration_cache_path),
                "engine_precision_tag": precision_tag,
                "agent_export_mode": "dynamic_agent_dim",
                "valid_voxel_mask_enabled": True,
                "pointpillar_scatter_plugin_enabled": True,
            }
        if args.mixed_heads_fp16:
            eval_path = dirs["evaluation"] / f"trt_int8_ap_report_dynamic_fixed_k_plugin_mixed_heads_fp16_{frames}.json"
            bench_path = dirs["benchmark"] / f"dynamic_fixed_k_plugin_int8_mixed_heads_fp16_{frames}.json"
        else:
            eval_path = dirs["evaluation"] / f"trt_int8_ap_report_dynamic_fixed_k_plugin_calib{int(args.calibration_frames)}_{frames}.json"
            bench_path = dirs["benchmark"] / f"dynamic_fixed_k_plugin_int8_calib{int(args.calibration_frames)}_{frames}.json"
        save_json(report, eval_path)
        save_json(
            {
                "precision": "int8",
                "precision_strategy": "mixed_heads_fp16" if args.mixed_heads_fp16 else "native",
                "calibration_frames": int(args.calibration_frames),
                "eval_frames": frames,
                "execute_ms": report.get("execute_ms") or report.get("execute_ms p50/p90/p95/p99/mean/max"),
                "forward_ms": report.get("forward_ms") or report.get("forward_ms p50/p90/p95/p99/mean/max"),
                "total_runner_ms": report.get("total_runner_ms") or report.get("total_runner_ms p50/p90/p95/p99/mean/max"),
                "FPS": report.get("FPS") or report.get("fps"),
                "record_len_distribution": report.get("record_len_distribution"),
                "bucket_distribution": report.get("bucket_distribution"),
                "engine_path": report.get("engine_path"),
                "engine_precision_tag": report.get("engine_precision_tag"),
                "success": report.get("success"),
                "error": report.get("error"),
            },
            bench_path,
        )
        reports[str(frames)] = report
    return {"reports": reports}


def main(argv: list[str] | None = None) -> int:
    report = evaluate(parse_args(argv))
    print(report)
    return 0 if any(item.get("success") for item in report["reports"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
