from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_lidar_pyramid_trt_engine import benchmark_engine
from build_lidar_pyramid_trt_engine import build_engine
from export_lidar_pyramid_onnx import export_lidar_pyramid_onnx
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRT_ROOT,
    INT8_NOT_IMPLEMENTED_MESSAGE,
    collect_env_report,
    create_quant_deploy_run_dirs,
    detect_special_ops_in_onnx,
    dirs_for_summary,
    parse_precisions,
    read_json,
    save_csv,
    save_json,
    write_summary_files,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-key LiDAR pyramid ONNX -> TensorRT -> benchmark runner.")
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default=None)
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16", "int8"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no_tf32", action="store_true", default=True)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--strict_fp16", action="store_true")
    parser.add_argument("--calib_dir", default=None)
    parser.add_argument("--calib_num_frames", type=int, default=0)
    parser.add_argument("--calib_cache", default=None)
    parser.add_argument("--qdq_onnx_path", default=None)
    parser.add_argument("--int8_mode", default="qdq", choices=["qdq"])
    parser.add_argument("--allow_fp16_fallback", action="store_true")
    parser.add_argument("--allow_synthetic_fallback", action="store_true")
    parser.add_argument("--bev_warp_export_mode", default="exportable_grid", choices=["original", "exportable_grid"])
    parser.add_argument("--pillar_vfe_export_fix", default="explicit_squeeze", choices=["none", "explicit_squeeze"])
    parser.add_argument("--pyramid_forward_export_mode", default="fixed_static", choices=["original", "fixed_static"])
    return parser.parse_args(argv)


def _precision_summary_from_files(dirs: dict[str, Path], precision: str) -> dict[str, Any]:
    item = read_json(dirs["summary"] / f"summary_{precision}.json", default={}) or {}
    benchmark = read_json(dirs[f"benchmark_{precision}"] / f"benchmark_{precision}.json", default={}) or {}
    engine_name = "lidar_pyramid_int8_qdq.engine" if precision == "int8" else f"lidar_pyramid_{precision}.engine"
    engine_path = dirs[f"engine_{precision}"] / engine_name
    build_success = bool(item.get("build_success", False))
    error = item.get("error")
    if build_success:
        error = benchmark.get("error") or error
    return {
        "implemented": precision in {"fp32", "fp16"},
        "build_success": build_success,
        "benchmark_success": bool(benchmark.get("success", item.get("benchmark_success", False))),
        "engine_path": str(engine_path) if engine_path.exists() else item.get("engine_path"),
        "engine_size_MB": benchmark.get("engine_size_MB", item.get("engine_size_MB")),
        "forward_p50_ms": benchmark.get("forward_p50_ms", item.get("forward_p50_ms")),
        "forward_p90_ms": benchmark.get("forward_p90_ms", item.get("forward_p90_ms")),
        "forward_p95_ms": benchmark.get("forward_p95_ms", item.get("forward_p95_ms")),
        "fps": benchmark.get("fps", item.get("fps")),
        "speedup_vs_fp32": None,
        "error": error,
    }


def _write_skip_precision(dirs: dict[str, Path], precision: str, error: str) -> None:
    log_path = dirs["logs_build"] / f"build_{precision}.log"
    log_path.write_text(error + "\n", encoding="utf-8")
    summary = {
        "implemented": precision in {"fp32", "fp16"},
        "build_success": False,
        "benchmark_success": False,
        "engine_path": None,
        "engine_size_MB": None,
        "forward_p50_ms": None,
        "forward_p90_ms": None,
        "forward_p95_ms": None,
        "fps": None,
        "speedup_vs_fp32": None,
        "error": error,
    }
    save_json(summary, dirs["summary"] / f"summary_{precision}.json")


def _write_skip_benchmark(dirs: dict[str, Path], precision: str, error: str, num_frames: int, warmup_frames: int) -> None:
    bench_dir = dirs[f"benchmark_{precision}"]
    payload = {
        "precision": precision,
        "engine_path": None,
        "num_frames": num_frames,
        "warmup_frames": warmup_frames,
        "forward_mean_ms": None,
        "forward_p50_ms": None,
        "forward_p90_ms": None,
        "forward_p95_ms": None,
        "forward_min_ms": None,
        "forward_max_ms": None,
        "fps": None,
        "engine_size_MB": None,
        "latency_scope": "engine_forward_only",
        "data_loading_time_ms": None,
        "data_to_gpu_time_ms": None,
        "forward_time_ms": None,
        "postprocess_time_ms": None,
        "total_time_ms": None,
        "success": False,
        "error": error,
    }
    save_json(payload, bench_dir / f"benchmark_{precision}.json")
    from quant_deploy_utils import save_csv

    save_csv([payload], bench_dir / f"benchmark_{precision}.csv")
    save_csv([{"precision": precision, "latency_scope": payload["latency_scope"], "source": "skipped", "error": error}], bench_dir / f"latency_raw_{precision}.csv")
    (dirs["logs_benchmark"] / f"benchmark_{precision}.log").write_text(error + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    precisions = parse_precisions(args.precision, args.precisions)
    dirs = create_quant_deploy_run_dirs(args.output_dir, args.run_name, args.overwrite)
    env_report = collect_env_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    save_json(env_report, dirs["debug"] / "env_report.json")
    save_json(
        {
            "checkpoint": args.checkpoint,
            "hypes_yaml": args.hypes_yaml,
            "heal_repo": args.heal_repo,
            "precisions": precisions,
            "device": args.device,
            "num_frames": args.num_frames,
            "warmup_frames": args.warmup_frames,
            "max_cav": args.max_cav,
            "opset": args.opset,
            "strict_fp16": args.strict_fp16,
            "no_tf32": bool(args.no_tf32 and not args.allow_tf32),
            "int8_mode": args.int8_mode,
            "allow_fp16_fallback": args.allow_fp16_fallback,
            "bev_warp_export_mode": args.bev_warp_export_mode,
            "pillar_vfe_export_fix": args.pillar_vfe_export_fix,
            "pyramid_forward_export_mode": args.pyramid_forward_export_mode,
            "trt_root": args.trt_root,
            "trtexec_path": args.trtexec_path,
            "output_root": str(dirs["output_root"]),
        },
        dirs["configs"] / "run_config.json",
    )
    save_json({"precisions": precisions, "strict_fp16": args.strict_fp16, "no_tf32": bool(args.no_tf32 and not args.allow_tf32)}, dirs["configs"] / "precision_config.json")
    save_json({"int8_qdq": "reserved", "calib_dir": args.calib_dir, "calib_num_frames": args.calib_num_frames}, dirs["configs"] / "deploy_config.json")
    save_json({"status": "reserved_not_generated"}, dirs["calibration"] / "dataset_manifest.json")
    for precision in ("fp32", "fp16", "int8"):
        eval_payload = {
            "precision": precision,
            "evaluation_status": "not_run",
            "reason": "current stage only benchmarks TensorRT engine forward latency",
            "ap_0_3": None,
            "ap_0_5": None,
            "ap_0_7": None,
            "map": None,
        }
        save_json(eval_payload, dirs[f"evaluation_{precision}"] / f"eval_metrics_{precision}.json")
        save_csv([eval_payload], dirs[f"evaluation_{precision}"] / f"eval_metrics_{precision}.csv")
        (dirs[f"evaluation_{precision}"] / "pred_outputs").mkdir(parents=True, exist_ok=True)
        (dirs["logs_evaluation"] / f"evaluate_{precision}.log").write_text(eval_payload["reason"] + "\n", encoding="utf-8")
        (dirs[f"evaluation_{precision}"] / f"eval_log_{precision}.txt").write_text(eval_payload["reason"] + "\n", encoding="utf-8")

    export_args = SimpleNamespace(
        hypes_yaml=args.hypes_yaml,
        checkpoint=args.checkpoint,
        heal_repo=args.heal_repo,
        output_dir=args.output_dir,
        output_root=str(dirs["output_root"]),
        run_name=args.run_name,
        device=args.device,
        num_frames=1,
        max_cav=args.max_cav,
        opset=args.opset,
        overwrite=True,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        trt_root=args.trt_root,
        trtexec_path=args.trtexec_path,
        bev_warp_export_mode=args.bev_warp_export_mode,
        pillar_vfe_export_fix=args.pillar_vfe_export_fix,
        pyramid_forward_export_mode=args.pyramid_forward_export_mode,
    )
    export_result = export_lidar_pyramid_onnx(export_args)
    onnx_path = dirs["onnx_fp32"] / "lidar_pyramid_fp32_dynamic.onnx"

    for precision in precisions:
        if precision == "int8":
            build_args = SimpleNamespace(
                onnx_path=str(onnx_path),
                output_root=str(dirs["output_root"]),
                precision="int8",
                profile_shapes_json=None,
                trt_root=args.trt_root,
                trtexec_path=args.trtexec_path,
                timeout=args.timeout,
                no_tf32=args.no_tf32,
                allow_tf32=args.allow_tf32,
                strict_fp16=args.strict_fp16,
                calib_dir=args.calib_dir,
                calib_num_frames=args.calib_num_frames,
                calib_cache=args.calib_cache,
                qdq_onnx_path=args.qdq_onnx_path,
                int8_mode=args.int8_mode,
                allow_fp16_fallback=args.allow_fp16_fallback,
            )
            build_engine(build_args)
            benchmark_engine(SimpleNamespace(output_root=str(dirs["output_root"]), precision="int8", engine_path=None, num_frames=args.num_frames, warmup_frames=args.warmup_frames, device=args.device, trt_root=args.trt_root, trtexec_path=args.trtexec_path, timeout=args.timeout))
            continue
        if not export_result["success"]:
            skip_error = f"skipped because ONNX export failed: {export_result.get('error')}"
            _write_skip_precision(dirs, precision, skip_error)
            _write_skip_benchmark(dirs, precision, skip_error, args.num_frames, args.warmup_frames)
            continue
        build_args = SimpleNamespace(
            onnx_path=str(onnx_path),
            output_root=str(dirs["output_root"]),
            precision=precision,
            profile_shapes_json=None,
            trt_root=args.trt_root,
            trtexec_path=args.trtexec_path,
            timeout=args.timeout,
            no_tf32=args.no_tf32,
            allow_tf32=args.allow_tf32,
            strict_fp16=args.strict_fp16,
            calib_dir=args.calib_dir,
            calib_num_frames=args.calib_num_frames,
            calib_cache=args.calib_cache,
            qdq_onnx_path=args.qdq_onnx_path,
            int8_mode=args.int8_mode,
            allow_fp16_fallback=args.allow_fp16_fallback,
        )
        build_result = build_engine(build_args)
        benchmark_engine(
            SimpleNamespace(
                output_root=str(dirs["output_root"]),
                precision=precision,
                engine_path=build_result.get("engine_path") if build_result.get("build_success") else None,
                num_frames=args.num_frames,
                warmup_frames=args.warmup_frames,
                device=args.device,
                trt_root=args.trt_root,
                trtexec_path=args.trtexec_path,
                timeout=args.timeout,
            )
        )

    precision_items = {precision: _precision_summary_from_files(dirs, precision) for precision in precisions}
    fp32_p50 = precision_items.get("fp32", {}).get("forward_p50_ms")
    if fp32_p50:
        for precision, item in precision_items.items():
            p50 = item.get("forward_p50_ms")
            item["speedup_vs_fp32"] = float(fp32_p50) / float(p50) if p50 else None
            save_json(item, dirs["summary"] / f"summary_{precision}.json")
    if "int8" in precision_items:
        precision_items["int8"]["implemented"] = False
        precision_items["int8"]["error"] = precision_items["int8"].get("error") or INT8_NOT_IMPLEMENTED_MESSAGE

    special_ops = detect_special_ops_in_onnx(onnx_path) if onnx_path.exists() else read_json(dirs["debug"] / "special_ops_report.json", default={})
    summary = {
        "model": "lidar_pyramid",
        "checkpoint": args.checkpoint,
        "hypes_yaml": args.hypes_yaml,
        "output_root": str(dirs["output_root"]),
        "dirs": dirs_for_summary(dirs),
        "onnx_path": str(onnx_path) if onnx_path.exists() else None,
        "pyramid_forward_export_mode": args.pyramid_forward_export_mode,
        "fixed_pyramid_wrapper_source": export_result.get("fixed_pyramid_wrapper_source"),
        "fixed_pyramid_forward_report": export_result.get("fixed_pyramid_forward_report"),
        "onnx_export": {
            "success": bool(export_result["success"]),
            "error": export_result.get("error"),
            "export_boundary": export_result.get("export_boundary", "full_model"),
            "opset": args.opset,
        },
        "precisions": precision_items,
        "detected_special_ops": special_ops or {"GridSample": [], "AffineGrid": [], "Scatter": [], "Inverse": [], "unsupported_ops": []},
        "env_report": env_report,
        "bev_warp_export_mode": args.bev_warp_export_mode,
        "pillar_vfe_export_fix": args.pillar_vfe_export_fix,
        "evaluation_status": "not_run",
        "evaluation_reason": "current stage only benchmarks TensorRT engine forward latency",
    }
    write_summary_files(summary, dirs)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_pipeline(args)
    print(summary["output_root"])
    if not summary["onnx_export"]["success"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
