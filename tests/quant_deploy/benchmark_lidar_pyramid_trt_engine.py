from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import (
    DEFAULT_TRT_ROOT,
    INT8_NOT_IMPLEMENTED_MESSAGE,
    collect_env_report,
    default_benchmark_result,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    read_json,
    run_command,
    save_csv,
    save_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engine forward latency for lidar_pyramid.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    parser.add_argument("--engine_path", default=None)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args(argv)


def _default_engine_path(dirs: dict[str, Path], precision: str) -> Path:
    name = "lidar_pyramid_int8_qdq.engine" if precision == "int8" else f"lidar_pyramid_{precision}.engine"
    return dirs[f"engine_{precision}"] / name


def _parse_latency_from_trtexec(log_text: str) -> dict[str, float | None]:
    metrics = {
        "forward_mean_ms": None,
        "forward_p50_ms": None,
        "forward_p90_ms": None,
        "forward_p95_ms": None,
        "forward_min_ms": None,
        "forward_max_ms": None,
    }
    lines = [line.strip() for line in log_text.splitlines()]
    target_lines = [line for line in lines if "GPU Compute Time" in line or "Latency" in line or "GPU latency" in line]
    text = "\n".join(target_lines or lines)
    patterns = {
        "forward_min_ms": r"min\s*=\s*([0-9.]+)\s*ms",
        "forward_max_ms": r"max\s*=\s*([0-9.]+)\s*ms",
        "forward_mean_ms": r"mean\s*=\s*([0-9.]+)\s*ms",
        "forward_p50_ms": r"median\s*=\s*([0-9.]+)\s*ms",
        "forward_p90_ms": r"percentile\(90%\)\s*=\s*([0-9.]+)\s*ms",
        "forward_p95_ms": r"percentile\(95%\)\s*=\s*([0-9.]+)\s*ms",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))
    if metrics["forward_p50_ms"] is None and metrics["forward_mean_ms"] is not None:
        metrics["forward_p50_ms"] = metrics["forward_mean_ms"]
    return metrics


def benchmark_engine(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    precision = args.precision.lower()
    engine_path = Path(args.engine_path) if args.engine_path else _default_engine_path(dirs, precision)
    bench_dir = dirs[f"benchmark_{precision}"]
    log_path = dirs["logs_benchmark"] / f"benchmark_{precision}.log"
    result = default_benchmark_result(precision, engine_path, args.num_frames, args.warmup_frames)
    result["latency_scope"] = "engine_forward_only"
    env_report = collect_env_report(trt_root=getattr(args, "trt_root", None), explicit_trtexec=getattr(args, "trtexec_path", None))
    save_json(env_report, dirs["debug"] / "env_report.json")

    if precision == "int8":
        result.update({"success": False, "error": INT8_NOT_IMPLEMENTED_MESSAGE})
        log_path.write_text(INT8_NOT_IMPLEMENTED_MESSAGE + "\n", encoding="utf-8")
    elif not engine_path.exists():
        result.update({"success": False, "error": f"engine file does not exist: {engine_path}"})
        log_path.write_text(result["error"] + "\n", encoding="utf-8")
    else:
        trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
        result["trtexec"] = trtexec_report
        if not trtexec_report["trtexec_found"]:
            result.update({"success": False, "error": f"trtexec not found. {trtexec_report['suggestion']}"})
            log_path.write_text(result["error"] + "\n" + str(trtexec_report) + "\n", encoding="utf-8")
        else:
            cmd = [
                trtexec_report["trtexec_path"],
                f"--loadEngine={engine_path}",
                f"--warmUp={max(0, args.warmup_frames)}",
                f"--iterations={max(1, args.num_frames)}",
                "--verbose",
            ]
            command_result = run_command(cmd, log_path, timeout=args.timeout)
            log_text = Path(log_path).read_text(encoding="utf-8")
            result.update(_parse_latency_from_trtexec(log_text))
            result["success"] = bool(command_result["success"])
            result["error"] = command_result["error"]
            if result["forward_p50_ms"]:
                result["fps"] = 1000.0 / float(result["forward_p50_ms"])
            if not result["success"] and not result["error"]:
                result["error"] = "trtexec benchmark failed."

    json_path = bench_dir / f"benchmark_{precision}.json"
    csv_path = bench_dir / f"benchmark_{precision}.csv"
    raw_path = bench_dir / f"latency_raw_{precision}.csv"
    save_json(result, json_path)
    save_csv([result], csv_path)
    save_csv(
        [
            {
                "precision": precision,
                "latency_scope": result["latency_scope"],
                "forward_mean_ms": result["forward_mean_ms"],
                "forward_p50_ms": result["forward_p50_ms"],
                "forward_p90_ms": result["forward_p90_ms"],
                "forward_p95_ms": result["forward_p95_ms"],
                "source": "trtexec_summary",
            }
        ],
        raw_path,
    )
    existing_summary = read_json(dirs["summary"] / f"summary_{precision}.json", default={}) or {}
    build_success = bool(existing_summary.get("build_success", engine_path.exists()))
    summary_error = result["error"] if build_success else existing_summary.get("error", result["error"])
    save_json(
        {
            "precision": precision,
            "implemented": precision in {"fp32", "fp16"},
            "build_success": build_success,
            "benchmark_success": result["success"],
            "engine_path": str(engine_path),
            "engine_size_MB": result["engine_size_MB"],
            "forward_p50_ms": result["forward_p50_ms"],
            "forward_p90_ms": result["forward_p90_ms"],
            "forward_p95_ms": result["forward_p95_ms"],
            "fps": result["fps"],
            "speedup_vs_fp32": None,
            "error": summary_error,
        },
        dirs["summary"] / f"summary_{precision}.json",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = benchmark_engine(args)
    if not result["success"]:
        print(result["error"], file=sys.stderr)
        return 2 if args.precision != "int8" else 0
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
