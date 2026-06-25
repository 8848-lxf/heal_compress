from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import (
    DEFAULT_TRT_ROOT,
    INT8_NOT_IMPLEMENTED_MESSAGE,
    build_trtexec_command,
    collect_env_report,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    load_profile_shapes,
    parse_trtexec_failure,
    read_json,
    run_command,
    save_json,
    write_debug_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT engine for lidar_pyramid ONNX.")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    parser.add_argument("--profile_shapes_json", default=None)
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
    return parser.parse_args(argv)


def _engine_name(precision: str) -> str:
    if precision == "int8":
        return "lidar_pyramid_int8_qdq.engine"
    return f"lidar_pyramid_{precision}.engine"


def build_engine(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    precision = args.precision.lower()
    engine_dir = dirs[f"engine_{precision}"]
    engine_path = engine_dir / _engine_name(precision)
    layerinfo_path = engine_dir / f"layerinfo_{precision}.json"
    log_path = dirs["logs_build"] / f"build_{precision}.log"
    meta_path = engine_dir / f"engine_meta_{precision}.json"
    summary_path = dirs["summary"] / f"summary_{precision}.json"
    result: dict[str, Any] = {
        "precision": precision,
        "implemented": precision in {"fp32", "fp16"},
        "build_success": False,
        "benchmark_success": False,
        "engine_path": str(engine_path),
        "engine_size_MB": None,
        "layerinfo_path": str(layerinfo_path),
        "log_path": str(log_path),
        "error": None,
    }
    env_report = collect_env_report(trt_root=getattr(args, "trt_root", None), explicit_trtexec=getattr(args, "trtexec_path", None))
    save_json(env_report, dirs["debug"] / "env_report.json")
    result["env_report"] = env_report

    if precision == "int8":
        log_path.write_text(INT8_NOT_IMPLEMENTED_MESSAGE + "\n", encoding="utf-8")
        result.update({"implemented": False, "error": INT8_NOT_IMPLEMENTED_MESSAGE})
        save_json(result, meta_path)
        save_json(result, summary_path)
        return result

    profile_shapes = load_profile_shapes(dirs, args.profile_shapes_json)
    save_json(profile_shapes or {}, dirs["configs"] / "profile_shapes.json")
    no_tf32 = bool(args.no_tf32 and not args.allow_tf32)
    trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result["trtexec"] = trtexec_report
    if not trtexec_report["trtexec_found"]:
        result["error"] = f"trtexec not found. {trtexec_report['suggestion']}"
        log_path.write_text(result["error"] + "\n" + str(trtexec_report) + "\n", encoding="utf-8")
        save_json(result, meta_path)
        save_json(result, summary_path)
        save_json(trtexec_report, dirs["debug"] / "trtexec_report.json")
        return result
    try:
        cmd = build_trtexec_command(
            precision=precision,
            onnx_path=args.onnx_path,
            engine_path=engine_path,
            layerinfo_path=layerinfo_path,
            profile_shapes=profile_shapes,
            trtexec_path=trtexec_report["trtexec_path"],
            no_tf32=no_tf32,
        )
    except Exception as exc:
        result["error"] = str(exc)
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        save_json(result, meta_path)
        save_json(result, summary_path)
        return result

    command_result = run_command(cmd, log_path, timeout=args.timeout)
    build_log_text = Path(log_path).read_text(encoding="utf-8")
    failure = parse_trtexec_failure(build_log_text)
    result.update(
        {
            "command": cmd,
            "returncode": command_result["returncode"],
            "build_success": bool(command_result["success"] and engine_path.exists()),
            "error": command_result["error"],
            "unsupported_ops": failure["unsupported_ops"],
            "failed_nodes": failure["failed_nodes"],
        }
    )
    if not result["build_success"] and failure["unsupported_ops"]:
        result["error"] = failure["unsupported_ops"][0].get("line") or result["error"]
    if engine_path.exists():
        result["engine_size_MB"] = engine_path.stat().st_size / (1024 * 1024)
    if not result["build_success"] and not result["error"]:
        result["error"] = "trtexec did not produce the expected engine file."

    strict_result = None
    if precision == "fp16" and args.strict_fp16:
        strict_engine_path = engine_dir / "lidar_pyramid_fp16_strict.engine"
        strict_layerinfo_path = engine_dir / "layerinfo_fp16_strict.json"
        strict_log_path = dirs["logs_build"] / "build_fp16_strict.log"
        strict_cmd = build_trtexec_command(
            precision="fp16",
            onnx_path=args.onnx_path,
            engine_path=strict_engine_path,
            layerinfo_path=strict_layerinfo_path,
            profile_shapes=profile_shapes,
            trtexec_path=trtexec_report["trtexec_path"],
            strict_fp16=True,
        )
        strict_command_result = run_command(strict_cmd, strict_log_path, timeout=args.timeout)
        strict_result = {
            "engine_path": str(strict_engine_path),
            "layerinfo_path": str(strict_layerinfo_path),
            "log_path": str(strict_log_path),
            "command": strict_cmd,
            "success": bool(strict_command_result["success"] and strict_engine_path.exists()),
            "error": strict_command_result["error"],
        }
    result["strict_fp16"] = strict_result

    save_json(result, meta_path)
    save_json(result, summary_path)
    write_debug_reports(dirs, build_log=build_log_text)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_engine(args)
    if not result["build_success"]:
        print(result.get("error") or "TensorRT build failed", file=sys.stderr)
        return 2 if result["precision"] != "int8" else 0
    print(result["engine_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
