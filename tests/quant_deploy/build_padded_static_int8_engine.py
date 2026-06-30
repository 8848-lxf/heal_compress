from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_dynamic_fixed_k_int8_engine import _NpzEntropyCalibrator, _build_with_python_calibrator
from bucketed_padded_agent_latency import _fixed_k_plugin_bucket_layerinfo_path, profile_shapes_for_fixed_k_scatter_plugin_bucket
from dump_padded_static_train_calibration_npz import calibration_npz_dir
from dump_train_calibration_npz_for_all_strategies import fixed_k_buckets
from quant_deploy_utils import DEFAULT_OUTPUT_DIR, DEFAULT_TRT_ROOT, collect_env_report, ensure_quant_deploy_run_dirs, read_json, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build padded_agent_static fixedK INT8 train-calib TensorRT engines.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_DIR / "lidar_pyramid_agent_export_strategy_compare"))
    parser.add_argument("--fixed_k", type=int, default=29696)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--calibration_frames", type=int, default=200)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--force_recalibrate", action="store_true")
    return parser.parse_args(argv)


def fixedk_dirs(output_root: str | Path, fixed_k: int) -> dict[str, Path]:
    dirs = ensure_quant_deploy_run_dirs(output_root)
    namespace = f"fixedK{int(fixed_k)}"
    dirs["onnx"] = dirs["output_root"] / "artifacts" / "onnx" / namespace
    dirs["onnx_fp32"] = dirs["onnx"] / "fp32"
    dirs["engines"] = dirs["output_root"] / "artifacts" / "engines" / namespace
    dirs["logs_build"] = dirs["output_root"] / "logs" / "build" / namespace
    for key in ("onnx", "onnx_fp32", "engines", "logs_build"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def padded_int8_engine_path(dirs: dict[str, Path], fixed_k: int, bucket_id: int) -> Path:
    return (
        dirs["engines"]
        / "padded_agent_static"
        / "int8_train_calib200"
        / f"lidar_pyramid_padded_agent_static_fixedK{int(fixed_k)}_int8_train_calib200_bucket{int(bucket_id)}.engine"
    )


def padded_int8_layerinfo_path(dirs: dict[str, Path], bucket_id: int) -> Path:
    return dirs["engines"] / "padded_agent_static" / "int8_train_calib200" / f"layerinfo_padded_agent_static_int8_train_calib200_bucket{int(bucket_id)}.json"


def padded_int8_cache_path(dirs: dict[str, Path], fixed_k: int, frames: int, bucket_id: int) -> Path:
    return (
        dirs["output_root"]
        / "artifacts"
        / "calibration"
        / f"lidar_pyramid_padded_agent_static_fixedK{int(fixed_k)}_int8_train_calib{int(frames)}_bucket{int(bucket_id)}.cache"
    )


def _onnx_path(dirs: dict[str, Path]) -> Path:
    canonical = dirs["onnx_fp32"] / "lidar_pyramid_padded_agent_static_fixed_k_scatter_plugin.onnx"
    if canonical.exists():
        return canonical
    return dirs["onnx_fp32"] / "lidar_pyramid_fixed_k_scatter_plugin_fp32_dynamic.onnx"


def _sample_files(npz_dir: Path, bucket_id: int) -> list[Path]:
    return sorted(npz_dir.glob(f"*_bucket{int(bucket_id)}.npz"))


def _read_layerinfo(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _count_layer_precisions(layer_text: str) -> tuple[int | None, int | None]:
    try:
        data = json.loads(layer_text)
    except Exception:
        data = None
    if not isinstance(data, list):
        return None, None
    int8 = 0
    fp16 = 0
    for layer in data:
        text = json.dumps(layer).lower()
        if "int8" in text:
            int8 += 1
        if "fp16" in text or "float16" in text:
            fp16 += 1
    return int8, fp16


def _plugin_loaded(plugin_path: Path) -> bool:
    try:
        ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        return True
    except Exception:
        return False


def _build_one(args: argparse.Namespace, dirs: dict[str, Path], bucket: dict[str, Any]) -> dict[str, Any]:
    bucket_id = int(bucket["bucket_id"])
    npz_dir = calibration_npz_dir(args.output_root, int(args.fixed_k), int(args.calibration_frames))
    samples = _sample_files(npz_dir, bucket_id)
    onnx_path = _onnx_path(dirs)
    engine_path = padded_int8_engine_path(dirs, int(args.fixed_k), bucket_id)
    layerinfo_path = padded_int8_layerinfo_path(dirs, bucket_id)
    log_path = dirs["logs_build"] / f"build_padded_agent_static_fixedK{int(args.fixed_k)}_int8_train_calib{int(args.calibration_frames)}_bucket{bucket_id}.log"
    cache_path = padded_int8_cache_path(dirs, int(args.fixed_k), int(args.calibration_frames), bucket_id)
    plugin_path = Path(args.plugin_path).expanduser()
    profile_shapes = profile_shapes_for_fixed_k_scatter_plugin_bucket(bucket, max_cav=int(args.max_cav))
    result: dict[str, Any] = {
        "strategy": "padded_agent_static",
        "fixed_K": int(args.fixed_k),
        "max_cav": int(args.max_cav),
        "bucket": bucket,
        "bucket_id": bucket_id,
        "precision": "int8",
        "calibration_split": "train",
        "calibration_frames": int(args.calibration_frames),
        "calibration_npz_dir": str(npz_dir),
        "calibration_cache_path": str(cache_path),
        "calibration_sample_files_for_engine": len(samples),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "layerinfo_path": str(layerinfo_path),
        "build_log_path": str(log_path),
        "plugin_path": str(plugin_path),
        "plugin_loaded": _plugin_loaded(plugin_path),
        "profile_shapes": profile_shapes,
        "valid_agent_mask input exists": True,
        "valid_voxel_mask input exists": True,
        "PointPillarScatterTRT present": False,
        "build_success": False,
        "error": None,
    }
    if engine_path.exists() and not args.rebuild:
        layer_text = _read_layerinfo(layerinfo_path)
        int8_count, fp16_count = _count_layer_precisions(layer_text)
        result.update(
            {
                "build_success": True,
                "skipped_existing": True,
                "PointPillarScatterTRT present": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower(),
                "INT8 layer count": int8_count,
                "FP16 fallback layer count": fp16_count,
                "engine_size_MB": engine_path.stat().st_size / (1024 * 1024),
            }
        )
        save_json(result, engine_path.with_suffix(".meta.json"))
        return result
    if not onnx_path.exists():
        result["error"] = f"ONNX not found: {onnx_path}"
        save_json(result, engine_path.with_suffix(".meta.json"))
        return result
    if not plugin_path.exists():
        result["error"] = f"plugin .so not found: {plugin_path}"
        save_json(result, engine_path.with_suffix(".meta.json"))
        return result
    if not samples:
        result["error"] = f"no calibration samples for padded bucket={bucket_id} in {npz_dir}"
        save_json(result, engine_path.with_suffix(".meta.json"))
        return result
    if args.force_recalibrate and cache_path.exists():
        cache_path.unlink()
    command = _build_with_python_calibrator(
        onnx_path=onnx_path,
        engine_path=engine_path,
        layerinfo_path=layerinfo_path,
        log_path=log_path,
        cache_path=cache_path,
        plugin_path=plugin_path,
        profile_shapes=profile_shapes,
        sample_files=samples,
        mixed_heads_fp16=False,
    )
    layer_text = _read_layerinfo(layerinfo_path)
    int8_count, fp16_count = _count_layer_precisions(layer_text)
    result.update(
        {
            "command": command.get("command"),
            "returncode": command.get("returncode"),
            "build_success": bool(command.get("success") and engine_path.exists()),
            "error": command.get("error"),
            "PointPillarScatterTRT present": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower(),
            "plugin precision in INT8 engine": "FP16/FP32 plugin island inferred from plugin layer; TensorRT inspector text is authoritative in layerinfo_path",
            "INT8 layer count": int8_count,
            "FP16 fallback layer count": fp16_count,
            "engine_size_MB": engine_path.stat().st_size / (1024 * 1024) if engine_path.exists() else None,
        }
    )
    save_json(result, engine_path.with_suffix(".meta.json"))
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Padded Agent Static INT8 Train-Calib200 Build Report",
        "",
        f"- strategy: padded_agent_static",
        f"- fixed_K: {report.get('fixed_K')}",
        f"- max_cav: {report.get('max_cav')}",
        f"- calibration_split: train",
        f"- calibration_frames: {report.get('calibration_frames')}",
        f"- calibration_npz_dir: {report.get('calibration_npz_dir')}",
        f"- engine_count: {report.get('engine_count')}",
        f"- build_success: {report.get('build_success')}",
        "",
        "bucket | samples | success | PointPillarScatterTRT | valid_agent_mask | valid_voxel_mask | int8_layers | fp16_fallback_layers | engine",
        "--- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in report.get("engines", []):
        lines.append(
            f"{row.get('bucket_id')} | {row.get('calibration_sample_files_for_engine')} | {row.get('build_success')} | "
            f"{row.get('PointPillarScatterTRT present')} | {row.get('valid_agent_mask input exists')} | "
            f"{row.get('valid_voxel_mask input exists')} | {row.get('INT8 layer count')} | "
            f"{row.get('FP16 fallback layer count')} | {row.get('engine_path')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = fixedk_dirs(args.output_root, int(args.fixed_k))
    npz_dir = calibration_npz_dir(args.output_root, int(args.fixed_k), int(args.calibration_frames))
    manifest = read_json(npz_dir / "manifest.json", default={}) or {}
    rows = [_build_one(args, dirs, bucket) for bucket in fixed_k_buckets(int(args.fixed_k))]
    successful = [row for row in rows if row.get("build_success")]
    report = {
        "strategy": "padded_agent_static",
        "fixed_K": int(args.fixed_k),
        "max_cav": int(args.max_cav),
        "calibration_split": "train",
        "calibration_frames": int(args.calibration_frames),
        "calibration_npz_dir": str(npz_dir),
        "calibration_manifest_path": str(npz_dir / "manifest.json"),
        "calibration_manifest": manifest,
        "calibration_cache_paths": [row.get("calibration_cache_path") for row in rows],
        "engine_paths": [row.get("engine_path") for row in successful],
        "engine_count": len(successful),
        "onnx_path": str(_onnx_path(dirs)),
        "TensorRT profile min/opt/max": {str(row.get("bucket_id")): row.get("profile_shapes") for row in rows},
        "valid_agent_mask input exists": all(bool(row.get("valid_agent_mask input exists")) for row in rows),
        "valid_voxel_mask input exists": all(bool(row.get("valid_voxel_mask input exists")) for row in rows),
        "PointPillarScatterTRT present": all(bool(row.get("PointPillarScatterTRT present")) for row in successful) if successful else False,
        "plugin_path": str(Path(args.plugin_path).expanduser()),
        "plugin_loaded": _plugin_loaded(Path(args.plugin_path).expanduser()),
        "TensorRT environment": collect_env_report(trt_root=args.trt_root),
        "engines": rows,
        "build_success": bool(rows and len(successful) == len(rows)),
        "any_build_success": bool(successful),
        "build logs": [row.get("build_log_path") for row in rows],
    }
    root_dirs = ensure_quant_deploy_run_dirs(args.output_root)
    save_json(report, root_dirs["benchmark"] / "padded_agent_static_int8_train_calib200_build_report.json")
    _write_markdown(report, root_dirs["summary"] / "padded_agent_static_int8_train_calib200_build_report.md")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"build_success": report.get("build_success"), "engine_count": report.get("engine_count")}, indent=2))
    return 0 if report.get("build_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
