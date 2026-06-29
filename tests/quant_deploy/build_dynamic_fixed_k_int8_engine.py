from __future__ import annotations

import argparse
import ctypes
import traceback
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_mode_fixed_k_plugin_ablation import FIXED_K_BUCKETS, _dynamic_engine_path, _dynamic_layerinfo_path, _profile_shapes_dynamic_fixed_k
from quant_deploy_utils import (
    DEFAULT_TRT_ROOT,
    build_trtexec_command,
    collect_env_report,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    parse_trtexec_failure,
    read_json,
    run_command,
    save_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic fixed-K PointPillarScatterTRT INT8 engines.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calib_npz_dir", required=True)
    parser.add_argument("--calib_cache", required=True)
    parser.add_argument("--calibration_frames", type=int, required=True)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--builder", choices=["python", "trtexec"], default="python")
    parser.add_argument("--mixed_heads_fp16", action="store_true")
    parser.add_argument("--force_recalibrate", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args(argv)


def _dynamic_onnx_path(dirs: dict[str, Path], fixed_n: int) -> Path:
    return dirs["onnx_fp32"] / f"lidar_pyramid_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin.onnx"


def _compat_engine_path(dirs: dict[str, Path], calib_frames: int) -> Path:
    return dirs["engine_int8"] / f"lidar_pyramid_dynamic_fixed_k_scatter_plugin_int8_calib{calib_frames}.engine"


def _precision_tag(calibration_frames: int, mixed_heads_fp16: bool = False) -> str:
    suffix = "_mixed_heads_fp16" if mixed_heads_fp16 else ""
    return f"int8_calib{int(calibration_frames)}{suffix}"


def _cache_path(args: argparse.Namespace, fixed_n: int, bucket_id: int) -> Path:
    base = Path(args.calib_cache)
    return base.with_name(f"{base.stem}_N{fixed_n}_bucket{bucket_id}{base.suffix}")


def _input_files(calib_npz_dir: Path, fixed_n: int, bucket_id: int) -> list[Path]:
    pattern = f"*_N{fixed_n}_bucket{bucket_id}.npz"
    return sorted(calib_npz_dir.glob(pattern))


def _trt_dtype_to_numpy(dtype: Any, trt_module: Any) -> Any:
    mapping = {
        getattr(trt_module, "float32"): np.float32,
        getattr(trt_module, "float16"): np.float16,
        getattr(trt_module, "int32"): np.int32,
        getattr(trt_module, "bool"): np.bool_,
    }
    if hasattr(trt_module, "int8"):
        mapping[getattr(trt_module, "int8")] = np.int8
    return mapping.get(dtype, np.float32)


def _shape_list(profile_shapes: dict[str, Any], name: str, key: str) -> list[int]:
    return [int(v) for v in profile_shapes[name][key]]


class _NpzEntropyCalibrator:
    def __init__(
        self,
        trt_module: Any,
        input_names: list[str],
        input_dtypes: dict[str, Any],
        sample_files: list[Path],
        cache_path: Path,
    ) -> None:
        import torch

        class _Impl(trt_module.IInt8EntropyCalibrator2):
            def __init__(self, outer: "_NpzEntropyCalibrator") -> None:
                trt_module.IInt8EntropyCalibrator2.__init__(self)
                self.outer = outer

            def get_batch_size(self) -> int:
                return 1

            def get_batch(self, names: list[str]) -> list[int] | None:
                return self.outer.get_batch(names)

            def read_calibration_cache(self) -> bytes | None:
                return self.outer.read_calibration_cache()

            def write_calibration_cache(self, cache: bytes) -> None:
                self.outer.write_calibration_cache(cache)

        self.trt = trt_module
        self.torch = torch
        self.input_names = list(input_names)
        self.input_dtypes = dict(input_dtypes)
        self.sample_files = list(sample_files)
        self.cache_path = Path(cache_path)
        self.index = 0
        self.device_tensors: list[Any] = []
        self.impl = _Impl(self)

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index >= len(self.sample_files):
            return None
        sample_path = self.sample_files[self.index]
        self.index += 1
        sample = np.load(sample_path)
        ptrs: list[int] = []
        self.device_tensors = []
        for name in names:
            if name not in sample:
                raise KeyError(f"calibration sample {sample_path} is missing input '{name}'")
            dtype = _trt_dtype_to_numpy(self.input_dtypes.get(name), self.trt)
            array = np.ascontiguousarray(sample[name].astype(dtype, copy=False))
            tensor = self.torch.as_tensor(array, device="cuda")
            self.device_tensors.append(tensor)
            ptrs.append(int(tensor.data_ptr()))
        return ptrs

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.exists() and self.cache_path.stat().st_size > 0:
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)


def _build_with_python_calibrator(
    *,
    onnx_path: Path,
    engine_path: Path,
    layerinfo_path: Path,
    log_path: Path,
    cache_path: Path,
    plugin_path: Path,
    profile_shapes: dict[str, Any],
    sample_files: list[Path],
    mixed_heads_fp16: bool,
) -> dict[str, Any]:
    started = time.time()
    log_lines: list[str] = []
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, "")
        ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
        config = builder.create_builder_config()
        if hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        if hasattr(trt, "ProfilingVerbosity"):
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        if mixed_heads_fp16:
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            for idx in range(network.num_layers):
                layer = network.get_layer(idx)
                lname = layer.name.lower()
                if any(token in lname for token in ("cls", "reg", "dir", "head")):
                    layer.precision = trt.float16
                    for out_idx in range(layer.num_outputs):
                        layer.set_output_type(out_idx, trt.float16)

        profile = builder.create_optimization_profile()
        input_names: list[str] = []
        input_dtypes: dict[str, Any] = {}
        for idx in range(network.num_inputs):
            tensor = network.get_input(idx)
            name = tensor.name
            if name not in profile_shapes:
                raise KeyError(f"Missing TensorRT profile shape for input '{name}'")
            input_names.append(name)
            input_dtypes[name] = tensor.dtype
            profile.set_shape(
                name,
                tuple(_shape_list(profile_shapes, name, "min")),
                tuple(_shape_list(profile_shapes, name, "opt")),
                tuple(_shape_list(profile_shapes, name, "max")),
            )
        config.add_optimization_profile(profile)
        try:
            config.set_calibration_profile(profile)
        except Exception as exc:
            log_lines.append(f"set_calibration_profile warning: {exc!r}")

        calibrator = _NpzEntropyCalibrator(trt, input_names, input_dtypes, sample_files, cache_path)
        config.int8_calibrator = calibrator.impl

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(serialized))

        layerinfo_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(bytes(serialized))
            inspector = engine.create_engine_inspector()
            layerinfo = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            layerinfo_path.write_text(layerinfo, encoding="utf-8")
        except Exception as exc:
            layerinfo_path.write_text("[]\n", encoding="utf-8")
            log_lines.append(f"engine inspector failed: {exc!r}")

        elapsed = time.time() - started
        log_lines.extend(
            [
                f"build_method=tensorrt_python_calibrator",
                f"onnx_path={onnx_path}",
                f"engine_path={engine_path}",
                f"layerinfo_path={layerinfo_path}",
                f"calibration_cache_path={cache_path}",
                f"calibration_samples={len(sample_files)}",
                f"input_names={input_names}",
                f"elapsed_seconds={elapsed:.3f}",
            ]
        )
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {"success": True, "returncode": 0, "elapsed_seconds": elapsed, "error": None, "log_path": str(log_path), "command": ["tensorrt_python_builder", str(onnx_path)]}
    except Exception as exc:
        elapsed = time.time() - started
        log_lines.extend([f"build_method=tensorrt_python_calibrator", f"error={exc}", traceback.format_exc(), f"elapsed_seconds={elapsed:.3f}"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {"success": False, "returncode": 1, "elapsed_seconds": elapsed, "error": str(exc), "log_path": str(log_path), "command": ["tensorrt_python_builder", str(onnx_path)]}


def _read_layerinfo(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _plugin_loaded(plugin_path: Path) -> bool:
    try:
        ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        return True
    except Exception:
        return False


def _build_one(args: argparse.Namespace, dirs: dict[str, Path], fixed_n: int, bucket: dict[str, Any]) -> dict[str, Any]:
    bucket_id = int(bucket["bucket_id"])
    precision_tag = _precision_tag(int(args.calibration_frames), bool(args.mixed_heads_fp16))
    onnx_path = _dynamic_onnx_path(dirs, fixed_n)
    engine_path = _dynamic_engine_path(dirs, precision_tag, fixed_n, bucket_id)
    layerinfo_path = _dynamic_layerinfo_path(dirs, precision_tag, fixed_n, bucket_id)
    log_path = dirs["logs_build"] / f"build_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin_bucket{bucket_id}_{precision_tag}.log"
    cache_path = _cache_path(args, fixed_n, bucket_id)
    plugin_path = Path(args.plugin_path)
    env = collect_env_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    trtexec = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    samples = _input_files(Path(args.calib_npz_dir), fixed_n, bucket_id)
    result: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "precision": "int8",
        "engine_precision_tag": precision_tag,
        "calibration_frames": int(args.calibration_frames),
        "calibration_npz_dir": str(args.calib_npz_dir),
        "calibration_cache_path": str(cache_path),
        "calibration_sample_files_for_engine": len(samples),
        "plugin_path": str(plugin_path),
        "plugin_loaded": _plugin_loaded(plugin_path),
        "PointPillarScatterTRT layer exists": False,
        "valid_voxel_mask input exists": False,
        "TensorRT version": env.get("tensorrt_version"),
        "builder_flags": ["INT8", "FP16_FALLBACK_ALLOWED_BY_TRT_BUILDER"]
        + (["PREFER_PRECISION_CONSTRAINTS"] if args.mixed_heads_fp16 else []),
        "build_method": "tensorrt_python_calibrator" if args.builder == "python" else "trtexec",
        "enabled INT8": True,
        "enabled FP16 fallback": True,
        "strict_types": False,
        "prefer_precision_constraints": bool(args.mixed_heads_fp16),
        "obey_precision_constraints": False,
        "mixed_heads_fp16": bool(args.mixed_heads_fp16),
        "forced_fp16_layer_patterns": ["cls", "reg", "dir", "head"] if args.mixed_heads_fp16 else [],
        "force_recalibrate": bool(args.force_recalibrate),
        "build_success": False,
        "build_log_path": str(log_path),
        "trtexec": trtexec,
        "error": None,
    }
    if engine_path.exists() and not args.rebuild:
        layer_text = _read_layerinfo(layerinfo_path)
        result.update(
            {
                "build_success": True,
                "skipped_existing": True,
                "PointPillarScatterTRT layer exists": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower(),
                "valid_voxel_mask input exists": "valid_voxel_mask" in layer_text,
                "engine_size_MB": engine_path.stat().st_size / (1024 * 1024),
            }
        )
        save_json(result, engine_path.with_suffix(".meta.json"))
        return result
    if not trtexec.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec.get('suggestion')}"
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
        result["error"] = f"no calibration samples for N={fixed_n}, bucket={bucket_id} in {args.calib_npz_dir}"
        save_json(result, engine_path.with_suffix(".meta.json"))
        return result

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    layerinfo_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force_recalibrate and cache_path.exists():
        cache_path.unlink()
    profile_shapes = _profile_shapes_dynamic_fixed_k(bucket, fixed_n)
    if args.builder == "python":
        command = _build_with_python_calibrator(
            onnx_path=onnx_path,
            engine_path=engine_path,
            layerinfo_path=layerinfo_path,
            log_path=log_path,
            cache_path=cache_path,
            plugin_path=plugin_path,
            profile_shapes=profile_shapes,
            sample_files=samples,
            mixed_heads_fp16=bool(args.mixed_heads_fp16),
        )
        cmd = command.get("command", [])
    else:
        cmd = build_trtexec_command(
            precision="int8",
            onnx_path=onnx_path,
            engine_path=engine_path,
            layerinfo_path=layerinfo_path,
            profile_shapes=profile_shapes,
            trtexec_path=trtexec["trtexec_path"],
            no_tf32=True,
            calib_cache=cache_path,
            int8_mode="native_trt",
            allow_fp16_fallback=True,
            skip_inference=True,
            static_plugins=[str(plugin_path)],
        )
        if args.mixed_heads_fp16:
            cmd.extend(["--precisionConstraints=prefer", "--layerPrecisions=*cls*:fp16,*reg*:fp16,*dir*:fp16,*head*:fp16"])
        command = run_command(cmd, log_path, timeout=int(args.timeout))
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    layer_text = _read_layerinfo(layerinfo_path)
    failure = parse_trtexec_failure(log_text)
    result.update(
        {
            "command": cmd,
            "returncode": command.get("returncode"),
            "build_success": bool(command.get("success") and engine_path.exists()),
            "error": command.get("error"),
            "unsupported_ops": failure.get("unsupported_ops", []),
            "failed_nodes": failure.get("failed_nodes", []),
            "PointPillarScatterTRT layer exists": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower() or "PointPillarScatterTRT" in log_text,
            "valid_voxel_mask input exists": "valid_voxel_mask" in layer_text or "valid_voxel_mask" in log_text,
            "engine_size_MB": engine_path.stat().st_size / (1024 * 1024) if engine_path.exists() else None,
        }
    )
    if not result["build_success"] and not result["error"]:
        result["error"] = "trtexec did not produce an engine"
    save_json(result, engine_path.with_suffix(".meta.json"))
    return result


def _write_report(dirs: dict[str, Path], report: dict[str, Any], calib_frames: int) -> None:
    if report.get("mixed_heads_fp16"):
        json_path = dirs["benchmark"] / "dynamic_fixed_k_int8_mixed_precision_build_report.json"
        md_path = dirs["summary"] / "dynamic_fixed_k_int8_mixed_precision_report.md"
    else:
        json_path = dirs["benchmark"] / f"dynamic_fixed_k_int8_engine_build_calib{calib_frames}.json"
        md_path = dirs["summary"] / "dynamic_fixed_k_int8_engine_build_report.md"
    save_json(report, json_path)
    lines = [
        "# Dynamic Fixed-K INT8 Engine Build Report",
        "",
        "N | bucket | success | PointPillarScatterTRT | valid_voxel_mask | engine",
        "--- | --- | --- | --- | --- | ---",
    ]
    for row in report.get("engines", []):
        lines.append(
            f"{row.get('fixed_N')} | {row.get('bucket_id')} | {row.get('build_success')} | "
            f"{row.get('PointPillarScatterTRT layer exists')} | {row.get('valid_voxel_mask input exists')} | {row.get('engine_path')}"
        )
    lines.extend(["", f"- any_build_success: {report.get('any_build_success')}", f"- all_build_success: {report.get('all_build_success')}"])
    if report.get("mixed_heads_fp16"):
        lines.extend(
            [
                "",
                "- strategy: INT8 backbone with detection heads requested FP16",
                "- forced_fp16_layer_patterns: cls, reg, dir, head",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    rows = []
    for fixed_n in (1, 2):
        for bucket in FIXED_K_BUCKETS:
            row = _build_one(args, dirs, fixed_n, bucket)
            row["fixed_N"] = fixed_n
            row["bucket_id"] = int(bucket["bucket_id"])
            rows.append(row)
    successful = [row for row in rows if row.get("build_success")]
    if successful:
        compat = _compat_engine_path(dirs, int(args.calibration_frames))
        compat.parent.mkdir(parents=True, exist_ok=True)
        try:
            if compat.exists() or compat.is_symlink():
                compat.unlink()
            compat.symlink_to(Path(successful[0]["engine_path"]).resolve())
        except OSError:
            pass
    report = {
        "agent_export_mode": "dynamic_agent_dim",
        "fixed_k_bucket_router": True,
        "pointpillar_scatter_plugin_enabled": True,
        "valid_voxel_mask_enabled": True,
        "precision": "int8",
        "engine_precision_tag": _precision_tag(int(args.calibration_frames), bool(args.mixed_heads_fp16)),
        "mixed_heads_fp16": bool(args.mixed_heads_fp16),
        "forced_fp16_layer_patterns": ["cls", "reg", "dir", "head"] if args.mixed_heads_fp16 else [],
        "calibration_frames": int(args.calibration_frames),
        "calibration_npz_dir": str(args.calib_npz_dir),
        "calibration_cache_path": str(args.calib_cache),
        "plugin_path": str(args.plugin_path),
        "engines": rows,
        "any_build_success": bool(successful),
        "all_build_success": bool(rows and len(successful) == len(rows)),
    }
    _write_report(dirs, report, int(args.calibration_frames))
    return report


def main(argv: list[str] | None = None) -> int:
    report = build_all(parse_args(argv))
    print(report)
    return 0 if report.get("any_build_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
