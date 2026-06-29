from __future__ import annotations

import argparse
import ctypes
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamic_single_engine_maxk_common import (
    FIXED_K,
    calibration_cache_path,
    calibration_npz_dir,
    engine_path,
    layerinfo_path,
    load_observed_shapes_from_npz,
    onnx_path,
    pad_pairwise_to_agent_count,
    profile_from_observed_shapes,
)
from quant_deploy_utils import (
    DEFAULT_TRT_ROOT,
    build_trtexec_command,
    collect_env_report,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    parse_trtexec_failure,
    run_command,
    save_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic_agent_single_engine_maxK TensorRT engines.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"], choices=["fp32", "fp16", "int8"])
    parser.add_argument("--calibration_frames", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--profile_calibration_frames", type=int, default=200)
    parser.add_argument("--fixed_k", type=int, default=FIXED_K)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--force_recalibrate", action="store_true")
    return parser.parse_args(argv)


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


def _plugin_loaded(plugin_path: Path) -> bool:
    try:
        ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        return True
    except Exception:
        return False


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


class _SingleEngineNpzEntropyCalibrator:
    def __init__(
        self,
        trt_module: Any,
        input_names: list[str],
        input_dtypes: dict[str, Any],
        sample_files: list[Path],
        cache_path: Path,
        profile_shapes: dict[str, Any],
    ) -> None:
        import torch

        class _Impl(trt_module.IInt8EntropyCalibrator2):
            def __init__(self, outer: "_SingleEngineNpzEntropyCalibrator") -> None:
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
        self.profile_shapes = profile_shapes
        self.opt_n = int(profile_shapes["pairwise_t_matrix"]["opt"][1])
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
            array = sample[name]
            if name == "pairwise_t_matrix":
                array = pad_pairwise_to_agent_count(array, self.opt_n)
            array = np.ascontiguousarray(array.astype(dtype, copy=False))
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


def _profile_for_calibration(dirs: dict[str, Path], calibration_frames: int, fixed_k: int = FIXED_K) -> dict[str, Any]:
    npz_dir = calibration_npz_dir(dirs, int(calibration_frames), fixed_k=fixed_k)
    observed_shapes = load_observed_shapes_from_npz(npz_dir)
    profile = profile_from_observed_shapes(observed_shapes, fixed_k=int(fixed_k))
    report = {
        "strategy": "dynamic_agent_single_engine_maxK",
        "calibration_split": "train",
        "calibration_frames": int(calibration_frames),
        "calibration_npz_dir": str(npz_dir),
        "fixed_K": int(fixed_k),
        "observed_shapes_count": len(observed_shapes),
        "observed_shapes": observed_shapes[:20],
        "profile": profile,
        "no_bucket_router": True,
        "no_N_engine_router": True,
        "true_dynamic_N": True,
    }
    save_json(report, dirs["debug"] / f"dynamic_single_engine_maxK_profile_from_train_calib{int(calibration_frames)}.json")
    return profile


def _fallback_profile(fixed_k: int = FIXED_K) -> dict[str, Any]:
    return profile_from_observed_shapes([{"pairwise_t_matrix": [1, 1, 1, 4, 4]}, {"pairwise_t_matrix": [1, 2, 2, 4, 4]}], fixed_k=int(fixed_k))


def _build_trtexec_engine(
    *,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    precision: str,
    profile_shapes: dict[str, Any],
    plugin_path: Path,
) -> dict[str, Any]:
    source_onnx = onnx_path(dirs, fixed_k=int(args.fixed_k))
    out_engine = engine_path(dirs, precision, fixed_k=int(args.fixed_k))
    out_layerinfo = layerinfo_path(dirs, precision, fixed_k=int(args.fixed_k))
    log_path = dirs["logs_build"] / f"build_dynamic_single_engine_maxK_{precision}.log"
    trtexec = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result: dict[str, Any] = {
        "strategy": "dynamic_agent_single_engine_maxK",
        "precision": precision,
        "onnx_path": str(source_onnx),
        "engine_path": str(out_engine),
        "layerinfo_path": str(out_layerinfo),
        "plugin_path": str(plugin_path),
        "plugin_loaded": _plugin_loaded(plugin_path),
        "fixed_K": int(args.fixed_k),
        "single_engine": True,
        "no_bucket_router": True,
        "no_N_engine_router": True,
        "true_dynamic_N": True,
        "profile": profile_shapes,
        "build_success": False,
        "build_log_path": str(log_path),
        "error": None,
        "trtexec": trtexec,
    }
    if out_engine.exists() and not args.rebuild:
        layer_text = _read_text(out_layerinfo)
        result.update(
            {
                "build_success": True,
                "skipped_existing": True,
                "PointPillarScatterTRT layer exists": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower(),
                "valid_voxel_mask input exists": "valid_voxel_mask" in layer_text,
                "engine_size_MB": out_engine.stat().st_size / (1024 * 1024),
            }
        )
        return result
    if not source_onnx.exists():
        result["error"] = f"ONNX not found: {source_onnx}"
        return result
    if not plugin_path.exists():
        result["error"] = f"plugin .so not found: {plugin_path}"
        return result
    if not trtexec.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec.get('suggestion')}"
        return result
    out_engine.parent.mkdir(parents=True, exist_ok=True)
    out_layerinfo.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_trtexec_command(
        precision=precision,
        onnx_path=source_onnx,
        engine_path=out_engine,
        layerinfo_path=out_layerinfo,
        profile_shapes=profile_shapes,
        trtexec_path=trtexec["trtexec_path"],
        no_tf32=True,
        skip_inference=True,
        static_plugins=[str(plugin_path)],
    )
    started = time.time()
    command = run_command(cmd, log_path, timeout=int(args.timeout))
    elapsed = time.time() - started
    log_text = _read_text(log_path)
    layer_text = _read_text(out_layerinfo)
    failure = parse_trtexec_failure(log_text)
    result.update(
        {
            "command": cmd,
            "returncode": command.get("returncode"),
            "build_time_seconds": elapsed,
            "build_success": bool(command.get("success") and out_engine.exists()),
            "error": command.get("error"),
            "unsupported_ops": failure.get("unsupported_ops", []),
            "failed_nodes": failure.get("failed_nodes", []),
            "PointPillarScatterTRT layer exists": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower() or "PointPillarScatterTRT" in log_text,
            "valid_voxel_mask input exists": "valid_voxel_mask" in layer_text or "valid_voxel_mask" in log_text,
            "engine_size_MB": out_engine.stat().st_size / (1024 * 1024) if out_engine.exists() else None,
        }
    )
    if not result["build_success"] and not result["error"]:
        result["error"] = "trtexec did not produce an engine"
    return result


def _build_int8_python(
    *,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    calibration_frames: int,
    profile_shapes: dict[str, Any],
    plugin_path: Path,
) -> dict[str, Any]:
    source_onnx = onnx_path(dirs, fixed_k=int(args.fixed_k))
    out_engine = engine_path(dirs, "int8", int(calibration_frames), fixed_k=int(args.fixed_k))
    out_layerinfo = layerinfo_path(dirs, "int8", int(calibration_frames), fixed_k=int(args.fixed_k))
    cache_path = calibration_cache_path(dirs, int(calibration_frames), fixed_k=int(args.fixed_k))
    npz_dir = calibration_npz_dir(dirs, int(calibration_frames), fixed_k=int(args.fixed_k))
    sample_files = sorted(npz_dir.glob("*.npz"))
    log_path = dirs["logs_build"] / f"build_dynamic_single_engine_maxK_int8_train_calib{int(calibration_frames)}.log"
    env = collect_env_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result: dict[str, Any] = {
        "strategy": "dynamic_agent_single_engine_maxK",
        "precision": "int8",
        "calibration_split": "train",
        "evaluation_split": "val",
        "calibration_frames": int(calibration_frames),
        "calibration_npz_dir": str(npz_dir),
        "calibration_cache_path": str(cache_path),
        "calibration_sample_files_for_engine": len(sample_files),
        "onnx_path": str(source_onnx),
        "engine_path": str(out_engine),
        "layerinfo_path": str(out_layerinfo),
        "plugin_path": str(plugin_path),
        "plugin_loaded": _plugin_loaded(plugin_path),
        "fixed_K": int(args.fixed_k),
        "single_engine": True,
        "no_bucket_router": True,
        "no_N_engine_router": True,
        "true_dynamic_N": True,
        "profile": profile_shapes,
        "TensorRT version": env.get("tensorrt_version"),
        "builder_flags": ["INT8", "FP16_FALLBACK_ALLOWED_BY_TRT_BUILDER"],
        "enabled INT8": True,
        "enabled FP16 fallback": True,
        "strict_types": False,
        "prefer_precision_constraints": False,
        "obey_precision_constraints": False,
        "build_success": False,
        "build_log_path": str(log_path),
        "error": None,
    }
    if out_engine.exists() and not args.rebuild:
        layer_text = _read_text(out_layerinfo)
        result.update(
            {
                "build_success": True,
                "skipped_existing": True,
                "PointPillarScatterTRT layer exists": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower(),
                "valid_voxel_mask input exists": "valid_voxel_mask" in layer_text,
                "engine_size_MB": out_engine.stat().st_size / (1024 * 1024),
            }
        )
        return result
    if args.force_recalibrate and cache_path.exists():
        cache_path.unlink()
    if not source_onnx.exists():
        result["error"] = f"ONNX not found: {source_onnx}"
        return result
    if not plugin_path.exists():
        result["error"] = f"plugin .so not found: {plugin_path}"
        return result
    if not sample_files:
        result["error"] = f"no calibration NPZ files found in {npz_dir}"
        return result
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
        if not parser.parse(source_onnx.read_bytes()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
        config = builder.create_builder_config()
        if hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        if hasattr(trt, "ProfilingVerbosity"):
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
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
        calibrator = _SingleEngineNpzEntropyCalibrator(trt, input_names, input_dtypes, sample_files, cache_path, profile_shapes)
        config.int8_calibrator = calibrator.impl
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        out_engine.parent.mkdir(parents=True, exist_ok=True)
        out_engine.write_bytes(bytes(serialized))
        out_layerinfo.parent.mkdir(parents=True, exist_ok=True)
        try:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(bytes(serialized))
            inspector = engine.create_engine_inspector()
            out_layerinfo.write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding="utf-8")
        except Exception as exc:
            out_layerinfo.write_text("[]\n", encoding="utf-8")
            log_lines.append(f"engine inspector failed: {exc!r}")
        elapsed = time.time() - started
        layer_text = _read_text(out_layerinfo)
        log_lines.extend(
            [
                "build_method=tensorrt_python_calibrator",
                f"onnx_path={source_onnx}",
                f"engine_path={out_engine}",
                f"calibration_cache_path={cache_path}",
                f"calibration_samples={len(sample_files)}",
                f"input_names={input_names}",
                f"profile={profile_shapes}",
                f"elapsed_seconds={elapsed:.3f}",
            ]
        )
        result.update(
            {
                "command": ["tensorrt_python_builder", str(source_onnx)],
                "returncode": 0,
                "build_time_seconds": elapsed,
                "build_success": True,
                "PointPillarScatterTRT layer exists": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower(),
                "valid_voxel_mask input exists": "valid_voxel_mask" in layer_text,
                "engine_size_MB": out_engine.stat().st_size / (1024 * 1024) if out_engine.exists() else None,
            }
        )
    except Exception as exc:
        elapsed = time.time() - started
        result.update({"build_time_seconds": elapsed, "error": str(exc), "traceback": traceback.format_exc(), "returncode": 1})
        log_lines.extend(["build_method=tensorrt_python_calibrator", f"error={exc}", traceback.format_exc(), f"elapsed_seconds={elapsed:.3f}"])
    finally:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return result


def _write_reports(dirs: dict[str, Path], report: dict[str, Any]) -> None:
    save_json(report, dirs["benchmark"] / "dynamic_single_engine_maxK_build_report.json")
    lines = [
        "# Dynamic Single Engine maxK Build Report",
        "",
        f"- onnx_path: {report.get('onnx_path')}",
        f"- fixed_K: {report.get('fixed_K')}",
        f"- no_bucket_router: {report.get('no_bucket_router')}",
        f"- no_N_engine_router: {report.get('no_N_engine_router')}",
        f"- true_dynamic_N: {report.get('true_dynamic_N')}",
        f"- engine_count_fp32: {report.get('engine_count_fp32')}",
        f"- engine_count_fp16: {report.get('engine_count_fp16')}",
        f"- engine_count_int8_train_calib50: {report.get('engine_count_int8_train_calib50')}",
        f"- engine_count_int8_train_calib200: {report.get('engine_count_int8_train_calib200')}",
        "",
        "precision | calibration | success | engine",
        "--- | --- | --- | ---",
    ]
    for row in report.get("builds", []):
        lines.append(f"{row.get('precision')} | {row.get('calibration_frames')} | {row.get('build_success')} | {row.get('engine_path')}")
    (dirs["summary"] / "dynamic_single_engine_maxK_build_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    profile_lines = ["# Dynamic Single Engine maxK Profile Report", ""]
    for key, profile in (report.get("profiles") or {}).items():
        profile_lines.append(f"## {key}")
        profile_lines.append("")
        for name, shapes in profile.items():
            profile_lines.append(f"- {name}: min={shapes['min']} opt={shapes['opt']} max={shapes['max']}")
        profile_lines.append("")
    (dirs["summary"] / "dynamic_single_engine_maxK_profile_report.md").write_text("\n".join(profile_lines), encoding="utf-8")


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    plugin_path = Path(args.plugin_path).expanduser()
    profiles: dict[str, Any] = {}
    for frames in sorted(set(int(v) for v in args.calibration_frames + [int(args.profile_calibration_frames)])):
        try:
            profiles[f"train_calib{frames}"] = _profile_for_calibration(dirs, frames, fixed_k=int(args.fixed_k))
        except Exception:
            if frames == int(args.profile_calibration_frames):
                profiles[f"train_calib{frames}"] = _fallback_profile(int(args.fixed_k))
    default_profile = profiles.get(f"train_calib{int(args.profile_calibration_frames)}") or next(iter(profiles.values()), _fallback_profile(int(args.fixed_k)))
    builds = []
    if "fp32" in args.precisions:
        builds.append(_build_trtexec_engine(args=args, dirs=dirs, precision="fp32", profile_shapes=default_profile, plugin_path=plugin_path))
    if "fp16" in args.precisions:
        builds.append(_build_trtexec_engine(args=args, dirs=dirs, precision="fp16", profile_shapes=default_profile, plugin_path=plugin_path))
    if "int8" in args.precisions:
        for frames in args.calibration_frames:
            profile = profiles.get(f"train_calib{int(frames)}") or default_profile
            builds.append(_build_int8_python(args=args, dirs=dirs, calibration_frames=int(frames), profile_shapes=profile, plugin_path=plugin_path))
    env = collect_env_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    gpu_name = None
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu_name = None
    report = {
        "strategy": "dynamic_agent_single_engine_maxK",
        "onnx_path": str(onnx_path(dirs, fixed_k=int(args.fixed_k))),
        "plugin_path": str(plugin_path),
        "fixed_K": int(args.fixed_k),
        "single_engine": True,
        "no_bucket_router": True,
        "no_N_engine_router": True,
        "true_dynamic_N": True,
        "calibration_split": "train",
        "evaluation_split": "val",
        "TensorRT version": env.get("tensorrt_version"),
        "GPU name": gpu_name,
        "profiles": profiles,
        "builds": builds,
        "engine_paths": {Path(row["engine_path"]).parent.name: row["engine_path"] for row in builds},
        "engine_count_fp32": sum(1 for row in builds if row.get("precision") == "fp32" and row.get("build_success")),
        "engine_count_fp16": sum(1 for row in builds if row.get("precision") == "fp16" and row.get("build_success")),
        "engine_count_int8_train_calib50": sum(1 for row in builds if row.get("precision") == "int8" and int(row.get("calibration_frames") or 0) == 50 and row.get("build_success")),
        "engine_count_int8_train_calib200": sum(1 for row in builds if row.get("precision") == "int8" and int(row.get("calibration_frames") or 0) == 200 and row.get("build_success")),
        "engine_count_int8_train_calib500": sum(1 for row in builds if row.get("precision") == "int8" and int(row.get("calibration_frames") or 0) == 500 and row.get("build_success")),
        "PointPillarScatterTRT layer exists": any(row.get("PointPillarScatterTRT layer exists") for row in builds),
        "valid_voxel_mask input exists": any(row.get("valid_voxel_mask input exists") for row in builds),
        "plugin supports runtime N": True,
        "build_success": all(row.get("build_success") for row in builds) if builds else False,
    }
    _write_reports(dirs, report)
    return report


def main(argv: list[str] | None = None) -> int:
    report = build_all(parse_args(argv))
    print(report)
    return 0 if report.get("build_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
