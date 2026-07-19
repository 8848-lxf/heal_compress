"""Isolated TensorRT EntropyCalibration2 worker for production explicit Q/DQ."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def _validate_modelopt_toolchain() -> dict[str, Any]:
    prefix = Path(os.environ.get("CONDA_PREFIX", "")).expanduser().resolve()
    if os.environ.get("CONDA_DEFAULT_ENV") != "modelopt" or not prefix.is_dir():
        raise RuntimeError(
            f"modelopt_environment_not_active:{os.environ.get('CONDA_DEFAULT_ENV')}:{prefix}"
        )
    paths = {name: shutil.which(name) for name in ("python", "nvcc", "gcc", "g++")}
    if any(value is None for value in paths.values()):
        raise RuntimeError(f"modelopt_tool_missing:{paths}")
    realpaths = {name: str(Path(str(value)).resolve()) for name, value in paths.items()}
    prefix_text = str(prefix) + os.sep
    outside = {name: value for name, value in realpaths.items() if not value.startswith(prefix_text)}
    if outside:
        raise RuntimeError(f"modelopt_toolchain_not_isolated:{outside}")
    cuda_home = Path(os.environ.get("CUDA_HOME", "")).expanduser().resolve()
    if cuda_home != prefix:
        raise RuntimeError(f"modelopt_cuda_home_mismatch:{cuda_home}!={prefix}")
    return {
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": str(prefix),
        "tool_paths": paths,
        "tool_realpaths": realpaths,
        "nvcc_version": _tool_version(["nvcc", "--version"]),
        "gcc_version": _tool_version(["gcc", "--version"]).splitlines()[0],
        "gxx_version": _tool_version(["g++", "--version"]).splitlines()[0],
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CC": os.environ.get("CC", ""),
        "CXX": os.environ.get("CXX", ""),
        "CUDACXX": os.environ.get("CUDACXX", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CMAKE_PREFIX_PATH": os.environ.get("CMAKE_PREFIX_PATH", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _resolve_and_verify_samples(
    manifest_path: str | Path,
    *,
    num_batches: int,
    fixed_k: int,
    input_names: list[str] | None = None,
) -> tuple[list[Path], list[int], dict[str, Any]]:
    import numpy as np

    from .calibration_provider import fixed_k_calibration_npz_manifest_identity

    identity = fixed_k_calibration_npz_manifest_identity(
        manifest_path,
        num_batches=int(num_batches),
        fixed_k=int(fixed_k),
        input_names=input_names,
    )
    expected_inputs = tuple(str(value) for value in identity["input_names"])
    manifest = Path(identity["manifest_path"])
    samples: list[Path] = []
    agent_counts: list[int] = []
    verified: list[dict[str, Any]] = []
    for row in identity["files"]:
        raw_path = Path(str(row["path"])).expanduser()
        candidates = [raw_path, manifest.parent / str(row["name"]), manifest.parent / raw_path.name]
        source = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if source is None:
            raise RuntimeError(f"calibration_npz_file_missing:{row['index']}:{row['name']}")
        size = int(source.stat().st_size)
        digest = _sha256_file(source)
        if int(row["bytes"]) > 0 and size != int(row["bytes"]):
            raise RuntimeError(
                f"calibration_npz_file_size_mismatch:{row['name']}:{size}!={row['bytes']}"
            )
        if digest != str(row["sha256"]):
            raise RuntimeError(
                f"calibration_npz_file_hash_mismatch:{row['name']}:{digest}!={row['sha256']}"
            )
        with np.load(source) as values:
            missing = [name for name in expected_inputs if name not in values.files]
            if missing:
                raise RuntimeError(f"calibration_npz_inputs_missing:{row['name']}:{missing}")
            for name in ("voxel_features", "voxel_coords", "voxel_num_points", "valid_voxel_mask"):
                if int(values[name].shape[0]) != int(fixed_k):
                    raise RuntimeError(
                        f"calibration_npz_fixed_k_tensor_mismatch:{row['name']}:{name}:"
                        f"{values[name].shape[0]}!={int(fixed_k)}"
                    )
            pairwise_shape = tuple(int(value) for value in values["pairwise_t_matrix"].shape)
            if len(pairwise_shape) != 5 or pairwise_shape[1] != pairwise_shape[2]:
                raise RuntimeError(
                    f"calibration_npz_pairwise_shape_invalid:{row['name']}:{pairwise_shape}"
                )
            if "agent_mask" in expected_inputs:
                if pairwise_shape[1] != 2:
                    raise RuntimeError(
                        f"baseline_calibration_requires_static_two_agent_tensors:"
                        f"{row['name']}:{pairwise_shape}"
                    )
                agent_mask = np.asarray(values["agent_mask"])
                agent_mask_shape = tuple(int(value) for value in agent_mask.shape)
                if agent_mask_shape != (1, pairwise_shape[1]):
                    raise RuntimeError(
                        f"calibration_npz_agent_mask_shape_invalid:{row['name']}:"
                        f"{agent_mask_shape}:{pairwise_shape}"
                    )
                if (
                    not np.all(np.isfinite(agent_mask))
                    or not np.all((agent_mask == 0) | (agent_mask == 1))
                    or float(agent_mask[0, 0]) != 1.0
                    or float(agent_mask.sum()) < 1.0
                ):
                    raise RuntimeError(
                        f"calibration_npz_agent_mask_values_invalid:{row['name']}:"
                        f"{agent_mask.tolist()}"
                    )
            agent_counts.append(int(pairwise_shape[1]))
        samples.append(source)
        verified.append(
            {
                "index": int(row["index"]),
                "name": str(row["name"]),
                "path": str(source),
                "bytes": size,
                "sha256": digest,
            }
        )
    return samples, agent_counts, {**identity, "files_verified": True, "verified_files": verified}


def _profile(
    agent_counts: list[int],
    fixed_k: int,
    *,
    input_names: list[str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    if not agent_counts:
        raise RuntimeError("calibration_agent_counts_empty")
    counts = Counter(agent_counts)
    max_count = max(counts.values())
    opt_n = min(value for value, count in counts.items() if count == max_count)
    min_n = min(agent_counts)
    max_n = max(agent_counts)
    profile = {
        "voxel_features": {"min": [fixed_k, 32, 4], "opt": [fixed_k, 32, 4], "max": [fixed_k, 32, 4]},
        "voxel_coords": {"min": [fixed_k, 4], "opt": [fixed_k, 4], "max": [fixed_k, 4]},
        "voxel_num_points": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
        "pairwise_t_matrix": {
            "min": [1, min_n, min_n, 4, 4],
            "opt": [1, opt_n, opt_n, 4, 4],
            "max": [1, max_n, max_n, 4, 4],
        },
        "valid_voxel_mask": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
    }
    expected = list(input_names or profile)
    if "agent_mask" in expected:
        if set(agent_counts) != {2}:
            raise RuntimeError(
                f"baseline_calibration_profile_requires_static_two_agents:{agent_counts}"
            )
        profile["agent_mask"] = {
            "min": [1, min_n],
            "opt": [1, opt_n],
            "max": [1, max_n],
        }
    unknown = sorted(set(expected) - set(profile))
    if unknown:
        raise RuntimeError(f"calibration_profile_inputs_unsupported:{unknown}")
    return {name: profile[name] for name in expected}


def _pad_pairwise(array: Any, target_n: int) -> Any:
    import numpy as np

    source = np.asarray(array)
    n = int(source.shape[1])
    if n == int(target_n):
        return np.ascontiguousarray(source)
    if n > int(target_n):
        raise RuntimeError(f"pairwise_agent_count_exceeds_calibration_opt:{n}>{target_n}")
    result = np.zeros((1, target_n, target_n, 4, 4), dtype=source.dtype)
    result[:, :n, :n, :, :] = source
    for index in range(n, target_n):
        result[0, index, index] = np.eye(4, dtype=source.dtype)
    return np.ascontiguousarray(result)


def _pad_agent_mask(array: Any, target_n: int) -> Any:
    import numpy as np

    source = np.asarray(array)
    if source.ndim != 2 or source.shape[0] != 1:
        raise RuntimeError(f"agent_mask_shape_invalid:{source.shape}")
    if (
        not np.all(np.isfinite(source))
        or not np.all((source == 0) | (source == 1))
        or float(source[0, 0]) != 1.0
        or float(source.sum()) < 1.0
    ):
        raise RuntimeError(f"agent_mask_values_invalid:{source.tolist()}")
    n = int(source.shape[1])
    if n == int(target_n):
        return np.ascontiguousarray(source)
    if n > int(target_n):
        raise RuntimeError(f"agent_mask_count_exceeds_calibration_opt:{n}>{target_n}")
    result = np.zeros((1, target_n), dtype=source.dtype)
    result[:, :n] = source
    return np.ascontiguousarray(result)


def _trt_dtype_to_numpy(dtype: Any, trt: Any) -> Any:
    import numpy as np

    mapping = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int32: np.int32,
        trt.bool: np.bool_,
    }
    if hasattr(trt, "int8"):
        mapping[trt.int8] = np.int8
    return mapping.get(dtype, np.float32)


class _EntropyCalibrator:
    def __init__(
        self,
        trt: Any,
        *,
        input_names: list[str],
        input_dtypes: dict[str, Any],
        sample_files: list[Path],
        cache_path: Path,
        opt_agents: int,
    ) -> None:
        import torch

        class _Impl(trt.IInt8EntropyCalibrator2):
            def __init__(self, outer: "_EntropyCalibrator") -> None:
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.outer = outer

            def get_batch_size(self) -> int:
                return 1

            def get_batch(self, names: list[str]) -> list[int] | None:
                return self.outer.get_batch(names)

            def read_calibration_cache(self) -> bytes | None:
                return None

            def write_calibration_cache(self, cache: bytes) -> None:
                self.outer.write_calibration_cache(cache)

        self.trt = trt
        self.torch = torch
        self.input_names = input_names
        self.input_dtypes = input_dtypes
        self.sample_files = sample_files
        self.cache_path = cache_path
        self.opt_agents = int(opt_agents)
        self.index = 0
        self.device_tensors: list[Any] = []
        self.impl = _Impl(self)

    def get_batch(self, names: list[str]) -> list[int] | None:
        import numpy as np

        if self.index >= len(self.sample_files):
            return None
        sample_path = self.sample_files[self.index]
        self.index += 1
        pointers: list[int] = []
        self.device_tensors = []
        with np.load(sample_path) as sample:
            for name in names:
                if name not in sample:
                    raise RuntimeError(f"calibration_npz_input_missing_at_runtime:{sample_path}:{name}")
                array = sample[name]
                if name == "pairwise_t_matrix":
                    array = _pad_pairwise(array, self.opt_agents)
                elif name == "agent_mask":
                    array = _pad_agent_mask(array, self.opt_agents)
                dtype = _trt_dtype_to_numpy(self.input_dtypes[name], self.trt)
                array = np.ascontiguousarray(array.astype(dtype, copy=False))
                tensor = self.torch.as_tensor(array, device="cuda")
                self.device_tensors.append(tensor)
                pointers.append(int(tensor.data_ptr()))
        return pointers

    def write_calibration_cache(self, cache: bytes) -> None:
        if self.cache_path.exists():
            raise RuntimeError(f"fresh_calibration_cache_already_exists:{self.cache_path}")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)


def run(request: dict[str, Any]) -> dict[str, Any]:
    import torch
    import tensorrt as trt

    toolchain = _validate_modelopt_toolchain()
    onnx_path = Path(request["onnx_path"]).resolve()
    plugin_path = Path(request["plugin_path"]).resolve()
    cache_path = Path(request["cache_path"]).resolve()
    engine_path = Path(request["engine_path"]).resolve()
    fixed_k = int(request["fixed_k"])
    num_batches = int(request["num_batches"])
    for source, label in ((onnx_path, "onnx"), (plugin_path, "plugin")):
        if not source.is_file():
            raise RuntimeError(f"tensorrt_entropy_{label}_missing:{source}")
    if cache_path.exists() or engine_path.exists():
        raise RuntimeError("tensorrt_entropy_worker_refuses_existing_outputs")
    expected_inputs = [str(value) for value in request.get("input_names", [])] or None
    samples, agent_counts, calibration_identity = _resolve_and_verify_samples(
        request["calibration_npz_manifest"],
        num_batches=num_batches,
        fixed_k=fixed_k,
        input_names=expected_inputs,
    )
    profile_shapes = _profile(agent_counts, fixed_k, input_names=expected_inputs)
    ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("tensorrt_entropy_onnx_parse_failed:" + " | ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    input_names: list[str] = []
    input_dtypes: dict[str, Any] = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        name = str(tensor.name)
        if name not in profile_shapes:
            raise RuntimeError(f"tensorrt_entropy_profile_input_missing:{name}")
        input_names.append(name)
        input_dtypes[name] = tensor.dtype
        shapes = profile_shapes[name]
        profile.set_shape(name, tuple(shapes["min"]), tuple(shapes["opt"]), tuple(shapes["max"]))
    expected_network_inputs = set(profile_shapes)
    realized_network_inputs = set(input_names)
    if realized_network_inputs != expected_network_inputs:
        raise RuntimeError(
            f"tensorrt_entropy_network_input_contract_mismatch:"
            f"missing={sorted(expected_network_inputs - realized_network_inputs)}:"
            f"unknown={sorted(realized_network_inputs - expected_network_inputs)}"
        )
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)
    calibrator = _EntropyCalibrator(
        trt,
        input_names=input_names,
        input_dtypes=input_dtypes,
        sample_files=samples,
        cache_path=cache_path,
        opt_agents=int(profile_shapes["pairwise_t_matrix"]["opt"][1]),
    )
    config.int8_calibrator = calibrator.impl
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.time() - started
    if serialized is None:
        raise RuntimeError("tensorrt_entropy_build_serialized_network_returned_none")
    if not cache_path.is_file() or cache_path.stat().st_size <= 0:
        raise RuntimeError("tensorrt_entropy_calibration_cache_not_written")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    if engine is None:
        raise RuntimeError("tensorrt_entropy_calibration_engine_deserialize_failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("tensorrt_entropy_calibration_execution_context_failed")
    return {
        "status": "ok",
        "failure_reason": "",
        "dependencies": dict(request["dependencies"]),
        "semantics_version": request["dependencies"]["semantics_version"],
        "calibrator": "IInt8EntropyCalibrator2",
        "builder_flags": ["INT8", "FP16_FALLBACK_ALLOWED_BY_TRT_BUILDER"],
        "strict_types": False,
        "prefer_precision_constraints": False,
        "obey_precision_constraints": False,
        "fresh_calibration": True,
        "calibration_cache_reused": False,
        "calibration_input_provenance": calibration_identity,
        "calibration_sample_count": len(samples),
        "calibration_order": "npz_manifest_file_order",
        "profile": profile_shapes,
        "input_names": input_names,
        "onnx_path": str(onnx_path),
        "onnx_sha256": _sha256_file(onnx_path),
        "plugin_path": str(plugin_path),
        "plugin_sha256": _sha256_file(plugin_path),
        "calibration_cache_path": str(cache_path),
        "calibration_cache_sha256": _sha256_file(cache_path),
        "calibration_cache_bytes": cache_path.stat().st_size,
        "calibration_engine_path": str(engine_path),
        "calibration_engine_sha256": _sha256_file(engine_path),
        "calibration_engine_bytes": engine_path.stat().st_size,
        "calibration_engine_deserialize_passed": True,
        "calibration_execution_context_passed": True,
        "build_time_seconds": elapsed,
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability(0)),
        "toolchain": toolchain,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output_path = Path(request["output_path"])
    try:
        result = run(request)
        returncode = 0
    except Exception as exc:  # pragma: no cover - exercised in the isolated worker
        result = {
            "status": "failed",
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
            "dependencies": dict(request.get("dependencies", {})),
        }
        returncode = 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("status", "failure_reason", "calibration_cache_sha256")}, sort_keys=True))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
