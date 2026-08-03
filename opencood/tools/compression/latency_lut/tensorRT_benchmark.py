from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .schema import LatencyLUTKey, LatencyRecord
from .subgraph_exporter import SubgraphExportError, export_minimal_subgraph


SUCCESS_STATUSES = {"success", "ok"}


def file_hash(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_trtexec_latency(log_text: str) -> dict[str, float]:
    """Parse trtexec GPU Compute Time metrics from stdout/stderr text."""
    gpu_lines = [
        line.strip()
        for line in log_text.splitlines()
        if "GPU Compute Time:" in line and "Total GPU Compute Time" not in line
    ]
    if not gpu_lines:
        raise ValueError("trtexec log does not contain a GPU Compute Time summary")

    line = gpu_lines[-1]

    def extract(pattern: str) -> float | None:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if not match:
            return None
        return float(match.group(1))

    mean = extract(r"mean\s*=\s*([0-9.+\-eE]+)\s*ms")
    median = extract(r"median\s*=\s*([0-9.+\-eE]+)\s*ms")
    p90 = extract(r"percentile\(90%\)\s*=\s*([0-9.+\-eE]+)\s*ms")
    p95 = extract(r"percentile\(95%\)\s*=\s*([0-9.+\-eE]+)\s*ms")
    p99 = extract(r"percentile\(99%\)\s*=\s*([0-9.+\-eE]+)\s*ms")
    std = extract(r"(?:std|stddev|standard deviation)\s*=\s*([0-9.+\-eE]+)\s*ms")

    single_value = extract(r"GPU Compute Time:\s*([0-9.+\-eE]+)\s*ms")
    if mean is None and single_value is not None:
        mean = single_value
    if median is None and mean is not None:
        median = mean
    if p90 is None and mean is not None:
        p90 = mean
    if p95 is None and p90 is not None:
        p95 = p90
    if p99 is None and p95 is not None:
        p99 = p95
    if std is None:
        std = 0.0

    required = {
        "latency_p50_ms": median,
        "latency_p90_ms": p90,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "latency_mean_ms": mean,
        "latency_std_ms": std,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"trtexec GPU Compute Time summary is missing fields: {missing}; line={line}")
    return {name: float(value) for name, value in required.items() if value is not None}


def _zero_record(
    key: LatencyLUTKey,
    *,
    warmup: int,
    repeat: int,
    timing_method: str,
    status: str,
    error_message: str,
    engine_hash: str | None = None,
    onnx_hash: str | None = None,
    **metadata: Any,
) -> LatencyRecord:
    return LatencyRecord(
        key=key,
        latency_p50_ms=0.0,
        latency_p90_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        latency_mean_ms=0.0,
        latency_std_ms=0.0,
        num_warmup=warmup,
        num_repeat=repeat,
        timing_method=timing_method,
        engine_hash=engine_hash,
        onnx_hash=onnx_hash,
        status=status,
        error=error_message,
        error_message=error_message,
        metadata=metadata,
    )


def _resolve_trtexec(path: str | Path | None) -> str | None:
    if path:
        candidate = Path(path)
        if candidate.is_file():
            return str(candidate)
        return None
    return shutil.which("trtexec")


def _resolve_pointpillar_plugin(path: str | Path | None = None) -> str | None:
    if path:
        candidate = Path(path).expanduser()
        return str(candidate) if candidate.is_file() else None
    candidates = []
    candidates.extend(
        [
            Path("quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"),
            Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so"),
            Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build_clean/libpointpillar_scatter_trt.so"),
        ]
    )
    for candidate in candidates:
        if candidate.expanduser().is_file():
            return str(candidate.expanduser())
    return None


def _env_for_trtexec(trtexec: str, device: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(device))
    executable = Path(trtexec).resolve()
    candidate_lib_dirs = [
        executable.parent.parent / "lib",
        executable.parent.parent.parent / "lib",
    ]
    lib_dirs = [str(path) for path in candidate_lib_dirs if (path / "libnvinfer.so.10").exists() or (path / "libnvinfer_plugin.so.10").exists()]
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return env


def _trtexec_command(
    *,
    trtexec: str,
    onnx_path: str | Path,
    engine_path: str | Path,
    key: LatencyLUTKey,
    warmup: int,
    repeat: int,
    device: int | None,
    min_repeat_ms: int | None,
    iterations: int | None,
    allow_int8_fallback_to_fp16: bool,
    layer_info_path: str | Path | None = None,
    plugin_path: str | Path | None = None,
) -> list[str]:
    actual_iterations = int(iterations if iterations is not None else repeat)
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--warmUp={int(warmup)}",
        f"--iterations={actual_iterations}",
        "--useCudaGraph",
    ]
    if device is not None:
        # `_env_for_trtexec` maps the requested physical GPU to the only
        # visible CUDA device for the subprocess. In that namespace trtexec
        # must use device 0; passing the physical index makes parallel shards
        # fail with "Cannot find device ID <n>".
        cmd.append("--device=0")
    if min_repeat_ms is not None and int(min_repeat_ms) > 0:
        duration_s = max(1, int(math.ceil(int(min_repeat_ms) / 1000.0)))
        cmd.append(f"--duration={duration_s}")
    int8_requested = _is_int8_requested(key)
    if key.precision_profile == "TRT_FP16" or (int8_requested and (key.src_precision and "FP16" in key.src_precision or key.dst_precision and "FP16" in key.dst_precision)):
        cmd.append("--fp16")
    if int8_requested:
        cmd.extend(["--int8", "--dumpLayerInfo", "--profilingVerbosity=detailed"])
        if layer_info_path is not None:
            cmd.append(f"--exportLayerInfo={layer_info_path}")
    if plugin_path is not None:
        cmd.append(f"--staticPlugins={plugin_path}")
    return cmd


def build_trtexec_command_for_test(**kwargs: Any) -> list[str]:
    kwargs.setdefault("allow_int8_fallback_to_fp16", False)
    kwargs.setdefault("layer_info_path", None)
    kwargs.setdefault("plugin_path", None)
    return _trtexec_command(**kwargs)


def _is_int8_requested(key: LatencyLUTKey) -> bool:
    fields = [key.precision_profile, key.src_precision, key.dst_precision, key.tensor_dtype_before, key.tensor_dtype_after]
    return any("INT8" in str(value).upper() for value in fields if value is not None)


def parse_trtexec_precision(log_text: str, layer_info_text: str = "") -> dict[str, Any]:
    text = "\n".join([log_text or "", layer_info_text or ""])
    core_patterns = re.compile(r"\b(conv|convolution|gemm|matmul|matrixmultiply|fullyconnected)\b", re.IGNORECASE)
    int8 = re.compile(r"\b(INT8|Int8|int8)\b")
    fp16 = re.compile(r"\b(FP16|Half|kHALF|float16)\b", re.IGNORECASE)
    fp32 = re.compile(r"\b(FP32|Float|kFLOAT|float32)\b", re.IGNORECASE)
    observed_int8_layers = 0
    observed_fp16_layers = 0
    observed_fp32_layers = 0
    fallback_layers: list[str] = []
    for line in text.splitlines():
        if not core_patterns.search(line):
            continue
        if int8.search(line):
            observed_int8_layers += 1
        elif fp16.search(line):
            observed_fp16_layers += 1
            fallback_layers.append(line.strip()[:240])
        elif fp32.search(line):
            observed_fp32_layers += 1
            fallback_layers.append(line.strip()[:240])
    return {
        "observed_int8_layers": observed_int8_layers,
        "observed_fp16_layers": observed_fp16_layers,
        "observed_fp32_layers": observed_fp32_layers,
        "fallback_layers": fallback_layers[:20],
    }


def benchmark_key(
    key: LatencyLUTKey,
    *,
    output_dir: str | Path | None = None,
    onnx_dir: str | Path | None = None,
    engine_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    warmup: int = 50,
    repeat: int = 200,
    backend: str = "tensorrt",
    dry_run: bool = False,
    trtexec_path: str | Path | None = None,
    device: int | None = None,
    min_repeat_ms: int | None = None,
    iterations: int | None = None,
    allow_int8_fallback_to_fp16: bool = False,
    plugin_path: str | Path | None = None,
) -> LatencyRecord:
    root = Path(output_dir or "outputs/latency_lut/subgraphs")
    onnx_root = Path(onnx_dir) if onnx_dir is not None else root / "onnx"
    engine_root = Path(engine_dir) if engine_dir is not None else root / "engines"
    logs_root = Path(log_dir) if log_dir is not None else root / "logs"
    onnx_root.mkdir(parents=True, exist_ok=True)
    engine_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    if (key.plugin_flag or key.block_type == "plugin") and key.precision_profile == "TRT_INT8_QDQ":
        return _zero_record(
            key,
            warmup=warmup,
            repeat=repeat,
            timing_method="unavailable",
            status="skipped_plugin_int8_not_supported",
            error_message=f"{key.plugin_name or 'TensorRT plugin'} INT8 IO benchmark is not supported by the current plugin.",
            backend=backend,
        )

    # Resolve runtime prerequisites before exporting a subgraph.  This keeps a
    # missing TensorRT executable (or plugin) distinguishable from an ONNX/QDQ
    # contract failure and avoids writing an artifact that cannot be consumed.
    # Dry runs intentionally remain export-only and therefore do not require
    # either runtime dependency.
    resolved_plugin_path = None
    trtexec = None
    if not dry_run and backend == "tensorrt":
        if key.plugin_flag or key.block_type == "plugin":
            resolved_plugin_path = _resolve_pointpillar_plugin(plugin_path)
            if not resolved_plugin_path:
                return _zero_record(
                    key,
                    warmup=warmup,
                    repeat=repeat,
                    timing_method="unavailable",
                    status="skipped_plugin_not_available",
                    error_message=f"{key.plugin_name or 'TensorRT plugin'} shared library was not found.",
                    backend=backend,
                )
        trtexec = _resolve_trtexec(trtexec_path)
        if not trtexec:
            return _zero_record(
                key,
                warmup=warmup,
                repeat=repeat,
                timing_method="unavailable",
                status="skipped_trtexec_not_found",
                error_message=f"trtexec not found: {trtexec_path or 'PATH'}",
                backend=backend,
            )

    exported: dict[str, Any] | None = None
    try:
        exported = export_minimal_subgraph(key, onnx_root, dry_run=dry_run)
        onnx_hash = file_hash(exported.get("onnx_path", ""))
        if dry_run:
            return _zero_record(
                key,
                warmup=warmup,
                repeat=repeat,
                timing_method="dry_run",
                status="dry_run",
                error_message="dry-run only; no TensorRT engine was built",
                onnx_hash=onnx_hash,
                export=exported,
                backend=backend,
            )
        if backend != "tensorrt":
            return _zero_record(
                key,
                warmup=warmup,
                repeat=repeat,
                timing_method="unavailable",
                status="failed",
                error_message=f"unsupported backend: {backend}",
                onnx_hash=onnx_hash,
                backend=backend,
            )
        # ``trtexec`` was resolved above for real TensorRT runs.  The assertion
        # documents the ordering contract without inventing a fallback path.
        assert trtexec is not None

        engine_path = engine_root / f"{key.stable_hash()}.engine"
        log_path = logs_root / f"{key.stable_hash()}.log"
        layer_info_path = logs_root / f"{key.stable_hash()}.layerinfo.json"
        cmd = _trtexec_command(
            trtexec=trtexec,
            onnx_path=exported["onnx_path"],
            engine_path=engine_path,
            key=key,
            warmup=warmup,
            repeat=repeat,
            device=device,
            min_repeat_ms=min_repeat_ms,
            iterations=iterations,
            allow_int8_fallback_to_fp16=allow_int8_fallback_to_fp16,
            layer_info_path=layer_info_path if _is_int8_requested(key) else None,
            plugin_path=resolved_plugin_path,
        )
        env = _env_for_trtexec(trtexec, device)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False, env=env)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return _zero_record(
                key,
                warmup=warmup,
                repeat=repeat,
                timing_method="trtexec",
                status="failed",
                error_message=f"trtexec failed with returncode {proc.returncode}; see {log_path}",
                onnx_hash=onnx_hash,
                engine_hash=file_hash(engine_path),
                backend=backend,
                command=cmd,
                log_path=str(log_path),
                trtexec_returncode=proc.returncode,
            )
        parsed = parse_trtexec_latency(log_text)
        layer_info_text = layer_info_path.read_text(encoding="utf-8", errors="replace") if layer_info_path.is_file() else ""
        precision_info = parse_trtexec_precision(log_text, layer_info_text)
        exported_metadata = dict(exported.get("metadata") or {})
        int8_requested = _is_int8_requested(key)
        int8_verified = bool(
            int8_requested
            and (
                key.block_type == "precision_boundary"
                or int(precision_info.get("observed_int8_layers") or 0) > 0
            )
        )
        status = "success"
        error_message = None
        if int8_requested and not int8_verified:
            status = "skipped_int8_fallback_or_unverified"
            error_message = "TensorRT build completed, but no core Conv/Linear/GEMM INT8 layer was verified from trtexec layer info."
        return LatencyRecord(
            key=key,
            latency_p50_ms=parsed["latency_p50_ms"],
            latency_p90_ms=parsed["latency_p90_ms"],
            latency_p95_ms=parsed["latency_p95_ms"],
            latency_p99_ms=parsed["latency_p99_ms"],
            latency_mean_ms=parsed["latency_mean_ms"],
            latency_std_ms=parsed["latency_std_ms"],
            num_warmup=warmup,
            num_repeat=int(iterations if iterations is not None else repeat),
            timing_method="trtexec_gpu_compute_time",
            engine_hash=file_hash(engine_path),
            onnx_hash=onnx_hash,
            status=status,
            error=error_message,
            error_message=error_message,
            metadata={
                "backend": backend,
                "command": cmd,
                "log_path": str(log_path),
                "layer_info_path": str(layer_info_path) if layer_info_path.exists() else None,
                "onnx_path": str(exported["onnx_path"]),
                "engine_path": str(engine_path),
                "plugin_path": resolved_plugin_path,
                "requested_precision": key.precision_profile,
                "is_int8_verified": int8_verified,
                **precision_info,
                **exported_metadata,
            },
        )
    except SubgraphExportError as exc:
        return _zero_record(
            key,
            warmup=warmup,
            repeat=repeat,
            timing_method="unavailable",
            status="skipped_subgraph_not_implemented",
            error_message=str(exc),
            onnx_hash=file_hash(exported.get("onnx_path", "")) if exported else None,
            backend=backend,
            export=exported,
        )
    except Exception as exc:
        return _zero_record(
            key,
            warmup=warmup,
            repeat=repeat,
            timing_method="trtexec",
            status="failed",
            error_message=str(exc),
            onnx_hash=file_hash(exported.get("onnx_path", "")) if exported else None,
            backend=backend,
            export=exported,
        )
