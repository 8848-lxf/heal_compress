#!/usr/bin/env python3
"""Enumerate and pin native TensorRT tactics for the six real CoBEVT blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

from search.model_families.lidar_cobevt.native_tactic_evidence import (
    classify_tactic_kernel,
    classify_numerical_boundary,
    parse_editable_timing_log,
    realize_output_phenotype,
    select_f16a32_tactic,
    summarize_tactic_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _physical_device() -> dict[str, str]:
    import subprocess

    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return {"query": query, "cuda_visible_devices": visible, "target_arch": "SM89"}


class _RecordingLogger:
    def __new__(cls):
        import tensorrt as trt

        class Logger(trt.ILogger):
            def __init__(self):
                super().__init__()
                self.lines: list[str] = []

            def log(self, severity, message):  # noqa: ANN001
                if severity <= trt.ILogger.Severity.VERBOSE:
                    self.lines.append(str(message))

        return Logger()


def _cache_bytes(cache: Any) -> bytes:
    return bytes(cache.serialize())


def edit_timing_cache(cache_blob: bytes, key_text: str, tactic_hash: str) -> bytes:
    """Use the installed Python ITimingCache API, identical to the official sample."""

    import tensorrt as trt

    builder = trt.Builder(_RecordingLogger())
    config = builder.create_builder_config()
    cache = config.create_timing_cache(cache_blob)
    if cache is None:
        raise RuntimeError("timing_cache_create_failed")
    key = trt.TimingCacheKey.parse(str(key_text))
    value = trt.TimingCacheValue(int(str(tactic_hash), 16), 1.0)
    if not cache.update(key, value):
        raise RuntimeError(f"timing_cache_update_failed:{key_text}:{tactic_hash}")
    return _cache_bytes(cache)


def _make_network(module_name: str, role: str, a_shape: tuple[int, ...], b_shape: tuple[int, ...], trt: Any):
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = _BUILDER.create_network(flags)
    a = network.add_input("a", trt.float16, a_shape)
    b = network.add_input("b", trt.float16, b_shape)
    if a is None or b is None:
        raise RuntimeError("native_tactic_add_input_failed")
    op_b = trt.MatrixOperation.TRANSPOSE if role == "QK" else trt.MatrixOperation.NONE
    layer = network.add_matrix_multiply(a, trt.MatrixOperation.NONE, b, op_b)
    if layer is None:
        raise RuntimeError("native_tactic_add_matmul_failed")
    layer.name = f"{role.lower()}_matrix_multiply__{module_name.replace('.', '_')}"
    layer.get_output(0).name = "output"
    network.mark_output(layer.get_output(0))
    return network


_BUILDER: Any = None


def build_native_engine(
    *,
    module_name: str,
    role: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    engine_path: Path,
    cache_blob: bytes | None = None,
    error_on_cache_miss: bool = False,
) -> dict[str, Any]:
    import tensorrt as trt

    global _BUILDER
    logger = _RecordingLogger()
    _BUILDER = trt.Builder(logger)
    network = _make_network(module_name, role, a_shape, b_shape, trt)
    config = _BUILDER.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    if error_on_cache_miss:
        config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    cache = config.create_timing_cache(cache_blob or b"")
    if cache is None or not config.set_timing_cache(cache, True):
        raise RuntimeError("native_timing_cache_attach_failed")
    started = time.monotonic()
    serialized = _BUILDER.build_serialized_network(network, config)
    build_seconds = time.monotonic() - started
    if serialized is None:
        raise RuntimeError("native_tactic_engine_build_failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("native_tactic_engine_deserialize_failed")
    inspector = engine.create_engine_inspector()
    layer_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    cache_blob_out = _cache_bytes(cache)
    logs = list(logger.lines)
    records = parse_editable_timing_log(logs)
    return {
        "build_seconds": build_seconds,
        "engine_sha256": _sha256(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "cache_blob": cache_blob_out,
        "cache_sha256": hashlib.sha256(cache_blob_out).hexdigest(),
        "logs": logs,
        "profiling_records": records,
        "layer_info": layer_info,
    }


def _layer_metadata(layer_info: dict[str, Any], role: str, module_name: str) -> dict[str, Any]:
    token = f"{role.lower()}_matrix_multiply__{module_name.replace('.', '_')}"
    for layer in layer_info.get("Layers", []):
        name = str(layer.get("Name", ""))
        if token not in name and role.lower() not in name.lower():
            continue
        tactic_name = str(layer.get("TacticName", ""))
        if tactic_name:
            output_precision = "unknown"
            outputs = layer.get("Outputs", [])
            if outputs:
                dtype = str(outputs[0].get("Format/Datatype", ""))
                output_precision = "FP16" if "HALF" in dtype.upper() else "FP32" if "FLOAT" in dtype.upper() else dtype
            return {
                "tactic_name": tactic_name,
                "tactic_hash": re.findall(r"0x[0-9a-fA-F]+", tactic_name)[-1] if re.findall(r"0x[0-9a-fA-F]+", tactic_name) else None,
                "output_precision": output_precision,
            }
    return {"tactic_name": None, "tactic_hash": None, "output_precision": "unknown"}


def _layer_tactic(layer_info: dict[str, Any], role: str, module_name: str) -> tuple[str | None, str | None]:
    metadata = _layer_metadata(layer_info, role, module_name)
    return metadata["tactic_name"], metadata["tactic_hash"]


def _run_engine(engine_path: Path, a: torch.Tensor, b: torch.Tensor, device: torch.device) -> dict[str, Any]:
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("native_tactic_runtime_deserialize_failed")
    context = engine.create_execution_context()
    stream = torch.cuda.Stream(device=device)
    bindings: dict[str, torch.Tensor] = {}
    output = None
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = engine.get_tensor_dtype(name)
        torch_dtype = torch.float16 if dtype == trt.float16 else torch.float32
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            value = (a if name == "a" else b).to(device=device, dtype=torch_dtype).contiguous()
            context.set_input_shape(name, tuple(value.shape))
            bindings[name] = value
        else:
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
            output = torch.empty(shape, device=device, dtype=torch_dtype)
            bindings[name] = output
    if output is None:
        raise RuntimeError("native_tactic_runtime_output_missing")
    for name, value in bindings.items():
        context.set_tensor_address(name, int(value.data_ptr()))
    for _ in range(20):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    starts, ends = [], []
    for _ in range(50):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        context.execute_async_v3(stream.cuda_stream)
        end.record(stream)
        ends.append(end)
        starts.append(start)
    ends[-1].synchronize()
    times = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    return {
        "output": output.detach().cpu(),
        "p50_ms": float(torch.tensor(times).quantile(0.50)),
        "p90_ms": float(torch.tensor(times).quantile(0.90)),
        "p99_ms": float(torch.tensor(times).quantile(0.99)),
        "input_dtype": "FP16",
        "output_dtype": "FP16",
    }


def _numeric_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = value.float().reshape(-1)
    expected = reference.float().reshape(-1)
    diff = candidate - expected
    denominator = max(float(torch.linalg.vector_norm(expected)), 1.0e-12)
    return {
        "relative_l2": float(torch.linalg.vector_norm(diff)) / denominator,
        "max_abs": float(diff.abs().max()),
        "cosine": float(torch.nn.functional.cosine_similarity(candidate, expected, dim=0)),
        "finite": bool(torch.isfinite(candidate).all()),
        "reference_norm": float(torch.linalg.vector_norm(expected)),
        "output_zero_ratio": float((candidate == 0).float().mean()),
        "numerical_status": classify_numerical_boundary(
            finite=bool(torch.isfinite(candidate).all()),
            reference_norm=float(torch.linalg.vector_norm(expected)),
            output_zero_ratio=float((candidate == 0).float().mean()),
        ),
    }


def _capture_for_role(path: Path, role: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capture = torch.load(path, map_location="cpu", weights_only=False)
    tensors = capture["tensors"]
    if role == "QK":
        a, b = tensors["scaled_q"].half(), tensors["k"].half()
        reference = torch.matmul(a.float(), b.float().transpose(-1, -2))
    else:
        a, b = tensors["probability"].half(), tensors["v"].half()
        reference = torch.matmul(a.float(), b.float())
    return a, b, reference


def run_matrix(*, output_dir: Path, capture_manifest: Path, device: int) -> dict[str, Any]:
    manifest = json.loads(capture_manifest.read_text())
    selected: dict[str, Path] = {}
    for row in manifest["records"]:
        selected.setdefault(str(row["module_name"]), Path(str(row["capture_path"])))
    if len(selected) != 6:
        raise ValueError(f"native_tactic_expected_six_blocks:{len(selected)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_manifest.json", {"capture_manifest": str(capture_manifest), "capture_manifest_sha256": _sha256(capture_manifest), "device": device, "roles": ["QK", "AV"], "builder_flags": ["STRONGLY_TYPED", "EDITABLE_TIMING_CACHE", "NO_TF32"]})
    all_rows: list[dict[str, Any]] = []
    tactic_rows: list[dict[str, Any]] = []
    pin_rows: list[dict[str, Any]] = []
    device_obj = torch.device(f"cuda:{device}")
    torch.cuda.set_device(device_obj)
    for block, (module_name, capture_path) in enumerate(sorted(selected.items())):
        for role in ("QK", "AV"):
            a, b, reference = _capture_for_role(capture_path, role)
            slug = module_name.replace(".", "_")
            base_dir = output_dir / "microengines" / role.lower() / f"block_{block}"
            base_dir.mkdir(parents=True, exist_ok=True)
            base = build_native_engine(module_name=module_name, role=role, a_shape=tuple(a.shape), b_shape=tuple(b.shape), engine_path=base_dir / "baseline.engine")
            (base_dir / "baseline.cache").write_bytes(base["cache_blob"])
            (base_dir / "baseline_build.log").write_text("\n".join(base["logs"]) + "\n")
            _write_json(base_dir / "baseline_layer_info.json", base["layer_info"])
            records = base["profiling_records"]
            if len(records) != 1:
                raise RuntimeError(f"native_tactic_expected_one_profile_record:{module_name}:{role}:{len(records)}")
            record = records[0]
            target = select_f16a32_tactic(record)
            for row in summarize_tactic_records(records):
                tactic_rows.append({"role": role, "block": block, "module_name": module_name, **row})
            runtime = _run_engine(base_dir / "baseline.engine", a, b, device_obj)
            base_metrics = _numeric_metrics(runtime["output"], reference)
            baseline_metadata = _layer_metadata(base["layer_info"], role, module_name)
            baseline_tactic_name, baseline_tactic_hash = baseline_metadata["tactic_name"], baseline_metadata["tactic_hash"]
            base_result = {
                "role": role, "block": block, "module_name": module_name,
                "shape": list(a.shape), "baseline_engine_sha256": base["engine_sha256"],
                "baseline_cache_sha256": base["cache_sha256"],
                "available_tactic_count": len(record["available_tactics"]),
                "baseline_selected_hash": record.get("selected_tactic"),
                "baseline_realized_tactic_hash": baseline_tactic_hash,
                "baseline_realized_tactic_name": baseline_tactic_name,
                "baseline_output_precision": baseline_metadata["output_precision"],
                "baseline_p50_ms": runtime["p50_ms"], "baseline_p90_ms": runtime["p90_ms"], "baseline_p99_ms": runtime["p99_ms"],
                "baseline_relative_l2": base_metrics["relative_l2"], "baseline_cosine": base_metrics["cosine"], "baseline_finite": base_metrics["finite"], "baseline_reference_norm": base_metrics["reference_norm"], "baseline_output_zero_ratio": base_metrics["output_zero_ratio"], "baseline_numerical_status": base_metrics["numerical_status"],
                "target_tactic_hash": target.get("tactic_hash") if target else None,
                "target_kernel_name": target.get("kernel_name") if target else None,
                "target_phenotype": target["kernel_evidence"]["phenotype"] if target else "UNKNOWN_ACCUM",
                "target_evidence_level": target["kernel_evidence"]["evidence_level"] if target else "LEVEL_C_UNKNOWN",
                "pinning_stable": False,
            }
            if target is not None:
                edited = edit_timing_cache(base["cache_blob"], record["key"], target["tactic_hash"])
                cache_dir = output_dir / "timing_cache" / "edited"
                cache_dir.mkdir(parents=True, exist_ok=True)
                edited_path = cache_dir / f"{role.lower()}_block_{block}.cache"
                edited_path.write_bytes(edited)
                (output_dir / "timing_cache" / "original").mkdir(parents=True, exist_ok=True)
                (output_dir / "timing_cache" / "original" / f"{role.lower()}_block_{block}.cache").write_bytes(base["cache_blob"])
                rebuilds = []
                for repeat in range(3):
                    pinned_dir = base_dir / f"pinned_{repeat}"
                    pinned = build_native_engine(module_name=module_name, role=role, a_shape=tuple(a.shape), b_shape=tuple(b.shape), engine_path=pinned_dir / "engine.plan", cache_blob=edited, error_on_cache_miss=True)
                    (pinned_dir / "build.log").write_text("\n".join(pinned["logs"]) + "\n")
                    _write_json(pinned_dir / "layer_info.json", pinned["layer_info"])
                    realized_metadata = _layer_metadata(pinned["layer_info"], role, module_name)
                    realized_name, realized_hash = realized_metadata["tactic_name"], realized_metadata["tactic_hash"]
                    realized_hash = realized_hash or next((h for h in [t["tactic_hash"] for t in record["available_tactics"] if t["kernel_name"] == realized_name]), None)
                    rebuilds.append({"repeat": repeat, "requested_tactic_hash": target["tactic_hash"], "realized_tactic_hash": realized_hash, "realized_tactic_name": realized_name, "realized_output_precision": realized_metadata["output_precision"], "engine_sha256": pinned["engine_sha256"], "cache_sha256": pinned["cache_sha256"], "match": realized_name == target["kernel_name"] or realized_hash == target["tactic_hash"]})
                    if repeat == 0:
                        pinned_runtime = _run_engine(pinned_dir / "engine.plan", a, b, device_obj)
                        pinned_metrics = _numeric_metrics(pinned_runtime["output"], reference)
                        base_result.update({
                            "pinned_p50_ms": pinned_runtime["p50_ms"],
                            "pinned_p90_ms": pinned_runtime["p90_ms"],
                            "pinned_p99_ms": pinned_runtime["p99_ms"],
                            "pinned_relative_l2": pinned_metrics["relative_l2"],
                            "pinned_cosine": pinned_metrics["cosine"],
                            "pinned_finite": pinned_metrics["finite"],
                            "pinned_reference_norm": pinned_metrics["reference_norm"],
                            "pinned_output_zero_ratio": pinned_metrics["output_zero_ratio"],
                            "pinned_numerical_status": pinned_metrics["numerical_status"],
                        })
                base_result["pinning_rebuilds"] = rebuilds
                base_result["pinning_stable"] = len(rebuilds) == 3 and all(row["match"] for row in rebuilds)
                base_result["edited_cache_sha256"] = hashlib.sha256(edited).hexdigest()
                base_result["target_compute_phenotype"] = target["kernel_evidence"].get("compute_phenotype", "UNKNOWN_ACCUM")
                base_result["target_phenotype"] = realize_output_phenotype(target["kernel_evidence"], baseline_metadata["output_precision"])
            else:
                base_result["pinning_rebuilds"] = []
            all_rows.append(base_result)
            pin_rows.append(base_result)
    _write_csv(output_dir / "tactics" / "qk_available_tactics.csv", [row for row in tactic_rows if row["role"] == "QK"])
    _write_csv(output_dir / "tactics" / "av_available_tactics.csv", [row for row in tactic_rows if row["role"] == "AV"])
    _write_csv(output_dir / "tactics" / "unique_tactic_inventory.csv", tactic_rows)
    _write_csv(output_dir / "timing_cache" / "pinning_results.csv", pin_rows)
    _write_csv(output_dir / "timing_cache" / "deterministic_rebuild_results.csv", [sub for row in pin_rows for sub in row.get("pinning_rebuilds", [])])
    _write_csv(output_dir / "native_shape_pinning_matrix.csv", all_rows)
    _write_json(output_dir / "native_shape_pinning_matrix.json", all_rows)
    _write_json(
        output_dir / "tactics" / "enumeration_completeness.json",
        {
            "status": "partial",
            "record_count": len(tactic_rows),
            "shape_count": len(all_rows),
            "source": "sampleEditableTimingCache profiling table",
            "reason": (
                "TensorRT 10.9 exposes every candidate reported in the editable-cache "
                "profiling table, but its public API does not guarantee that disabled or "
                "internally filtered implementations are enumerable."
            ),
        },
    )
    _write_json(output_dir / "timing_cache" / "cache_sha256.json", {f"{row['role']}_block_{row['block']}": {"original": row.get("baseline_cache_sha256"), "edited": row.get("edited_cache_sha256")} for row in all_rows})
    return {"shape_count": len(all_rows), "tactic_count": len(tactic_rows), "stable": sum(bool(row["pinning_stable"]) for row in all_rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_matrix(output_dir=Path(args.output_dir).resolve(), capture_manifest=Path(args.capture_manifest).resolve(), device=args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
