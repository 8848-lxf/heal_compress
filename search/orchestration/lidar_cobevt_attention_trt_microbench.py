#!/usr/bin/env python3
"""Build and run native TensorRT QK/AV micro-engines on real captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from search.model_families.lidar_cobevt.attention_trt_realization import (
    classify_native_realization,
    infer_accumulator_from_tactics,
    native_micro_specs,
)


@dataclass(frozen=True)
class NativeCandidate:
    candidate_id: str
    module_name: str
    spec_id: str
    fresh_build: bool = True


@dataclass(frozen=True)
class MicroGraphPlan:
    spec_id: str
    family: str
    input_precision: str
    cast_inputs_to_fp32: bool
    native_int8_operands: bool
    dequantize_before_matmul: bool
    claimed_realized_phenotype: None = None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def native_candidate_matrix(modules: Iterable[str]) -> tuple[NativeCandidate, ...]:
    return tuple(
        NativeCandidate(
            candidate_id=f"{_slug(module)}__{spec.spec_id}",
            module_name=str(module),
            spec_id=spec.spec_id,
        )
        for module in modules
        for spec in native_micro_specs()
    )


def micro_graph_plan(spec_id: str) -> MicroGraphPlan:
    specs = {row.spec_id: row for row in native_micro_specs()}
    try:
        spec = specs[str(spec_id)]
    except KeyError as exc:
        raise ValueError(f"unknown_native_micro_spec:{spec_id}") from exc
    return MicroGraphPlan(
        spec_id=spec.spec_id,
        family=spec.family,
        input_precision=spec.input_precision,
        cast_inputs_to_fp32=spec.explicit_cast_to_fp32,
        native_int8_operands=spec.native_int8_quantize,
        dequantize_before_matmul=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _trt_dtype(trt: Any, precision: str) -> Any:
    mapping = {
        "FP32": trt.float32,
        "FP16": trt.float16,
        "BF16": trt.bfloat16,
        "INT8": trt.int8,
    }
    return mapping[precision]


def _torch_dtype(trt: Any, dtype: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
    }
    return mapping[dtype]


def _parse_inspector(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [{"raw": raw}]
    if isinstance(parsed, list):
        return [row if isinstance(row, dict) else {"raw": row} for row in parsed]
    if isinstance(parsed, dict):
        layers = parsed.get("Layers") or parsed.get("layers")
        if isinstance(layers, list):
            return [row for row in layers if isinstance(row, dict)]
        return [parsed]
    return [{"raw": raw}]


def _build_engine(
    *,
    plan: MicroGraphPlan,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    engine_path: Path,
    layer_info_path: Path,
) -> dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    dtype = _trt_dtype(trt, plan.input_precision)
    a = network.add_input("a", dtype, shape_a)
    b = network.add_input("b", dtype, shape_b)
    if a is None or b is None:
        raise RuntimeError("trt_add_input_failed")
    left, right = a, b
    if plan.cast_inputs_to_fp32:
        cast_a = network.add_cast(left, trt.float32)
        cast_b = network.add_cast(right, trt.float32)
        if cast_a is None or cast_b is None:
            raise RuntimeError("trt_add_cast_failed")
        cast_a.name = "cast_a_to_fp32"
        cast_b.name = "cast_b_to_fp32"
        left, right = cast_a.get_output(0), cast_b.get_output(0)

    op_b = trt.MatrixOperation.TRANSPOSE if plan.family == "QK" else trt.MatrixOperation.NONE
    matmul = network.add_matrix_multiply(left, trt.MatrixOperation.NONE, right, op_b)
    if matmul is None:
        raise RuntimeError("trt_add_matrix_multiply_failed")
    matmul.name = f"{plan.family.lower()}_matrix_multiply"
    output = matmul.get_output(0)
    output.name = "output"
    network.mark_output(output)

    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.monotonic() - started
    if serialized is None:
        raise RuntimeError("trt_build_serialized_network_failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("trt_deserialize_failed")
    inspector = engine.create_engine_inspector()
    raw_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layer_info_path.write_text(raw_info)
    return {
        "build_seconds": build_seconds,
        "engine_sha256": _sha256(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "layer_info": _parse_inspector(raw_info),
    }


def _quantize_int8(value: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = max(float(value.detach().abs().max()) / 127.0, 1.0e-12)
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _prepare_inputs(
    capture: dict[str, Any], plan: MicroGraphPlan
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    tensors = capture["tensors"]
    if plan.family == "QK":
        a = tensors["scaled_q"].contiguous()
        b = tensors["k"].contiguous()
        reference = torch.matmul(a.float(), b.float().transpose(-1, -2))
    else:
        a = tensors["probability"].contiguous()
        b = tensors["v"].contiguous()
        reference = torch.matmul(a.float(), b.float())
    recovery_scale = 1.0
    if plan.input_precision == "INT8":
        a, scale_a = _quantize_int8(a)
        b, scale_b = _quantize_int8(b)
        recovery_scale = scale_a * scale_b
    elif plan.input_precision == "FP16":
        a, b = a.half(), b.half()
    elif plan.input_precision == "BF16":
        a, b = a.bfloat16(), b.bfloat16()
    else:
        a, b = a.float(), b.float()
    return a, b, reference, recovery_scale


def _run_engine(
    engine_path: Path,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float], tuple[str, str], str]:
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("trt_deserialize_runtime_failed")
    context = engine.create_execution_context()
    stream = torch.cuda.Stream(device=device)
    input_values = {"a": a, "b": b}
    bindings: dict[str, torch.Tensor] = {}
    output: torch.Tensor | None = None
    input_precisions: list[str] = []
    output_precision = "unknown"
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = engine.get_tensor_dtype(name)
        torch_dtype = _torch_dtype(trt, dtype)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            value = input_values[name].to(device=device, dtype=torch_dtype).contiguous()
            context.set_input_shape(name, tuple(value.shape))
            bindings[name] = value
            input_precisions.append(str(dtype).upper())
        else:
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
            output = torch.empty(shape, device=device, dtype=torch_dtype)
            bindings[name] = output
            output_precision = str(dtype).upper()
    if output is None:
        raise RuntimeError("trt_output_missing")
    for name, value in bindings.items():
        if not context.set_tensor_address(name, int(value.data_ptr())):
            raise RuntimeError(f"trt_set_tensor_address_failed:{name}")
    for _ in range(warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("trt_warmup_failed")
    stream.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("trt_execute_failed")
        end.record(stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return (
        output.detach().cpu(),
        {
            "p50_ms": float(np.percentile(samples, 50)),
            "p90_ms": float(np.percentile(samples, 90)),
            "p99_ms": float(np.percentile(samples, 99)),
            "mean_ms": float(statistics.mean(samples)),
        },
        tuple(input_precisions),
        output_precision,
    )


def _precision_name(value: str) -> str:
    upper = value.upper()
    if "HALF" in upper or "FLOAT16" in upper:
        return "FP16"
    if "BF16" in upper:
        return "BF16"
    if "INT8" in upper:
        return "INT8"
    if "INT32" in upper:
        return "INT32"
    if "FLOAT" in upper:
        return "FP32"
    return "unknown"


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    cand = candidate.float().reshape(-1)
    ref = reference.float().reshape(-1)
    diff = cand - ref
    denominator = max(float(torch.linalg.vector_norm(ref)), 1.0e-12)
    cosine = float(torch.nn.functional.cosine_similarity(cand, ref, dim=0))
    return {
        "relative_l2": float(torch.linalg.vector_norm(diff)) / denominator,
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "cosine": cosine,
        "finite": bool(torch.isfinite(cand).all()),
    }


def _matmul_inputs(plan: MicroGraphPlan) -> tuple[str, str]:
    if plan.cast_inputs_to_fp32:
        return ("FP32", "FP32")
    return (plan.input_precision, plan.input_precision)


def _run_candidate(
    candidate: NativeCandidate,
    *,
    capture_path: Path,
    root: Path,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    plan = micro_graph_plan(candidate.spec_id)
    spec = {row.spec_id: row for row in native_micro_specs()}[candidate.spec_id]
    candidate_dir = root / "trt_microengines" / "candidates" / candidate.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    _write_json(candidate_dir / "candidate.json", {**asdict(candidate), **asdict(plan)})
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    a, b, reference, recovery_scale = _prepare_inputs(capture, plan)
    engine_path = candidate_dir / "model.engine"
    layer_info_path = candidate_dir / "engine_layer_info.json"
    base = {
        **asdict(candidate),
        **asdict(plan),
        "capture_path": str(capture_path),
        "capture_sha256": _sha256(capture_path),
        "requested_phenotype": spec.requested_phenotype,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
    }
    try:
        build = _build_engine(
            plan=plan,
            shape_a=tuple(a.shape),
            shape_b=tuple(b.shape),
            engine_path=engine_path,
            layer_info_path=layer_info_path,
        )
        output, latency, engine_inputs, engine_output = _run_engine(
            engine_path,
            a=a,
            b=b,
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
        recovered = output.float() * recovery_scale
        numerical = _metrics(recovered, reference)
        execution_layers = build.pop("layer_info")
        direct_accumulator_metadata = infer_accumulator_from_tactics(execution_layers)
        realization = classify_native_realization(
            requested_phenotype=spec.requested_phenotype,
            build_success=True,
            input_precisions=(plan.input_precision, plan.input_precision),
            matmul_input_precisions=_matmul_inputs(plan),
            output_precision=_precision_name(engine_output),
            execution_layers=execution_layers,
            direct_accumulator_metadata=direct_accumulator_metadata,
        )
        result = {
            **base,
            **build,
            **latency,
            **numerical,
            **realization,
            "build_success": True,
            "runtime_success": True,
            "engine_input_precisions": list(engine_inputs),
            "engine_output_precision": engine_output,
            "recovery_scale": recovery_scale,
            "execution_layer_count": len(execution_layers),
            "engine_path": str(engine_path),
            "layer_info_path": str(layer_info_path),
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            **base,
            "build_success": engine_path.exists(),
            "runtime_success": False,
            "failure_reason": f"{type(exc).__name__}:{exc}",
            "realized_phenotype": "unsupported_build" if not engine_path.exists() else "runtime_failure",
            "realized_accumulator_precision": "unknown",
            "evidence_level": "C",
            "requested_realized_match": False,
        }
        (candidate_dir / "failure.txt").write_text(result["failure_reason"] + "\n")
    _write_json(candidate_dir / "result.json", result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _representative_captures(manifest: dict[str, Any]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for row in manifest["records"]:
        selected.setdefault(str(row["module_name"]), Path(row["capture_path"]))
    if len(selected) != 6:
        raise ValueError(f"expected_six_attention_blocks:{len(selected)}")
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir).resolve()
    manifest = json.loads(Path(args.capture_manifest).read_text())
    captures = _representative_captures(manifest)
    matrix = native_candidate_matrix(sorted(captures))
    selected = [row for index, row in enumerate(matrix) if index % args.shard_count == args.shard_index]
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    rows = [
        _run_candidate(
            candidate,
            capture_path=captures[candidate.module_name],
            root=root,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for candidate in selected
    ]
    shard_dir = root / "trt_microengines" / "shards"
    _write_json(shard_dir / f"shard_{args.shard_index}.json", rows)
    _write_csv(shard_dir / f"shard_{args.shard_index}.csv", rows)
    return {
        "shard_index": args.shard_index,
        "candidate_count": len(rows),
        "build_success": sum(bool(row.get("build_success")) for row in rows),
        "runtime_success": sum(bool(row.get("runtime_success")) for row in rows),
    }


def aggregate(output_dir: Path, shard_count: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shard in range(shard_count):
        path = output_dir / "trt_microengines" / "shards" / f"shard_{shard}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing_native_micro_shard:{path}")
        rows.extend(json.loads(path.read_text()))
    rows.sort(key=lambda row: row["candidate_id"])
    _write_json(output_dir / "trt_realized_phenotype.json", rows)
    _write_csv(output_dir / "trt_realized_phenotype.csv", rows)
    casts = [row for row in rows if row.get("cast_inputs_to_fp32")]
    _write_json(output_dir / "cast_materialization_audit.json", casts)
    _write_csv(output_dir / "cast_materialization_audit.csv", casts)
    summary = {
        "candidate_count": len(rows),
        "build_success": sum(bool(row.get("build_success")) for row in rows),
        "runtime_success": sum(bool(row.get("runtime_success")) for row in rows),
        "requested_realized_match": sum(bool(row.get("requested_realized_match")) for row in rows),
    }
    _write_json(output_dir / "trt_microengines" / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        print(json.dumps(aggregate(Path(args.output_dir).resolve(), args.shard_count), indent=2))
    else:
        print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
