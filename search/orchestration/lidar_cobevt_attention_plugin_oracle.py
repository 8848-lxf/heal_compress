#!/usr/bin/env python3
"""Build Level-A cuBLASLt TensorRT plugin-oracle engines for real Attention shapes."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from search.orchestration.lidar_cobevt_attention_trt_microbench import (
    _metrics,
    _parse_inspector,
    _representative_captures,
    _run_engine,
    _sha256,
    _write_json,
)


@dataclass(frozen=True)
class PluginCandidate:
    candidate_id: str
    module_name: str
    family: str
    requested_phenotype: str = "F16A32"
    implementation: str = "plugin_oracle"


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def plugin_candidate_matrix(modules: Iterable[str]) -> tuple[PluginCandidate, ...]:
    return tuple(
        PluginCandidate(f"{_slug(module)}__PLUGIN_{family}_F16A32", str(module), family)
        for module in modules
        for family in ("QK", "AV")
    )


def _load_plugin(plugin_library: Path) -> None:
    ctypes.CDLL(str(plugin_library), mode=ctypes.RTLD_GLOBAL)


def select_plugin_creator(creators: Iterable[Any]) -> Any:
    matches = [
        creator
        for creator in creators
        if creator.name == "QKMixedAccumPlugin"
        and creator.plugin_version == "1"
        and creator.plugin_namespace == ""
    ]
    if not matches:
        raise ValueError("qk_mixed_accum_plugin_creator_missing")
    if len(matches) != 1:
        raise ValueError("qk_mixed_accum_plugin_creator_ambiguous")
    return matches[0]


def _creator() -> Any:
    import tensorrt as trt

    registry = trt.get_plugin_registry()
    return select_plugin_creator(registry.all_creators)


def _build_plugin_engine(
    *,
    family: str,
    scale: float,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    output_precision: str,
    engine_path: Path,
    layer_info_path: Path,
) -> dict[str, Any]:
    import tensorrt as trt

    builder = trt.Builder(trt.Logger(trt.Logger.VERBOSE))
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    left = network.add_input("a", trt.float16, shape_a)
    right = network.add_input("b", trt.float16, shape_b)
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(
                "family", np.array([0 if family == "QK" else 1], dtype=np.int32), trt.PluginFieldType.INT32
            ),
            trt.PluginField(
                "output_type",
                np.array([0 if output_precision == "FP32" else 1], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField("scale", np.array([scale], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        ]
    )
    plugin = _creator().create_plugin(f"{family.lower()}_mixed_accum", fields)
    if plugin is None:
        raise RuntimeError("qk_mixed_accum_plugin_create_failed")
    layer = network.add_plugin_v2([left, right], plugin)
    if layer is None:
        raise RuntimeError("qk_mixed_accum_add_plugin_failed")
    layer.name = f"{family.lower()}_mixed_accum_plugin_oracle"
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.monotonic() - started
    if serialized is None:
        raise RuntimeError("qk_mixed_accum_plugin_build_failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("qk_mixed_accum_plugin_deserialize_failed")
    inspector = engine.create_engine_inspector()
    layer_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layer_info_path.write_text(layer_info)
    return {
        "build_seconds": elapsed,
        "engine_sha256": _sha256(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "execution_layers": _parse_inspector(layer_info),
    }


def _capture_operands(capture: dict[str, Any], family: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    tensors = capture["tensors"]
    if family == "QK":
        left = tensors["q"].half().contiguous()
        right = tensors["k"].half().contiguous()
        scale = float(capture["scale"])
        reference = torch.matmul(tensors["q"].float() * scale, tensors["k"].float().transpose(-1, -2))
    else:
        left = tensors["probability"].half().contiguous()
        right = tensors["v"].half().contiguous()
        scale = 1.0
        reference = torch.matmul(tensors["probability"].float(), tensors["v"].float())
    return left, right, reference, scale


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir).resolve()
    plugin_library = Path(args.plugin_library).resolve()
    _load_plugin(plugin_library)
    manifest = json.loads(Path(args.capture_manifest).read_text())
    captures = _representative_captures(manifest)
    matrix = plugin_candidate_matrix(sorted(captures))
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    rows: list[dict[str, Any]] = []
    for candidate in matrix:
        candidate_dir = root / "plugins" / "microengines" / candidate.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        capture_path = captures[candidate.module_name]
        capture = torch.load(capture_path, map_location="cpu", weights_only=False)
        left, right, reference, scale = _capture_operands(capture, candidate.family)
        output_precision = "FP32" if candidate.family == "QK" else "FP16"
        engine_path = candidate_dir / "model.engine"
        info_path = candidate_dir / "engine_layer_info.json"
        base = {
            **asdict(candidate),
            "scale": scale,
            "output_precision": output_precision,
            "capture_path": str(capture_path),
            "plugin_library": str(plugin_library),
            "plugin_sha256": _sha256(plugin_library),
        }
        try:
            build = _build_plugin_engine(
                family=candidate.family,
                scale=scale,
                shape_a=tuple(left.shape),
                shape_b=tuple(right.shape),
                output_precision=output_precision,
                engine_path=engine_path,
                layer_info_path=info_path,
            )
            execution_layers = build.pop("execution_layers")
            output, latency, _, _ = _run_engine(
                engine_path,
                a=left,
                b=right,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            result = {
                **base,
                **build,
                **latency,
                **_metrics(output, reference),
                "build_success": True,
                "runtime_success": True,
                "realized_phenotype": "F16A32",
                "realized_accumulator_precision": "FP32",
                "evidence_level": "A",
                "requested_realized_match": True,
                "native_tensorrt": False,
                "execution_layer_count": len(execution_layers),
                "engine_path": str(engine_path),
                "layer_info_path": str(info_path),
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                **base,
                "build_success": engine_path.exists(),
                "runtime_success": False,
                "requested_realized_match": False,
                "evidence_level": "A",
                "failure_reason": f"{type(exc).__name__}:{exc}",
            }
            (candidate_dir / "failure.txt").write_text(result["failure_reason"] + "\n")
        _write_json(candidate_dir / "result.json", result)
        rows.append(result)
    _write_json(root / "plugins" / "plugin_oracle_matrix.json", rows)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (root / "plugins" / "plugin_oracle_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    summary = {
        "candidate_count": len(rows),
        "build_success": sum(bool(row.get("build_success")) for row in rows),
        "runtime_success": sum(bool(row.get("runtime_success")) for row in rows),
        "level_a_matches": sum(bool(row.get("requested_realized_match")) for row in rows),
    }
    _write_json(root / "plugins" / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--plugin-library", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
