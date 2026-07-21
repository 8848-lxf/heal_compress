"""Isolated device-resident H800 latency for accepted Transformer engines."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from search.integration.runtime_environment import runtime_cuda_index_for_physical
from search.orchestration.lidar_transformer_h800_inventory import MODEL_SPECS


TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compute_processes(physical_gpu: int) -> list[dict[str, Any]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) >= 2:
            rows.append(
                {
                    "pid": int(values[0]),
                    "process_name": values[1],
                    "used_memory_mib": float(values[2]) if len(values) > 2 else None,
                }
            )
    return rows


def audit_isolation(physical_gpu: int, seconds: int) -> dict[str, Any]:
    samples = []
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        processes = _compute_processes(physical_gpu)
        samples.append({"elapsed_seconds": time.monotonic() - started, "processes": processes})
        if processes:
            return {"isolated": False, "required_seconds": seconds, "samples": samples}
        time.sleep(min(5.0, max(0.1, seconds - (time.monotonic() - started))))
    return {"isolated": True, "required_seconds": seconds, "samples": samples}


def _gpu_telemetry(physical_gpu: int) -> dict[str, Any]:
    fields = (
        "uuid,name,clocks.current.graphics,clocks.current.memory,temperature.gpu,"
        "power.draw,power.limit,utilization.gpu,memory.used"
    )
    values = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().split(", ")
    keys = fields.split(",")
    return dict(zip(keys, values))


def _profile_directory(root: Path, model: str, descriptor: str) -> tuple[str, str, Path]:
    if ":" not in descriptor:
        raise ValueError(f"latency_profile_requires_section:{descriptor}")
    section, profile = descriptor.split(":", 1)
    directory = root / section / model / profile
    result_path = directory / "baseline_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"latency_profile_result_missing:{descriptor}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok" or int(result.get("requested_realized_conflict_count", -1)) != 0:
        raise RuntimeError(f"latency_profile_not_realized:{descriptor}:{result.get('status')}")
    engine = directory / "engine.plan"
    if not engine.is_file() or not engine.stat().st_size:
        raise FileNotFoundError(f"latency_engine_missing:{descriptor}")
    return section, profile, directory


def _real_inputs(model_name: str, device: Any) -> dict[str, Any]:
    from quantization.export.heal_lidar_cobevt import prepare_cobevt_maxk_inputs
    from search.model_family.export.heal_v2xvit import (
        HealV2XViTExportPolicy,
        prepare_v2xvit_fixed_k_inputs,
    )
    from search.orchestration.lidar_transformer_h800_inventory import _load, _real_batch

    spec = MODEL_SPECS[model_name]
    bundle, _ = _load(model_name, device)
    batch = _real_batch(bundle, str(spec["config"]), device)
    if model_name == "lidar_cobevt":
        inputs = prepare_cobevt_maxk_inputs(
            batch["ego"], fixed_k=int(spec["fixed_k"]), max_cav=2
        )
    else:
        inputs = prepare_v2xvit_fixed_k_inputs(
            batch["ego"],
            policy=HealV2XViTExportPolicy(fixed_k=int(spec["fixed_k"]), max_agents=2),
        )
    del batch, bundle
    return {str(name): tensor.contiguous() for name, tensor in inputs.items()}


def _engine_counts(directory: Path) -> dict[str, Any]:
    layer_path = directory / "engine_build" / "engine_layer_info.json"
    if not layer_path.is_file():
        return {"kernel_count": None, "cast_reformat_count": None, "role_latency_status": "layer_info_missing"}
    payload = json.loads(layer_path.read_text(encoding="utf-8"))
    layers = payload.get("Layers", payload if isinstance(payload, list) else [])
    cast_reformat = sum(
        any(token in " ".join(str(row.get(key, "")) for key in ("Name", "LayerType", "Metadata")).lower()
            for token in ("cast", "reformat"))
        for row in layers
    )
    return {
        "kernel_count": len(layers),
        "cast_reformat_count": cast_reformat,
        "role_latency_status": "not_separable_after_full_engine_fusion",
        "role_latency_additive": False,
        "qkv_projection_ms": None,
        "qk_ms": None,
        "softmax_ms": None,
        "av_ms": None,
        "output_projection_ms": None,
        "ffn1_ms": None,
        "ffn2_ms": None,
    }


def _time_engine(
    engine_path: Path,
    inputs: Mapping[str, Any],
    device: Any,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    import torch
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    runner = TensorRTEngineRunner(engine_path, device)
    runner.run(dict(inputs))
    stream = runner.stream
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            if not runner.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("formal_latency_warmup_failed")
    stream.synchronize()
    repetitions = []
    all_samples: list[float] = []
    for _ in range(repeats):
        samples = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                if not runner.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("formal_latency_execute_failed")
                end.record(stream)
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        all_samples.extend(samples)
        repetitions.append(
            {
                "p50_ms": float(np.percentile(samples, 50)),
                "p90_ms": float(np.percentile(samples, 90)),
                "p95_ms": float(np.percentile(samples, 95)),
                "p99_ms": float(np.percentile(samples, 99)),
                "mean_ms": float(statistics.mean(samples)),
                "std_ms": float(statistics.pstdev(samples)),
            }
        )
    del runner
    aggregate = {
        "p50_ms": float(np.percentile(all_samples, 50)),
        "p90_ms": float(np.percentile(all_samples, 90)),
        "p95_ms": float(np.percentile(all_samples, 95)),
        "p99_ms": float(np.percentile(all_samples, 99)),
        "mean_ms": float(statistics.mean(all_samples)),
        "std_ms": float(statistics.pstdev(all_samples)),
    }
    return aggregate, repetitions


def run_latency(
    *, output_root: Path, model_name: str, descriptors: tuple[str, ...],
    physical_gpu: int, plugin: Path, isolation_seconds: int,
    warmup: int, iterations: int, repeats: int,
) -> list[dict[str, Any]]:
    isolation = audit_isolation(physical_gpu, isolation_seconds)
    destination = output_root / "latency" / model_name
    _write_json(destination / "isolation_audit.json", isolation)
    if not isolation["isolated"]:
        raise RuntimeError("formal_latency_requires_five_minute_isolated_h800")
    import torch
    from search.integration.runtime_environment import load_tensorrt_runtime

    load_tensorrt_runtime(TRT_ROOT)
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    inputs = _real_inputs(model_name, device)
    torch.cuda.empty_cache()
    baseline = "baselines:B1_TRT_ATTN_FP32"
    ordered = (baseline, *descriptors, baseline)
    rows = []
    for replay_index, descriptor in enumerate(ordered):
        section, profile, directory = _profile_directory(output_root, model_name, descriptor)
        engine = directory / "engine.plan"
        before = _gpu_telemetry(physical_gpu)
        aggregate, repetitions = _time_engine(
            engine, inputs, device, warmup=warmup, iterations=iterations, repeats=repeats
        )
        after = _gpu_telemetry(physical_gpu)
        external = [row for row in _compute_processes(physical_gpu) if int(row["pid"]) != os.getpid()]
        if external:
            raise RuntimeError(f"external_process_during_formal_latency:{external}")
        rows.append(
            {
                "model": model_name,
                "section": section,
                "profile": profile,
                "descriptor": descriptor,
                "baseline_replay": descriptor == baseline,
                "replay_index": replay_index,
                "engine_sha256": _sha256(engine),
                "gpu_physical_index": physical_gpu,
                "gpu_uuid": before.get("uuid", ""),
                "fixed_k": int(MODEL_SPECS[model_name]["fixed_k"]),
                "warmup": warmup,
                "iterations": iterations,
                "repeats": repeats,
                "device_resident": True,
                "cuda_event": True,
                "formal": True,
                "telemetry_before": before,
                "telemetry_after": after,
                "repetitions": repetitions,
                **aggregate,
                **_engine_counts(directory),
            }
        )
        torch.cuda.empty_cache()
    replay = [row for row in rows if row["baseline_replay"]]
    drift = abs(float(replay[-1]["p50_ms"]) / float(replay[0]["p50_ms"]) - 1.0)
    if drift > 0.03:
        raise RuntimeError(f"formal_latency_baseline_replay_drift:{drift}")
    for row in rows:
        row["baseline_replay_p50_drift_ratio"] = drift
    _write_json(destination / "formal_latency.json", rows)
    _write_csv(output_root / f"formal_latency_{model_name.removeprefix('lidar_')}.csv", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--profiles", required=True, help="comma-separated section:profile descriptors")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--isolation-seconds", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    rows = run_latency(
        output_root=Path(args.output_root).resolve(), model_name=args.model,
        descriptors=tuple(value.strip() for value in args.profiles.split(",") if value.strip()),
        physical_gpu=args.physical_gpu, plugin=Path(args.plugin).resolve(),
        isolation_seconds=args.isolation_seconds, warmup=args.warmup,
        iterations=args.iterations, repeats=args.repeats,
    )
    print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
