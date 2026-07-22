"""Isolated full-engine latency with same-family/profile D0 replay baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

from search.model_families.transformer.dh_alignment_audit import latency_beneficial
from search.orchestration.lidar_transformer_h800_latency import (
    _compute_processes,
    _engine_counts,
    _gpu_telemetry,
    _real_inputs,
    _time_engine,
    audit_isolation,
)


TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    *, output_root: Path, model: str, family: str, profile: str,
    widths: tuple[int, ...], original_d_h: int, physical_gpu: int,
    plugin: Path, isolation_seconds: int = 300, warmup: int = 200,
    iterations: int = 2000, repeats: int = 5,
) -> list[dict[str, Any]]:
    isolation = audit_isolation(physical_gpu, isolation_seconds)
    destination = output_root / "latency" / model / family / profile
    _write_json(destination / "isolation_audit.json", isolation)
    if not isolation["isolated"]:
        raise RuntimeError("formal_latency_requires_five_minute_isolated_h800")
    import ctypes
    import torch
    from search.integration.runtime_environment import load_tensorrt_runtime, runtime_cuda_index_for_physical

    load_tensorrt_runtime(TRT_ROOT)
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime)
    device = torch.device(f"cuda:{runtime}")
    inputs = _real_inputs(model, device)
    order = (int(original_d_h), *tuple(int(value) for value in widths if int(value) != int(original_d_h)), int(original_d_h))
    rows: list[dict[str, Any]] = []
    for index, d_h in enumerate(order):
        directory = output_root / "engines" / model / family / f"dh_{d_h:03d}" / profile
        acceptance = json.loads((directory / "baseline_result.json").read_text(encoding="utf-8"))
        fixed500 = json.loads((directory / "evaluation" / "fixed500" / "evaluation_acceptance.json").read_text(encoding="utf-8"))
        if acceptance.get("status") != "ok" or fixed500.get("status") != "ok":
            raise RuntimeError(f"formal_latency_candidate_not_accepted:{d_h}")
        engine = directory / "engine.plan"
        before = _gpu_telemetry(physical_gpu)
        aggregate, repetition_rows = _time_engine(
            engine, inputs, device, warmup=warmup, iterations=iterations, repeats=repeats
        )
        after = _gpu_telemetry(physical_gpu)
        external = [row for row in _compute_processes(physical_gpu) if int(row["pid"]) != os.getpid()]
        if external:
            raise RuntimeError(f"external_process_during_formal_latency:{external}")
        repeat_p50 = [float(row["p50_ms"]) for row in repetition_rows]
        cv = statistics.pstdev(repeat_p50) / statistics.mean(repeat_p50) if statistics.mean(repeat_p50) else 0.0
        row = {
            "model": model,
            "attention_family": family,
            "profile": profile,
            "d_h": d_h,
            "baseline_replay": d_h == int(original_d_h),
            "replay_index": index,
            "engine_sha256": _sha256(engine),
            "gpu_physical_index": physical_gpu,
            "gpu_uuid": before.get("uuid", ""),
            "warmup": warmup,
            "iterations": iterations,
            "repeats": repeats,
            "repeat_p50_cv": cv,
            "device_resident": True,
            "cuda_event": True,
            "formal": True,
            "telemetry_before": before,
            "telemetry_after": after,
            "repetitions": repetition_rows,
            **aggregate,
            **_engine_counts(directory),
        }
        rows.append(row)
        _write_json(destination / "formal_latency_checkpoint.json", rows)
        torch.cuda.empty_cache()
    replay = [row for row in rows if row["baseline_replay"]]
    replay_drift = abs(float(replay[-1]["p50_ms"]) / float(replay[0]["p50_ms"]) - 1.0)
    if replay_drift > 0.03:
        raise RuntimeError(f"formal_latency_baseline_replay_drift:{replay_drift}")
    baseline = replay[0]
    for row in rows:
        row["baseline_replay_p50_drift_ratio"] = replay_drift
        row["speedup_profile"] = float(baseline["p50_ms"]) / float(row["p50_ms"])
        row.update(latency_beneficial(
            baseline_p50_ms=float(baseline["p50_ms"]),
            candidate_p50_ms=float(row["p50_ms"]),
            baseline_repeat_cv=float(baseline["repeat_p50_cv"]),
            baseline_replay_drift=replay_drift,
        ))
    by_width = {int(row["d_h"]): row for row in rows if not (row["baseline_replay"] and int(row["replay_index"]) == len(rows) - 1)}
    for width, row in by_width.items():
        neighbor = by_width.get(width + 1)
        row["speedup_neighbor"] = float(neighbor["p50_ms"]) / float(row["p50_ms"]) if neighbor else None
    _write_json(destination / "formal_latency.json", rows)
    _write_csv(destination / "formal_latency.csv", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--profile", choices=("P32", "P16", "P8"), required=True)
    parser.add_argument("--widths", required=True)
    parser.add_argument("--original-dh", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--isolation-seconds", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    rows = run(
        output_root=Path(args.output_root).resolve(), model=args.model,
        family=args.family, profile=args.profile,
        widths=tuple(int(value) for value in args.widths.split(",") if value),
        original_d_h=args.original_dh, physical_gpu=args.physical_gpu,
        plugin=Path(args.plugin).resolve(), isolation_seconds=args.isolation_seconds,
        warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
    )
    print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
