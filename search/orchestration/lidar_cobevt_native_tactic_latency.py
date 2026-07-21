#!/usr/bin/env python3
"""Device-resident formal latency for full CoBEVT TensorRT engines."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _telemetry() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _load_real_input(args: argparse.Namespace, device: Any) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from adapters.heal_lidar_adapter import HEALLiDARAdapter
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from quantization.export.heal_lidar_cobevt import prepare_cobevt_maxk_inputs
    from search.integration.evaluation_worker import _move

    manifest = json.loads(Path(args.manifest).read_text())
    frame_ids = [str(value) for value in manifest.get("evaluation_frame_ids", [])]
    if not frame_ids:
        raise RuntimeError("formal_latency_manifest_has_no_evaluation_frames")
    target = frame_ids[0]
    adapter = HEALLiDARAdapter(
        heal_repo=args.heal_root,
        config={"model": {"hypes_yaml": args.model_config}},
    )
    config_path = adapter._resolve_heal_path(args.model_config)
    hypes = adapter._absolutize_dataset_paths(yaml_utils.load_yaml(config_path))
    split_ids = [str(value) for value in json.loads(Path(hypes["validate_dir"]).read_text())]
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_batch_test,
        persistent_workers=True,
        prefetch_factor=2,
    )
    for index, batch in enumerate(loader):
        if index >= len(split_ids):
            break
        if split_ids[index] != target:
            continue
        batch = _move(batch, device)
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        return prepare_cobevt_maxk_inputs(ego, fixed_k=29696, max_cav=2)
    raise RuntimeError(f"formal_latency_frame_not_found:{target}")


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ctypes.CDLL(args.plugin_path, mode=ctypes.RTLD_GLOBAL)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    inputs = _load_real_input(args, device)
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    runner = TensorRTEngineRunner(args.engine, device)
    # The first call allocates, copies and binds stable device buffers.  Timed
    # execution below only launches the already-bound TensorRT context.
    runner.run(inputs)
    stream = runner.stream
    for _ in range(int(args.warmup)):
        if not runner.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("formal_latency_warmup_execute_failed")
    stream.synchronize()
    before = _telemetry()
    rounds: list[dict[str, Any]] = []
    all_samples: list[float] = []
    for repeat in range(int(args.repeats)):
        events = []
        for _ in range(int(args.iterations)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            if not runner.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("formal_latency_execute_failed")
            end.record(stream)
            events.append((start, end))
        events[-1][1].synchronize()
        values = [float(start.elapsed_time(end)) for start, end in events]
        all_samples.extend(values)
        rounds.append(
            {
                "repeat": repeat,
                "count": len(values),
                "p50_ms": _percentile(values, 50),
                "p90_ms": _percentile(values, 90),
                "p95_ms": _percentile(values, 95),
                "p99_ms": _percentile(values, 99),
                "mean_ms": float(statistics.fmean(values)),
                "std_ms": float(statistics.pstdev(values)),
            }
        )
    after = _telemetry()
    aggregate = {
        "p50_ms": _percentile(all_samples, 50),
        "p90_ms": _percentile(all_samples, 90),
        "p95_ms": _percentile(all_samples, 95),
        "p99_ms": _percentile(all_samples, 99),
        "mean_ms": float(statistics.fmean(all_samples)),
        "std_ms": float(statistics.pstdev(all_samples)),
    }
    report = {
        "status": "ok",
        "profile": args.profile,
        "physical_gpu": int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1")),
        "scope": "full_engine_forward_only_device_resident_cuda_event",
        "warmup_iterations": int(args.warmup),
        "timed_iterations_per_repeat": int(args.iterations),
        "repeats": int(args.repeats),
        "rounds": rounds,
        "aggregate": aggregate,
        "telemetry_before": before,
        "telemetry_after": after,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--heal-root", required=True)
    parser.add_argument("--plugin-path", required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = run_worker(args)
    print(json.dumps({"profile": result["profile"], **result["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
