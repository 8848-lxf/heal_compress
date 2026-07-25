#!/usr/bin/env python3
"""Serial 200-warmup/500x5 TensorRT forward latency for three controls."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(runner: Any, prepared: Any, repeat: int, phase: str) -> dict[str, Any]:
    values = []
    for _ in range(500):
        _outputs, profile = runner.run_profiled(prepared)
        values.append(float(profile["execute_async_ms"]))
    return {
        "repeat": repeat,
        "phase": phase,
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p90_ms": float(torch.tensor(values).quantile(0.90)),
        "p95_ms": float(torch.tensor(values).quantile(0.95)),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values),
    }


def run(args: argparse.Namespace) -> int:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"latency_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    import onnxruntime  # noqa: F401
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from adapters.heal_lidar_adapter import HEALLiDARAdapter
    from search.integration.evaluation_worker import _dataloader_worker_init, _move
    from search.model_family.export import HealV2XViTExportPolicy, prepare_v2xvit_fixed_k_inputs
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner
    inherited = json.loads(args.request_source.read_text(encoding="utf-8"))
    plugin_path = Path(inherited["plugin_path"]).resolve()
    if not plugin_path.is_file():
        raise RuntimeError(f"latency_plugin_missing:{plugin_path}")
    ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    adapter = HEALLiDARAdapter(heal_repo=inherited["heal_root"], config={"model": {"hypes_yaml": inherited["model_config"]}})
    hypes = adapter._absolutize_dataset_paths(yaml_utils.load_yaml(adapter._resolve_heal_path(inherited["model_config"])))
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_batch_test, pin_memory=True, prefetch_factor=2, persistent_workers=True, worker_init_fn=_dataloader_worker_init)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = set(str(x) for x in manifest["evaluation_frame_ids"][:1])
    split_ids = [str(x) for x in json.loads(Path(str(hypes["validate_dir"])).read_text(encoding="utf-8"))]
    prepared = None
    for index, batch in enumerate(loader):
        if split_ids[index] not in wanted:
            continue
        batch = _move(batch, device)
        policy = HealV2XViTExportPolicy(fixed_k=int(inherited["fixed_k"]), max_agents=int(inherited["max_agents"]))
        prepared = prepare_v2xvit_fixed_k_inputs(batch["ego"], policy=policy)
        break
    if prepared is None:
        raise RuntimeError("latency_reference_frame_not_found")
    results = {}
    runners = {}
    for control in args.controls:
        engine = args.output_root / "engines" / control / "candidate.plan"
        runner = TensorRTEngineRunner(engine, device)
        for _ in range(200):
            runner.run(prepared)
        torch.cuda.synchronize(device)
        runners[control] = runner
        results[control] = {
            "engine": str(engine),
            "engine_sha256": sha256(engine),
            "engine_size_bytes": engine.stat().st_size,
            "repeats": [],
            "gpu_uuid": args.gpu_uuid,
            "warmup_iterations": 200,
            "timed_iterations": 500,
            "repeat_count": 5,
            "scope": "TensorRT execute_async_ms only",
            "allocation_report": runner.allocation_report(),
        }
    # Each candidate measurement is bracketed by a baseline replay.  This is
    # deliberately serial: no engines execute concurrently on the device.
    baseline_drift = []
    candidates = [name for name in args.controls if name != "B0"]
    for repeat in range(5):
        pre = measure(runners["B0"], prepared, repeat, "baseline_pre")
        results["B0"]["repeats"].append(pre)
        for name in candidates:
            results[name]["repeats"].append(measure(runners[name], prepared, repeat, "candidate"))
        post = measure(runners["B0"], prepared, repeat, "baseline_post")
        results["B0"]["repeats"].append(post)
        baseline_drift.append({
            "repeat": repeat,
            "pre_p50_ms": pre["p50_ms"],
            "post_p50_ms": post["p50_ms"],
            "relative_drift": (post["p50_ms"] - pre["p50_ms"]) / pre["p50_ms"],
        })
    for control, row in results.items():
        all_values = [item["p50_ms"] for item in row["repeats"]]
        row["repeat_p50_ms"] = statistics.median(all_values)
        row["forward_p50_ms"] = statistics.median(all_values)
        row["fps"] = 1000.0 / row["forward_p50_ms"]
    b0 = results["B0"]["forward_p50_ms"]
    for name in args.controls:
        if name != "B0":
            results[name]["speedup_p50_vs_B0"] = b0 / results[name]["forward_p50_ms"]
    write(args.output_root / "reports" / args.report_name, {"protocol": {"warmup": 200, "timed": 500, "repeats": 5, "scope": "pure TensorRT execute_async_ms", "gpu_uuid": args.gpu_uuid, "serial_order": ["B0", *candidates, "B0_replay"]}, "baseline_replay_drift": baseline_drift, "controls": results})
    print(json.dumps({key: row["forward_p50_ms"] for key, row in results.items()}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--controls", nargs="+", choices=("B0", "S32", "JMIX-FRESH"), default=("B0", "S32", "JMIX-FRESH"))
    parser.add_argument("--report-name", default="latency_results.json")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
