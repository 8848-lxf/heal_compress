#!/usr/bin/env python3
"""Serial 200-warmup/500x5 TensorRT forward latency for three controls."""

from __future__ import annotations

import argparse
import ctypes
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


def _measure_repeat(runner: Any, prepared: Any, repeat: int) -> dict[str, float | int]:
    values = []
    for _ in range(500):
        _outputs, profile = runner.run_profiled(prepared)
        values.append(float(profile["execute_async_ms"]))
    tensor = torch.tensor(values)
    return {
        "repeat": int(repeat),
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p90_ms": float(tensor.quantile(0.90)),
        "p95_ms": float(tensor.quantile(0.95)),
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
    runners = {}
    for control in args.controls:
        engine = args.output_root / "engines" / control / "candidate.plan"
        runner = TensorRTEngineRunner(engine, device)
        for _ in range(200):
            runner.run(prepared)
        runners[control] = {"runner": runner, "engine": engine}
    torch.cuda.synchronize(device)
    if "B0" not in runners:
        raise RuntimeError("latency_matched_replay_requires_B0")

    baseline_replays = []
    candidate_repeats = {name: [] for name in args.controls if name != "B0"}
    matched_speedups = {name: [] for name in candidate_repeats}
    sequence = []
    for repeat in range(5):
        baseline = _measure_repeat(runners["B0"]["runner"], prepared, repeat)
        baseline["position"] = "round_start"
        baseline_replays.append(baseline)
        sequence.append({"round": repeat, "control": "B0", "position": "round_start"})
        for name in candidate_repeats:
            candidate = _measure_repeat(runners[name]["runner"], prepared, repeat)
            candidate_repeats[name].append(candidate)
            sequence.append({"round": repeat, "control": name, "position": "candidate"})
            replay = _measure_repeat(runners["B0"]["runner"], prepared, repeat)
            replay["position"] = f"after_{name}"
            baseline_replays.append(replay)
            sequence.append({"round": repeat, "control": "B0", "position": f"after_{name}"})
            matched = statistics.median(
                [float(baseline["p50_ms"]), float(replay["p50_ms"])]
            )
            matched_speedups[name].append(matched / float(candidate["p50_ms"]))
            baseline = replay

    results = {}
    baseline_values = [float(row["p50_ms"]) for row in baseline_replays]
    baseline_p50 = statistics.median(baseline_values)
    baseline_drift = (
        max(baseline_values) - min(baseline_values)
    ) / max(baseline_p50, 1.0e-12)
    results["B0"] = {
        "engine": str(runners["B0"]["engine"]),
        "repeats": baseline_replays,
        "repeat_p50_ms": baseline_p50,
        "forward_p50_ms": baseline_p50,
        "fps": 1000.0 / baseline_p50,
        "baseline_replay_drift": baseline_drift,
        "gpu_uuid": args.gpu_uuid,
        "warmup_iterations": 200,
        "timed_iterations": 500,
        "repeat_count": len(baseline_replays),
        "scope": "TensorRT execute_async_ms only",
        "allocation_report": runners["B0"]["runner"].allocation_report(),
    }
    for name, repeats in candidate_repeats.items():
        values = [float(row["p50_ms"]) for row in repeats]
        p50 = statistics.median(values)
        results[name] = {
            "engine": str(runners[name]["engine"]),
            "repeats": repeats,
            "repeat_p50_ms": p50,
            "forward_p50_ms": p50,
            "fps": 1000.0 / p50,
            "speedup_p50_vs_matched_B0": statistics.median(matched_speedups[name]),
            "matched_speedup_repeats": matched_speedups[name],
            "gpu_uuid": args.gpu_uuid,
            "warmup_iterations": 200,
            "timed_iterations": 500,
            "repeat_count": 5,
            "scope": "TensorRT execute_async_ms only",
            "allocation_report": runners[name]["runner"].allocation_report(),
        }
    write(args.output_root / "reports" / args.report_name, {"protocol": {"warmup": 200, "timed": 500, "candidate_repeats": 5, "baseline_replay": "B0_before_and_after_each_candidate", "measurement_sequence": sequence, "scope": "pure TensorRT execute_async_ms", "gpu_uuid": args.gpu_uuid}, "controls": results})
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
