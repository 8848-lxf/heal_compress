#!/usr/bin/env python3
"""Matched-B0 formal latency for six-budget V2X-ViT S32/JMIX engines."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
BUDGETS = ("030", "025", "020", "015", "010", "005")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _external_compute_pids(physical_gpu: int) -> list[int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return sorted(
        int(line.strip())
        for line in output.splitlines()
        if line.strip().isdigit() and int(line.strip()) != os.getpid()
    )


def _idle_audit(physical_gpu: int, seconds: int) -> dict[str, Any]:
    observations = []
    start = time.monotonic()
    while True:
        pids = _external_compute_pids(physical_gpu)
        observations.append({"elapsed_seconds": time.monotonic() - start, "external_compute_pids": pids})
        if pids:
            raise RuntimeError(f"formal_latency_gpu_not_isolated:{physical_gpu}:{pids}")
        if time.monotonic() - start >= seconds:
            break
        time.sleep(min(10, seconds))
    return {"physical_gpu": physical_gpu, "required_seconds": seconds, "observations": observations, "passed": True}


def _measure(runner: Any, prepared: Any, repeat: int) -> dict[str, Any]:
    values = []
    for _ in range(500):
        _outputs, profile = runner.run_profiled(prepared)
        values.append(float(profile["execute_async_ms"]))
    tensor = torch.tensor(values)
    return {
        "repeat": repeat,
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p90_ms": float(tensor.quantile(0.90)),
        "p95_ms": float(tensor.quantile(0.95)),
        "std_ms": statistics.pstdev(values),
    }


def _aggregate(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.median(float(row["mean_ms"]) for row in repeats),
        "p50_ms": statistics.median(float(row["p50_ms"]) for row in repeats),
        "p90_ms": statistics.median(float(row["p90_ms"]) for row in repeats),
        "p95_ms": statistics.median(float(row["p95_ms"]) for row in repeats),
        "repeat_p50_ms": [float(row["p50_ms"]) for row in repeats],
        "repeat_p50_cv": statistics.pstdev(float(row["p50_ms"]) for row in repeats)
        / max(statistics.mean(float(row["p50_ms"]) for row in repeats), 1.0e-12),
        "repeats": repeats,
    }


def _measure_budget(
    b0: Any,
    s32: Any,
    jmix: Any,
    prepared: Any,
    *,
    physical_gpu: int,
) -> dict[str, Any]:
    baseline_rows = []
    s32_rows = []
    jmix_rows = []
    s32_speedups = []
    jmix_speedups = []
    sequence = []
    process_audit = []
    for repeat in range(5):
        before_pids = _external_compute_pids(physical_gpu)
        process_audit.append({"repeat": repeat, "position": "round_start", "external_compute_pids": before_pids})
        if before_pids:
            raise RuntimeError(f"formal_latency_external_process:{physical_gpu}:{before_pids}")
        before = _measure(b0, prepared, repeat)
        baseline_rows.append({**before, "position": "round_start"})
        srow = _measure(s32, prepared, repeat)
        s32_rows.append(srow)
        after_s32 = _measure(b0, prepared, repeat)
        baseline_rows.append({**after_s32, "position": "after_S32"})
        matched_s32 = statistics.median((float(before["p50_ms"]), float(after_s32["p50_ms"])))
        s32_speedups.append(matched_s32 / float(srow["p50_ms"]))
        jrow = _measure(jmix, prepared, repeat)
        jmix_rows.append(jrow)
        after_jmix = _measure(b0, prepared, repeat)
        baseline_rows.append({**after_jmix, "position": "after_JMIX"})
        matched_jmix = statistics.median((float(after_s32["p50_ms"]), float(after_jmix["p50_ms"])))
        jmix_speedups.append(matched_jmix / float(jrow["p50_ms"]))
        after_pids = _external_compute_pids(physical_gpu)
        process_audit.append({"repeat": repeat, "position": "round_end", "external_compute_pids": after_pids})
        if after_pids:
            raise RuntimeError(f"formal_latency_external_process:{physical_gpu}:{after_pids}")
        sequence.extend(("B0", "S32", "B0", "JMIX", "B0"))
    baseline_p50 = [float(row["p50_ms"]) for row in baseline_rows]
    baseline_median = statistics.median(baseline_p50)
    drift = (max(baseline_p50) - min(baseline_p50)) / max(baseline_median, 1.0e-12)
    return {
        "B0": _aggregate(baseline_rows),
        "S32": {**_aggregate(s32_rows), "speedup_vs_matched_B0": statistics.median(s32_speedups)},
        "JMIX": {**_aggregate(jmix_rows), "speedup_vs_matched_B0": statistics.median(jmix_speedups)},
        "baseline_replay_drift": drift,
        "latency_batch_valid": drift <= 0.01,
        "measurement_sequence": sequence,
        "process_audit": process_audit,
    }


def _preload_tensorrt(root: Path) -> dict[str, Any]:
    manifest_path = root / "engines/B0/engine_build/engine_build_environment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("TensorRT_version") != "10.9.0.34":
        raise RuntimeError(f"unexpected_tensorrt_version:{manifest.get('TensorRT_version')}")
    trt_root = Path(str(manifest["TensorRT_root"])).resolve()
    library_dir = trt_root / "targets/x86_64-linux-gnu/lib"
    loaded = []
    for name in ("libnvinfer.so.10", "libnvonnxparser.so.10", "libnvinfer_plugin.so.10"):
        library = library_dir / name
        if not library.is_file():
            raise RuntimeError(f"tensorrt_library_missing:{library}")
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        loaded.append(str(library))
    return {
        "environment_manifest": str(manifest_path),
        "tensorrt_root": str(trt_root),
        "tensorrt_version": manifest["TensorRT_version"],
        "loaded_libraries": loaded,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    idle = _idle_audit(args.physical_gpu, args.idle_audit_seconds)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"latency_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    root = args.output_root.resolve()
    tensorrt_provenance = _preload_tensorrt(root)
    import onnxruntime  # noqa: F401
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from adapters.heal_lidar_adapter import HEALLiDARAdapter
    from search.integration.evaluation_worker import _dataloader_worker_init, _move
    from search.model_family.export import HealV2XViTExportPolicy, prepare_v2xvit_fixed_k_inputs
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    inherited = json.loads(args.request_source.read_text(encoding="utf-8"))
    plugin = Path(inherited["plugin_path"]).resolve()
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    adapter = HEALLiDARAdapter(
        heal_repo=inherited["heal_root"],
        config={"model": {"hypes_yaml": inherited["model_config"]}},
    )
    hypes = adapter._absolutize_dataset_paths(
        yaml_utils.load_yaml(adapter._resolve_heal_path(inherited["model_config"]))
    )
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_batch_test,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=_dataloader_worker_init,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = str(manifest["evaluation_frame_ids"][0])
    split_ids = [str(value) for value in json.loads(Path(str(hypes["validate_dir"])).read_text(encoding="utf-8"))]
    prepared = None
    for index, batch in enumerate(loader):
        if split_ids[index] != wanted:
            continue
        batch = _move(batch, device)
        prepared = prepare_v2xvit_fixed_k_inputs(
            batch["ego"],
            policy=HealV2XViTExportPolicy(fixed_k=27904, max_agents=2),
        )
        break
    if prepared is None:
        raise RuntimeError("latency_reference_frame_not_found")
    b0 = TensorRTEngineRunner(root / "engines/B0/candidate.plan", device)
    for _ in range(200):
        b0.run(prepared)
    rows = []
    for budget in BUDGETS:
        audit = json.loads((root / "budgets" / budget / "build_audit.json").read_text(encoding="utf-8"))
        s32 = TensorRTEngineRunner(Path(audit["selected_s32_dir"]) / "candidate.plan", device)
        jmix = TensorRTEngineRunner(Path(audit["selected_jmix_dir"]) / "candidate.plan", device)
        for runner in (s32, jmix):
            for _ in range(200):
                runner.run(prepared)
        measured = _measure_budget(b0, s32, jmix, prepared, physical_gpu=args.physical_gpu)
        rows.append({"budget": budget, **measured})
        del s32, jmix
        torch.cuda.empty_cache()
    result = {
        "schema_version": "v2xvit-six-budget-formal-latency-v1",
        "idle_audit": idle,
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": args.gpu_uuid,
        "tensorrt_provenance": tensorrt_provenance,
        "warmup_iterations": 200,
        "timed_iterations": 500,
        "candidate_repeats": 5,
        "device_resident": True,
        "scope": "TensorRT execute_async_ms only",
        "rows": rows,
        "all_batches_valid": all(row["latency_batch_valid"] for row in rows),
    }
    _write(root / "reports/six_budget_latency_raw.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--idle-audit-seconds", type=int, default=300)
    result = run(parser.parse_args())
    return 0 if result["all_batches_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
