#!/usr/bin/env python3
"""Matched B0 pre/post formal latency for six exact Greedy winners."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


LABELS = ("030", "025", "020", "015", "010", "005")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def gpu_telemetry(gpu_uuid: str) -> dict[str, Any]:
    fields = "uuid,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used"
    completed = subprocess.run(
        ["nvidia-smi", f"--id={gpu_uuid}", f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    return dict(zip(fields.split(","), values))


def measure(runner: Any, prepared: Any, repeat: int, phase: str) -> dict[str, Any]:
    values = []
    for _ in range(500):
        _outputs, profile = runner.run_profiled(prepared)
        values.append(float(profile["execute_async_ms"]))
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "repeat": repeat,
        "phase": phase,
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
        raise RuntimeError(f"six_budget_latency_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    import onnxruntime  # noqa: F401
    from adapters.heal_lidar_adapter import HEALLiDARAdapter
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from search.integration.evaluation_worker import _dataloader_worker_init, _move
    from search.model_family.export import (
        HealV2XViTExportPolicy,
        prepare_v2xvit_fixed_k_inputs,
    )
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    inherited = json.loads(args.request_json.read_text(encoding="utf-8"))
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
    split_ids = [
        str(value)
        for value in json.loads(Path(str(hypes["validate_dir"])).read_text())
    ]
    prepared = None
    for index, batch in enumerate(loader):
        if split_ids[index] != wanted:
            continue
        batch = _move(batch, device)
        policy = HealV2XViTExportPolicy(
            fixed_k=int(inherited["fixed_k"]),
            max_agents=int(inherited["max_agents"]),
        )
        prepared = prepare_v2xvit_fixed_k_inputs(batch["ego"], policy=policy)
        break
    if prepared is None:
        raise RuntimeError("latency_reference_frame_missing")

    root = args.output_root.resolve()
    b0_path = root / "engines/greedy_exact_winners/B0/candidate.plan"
    report_path = root / "reports/six_budget_latency.json"
    existing = json.loads(report_path.read_text()) if report_path.is_file() else {}
    results = dict(existing.get("budgets") or {})
    selected_labels = tuple(
        value.strip() for value in str(args.labels).split(",") if value.strip()
    )
    if not selected_labels or any(value not in LABELS for value in selected_labels):
        raise ValueError(f"invalid_latency_labels:{selected_labels}")
    for label in selected_labels:
        before_telemetry = gpu_telemetry(args.gpu_uuid)
        paths = {
            "B0": b0_path,
            "S32": root / f"engines/greedy_exact_winners/budget_{label}/S32/candidate.plan",
            "JMIX-FRESH": root / f"engines/greedy_exact_winners/budget_{label}/JMIX-FRESH/candidate.plan",
        }
        if not all(path.is_file() for path in paths.values()):
            raise RuntimeError(f"latency_engine_missing:budget_{label}")
        runners = {
            name: TensorRTEngineRunner(path, device) for name, path in paths.items()
        }
        # Establish a stable clock/thermal state before the protocol's 200
        # control-specific warmups.  These unmeasured B0 executions are reported
        # separately and never included in forward latency.
        for _ in range(int(args.stabilization_warmup)):
            runners["B0"].run(prepared)
        torch.cuda.synchronize(device)
        for runner in runners.values():
            for _ in range(200):
                runner.run(prepared)
        torch.cuda.synchronize(device)
        prior = results.get(label) or {}
        prior_attempts = list(prior.get("attempt_history") or prior.get("attempts") or [])
        attempts = []
        accepted = None
        for attempt in range(1, 4):
            rows = {name: [] for name in runners}
            drift = []
            for repeat in range(5):
                pre = measure(runners["B0"], prepared, repeat, "baseline_pre")
                rows["B0"].append(pre)
                rows["S32"].append(
                    measure(runners["S32"], prepared, repeat, "candidate")
                )
                rows["JMIX-FRESH"].append(
                    measure(
                        runners["JMIX-FRESH"], prepared, repeat, "candidate"
                    )
                )
                post = measure(runners["B0"], prepared, repeat, "baseline_post")
                rows["B0"].append(post)
                drift.append(
                    {
                        "repeat": repeat,
                        "pre_p50_ms": pre["p50_ms"],
                        "post_p50_ms": post["p50_ms"],
                        "relative_drift": (post["p50_ms"] - pre["p50_ms"])
                        / pre["p50_ms"],
                    }
                )
            valid = max(abs(row["relative_drift"]) for row in drift) <= 0.01
            attempt_row = {"attempt": attempt, "valid": valid, "drift": drift, "rows": rows}
            attempts.append(attempt_row)
            if valid:
                accepted = attempt_row
                break
        if accepted is None:
            results[label] = {
                "status": "invalid_baseline_replay_drift",
                "attempt_history": [*prior_attempts, *attempts],
                "gpu_telemetry_before": before_telemetry,
                "gpu_telemetry_after": gpu_telemetry(args.gpu_uuid),
            }
            print(json.dumps({"budget": label, "status": "invalid_drift"}), flush=True)
            write(report_path, {"gpu_uuid": args.gpu_uuid, "budgets": results})
            del runners
            torch.cuda.empty_cache()
            continue
        controls = {}
        for name, path in paths.items():
            p50_values = [row["p50_ms"] for row in accepted["rows"][name]]
            controls[name] = {
                "engine": str(path),
                "engine_sha256": sha256(path),
                "engine_size_bytes": path.stat().st_size,
                "forward_p50_ms": statistics.median(p50_values),
                "fps": 1000.0 / statistics.median(p50_values),
                "repeat_rows": accepted["rows"][name],
            }
        baseline = controls["B0"]["forward_p50_ms"]
        for name in ("S32", "JMIX-FRESH"):
            controls[name]["speedup_p50_vs_B0"] = (
                baseline / controls[name]["forward_p50_ms"]
            )
        results[label] = {
            "status": "ok",
            "accepted_attempt": accepted["attempt"],
            "attempt_history": [*prior_attempts, *attempts],
            "baseline_replay_drift": accepted["drift"],
            "controls": controls,
            "gpu_uuid": args.gpu_uuid,
            "gpu_telemetry_before": before_telemetry,
            "gpu_telemetry_after": gpu_telemetry(args.gpu_uuid),
            "protocol": {
                "warmup": 200,
                "unmeasured_b0_stabilization_warmup": int(args.stabilization_warmup),
                "timed_iterations": 500,
                "repeats": 5,
                "scope": "TensorRT execute_async_ms only",
                "serial": True,
            },
        }
        print(
            json.dumps(
                {
                    "budget": label,
                    "B0": controls["B0"]["forward_p50_ms"],
                    "S32": controls["S32"]["forward_p50_ms"],
                    "JMIX": controls["JMIX-FRESH"]["forward_p50_ms"],
                    "S32_speedup": controls["S32"]["speedup_p50_vs_B0"],
                    "JMIX_speedup": controls["JMIX-FRESH"]["speedup_p50_vs_B0"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        write(report_path, {"gpu_uuid": args.gpu_uuid, "budgets": results})
        del runners
        torch.cuda.empty_cache()
    write(
        report_path,
        {"gpu_uuid": args.gpu_uuid, "budgets": results},
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--labels", default=",".join(LABELS))
    parser.add_argument("--stabilization-warmup", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
