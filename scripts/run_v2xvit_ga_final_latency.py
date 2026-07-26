#!/usr/bin/env python3
"""Matched B0 pre/post formal latency for Greedy versus final GA winners."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_v2xvit_six_budget_latency import gpu_telemetry, measure, sha256, write


def load_budget_summary(root: Path, label: str) -> dict[str, Any]:
    """Load one completed formal-budget summary without consulting stale aggregates."""
    summary_path = root / f"ga/budget_{label}/seed_0/budget_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"formal_budget_summary_missing:{label}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "greedy_anchor" not in summary or "final_winner" not in summary:
        raise RuntimeError(f"formal_budget_summary_incomplete:{label}")
    return summary


def run(args: argparse.Namespace) -> int:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("ga_final_latency_requires_one_visible_gpu")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    import onnxruntime  # noqa: F401
    from adapters.heal_lidar_adapter import HEALLiDARAdapter
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from search.integration.evaluation_worker import _dataloader_worker_init, _move
    from search.model_family.export import HealV2XViTExportPolicy, prepare_v2xvit_fixed_k_inputs
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    root = args.output_root.resolve()
    request_path = args.request_json or (
        root / "evaluation_fixed500/B0/evaluation_request.json"
    )
    request = json.loads(Path(request_path).read_text())
    ctypes.CDLL(str(Path(request["plugin_path"])), mode=ctypes.RTLD_GLOBAL)
    adapter = HEALLiDARAdapter(
        heal_repo=request["heal_root"], config={"model": {"hypes_yaml": request["model_config"]}}
    )
    hypes = adapter._absolutize_dataset_paths(yaml_utils.load_yaml(adapter._resolve_heal_path(request["model_config"])))
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
                        collate_fn=dataset.collate_batch_test, pin_memory=True,
                        prefetch_factor=2, persistent_workers=True,
                        worker_init_fn=_dataloader_worker_init)
    manifest = json.loads(args.manifest.read_text())
    wanted = str(manifest["evaluation_frame_ids"][0])
    split_ids = [str(value) for value in json.loads(Path(str(hypes["validate_dir"])).read_text())]
    prepared = None
    for index, batch in enumerate(loader):
        if split_ids[index] != wanted:
            continue
        batch = _move(batch, device)
        prepared = prepare_v2xvit_fixed_k_inputs(
            batch["ego"], policy=HealV2XViTExportPolicy(
                fixed_k=int(request["fixed_k"]), max_agents=int(request["max_agents"])
            )
        )
        break
    if prepared is None:
        raise RuntimeError("ga_final_latency_frame_missing")

    output_path = args.output_json or (
        root / "reports/greedy_vs_ga_latency.json"
    )
    results = json.loads(output_path.read_text()).get("budgets", {}) if output_path.is_file() else {}
    labels = tuple(value.strip() for value in args.labels.split(",") if value.strip())
    b0 = args.b0_engine or (
        root / "engines/greedy_exact_winners/B0/candidate.plan"
    )
    for label in labels:
        budget = load_budget_summary(root, label)
        greedy_hash = budget["greedy_anchor"]["complete_phenotype_hash"]
        ga = budget["final_winner"]
        greedy_path = Path(budget["greedy_anchor"]["metadata"]["engine_path"])
        ga_path = (
            greedy_path
            if ga["complete_phenotype_hash"] == greedy_hash
            else Path(ga["metadata"]["engine_path"])
        )
        paths = {
            "B0": b0,
            "Greedy": greedy_path,
            "GA-final": ga_path,
        }
        unique_paths = {str(path): path for path in paths.values()}
        path_runners = {key: TensorRTEngineRunner(path, device) for key, path in unique_paths.items()}
        runners = {name: path_runners[str(path)] for name, path in paths.items()}
        for runner in path_runners.values():
            for _ in range(200):
                runner.run(prepared)
        torch.cuda.synchronize(device)
        attempts = []
        accepted = None
        for attempt in range(1, 4):
            rows = {name: [] for name in paths}
            drift = []
            for repeat in range(5):
                pre = measure(runners["B0"], prepared, repeat, "baseline_pre")
                rows["B0"].append(pre)
                rows["Greedy"].append(measure(runners["Greedy"], prepared, repeat, "candidate"))
                if paths["GA-final"].resolve() == paths["Greedy"].resolve():
                    rows["GA-final"].append(dict(rows["Greedy"][-1]))
                else:
                    rows["GA-final"].append(measure(runners["GA-final"], prepared, repeat, "candidate"))
                post = measure(runners["B0"], prepared, repeat, "baseline_post")
                rows["B0"].append(post)
                drift.append({"repeat": repeat, "pre_p50_ms": pre["p50_ms"],
                              "post_p50_ms": post["p50_ms"],
                              "relative_drift": (post["p50_ms"] - pre["p50_ms"]) / pre["p50_ms"]})
            attempt_row = {"attempt": attempt, "rows": rows, "drift": drift,
                           "valid": max(abs(item["relative_drift"]) for item in drift) <= 0.01}
            attempts.append(attempt_row)
            if attempt_row["valid"]:
                accepted = attempt_row
                break
        if accepted is None:
            results[label] = {"status": "invalid_baseline_replay_drift", "attempts": attempts}
            write(output_path, {"gpu_uuid": args.gpu_uuid, "budgets": results})
            continue
        controls = {}
        baseline_p50 = statistics.median(row["p50_ms"] for row in accepted["rows"]["B0"])
        for name, path in paths.items():
            p50 = statistics.median(row["p50_ms"] for row in accepted["rows"][name])
            controls[name] = {
                "candidate_hash": greedy_hash if name == "Greedy" else (
                    ga["complete_phenotype_hash"] if name == "GA-final" else "B0"),
                "engine": str(path), "engine_sha256": sha256(path),
                "engine_size_bytes": path.stat().st_size,
                "p50_ms": p50, "fps": 1000.0 / p50,
                "speedup_vs_B0": baseline_p50 / p50,
                "repeat_rows": accepted["rows"][name],
            }
        results[label] = {
            "status": "ok", "budget": budget["budget"], "controls": controls,
            "attempts": attempts, "baseline_replay_drift": accepted["drift"],
            "gpu_uuid": args.gpu_uuid, "gpu_telemetry": gpu_telemetry(args.gpu_uuid),
            "protocol": {"warmup": 200, "timed": 500, "repeats": 5,
                         "baseline_pre_post": True, "drift_limit": 0.01},
        }
        write(output_path, {"gpu_uuid": args.gpu_uuid, "budgets": results})
        del runners, path_runners
        torch.cuda.empty_cache()
    rows = []
    for label, result in results.items():
        if result.get("status") != "ok":
            continue
        for control, metric in result["controls"].items():
            rows.append({"budget": result["budget"], "control": control,
                         "candidate_hash": metric["candidate_hash"], "p50_ms": metric["p50_ms"],
                         "fps": metric["fps"], "speedup_vs_B0": metric["speedup_vs_B0"]})
    if rows:
        with (root / "reports/greedy_vs_ga_latency.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--b0-engine", type=Path)
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
