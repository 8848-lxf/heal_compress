#!/usr/bin/env python3
"""Wait for all six formal budgets and fixed500 results, then run isolated latency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


LABELS = ("030", "025", "020", "015", "010", "005")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluation_path(root: Path, label: str, control: str, candidate_hash: str) -> Path:
    return root / f"{control}_final_fixed500_partial/budget_{label}/{candidate_hash}/evaluation.json"


def completion_state(root: Path) -> dict[str, Any]:
    """Return a fail-closed readiness audit for six summaries and twelve evaluations."""
    state: dict[str, Any] = {"ready": True, "budgets": {}, "failures": []}
    for label in LABELS:
        failure = root / f"ga/budget_{label}/failure.json"
        summary_path = root / f"ga/budget_{label}/seed_0/budget_summary.json"
        row: dict[str, Any] = {"summary": str(summary_path), "ready": False}
        if failure.is_file():
            row["failure"] = json.loads(failure.read_text(encoding="utf-8"))
            state["failures"].append(f"formal_budget_failed:{label}")
            state["ready"] = False
            state["budgets"][label] = row
            continue
        if not summary_path.is_file():
            state["ready"] = False
            state["budgets"][label] = row
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        try:
            hashes = {
                "greedy": summary["greedy_anchor"]["complete_phenotype_hash"],
                "ga": summary["final_winner"]["complete_phenotype_hash"],
            }
        except KeyError as exc:
            state["failures"].append(f"formal_budget_summary_incomplete:{label}:{exc}")
            state["ready"] = False
            state["budgets"][label] = row
            continue
        row["candidate_hashes"] = hashes
        row["evaluations"] = {}
        row["ready"] = True
        for control, candidate_hash in hashes.items():
            path = _evaluation_path(root, label, control, candidate_hash)
            evaluation_ready = False
            if path.is_file():
                metric = json.loads(path.read_text(encoding="utf-8"))
                evaluation_ready = (
                    metric.get("status") == "ok"
                    and metric.get("num_evaluated_frames") == 500
                    and metric.get("num_skipped_frames") == 0
                )
                if not evaluation_ready:
                    state["failures"].append(f"fixed500_invalid:{label}:{control}")
            row["evaluations"][control] = {"path": str(path), "ready": evaluation_ready}
            row["ready"] = row["ready"] and evaluation_ready
        state["ready"] = state["ready"] and row["ready"]
        state["budgets"][label] = row
    return state


def gpu_compute_processes(gpu_uuid: str) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 4 and fields[0] == gpu_uuid:
            rows.append({"gpu_uuid": fields[0], "pid": fields[1], "process_name": fields[2],
                         "used_memory_mib": fields[3]})
    return rows


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    status_path = root / "reports/v2xvit_six_budget_final_latency_watcher.json"
    while True:
        state = completion_state(root)
        state["timestamp"] = time.time()
        _write(status_path, state)
        if state["failures"]:
            return 2
        if state["ready"]:
            break
        time.sleep(args.poll_seconds)

    idle_checks = 0
    while idle_checks < args.idle_checks_required:
        processes = gpu_compute_processes(args.gpu_uuid)
        if processes:
            idle_checks = 0
        else:
            idle_checks += 1
        _write(status_path, {
            **completion_state(root), "timestamp": time.time(), "gpu_uuid": args.gpu_uuid,
            "gpu_processes": processes, "idle_checks": idle_checks,
            "idle_checks_required": args.idle_checks_required,
        })
        if idle_checks < args.idle_checks_required:
            time.sleep(args.poll_seconds)

    output_json = root / "reports/v2xvit_six_budget_final_latency.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_v2xvit_ga_final_latency.py")),
        "--output-root", str(root),
        "--manifest", str(args.manifest),
        "--gpu-uuid", args.gpu_uuid,
        "--labels", ",".join(LABELS),
        "--b0-engine", str(args.b0_engine),
        "--request-json", str(args.request_json),
        "--output-json", str(output_json),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    log_path = root / "logs/v2xvit_six_budget_final_latency.log"
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    final = completion_state(root)
    final.update({
        "timestamp": time.time(), "gpu_uuid": args.gpu_uuid, "physical_gpu": args.physical_gpu,
        "latency_command": command, "latency_returncode": result.returncode,
        "latency_output": str(output_json), "status": "complete" if result.returncode == 0 else "failed",
    })
    _write(status_path, final)
    if result.returncode == 0:
        summary_script = Path(__file__).with_name("summarize_v2xvit_six_budget_results.py")
        subprocess.run(
            [sys.executable, str(summary_script), "--output-root", str(root), "--require-complete"],
            env=env,
            check=True,
        )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--b0-engine", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--idle-checks-required", type=int, default=10)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
