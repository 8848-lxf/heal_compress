"""Isolated one-GPU formal latency replay for full-validation deployments."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..anchors.joint_taylor_sweep import assert_formal_latency_isolation
from ..hashing import canonical_json_hash


def query_active_deployment_process_commands() -> list[str]:
    """Report foreign deployment workers while excluding this process tree."""

    excluded = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except (FileNotFoundError, IndexError, ValueError):
            break
    completed = subprocess.run(
        [
            "pgrep",
            "-af",
            "candidate_worker|stage2_process_pool|evaluation_worker|calibrat|trt_build_worker|trtexec",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )
    commands = []
    for line in completed.stdout.splitlines():
        head, _, command = line.partition(" ")
        try:
            process_id = int(head)
        except ValueError:
            process_id = -1
        if process_id in excluded:
            continue
        if "pgrep -af" not in command:
            commands.append(line)
    return commands


def select_formal_latency_gpu(
    gpu_rows: Sequence[Mapping[str, Any]],
    *,
    requested_gpu_id: int | None = None,
    max_memory_fraction: float = 0.50,
    max_utilization_pct: int = 20,
) -> dict[str, Any]:
    eligible = []
    for source in gpu_rows:
        row = dict(source)
        index = int(row.get("index", -1))
        if requested_gpu_id is not None and index != int(requested_gpu_id):
            continue
        total = float(row.get("memory_total_mib", 0.0) or 0.0)
        used = float(row.get("memory_used_mib", 0.0) or 0.0)
        fraction = used / total if total > 0 else float("inf")
        if (
            fraction <= float(max_memory_fraction)
            and int(row.get("utilization_gpu_pct", 0) or 0)
            <= int(max_utilization_pct)
            and not list(row.get("processes", []) or [])
        ):
            eligible.append({**row, "memory_fraction": fraction})
    if not eligible:
        raise RuntimeError("formal_latency_no_isolated_gpu_available")
    return sorted(
        eligible,
        key=lambda row: (
            int(row.get("utilization_gpu_pct", 0)),
            float(row["memory_fraction"]),
            int(row["index"]),
        ),
    )[0]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_formal_latency_replay(
    *,
    rows: Sequence[Mapping[str, Any]],
    strict_fp32_reference: Mapping[str, Any],
    stage2_pool: Any,
    run_dir: str | Path,
    selected_gpu_id: int,
    selected_gpu_uuid: str,
    active_process_commands: Sequence[str],
    gpu_processes: Sequence[Mapping[str, Any]],
    required_evaluated_frames: int = 1789,
    required_skipped_frames: int = 0,
) -> dict[str, Any]:
    """Replay strict FP32 first and all candidate engines on one isolated GPU."""

    assert_formal_latency_isolation(
        active_process_commands=active_process_commands,
        selected_gpu_uuid=str(selected_gpu_uuid),
        gpu_processes=gpu_processes,
    )
    if int(getattr(stage2_pool, "parallelism", 1)) != 1:
        raise RuntimeError("formal_latency_requires_single_worker_pool")
    reference = dict(strict_fp32_reference)
    if (
        str(reference.get("status", "")) != "ok"
        or str(reference.get("reference_precision", "")) != "strict_fp32"
        or not str(reference.get("engine_path", ""))
        or not str(reference.get("engine_hash", ""))
        or not str(reference.get("reference_hash", ""))
    ):
        raise RuntimeError("formal_latency_strict_fp32_reference_invalid")
    unique: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        identity = str(
            row.get("deployment_identity")
            or row.get("deployment_hash")
            or row.get("engine_hash")
        )
        if not identity:
            raise ValueError("formal_latency_deployment_identity_missing")
        unique.setdefault(identity, row)
    reference_precision_hash = "strict_fp32_reference_profile"
    tasks = [
        {
            "task_protocol": "formal_latency",
            "task_cache_key": canonical_json_hash(
                {
                    "protocol": "formal_latency",
                    "deployment_identity": "strict_fp32_reference",
                    "engine_hash": str(reference["engine_hash"]),
                    "required_evaluated_frames": int(required_evaluated_frames),
                }
            ),
            "candidate_hash": "strict_fp32_reference",
            "engine_path": str(reference["engine_path"]),
            "output_dir": str(
                (Path(run_dir) / "strict_fp32_reference").resolve()
            ),
            "deployment_metadata": {
                "candidate_source": "baseline",
                "reference_hash": str(reference["reference_hash"]),
                "raw_precision_gene_hash": reference_precision_hash,
                "repaired_precision_gene_hash": reference_precision_hash,
                "requested_precision_profile_hash": reference_precision_hash,
                "realized_precision_profile_hash": reference_precision_hash,
                "precision_identity_passed": True,
            },
        }
    ]
    for identity, row in unique.items():
        candidate_hash = str(row["candidate_hash"])
        tasks.append(
            {
                "task_protocol": "formal_latency",
                "task_cache_key": canonical_json_hash(
                    {
                        "protocol": "formal_latency",
                        "deployment_identity": identity,
                        "engine_hash": str(row.get("engine_hash", "")),
                        "required_evaluated_frames": int(
                            required_evaluated_frames
                        ),
                    }
                ),
                "candidate_hash": candidate_hash,
                "engine_path": str(row["engine_path"]),
                "output_dir": str(
                    (Path(run_dir) / candidate_hash).resolve()
                ),
                "deployment_metadata": row,
            }
        )
    raw_results = [dict(row) for row in stage2_pool.map_tasks(tasks)]
    if len(raw_results) != len(tasks):
        raise RuntimeError("formal_latency_result_count_mismatch")

    def normalize(source: dict[str, Any]) -> dict[str, Any]:
        evaluated = int(
            source.get("evaluated", source.get("num_evaluated_frames", -1))
        )
        skipped = int(
            source.get("skipped", source.get("num_skipped_frames", -1))
        )
        p50 = float(source.get("forward_p50_ms", float("nan")))
        passed = (
            str(source.get("status", "")) == "ok"
            and evaluated == int(required_evaluated_frames)
            and skipped == int(required_skipped_frames)
            and math.isfinite(p50)
            and p50 > 0.0
            and int(source.get("worker_gpu_id", selected_gpu_id))
            == int(selected_gpu_id)
        )
        return {
            **source,
            "formal_latency_success": passed,
            "formal_latency_p50_ms": p50,
            "formal_latency_p95_ms": source.get("forward_p95_ms"),
            "formal_latency_gpu_id": int(selected_gpu_id),
            "formal_latency_gpu_uuid": str(selected_gpu_uuid),
            "formal_latency_evaluated_frames": evaluated,
            "formal_latency_skipped_frames": skipped,
        }

    reference_result = normalize(raw_results[0])
    if not reference_result["formal_latency_success"]:
        raise RuntimeError("formal_latency_strict_fp32_reference_failed")
    normalized = [normalize(row) for row in raw_results[1:]]
    report = {
        "selected_gpu_id": int(selected_gpu_id),
        "selected_gpu_uuid": str(selected_gpu_uuid),
        "strict_fp32_reference_hash": str(reference["reference_hash"]),
        "strict_fp32_formal_p50_ms": float(
            reference_result["formal_latency_p50_ms"]
        ),
        "reference_result": reference_result,
        "unique_deployment_count": len(unique),
        "successful_candidate_count": sum(
            bool(row["formal_latency_success"]) for row in normalized
        ),
        "results": normalized,
        "isolation": {
            "active_process_commands": list(active_process_commands),
            "gpu_processes": [dict(row) for row in gpu_processes],
            "passed": True,
        },
    }
    destination = Path(run_dir)
    _write_json(destination / "formal_latency_replay.json", report)
    _write_csv(destination / "formal_latency_replay.csv", normalized)
    return report
