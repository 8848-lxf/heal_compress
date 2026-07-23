#!/usr/bin/env python3
"""Capture read-only provenance for protected search processes.

The audit deliberately uses only ``/proc`` and ``nvidia-smi``.  It never sends
signals, opens process files for writing, or mutates an existing run directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = (
    "fcooper",
    "heal_lidar_disco",
    "h800_heal_lidar_disco",
    "run_heal_lidar_runtime_graph",
    "generation_finalization",
    "tensorrt_entropy_calibration_worker",
    "lidar_transformer_dh",
)

PATH_FLAGS = {
    "--artifact-root",
    "--calibration-dir",
    "--calibration-summary",
    "--formal-root",
    "--initial-checkpoint",
    "--model-dir",
    "--output-dir",
    "--output-root",
    "--report",
    "--request",
    "--request-json",
    "--source-checkpoint",
    "--source-config",
    "--work-root",
}


def _read_bytes(path: Path) -> tuple[bytes | None, str | None]:
    try:
        return path.read_bytes(), None
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _read_link(path: Path) -> tuple[str | None, str | None]:
    try:
        return os.readlink(path), None
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _decode_nul(data: bytes | None) -> list[str]:
    if not data:
        return []
    return [part.decode("utf-8", errors="replace") for part in data.split(b"\0") if part]


def _run_csv(command: list[str]) -> tuple[list[list[str]], str | None]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"{type(exc).__name__}:{exc}"
    if completed.returncode != 0:
        return [], f"returncode={completed.returncode}:{completed.stderr.strip()}"
    rows = [
        [part.strip() for part in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    return rows, None


def _gpu_process_map() -> tuple[dict[int, list[dict[str, Any]]], dict[str, int], list[str]]:
    errors: list[str] = []
    gpu_rows, gpu_error = _run_csv(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_error:
        errors.append(f"gpu_query:{gpu_error}")
    uuid_to_index: dict[str, int] = {}
    for row in gpu_rows:
        if len(row) >= 2:
            try:
                uuid_to_index[row[1]] = int(row[0])
            except ValueError:
                continue

    process_rows, process_error = _run_csv(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if process_error:
        errors.append(f"compute_query:{process_error}")
    result: dict[int, list[dict[str, Any]]] = {}
    for row in process_rows:
        if len(row) < 4:
            continue
        try:
            pid = int(row[0])
            memory_mib: int | str = int(row[3])
        except ValueError:
            continue
        result.setdefault(pid, []).append(
            {
                "gpu_uuid": row[1],
                "gpu_index": uuid_to_index.get(row[1]),
                "process_name": row[2],
                "used_memory_mib": memory_mib,
            }
        )
    return result, uuid_to_index, errors


def _extract_declared_paths(argv: list[str]) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in PATH_FLAGS and index + 1 < len(argv):
            paths.setdefault(token.removeprefix("--"), []).append(argv[index + 1])
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag in PATH_FLAGS:
                paths.setdefault(flag.removeprefix("--"), []).append(value)
        index += 1
    return paths


def _open_artifact_paths(pid: int) -> tuple[list[str], list[str]]:
    fd_root = Path("/proc") / str(pid) / "fd"
    values: set[str] = set()
    errors: list[str] = []
    try:
        entries: Iterable[Path] = fd_root.iterdir()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as exc:
        return [], [f"fd_list:{type(exc).__name__}:{exc}"]
    for entry in entries:
        target, error = _read_link(entry)
        if error:
            continue
        if target and any(marker in target for marker in ("/outputs/", "/results/", "/search_artifacts/")):
            values.add(target)
    return sorted(values), errors


def _model_from_command(command: str) -> tuple[str, bool]:
    lower = command.lower()
    has_fcooper = "fcooper" in lower
    has_disco = "disco" in lower
    if has_fcooper and has_disco:
        return "fcooper_disconet_queue", True
    if has_fcooper:
        return "fcooper", True
    if has_disco:
        return "disconet", True
    if "lidar_transformer_dh" in lower:
        return "transformer_alignment", False
    if "tensorrt_entropy_calibration_worker" in lower:
        return "unknown_entropy_calibration", False
    return "unknown_search", False


def _command_gpu_args(argv: list[str]) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token in {"--gpu", "--gpus", "--gpu-id", "--physical-gpu"} and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif re.match(r"^--(?:gpu|gpus|gpu-id|physical-gpu)=", token):
            values.append(token.split("=", 1)[1])
    return values


def _capture_process(pid: int, gpu_map: dict[int, list[dict[str, Any]]]) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    cmdline_data, cmdline_error = _read_bytes(proc / "cmdline")
    argv = _decode_nul(cmdline_data)
    if not argv:
        return None
    command = " ".join(argv)
    lower = command.lower()
    if not any(pattern in lower for pattern in DEFAULT_PATTERNS):
        return None

    cwd, cwd_error = _read_link(proc / "cwd")
    environ_data, environ_error = _read_bytes(proc / "environ")
    environment = _decode_nul(environ_data)
    cuda_visible = [
        value.split("=", 1)[1]
        for value in environment
        if value.startswith("CUDA_VISIBLE_DEVICES=")
    ]
    open_paths, fd_errors = _open_artifact_paths(pid)
    model, protected = _model_from_command(command)
    errors = [
        value
        for value in (
            f"cmdline:{cmdline_error}" if cmdline_error else None,
            f"cwd:{cwd_error}" if cwd_error else None,
            f"environ:{environ_error}" if environ_error else None,
            *fd_errors,
        )
        if value
    ]
    return {
        "pid": pid,
        "command_line": command,
        "cwd": cwd,
        "model": model,
        "declared_paths": _extract_declared_paths(argv),
        "open_output_or_log_paths": open_paths,
        "gpu": {
            "cuda_visible_devices": cuda_visible,
            "command_arguments": _command_gpu_args(argv),
            "nvidia_smi": gpu_map.get(pid, []),
        },
        "belongs_to_disconet_or_fcooper": protected,
        "task_touched_process": False,
        "task_touched_declared_or_open_paths": False,
        "read_errors": errors,
    }


def capture(*, phase: str, task_worktree: Path, task_run_root: Path) -> dict[str, Any]:
    gpu_map, uuid_to_index, gpu_errors = _gpu_process_map()
    processes: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        row = _capture_process(pid, gpu_map)
        if row is not None:
            processes.append(row)
    processes.sort(key=lambda row: int(row["pid"]))
    return {
        "schema_version": "h800-active-search-process-audit-v1",
        "phase": phase,
        "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "audit_mode": "read_only",
        "task_worktree": str(task_worktree.resolve()),
        "task_run_root": str(task_run_root.resolve()),
        "match_patterns": list(DEFAULT_PATTERNS),
        "gpu_uuid_to_index": uuid_to_index,
        "gpu_query_errors": gpu_errors,
        "process_count": len(processes),
        "protected_disconet_or_fcooper_process_count": sum(
            bool(row["belongs_to_disconet_or_fcooper"]) for row in processes
        ),
        "processes": processes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--task-worktree", type=Path, required=True)
    parser.add_argument("--task-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = capture(
        phase=args.phase,
        task_worktree=args.task_worktree,
        task_run_root=args.task_run_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
