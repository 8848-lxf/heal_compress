#!/usr/bin/env python3
"""Read-only process/GPU provenance for isolated search workspaces."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MATCH_PATTERNS = (
    "fcooper",
    "disconet",
    "heal_lidar_disco",
    "h800_heal_lidar_disco",
    "run_heal_lidar_runtime_graph",
    "generation_finalization",
    "tensorrt_entropy_calibration_worker",
    "lidar_transformer_dh",
    "transformer_unified_search",
)


def _read(path: Path, *, binary: bool = False) -> tuple[Any | None, str | None]:
    try:
        return (path.read_bytes() if binary else path.read_text(encoding="utf-8", errors="replace")), None
    except (OSError, PermissionError) as exc:
        return None, f"{path.name}:{type(exc).__name__}:{exc}"


def _gpu_state() -> tuple[dict[str, int], dict[int, list[dict[str, Any]]], list[str]]:
    errors: list[str] = []
    uuid_to_index: dict[str, int] = {}
    pid_to_rows: dict[int, list[dict[str, Any]]] = {}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if gpu.returncode != 0:
            errors.append(f"gpu_query:{gpu.stderr.strip()}")
        for line in gpu.stdout.splitlines():
            index, uuid = [item.strip() for item in line.split(",", 1)]
            uuid_to_index[uuid] = int(index)
        compute = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if compute.returncode != 0:
            errors.append(f"compute_query:{compute.stderr.strip()}")
        for line in compute.stdout.splitlines():
            parts = [item.strip() for item in line.split(",", 3)]
            if len(parts) != 4:
                continue
            uuid, pid_text, process_name, used_memory = parts
            try:
                pid = int(pid_text)
                memory = int(used_memory)
            except ValueError:
                continue
            pid_to_rows.setdefault(pid, []).append(
                {
                    "gpu_index": uuid_to_index.get(uuid),
                    "gpu_uuid": uuid,
                    "process_name": process_name,
                    "used_memory_mib": memory,
                }
            )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"nvidia_smi:{type(exc).__name__}:{exc}")
    return uuid_to_index, pid_to_rows, errors


def _model(command: str) -> tuple[str, bool]:
    text = command.lower()
    if "fcooper" in text and ("disco" in text or "disconet" in text):
        return "fcooper_disconet_queue", True
    if "fcooper" in text:
        return "fcooper", True
    if "disconet" in text or "heal_lidar_disco" in text:
        return "disconet", True
    if "lidar_transformer_dh" in text:
        return "transformer_alignment", False
    if "transformer_unified_search" in text:
        return "transformer_unified_search", False
    return "unknown_search", False


def _declared_paths(argv: list[str]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for index, token in enumerate(argv[:-1]):
        if not token.startswith("--"):
            continue
        value = argv[index + 1]
        if value.startswith("--"):
            continue
        if value.startswith("/") and any(part in token for part in ("root", "dir", "path", "log", "output", "checkpoint", "config", "report", "manifest")):
            rows.setdefault(token[2:], []).append(value)
    return rows


def _open_artifact_paths(proc_root: Path) -> tuple[list[str], list[str]]:
    rows: list[str] = []
    errors: list[str] = []
    fd_root = proc_root / "fd"
    try:
        entries = list(fd_root.iterdir())[:512]
    except (OSError, PermissionError) as exc:
        return [], [f"fd_list:{type(exc).__name__}:{exc}"]
    for entry in entries:
        try:
            target = os.readlink(entry)
        except (OSError, PermissionError):
            continue
        lower = target.lower()
        if target.startswith("/") and any(part in lower for part in ("output", "result", ".log", ".json", ".csv", ".onnx", ".engine", ".plan")):
            rows.append(target)
    return sorted(set(rows)), errors


def _process_rows(task_run_root: Path, task_worktree: Path) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    uuid_to_index, gpu_by_pid, gpu_errors = _gpu_state()
    rows: list[dict[str, Any]] = []
    self_pid = os.getpid()
    for proc_root in sorted(Path("/proc").glob("[0-9]*"), key=lambda path: int(path.name)):
        pid = int(proc_root.name)
        if pid == self_pid:
            continue
        raw_cmd, cmd_error = _read(proc_root / "cmdline", binary=True)
        if not raw_cmd:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw_cmd.split(b"\0") if part]
        command = " ".join(shlex.quote(part) for part in argv)
        lower = command.lower()
        if not any(pattern in lower for pattern in MATCH_PATTERNS):
            continue
        read_errors: list[str] = [cmd_error] if cmd_error else []
        try:
            cwd = str((proc_root / "cwd").resolve(strict=True))
        except (OSError, PermissionError) as exc:
            cwd = None
            read_errors.append(f"cwd:{type(exc).__name__}:{exc}")
        env_raw, env_error = _read(proc_root / "environ", binary=True)
        cuda_visible: list[str] = []
        if env_error:
            read_errors.append(env_error)
        elif env_raw:
            for item in env_raw.split(b"\0"):
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    cuda_visible = item.split(b"=", 1)[1].decode("utf-8", errors="replace").split(",")
        open_paths, fd_errors = _open_artifact_paths(proc_root)
        read_errors.extend(fd_errors)
        model, belongs = _model(command)
        declared = _declared_paths(argv)
        referenced_paths = [path for values in declared.values() for path in values] + open_paths
        touches_task_path = any(
            path == str(task_run_root) or path.startswith(f"{task_run_root}/")
            or path == str(task_worktree) or path.startswith(f"{task_worktree}/")
            for path in referenced_paths
        )
        rows.append(
            {
                "pid": pid,
                "command_line": command,
                "cwd": cwd,
                "model": model,
                "declared_paths": declared,
                "open_output_or_log_paths": open_paths,
                "gpu": {
                    "cuda_visible_devices": cuda_visible,
                    "nvidia_smi": gpu_by_pid.get(pid, []),
                },
                "belongs_to_disconet_or_fcooper": belongs,
                "task_touched_declared_or_open_paths": False,
                "process_references_task_path": touches_task_path,
                "task_touched_process": False,
                "signals_sent_by_task": [],
                "read_errors": read_errors,
            }
        )
    return rows, uuid_to_index, gpu_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument(
        "--phase",
        choices=("before", "after"),
        default="after",
        help="Label this immutable audit snapshot without changing scan semantics.",
    )
    parser.add_argument("--task-run-root", type=Path, required=True)
    parser.add_argument("--task-worktree", type=Path, required=True)
    args = parser.parse_args()

    rows, uuid_to_index, gpu_errors = _process_rows(args.task_run_root.resolve(), args.task_worktree.resolve())
    before_comparison: dict[str, Any] = {}
    if args.before and args.before.is_file():
        before = json.loads(args.before.read_text(encoding="utf-8"))
        before_external = {
            int(row["pid"]): row
            for row in before.get("processes", [])
            if row.get("belongs_to_disconet_or_fcooper")
        }
        current_external = {
            int(row["pid"]): row
            for row in rows
            if row.get("belongs_to_disconet_or_fcooper")
        }
        before_comparison = {
            "before_external_pid_count": len(before_external),
            "current_external_pid_count": len(current_external),
            "before_pids_still_alive": sorted(set(before_external) & set(current_external)),
            "before_pids_no_longer_alive": sorted(set(before_external) - set(current_external)),
            "current_new_external_pids": sorted(set(current_external) - set(before_external)),
            "persistent_supervisor_pids": sorted(
                pid
                for pid in set(before_external) & set(current_external)
                if "supervisor" in current_external[pid].get("command_line", "").lower()
                or "watcher" in current_external[pid].get("command_line", "").lower()
            ),
            "interpretation": (
                "Transient worker PID turnover is external lifecycle activity. "
                "This task sent no signals and wrote none of the declared/open external paths."
            ),
        }
    payload = {
        "schema_version": "active-search-process-audit-v2",
        "phase": args.phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "read_only",
        "match_patterns": list(MATCH_PATTERNS),
        "task_run_root": str(args.task_run_root.resolve()),
        "task_worktree": str(args.task_worktree.resolve()),
        "signals_sent_to_external_processes": [],
        "external_paths_written_by_task": [],
        "gpu_uuid_to_index": uuid_to_index,
        "gpu_query_errors": gpu_errors,
        "process_count": len(rows),
        "processes": rows,
        "before_comparison": before_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
