#!/usr/bin/env python3
"""Select an idle GPU and record latency-benchmark GPU state."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - torch is available in the target env.
    torch = None  # type: ignore[assignment]


@dataclass
class GpuProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mb: int


@dataclass
class GpuSnapshot:
    index: int
    uuid: str
    name: str
    utilization_gpu: int
    memory_used_mb: int
    memory_total_mb: int
    temperature_gpu: int = 0
    power_draw_w: float = 0.0
    driver_version: str = ""
    processes: list[GpuProcess] | None = None

    @property
    def memory_ratio(self) -> float:
        return float(self.memory_used_mb) / float(self.memory_total_mb) if self.memory_total_mb > 0 else 1.0


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _parse_int(value: str) -> int:
    text = str(value).strip()
    if text in {"", "N/A", "[Not Supported]", "No running processes found"}:
        return 0
    return int(float(text.replace("MiB", "").replace("%", "").replace("W", "").strip()))


def _parse_float(value: str) -> float:
    text = str(value).strip()
    if text in {"", "N/A", "[Not Supported]", "No running processes found"}:
        return 0.0
    return float(text.replace("MiB", "").replace("%", "").replace("W", "").strip())


def parse_gpu_query_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "index": _parse_int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "utilization_gpu": _parse_int(parts[3]),
                "memory_used_mb": _parse_int(parts[4]),
                "memory_total_mb": _parse_int(parts[5]),
                "temperature_gpu": _parse_int(parts[6]) if len(parts) > 6 else 0,
                "power_draw_w": _parse_float(parts[7]) if len(parts) > 7 else 0.0,
                "driver_version": parts[8] if len(parts) > 8 else "",
            }
        )
    return rows


def parse_process_query_csv(text: str) -> list[GpuProcess]:
    processes: list[GpuProcess] = []
    for line in text.splitlines():
        if not line.strip() or "No running processes found" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            processes.append(
                GpuProcess(
                    gpu_uuid=parts[0],
                    pid=_parse_int(parts[1]),
                    process_name=parts[2],
                    used_memory_mb=_parse_int(parts[3]),
                )
            )
        except ValueError:
            continue
    return processes


def snapshots_from_query_rows(gpu_rows: list[dict[str, Any]], processes: list[GpuProcess]) -> list[GpuSnapshot]:
    by_uuid: dict[str, list[GpuProcess]] = {}
    for process in processes:
        by_uuid.setdefault(process.gpu_uuid, []).append(process)
    return [
        GpuSnapshot(
            index=int(row["index"]),
            uuid=str(row["uuid"]),
            name=str(row["name"]),
            utilization_gpu=int(row["utilization_gpu"]),
            memory_used_mb=int(row["memory_used_mb"]),
            memory_total_mb=int(row["memory_total_mb"]),
            temperature_gpu=int(row.get("temperature_gpu", 0) or 0),
            power_draw_w=float(row.get("power_draw_w", 0.0) or 0.0),
            driver_version=str(row.get("driver_version", "")),
            processes=by_uuid.get(str(row["uuid"]), []),
        )
        for row in gpu_rows
    ]


def _run_nvidia_smi(args: list[str], *, timeout_sec: int = 10) -> str:
    try:
        completed = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"nvidia-smi timed out after {timeout_sec}s: {' '.join(args)}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        if "No running processes found" in message:
            return ""
        raise RuntimeError(f"nvidia-smi failed: {' '.join(args)}: {message}")
    return completed.stdout


def query_gpu_snapshots() -> list[GpuSnapshot]:
    gpu_text = _run_nvidia_smi(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    process_text = _run_nvidia_smi(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return snapshots_from_query_rows(parse_gpu_query_csv(gpu_text), parse_process_query_csv(process_text))


def snapshot_to_dict(snapshot: GpuSnapshot) -> dict[str, Any]:
    data = asdict(snapshot)
    data["memory_ratio"] = snapshot.memory_ratio
    data["processes"] = [asdict(process) for process in (snapshot.processes or [])]
    data["process_count"] = len(snapshot.processes or [])
    return data


def choose_idle_gpu(
    snapshots: list[GpuSnapshot],
    *,
    max_utilization: int = 5,
    max_memory_ratio: float = 0.20,
    requested_index: int | None = None,
    allow_busy_gpu: bool = False,
) -> tuple[GpuSnapshot | None, str]:
    by_index = {snapshot.index: snapshot for snapshot in snapshots}
    if requested_index is not None:
        snapshot = by_index.get(int(requested_index))
        if snapshot is None:
            return None, f"requested GPU {requested_index} was not found"
        idle = snapshot.utilization_gpu <= int(max_utilization) and snapshot.memory_ratio <= float(max_memory_ratio)
        if idle:
            return snapshot, "requested GPU is idle"
        if allow_busy_gpu:
            return snapshot, "requested GPU accepted because allow_busy_gpu=true"
        return (
            None,
            f"requested GPU {requested_index} is busy: util={snapshot.utilization_gpu}% "
            f"memory_ratio={snapshot.memory_ratio:.4f}",
        )

    candidates = [
        snapshot
        for snapshot in snapshots
        if snapshot.utilization_gpu <= int(max_utilization) and snapshot.memory_ratio <= float(max_memory_ratio)
    ]
    if not candidates:
        return None, "idle_gpu_not_found"
    candidates.sort(key=lambda item: (item.memory_ratio, item.utilization_gpu, item.index))
    return candidates[0], "selected idle GPU with lowest memory ratio/utilization"


def wait_for_idle_gpu(
    *,
    max_utilization: int,
    max_memory_ratio: float,
    wait_timeout_minutes: float,
    poll_seconds: int,
    requested_index: int | None = None,
    allow_busy_gpu: bool = False,
) -> tuple[GpuSnapshot, str, list[dict[str, Any]]]:
    deadline = time.time() + max(0.0, float(wait_timeout_minutes)) * 60.0
    attempts: list[dict[str, Any]] = []
    while True:
        snapshots: list[GpuSnapshot] = []
        try:
            snapshots = query_gpu_snapshots()
            selected, reason = choose_idle_gpu(
                snapshots,
                max_utilization=max_utilization,
                max_memory_ratio=max_memory_ratio,
                requested_index=requested_index,
                allow_busy_gpu=allow_busy_gpu,
            )
        except Exception as exc:
            selected = None
            reason = f"gpu_query_failed: {exc}"
        attempts.append(
            {
                "timestamp": _now(),
                "reason": reason,
                "snapshots": [snapshot_to_dict(snapshot) for snapshot in snapshots],
            }
        )
        if selected is not None:
            return selected, reason, attempts
        if time.time() >= deadline:
            raise TimeoutError(f"idle_gpu_not_found after {wait_timeout_minutes} minutes: {reason}")
        time.sleep(max(1, int(poll_seconds)))


def collect_gpu_state(selected_gpu_index: int | None = None) -> dict[str, Any]:
    snapshots = query_gpu_snapshots()
    selected = None
    if selected_gpu_index is not None:
        selected = next((snapshot for snapshot in snapshots if snapshot.index == int(selected_gpu_index)), None)
    torch_info = {
        "torch_version": getattr(torch, "__version__", "") if torch is not None else "",
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", "") if torch is not None else "",
        "cudnn_version": torch.backends.cudnn.version() if torch is not None and torch.backends.cudnn.is_available() else "",
    }
    return {
        "timestamp": _now(),
        "selected_gpu_index": selected_gpu_index,
        "gpu_name": selected.name if selected is not None else "",
        "utilization.gpu": selected.utilization_gpu if selected is not None else None,
        "memory.used": selected.memory_used_mb if selected is not None else None,
        "memory.total": selected.memory_total_mb if selected is not None else None,
        "temperature.gpu": selected.temperature_gpu if selected is not None else None,
        "power.draw": selected.power_draw_w if selected is not None else None,
        "running_processes": [asdict(process) for process in (selected.processes or [])] if selected is not None else [],
        "driver_version": selected.driver_version if selected is not None else "",
        "python_version": platform.python_version(),
        "all_gpus": [snapshot_to_dict(snapshot) for snapshot in snapshots],
        **torch_info,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _device_to_index(device: str | None) -> int | None:
    if not device:
        return None
    text = str(device)
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    if text.isdigit():
        return int(text)
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select an idle GPU for latency benchmark.")
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--max-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--device", default="")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    requested_index = _device_to_index(args.device)
    try:
        if args.wait:
            selected, reason, attempts = wait_for_idle_gpu(
                max_utilization=args.max_utilization,
                max_memory_ratio=args.max_memory_ratio,
                wait_timeout_minutes=args.wait_timeout_minutes,
                poll_seconds=args.poll_seconds,
                requested_index=requested_index,
                allow_busy_gpu=bool(args.allow_busy_gpu),
            )
        else:
            snapshots = query_gpu_snapshots()
            selected, reason = choose_idle_gpu(
                snapshots,
                max_utilization=args.max_utilization,
                max_memory_ratio=args.max_memory_ratio,
                requested_index=requested_index,
                allow_busy_gpu=bool(args.allow_busy_gpu),
            )
            attempts = [{"timestamp": _now(), "reason": reason, "snapshots": [snapshot_to_dict(s) for s in snapshots]}]
            if selected is None:
                raise RuntimeError(reason)
    except Exception as exc:
        failure = {
            "success": False,
            "failure_reason": "idle_gpu_not_found",
            "error": str(exc),
            "timestamp": _now(),
            "attempts": locals().get("attempts", []),
        }
        _write_json(output, failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 2

    payload = {
        "success": True,
        "selected_gpu_index": selected.index,
        "selected_gpu": snapshot_to_dict(selected),
        "selected_gpu_reason": reason,
        "attempts": attempts,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
