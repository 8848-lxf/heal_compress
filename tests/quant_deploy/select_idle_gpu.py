from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    processes: list[GpuProcess]


def _parse_int(value: str) -> int:
    text = str(value).strip()
    if text in {"", "[Not Supported]", "N/A", "No running processes found"}:
        return 0
    return int(float(text.replace("MiB", "").replace("%", "").strip()))


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


def _run_nvidia_smi(args: list[str], *, timeout_sec: int | None = None) -> str:
    timeout = int(timeout_sec or os.environ.get("NVIDIA_SMI_QUERY_TIMEOUT_SEC", "10"))
    try:
        completed = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"nvidia-smi timed out after {timeout}s: {' '.join(args)}") from exc
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
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    proc_text = _run_nvidia_smi(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = parse_process_query_csv(proc_text)
    by_uuid: dict[str, list[GpuProcess]] = {}
    for process in processes:
        by_uuid.setdefault(process.gpu_uuid, []).append(process)
    snapshots: list[GpuSnapshot] = []
    for row in parse_gpu_query_csv(gpu_text):
        snapshots.append(
            GpuSnapshot(
                index=int(row["index"]),
                uuid=str(row["uuid"]),
                name=str(row["name"]),
                utilization_gpu=int(row["utilization_gpu"]),
                memory_used_mb=int(row["memory_used_mb"]),
                memory_total_mb=int(row["memory_total_mb"]),
                processes=by_uuid.get(str(row["uuid"]), []),
            )
        )
    return snapshots


def snapshot_to_dict(snapshot: GpuSnapshot) -> dict[str, Any]:
    data = asdict(snapshot)
    data["process_count"] = len(snapshot.processes)
    return data


def snapshots_to_dicts(snapshots: list[GpuSnapshot]) -> list[dict[str, Any]]:
    return [snapshot_to_dict(snapshot) for snapshot in snapshots]


def idle_gpu_candidates(
    snapshots: list[GpuSnapshot],
    *,
    util_threshold: int,
    mem_threshold_mb: int,
) -> list[GpuSnapshot]:
    return [
        snapshot
        for snapshot in snapshots
        if snapshot.utilization_gpu <= int(util_threshold)
        and snapshot.memory_used_mb <= int(mem_threshold_mb)
        and not snapshot.processes
    ]


def choose_idle_gpu(
    snapshots: list[GpuSnapshot],
    *,
    util_threshold: int = 5,
    mem_threshold_mb: int = 2000,
    gpu_index: int | None = None,
    allow_busy_gpu: bool = False,
) -> tuple[GpuSnapshot | None, str]:
    by_index = {snapshot.index: snapshot for snapshot in snapshots}
    if gpu_index is not None:
        selected = by_index.get(int(gpu_index))
        if selected is None:
            return None, f"requested GPU {gpu_index} was not found"
        idle = (
            selected.utilization_gpu <= int(util_threshold)
            and selected.memory_used_mb <= int(mem_threshold_mb)
            and not selected.processes
        )
        if idle or allow_busy_gpu:
            reason = "requested GPU is idle" if idle else "requested GPU accepted because allow_busy_gpu=true"
            return selected, reason
        return None, (
            f"requested GPU {gpu_index} is busy: util={selected.utilization_gpu}% "
            f"mem={selected.memory_used_mb} MiB processes={len(selected.processes)}"
        )
    candidates = idle_gpu_candidates(snapshots, util_threshold=util_threshold, mem_threshold_mb=mem_threshold_mb)
    if not candidates:
        return None, "no GPU satisfied idle thresholds"
    candidates = sorted(candidates, key=lambda item: (item.memory_used_mb, item.utilization_gpu, item.index))
    selected = candidates[0]
    return selected, (
        f"selected lowest memory/util idle GPU: mem={selected.memory_used_mb} MiB "
        f"util={selected.utilization_gpu}%"
    )


def wait_for_idle_gpu(
    *,
    util_threshold: int = 5,
    mem_threshold_mb: int = 2000,
    wait_timeout_sec: int = 3600,
    poll_interval_sec: int = 30,
    gpu_index: int | None = None,
    allow_busy_gpu: bool = False,
) -> tuple[GpuSnapshot, str, list[dict[str, Any]]]:
    start = time.time()
    attempts: list[dict[str, Any]] = []
    while True:
        snapshots: list[GpuSnapshot] = []
        try:
            snapshots = query_gpu_snapshots()
            selected, reason = choose_idle_gpu(
                snapshots,
                util_threshold=util_threshold,
                mem_threshold_mb=mem_threshold_mb,
                gpu_index=gpu_index,
                allow_busy_gpu=allow_busy_gpu,
            )
        except Exception as exc:
            selected = None
            reason = f"gpu query failed: {exc}"
        attempts.append(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "reason": reason,
                "snapshots": snapshots_to_dicts(snapshots),
            }
        )
        if selected is not None:
            return selected, reason, attempts
        if time.time() - start >= int(wait_timeout_sec):
            raise TimeoutError(f"timed out waiting for idle GPU after {wait_timeout_sec}s: {reason}")
        time.sleep(max(1, int(poll_interval_sec)))


class GpuTelemetryMonitor:
    def __init__(
        self,
        *,
        selected_gpu_index: int,
        selected_gpu_uuid: str,
        own_pid: int | None = None,
        poll_interval_sec: int = 30,
    ) -> None:
        self.selected_gpu_index = int(selected_gpu_index)
        self.selected_gpu_uuid = str(selected_gpu_uuid)
        self.own_pid = int(own_pid or os.getpid())
        self.poll_interval_sec = max(1, int(poll_interval_sec))
        self.samples: list[dict[str, Any]] = []
        self.contention_events: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        snapshots = query_gpu_snapshots()
        selected = next((item for item in snapshots if item.index == self.selected_gpu_index), None)
        if selected is None:
            event = {"time": time.time(), "reason": "selected_gpu_missing"}
            self.contention_events.append(event)
            return
        other_processes = [process for process in selected.processes if int(process.pid) != self.own_pid]
        sample = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "snapshot": snapshot_to_dict(selected),
            "other_processes": [asdict(process) for process in other_processes],
            "other_processes_detected": bool(other_processes),
            "high_utilization": selected.utilization_gpu >= 90,
        }
        self.samples.append(sample)
        if other_processes:
            self.contention_events.append(
                {
                    "time": sample["time"],
                    "reason": "other_compute_process_detected",
                    "utilization_gpu": selected.utilization_gpu,
                    "other_processes": sample["other_processes"],
                }
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as exc:
                self.contention_events.append(
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "reason": "gpu_monitor_sample_failed",
                        "error": str(exc),
                    }
                )
            self._stop.wait(self.poll_interval_sec)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2, self.poll_interval_sec + 1))
        try:
            self._sample_once()
        except Exception:
            pass

    def report(self) -> dict[str, Any]:
        other_processes = [
            process
            for sample in self.samples
            for process in sample.get("other_processes", [])
        ]
        contention = bool(self.contention_events)
        return {
            "selected_gpu": self.selected_gpu_index,
            "selected_gpu_uuid": self.selected_gpu_uuid,
            "polling_interval": self.poll_interval_sec,
            "samples": self.samples,
            "contention_events": self.contention_events,
            "other_processes_detected": bool(other_processes),
            "other_processes": other_processes,
            "gpu_contention_detected": contention,
            "unreliable_latency": contention,
        }


def write_gpu_selection_reports(report: dict[str, Any], *, debug_path: Path, summary_path: Path) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# GPU Selection and Contention Report",
        "",
        f"- selected_gpu: {report.get('selected_gpu')}",
        f"- selected_gpu_reason: {report.get('selected_gpu_reason')}",
        f"- CUDA_VISIBLE_DEVICES: {report.get('CUDA_VISIBLE_DEVICES')}",
        f"- polling_interval: {report.get('polling_interval')}",
        f"- gpu_contention_detected: {report.get('gpu_contention_detected')}",
        f"- unreliable_latency: {report.get('unreliable_latency')}",
        f"- other_processes_detected: {report.get('other_processes_detected')}",
        f"- run_start_time: {report.get('run_start_time')}",
        f"- run_end_time: {report.get('run_end_time')}",
        "",
        "## Contention Events",
        "",
    ]
    events = report.get("contention_events") or []
    if not events:
        lines.append("- none")
    else:
        for event in events:
            lines.append(f"- {event}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select an idle GPU using nvidia-smi.")
    parser.add_argument("--gpu_idle_util_threshold", type=int, default=5)
    parser.add_argument("--gpu_idle_mem_threshold_mb", type=int, default=2000)
    parser.add_argument("--gpu_wait_timeout_sec", type=int, default=3600)
    parser.add_argument("--gpu_poll_interval_sec", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=None)
    parser.add_argument("--allow_busy_gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selected, reason, attempts = wait_for_idle_gpu(
            util_threshold=args.gpu_idle_util_threshold,
            mem_threshold_mb=args.gpu_idle_mem_threshold_mb,
            wait_timeout_sec=args.gpu_wait_timeout_sec,
            poll_interval_sec=args.gpu_poll_interval_sec,
            gpu_index=args.gpu_index,
            allow_busy_gpu=args.allow_busy_gpu,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(exc),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.index)
    print(
        json.dumps(
            {
                "success": True,
                "selected_gpu": snapshot_to_dict(selected),
                "selected_gpu_reason": reason,
                "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                "attempts": attempts,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
