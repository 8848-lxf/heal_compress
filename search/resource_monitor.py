"""Low-overhead, machine-readable resource accounting for formal searches."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Iterator, Sequence


class SearchResourceMonitor:
    """Record phase intervals and sample per-GPU utilization/memory.

    ``allocated_gpu_hours`` is computed from exact measured phase intervals and
    the explicitly assigned GPU set.  ``utilization_weighted_gpu_hours`` and
    peak device memory are sampled observations and retain their sampling
    interval in the report so the two notions are never conflated.
    """

    schema_version = "search-resource-accounting-v1"

    def __init__(
        self,
        output_dir: str | Path,
        gpu_ids: Sequence[int],
        *,
        sample_interval_seconds: float = 0.5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_ids = tuple(sorted({int(value) for value in gpu_ids}))
        self.sample_interval_seconds = max(0.1, float(sample_interval_seconds))
        self.events_path = self.output_dir / "resource_events.jsonl"
        self.samples_path = self.output_dir / "gpu_samples.jsonl"
        self.summary_path = self.output_dir / "resource_summary.json"
        self._events: list[dict[str, Any]] = []
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._started_wall = time.time()
        self._started_monotonic = time.perf_counter()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="search-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _query_gpus(self) -> list[dict[str, Any]]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(2.0, self.sample_interval_seconds * 4.0),
        )
        if completed.returncode != 0:
            return []
        rows: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) < 5:
                continue
            try:
                gpu_id = int(values[0])
                if gpu_id not in self.gpu_ids:
                    continue
                rows.append(
                    {
                        "gpu_id": gpu_id,
                        "utilization_pct": float(values[1]),
                        "memory_used_mib": float(values[2]),
                        "memory_total_mib": float(values[3]),
                        "power_watts": float(values[4]),
                    }
                )
            except ValueError:
                continue
        return rows

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            now_wall = time.time()
            now_monotonic = time.perf_counter()
            for gpu in self._query_gpus():
                row = {
                    "schema_version": self.schema_version,
                    "timestamp_unix": now_wall,
                    "elapsed_seconds": now_monotonic - self._started_monotonic,
                    **gpu,
                }
                with self._lock:
                    self._samples.append(row)
                    self._append_jsonl(self.samples_path, row)
            self._stop.wait(self.sample_interval_seconds)

    def record_event(self, event: str, **metadata: Any) -> None:
        row = {
            "schema_version": self.schema_version,
            "event": str(event),
            "timestamp_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - self._started_monotonic,
            **metadata,
        }
        with self._lock:
            self._events.append(row)
            self._append_jsonl(self.events_path, row)

    def record_completed_phase(
        self,
        name: str,
        elapsed_seconds: float,
        gpu_ids: Sequence[int],
        **metadata: Any,
    ) -> None:
        devices = sorted({int(value) for value in gpu_ids})
        elapsed = max(0.0, float(elapsed_seconds))
        self.record_event(
            "phase_complete",
            phase=str(name),
            phase_elapsed_seconds=elapsed,
            gpu_ids=devices,
            allocated_gpu_hours=elapsed * len(devices) / 3600.0,
            **metadata,
        )

    @contextmanager
    def phase(
        self,
        name: str,
        gpu_ids: Sequence[int],
        **metadata: Any,
    ) -> Iterator[None]:
        started = time.perf_counter()
        devices = sorted({int(value) for value in gpu_ids})
        self.record_event("phase_start", phase=str(name), gpu_ids=devices, **metadata)
        try:
            yield
        finally:
            self.record_completed_phase(
                name,
                time.perf_counter() - started,
                devices,
                **metadata,
            )

    def _summary(self) -> dict[str, Any]:
        phases: dict[str, dict[str, Any]] = {}
        for row in self._events:
            if row.get("event") != "phase_complete":
                continue
            name = str(row.get("phase", "unknown"))
            phase = phases.setdefault(
                name,
                {
                    "invocation_count": 0,
                    "elapsed_seconds": 0.0,
                    "allocated_gpu_hours": 0.0,
                },
            )
            phase["invocation_count"] += 1
            phase["elapsed_seconds"] += float(row.get("phase_elapsed_seconds", 0.0))
            phase["allocated_gpu_hours"] += float(row.get("allocated_gpu_hours", 0.0))
        by_gpu: dict[int, list[dict[str, Any]]] = {}
        for row in self._samples:
            by_gpu.setdefault(int(row["gpu_id"]), []).append(row)
        gpu_summary: dict[str, Any] = {}
        for gpu_id, rows in sorted(by_gpu.items()):
            rows.sort(key=lambda row: float(row["elapsed_seconds"]))
            utilization_seconds = 0.0
            energy_joules = 0.0
            for left, right in zip(rows, rows[1:]):
                delta = max(
                    0.0,
                    min(
                        float(right["elapsed_seconds"]) - float(left["elapsed_seconds"]),
                        self.sample_interval_seconds * 2.5,
                    ),
                )
                utilization_seconds += delta * float(left["utilization_pct"]) / 100.0
                energy_joules += delta * float(left["power_watts"])
            gpu_summary[str(gpu_id)] = {
                "sample_count": len(rows),
                "peak_memory_used_mib": max(
                    (float(row["memory_used_mib"]) for row in rows),
                    default=0.0,
                ),
                "utilization_weighted_gpu_hours": utilization_seconds / 3600.0,
                "sampled_energy_kwh": energy_joules / 3_600_000.0,
            }
        return {
            "schema_version": self.schema_version,
            "started_timestamp_unix": self._started_wall,
            "finished_timestamp_unix": time.time(),
            "wall_elapsed_seconds": time.perf_counter() - self._started_monotonic,
            "gpu_ids": list(self.gpu_ids),
            "sample_interval_seconds": self.sample_interval_seconds,
            "phase_totals": phases,
            "gpu_summary": gpu_summary,
            "total_allocated_gpu_hours": sum(
                float(row["allocated_gpu_hours"]) for row in phases.values()
            ),
            "total_utilization_weighted_gpu_hours": sum(
                float(row["utilization_weighted_gpu_hours"])
                for row in gpu_summary.values()
            ),
            "peak_memory_used_mib": max(
                (float(row["peak_memory_used_mib"]) for row in gpu_summary.values()),
                default=0.0,
            ),
            "accounting_note": (
                "allocated_gpu_hours uses exact phase wall intervals times assigned devices; "
                "utilization-weighted GPU-hours, energy and peak VRAM are nvidia-smi samples"
            ),
        }

    def close(self) -> dict[str, Any]:
        if self._closed:
            if self.summary_path.is_file():
                return json.loads(self.summary_path.read_text(encoding="utf-8"))
            return self._summary()
        self._closed = True
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(2.0, self.sample_interval_seconds * 3.0))
        with self._lock:
            summary = self._summary()
            self.summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return summary


__all__ = ["SearchResourceMonitor"]
