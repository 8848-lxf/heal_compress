"""Persistent one-process-per-GPU Stage-2 task pool."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class PersistentStage2ProcessPool:
    """Keep one isolated worker alive on each configured physical GPU."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        gpu_ids: Sequence[int],
        worker_payload: dict[str, Any],
        worker_command: Sequence[str] | None = None,
        startup_timeout_seconds: float = 900.0,
        task_timeout_seconds: float = 7200.0,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.gpu_ids = [int(value) for value in gpu_ids]
        if not self.gpu_ids:
            raise ValueError("stage2_worker_gpu_ids_required")
        if len(self.gpu_ids) != len(set(self.gpu_ids)):
            raise ValueError("stage2_worker_gpu_ids_must_be_unique")
        self.worker_payload = dict(worker_payload)
        self._pool_signature = hashlib.sha256(
            json.dumps(
                {
                    "gpu_ids": self.gpu_ids,
                    "worker_payload": self.worker_payload,
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        self.worker_command = list(
            worker_command
            or [sys.executable, "-m", "search.stage2.candidate_worker"]
        )
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.task_timeout_seconds = float(task_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.root = self.run_dir / "stage2_workers"
        self.control = self.root / "control"
        self.results = self.root / "results"
        self._workers: list[dict[str, Any]] = []
        self._sequence = 0
        self._cursor = 0
        self._started = False
        self._result_cache: dict[str, dict[str, Any]] = {}
        self._load_result_cache()

    @property
    def parallelism(self) -> int:
        return len(self.gpu_ids)

    def _load_result_cache(self) -> None:
        if not self.results.is_dir():
            return
        for path in sorted(self.results.glob("task_*.json")):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cache_key = str(result.get("candidate_hash", ""))
            if (
                cache_key
                and str(result.get("status", "")) == "ok"
                and str(result.get("pool_signature", ""))
                == self._pool_signature
            ):
                result.setdefault("pool_task_id", path.stem)
                self._result_cache[cache_key] = result

    def start(self) -> "PersistentStage2ProcessPool":
        if self._started:
            return self
        self.control.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        for gpu_id in self.gpu_ids:
            worker_dir = self.root / f"gpu_{gpu_id:03d}"
            queue_dir = worker_dir / "queue"
            queue_dir.mkdir(parents=True, exist_ok=True)
            ready_path = worker_dir / "ready.json"
            stop_path = worker_dir / "stop"
            for stale in (ready_path, stop_path):
                stale.unlink(missing_ok=True)
            init_path = self.control / f"worker_gpu_{gpu_id:03d}.json"
            init = {
                **self.worker_payload,
                "gpu_id": gpu_id,
                "worker_dir": str(worker_dir),
                "queue_dir": str(queue_dir),
                "ready_path": str(ready_path),
                "stop_path": str(stop_path),
                "poll_interval_seconds": self.poll_interval_seconds,
            }
            _atomic_write_json(init_path, init)
            log_path = worker_dir / "worker.log"
            log_handle = log_path.open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment["STAGE2_PHYSICAL_GPU_ID"] = str(gpu_id)
            process = subprocess.Popen(
                [*self.worker_command, "--request", str(init_path)],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._workers.append(
                {
                    "gpu_id": gpu_id,
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": str(log_path),
                    "queue_dir": queue_dir,
                    "ready_path": ready_path,
                    "stop_path": stop_path,
                    "init_path": str(init_path),
                }
            )
        self._wait_until_ready()
        self._started = True
        _atomic_write_json(
            self.root / "pool_manifest.json",
            {
                "status": "ready",
                "gpu_ids": self.gpu_ids,
                "parallelism": self.parallelism,
                "pool_signature": self._pool_signature,
                "workers": [
                    {
                        "gpu_id": row["gpu_id"],
                        "pid": row["process"].pid,
                        "log_path": row["log_path"],
                        "init_path": row["init_path"],
                    }
                    for row in self._workers
                ],
                "isolation": "one_persistent_process_per_physical_gpu",
                "nested_tensorrt_cuda_visible_devices": True,
            },
        )
        return self

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            failed = [
                row
                for row in self._workers
                if row["process"].poll() is not None
                and not row["ready_path"].is_file()
            ]
            if failed:
                details = ",".join(
                    f"gpu={row['gpu_id']}:rc={row['process'].returncode}:log={row['log_path']}"
                    for row in failed
                )
                self.close()
                raise RuntimeError(f"stage2_worker_start_failed:{details}")
            if all(row["ready_path"].is_file() for row in self._workers):
                return
            time.sleep(self.poll_interval_seconds)
        self.close()
        raise TimeoutError("stage2_worker_start_timeout")

    def map_tasks(self, tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        self.start()
        ordered_tasks = [dict(task) for task in tasks]
        ordered_results: list[dict[str, Any] | None] = [None] * len(ordered_tasks)
        uncached: list[tuple[int, dict[str, Any]]] = []
        for index, task in enumerate(ordered_tasks):
            cache_key = str(task.get("candidate_hash", ""))
            cached = self._result_cache.get(cache_key) if cache_key else None
            if cached is None:
                uncached.append((index, task))
                continue
            reused = dict(cached)
            reused["pool_cache_hit"] = True
            reused["reused_pool_task_id"] = str(cached.get("pool_task_id", ""))
            if task.get("output_dir"):
                reused["pool_reuse_requested_output_dir"] = str(task["output_dir"])
            ordered_results[index] = reused

        pending: dict[str, dict[str, Any]] = {}
        next_uncached = 0

        def dispatch(worker: dict[str, Any]) -> None:
            nonlocal next_uncached
            result_index, task = uncached[next_uncached]
            next_uncached += 1
            self._sequence += 1
            task_id = f"task_{self._sequence:08d}"
            result_path = self.results / f"{task_id}.json"
            payload = {
                **task,
                "task_id": task_id,
                "assigned_gpu_id": worker["gpu_id"],
                "pool_signature": self._pool_signature,
                "result_path": str(result_path),
            }
            task_path = worker["queue_dir"] / f"{task_id}.task.json"
            _atomic_write_json(task_path, payload)
            pending[task_id] = {
                "worker": worker,
                "task_id": task_id,
                "result_index": result_index,
                "result_path": result_path,
                "started_at": time.monotonic(),
            }

        initial_workers = [
            self._workers[(self._cursor + offset) % self.parallelism]
            for offset in range(min(len(uncached), self.parallelism))
        ]
        self._cursor = (self._cursor + len(initial_workers)) % self.parallelism
        for worker in initial_workers:
            dispatch(worker)

        while pending:
            progressed = False
            for task_id, item in list(pending.items()):
                worker = item["worker"]
                if worker["process"].poll() is not None:
                    raise RuntimeError(
                        "stage2_worker_died:"
                        f"gpu={worker['gpu_id']}:rc={worker['process'].returncode}:"
                        f"task={task_id}:log={worker['log_path']}"
                    )
                if time.monotonic() - float(item["started_at"]) > self.task_timeout_seconds:
                    raise TimeoutError(f"stage2_worker_task_timeout:['{task_id}']")
                result_path = item["result_path"]
                if not result_path.is_file():
                    continue
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                result.setdefault("worker_gpu_id", worker["gpu_id"])
                result["pool_task_id"] = task_id
                result["pool_signature"] = self._pool_signature
                result["pool_cache_hit"] = False
                _atomic_write_json(result_path, result)
                ordered_results[int(item["result_index"])] = result
                cache_key = str(result.get("candidate_hash", ""))
                if cache_key and str(result.get("status", "")) == "ok":
                    self._result_cache[cache_key] = dict(result)
                del pending[task_id]
                if next_uncached < len(uncached):
                    dispatch(worker)
                progressed = True
            if pending and not progressed:
                time.sleep(self.poll_interval_seconds)

        if any(result is None for result in ordered_results):
            raise RuntimeError("stage2_pool_internal_result_ordering_failure")
        return [result for result in ordered_results if result is not None]

    def _wait_for_results(self, pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.task_timeout_seconds
        resolved: dict[str, dict[str, Any]] = {}
        while time.monotonic() < deadline and len(resolved) < len(pending):
            for item in pending:
                task_id = str(item["task_id"])
                if task_id in resolved:
                    continue
                worker = item["worker"]
                if worker["process"].poll() is not None:
                    raise RuntimeError(
                        "stage2_worker_died:"
                        f"gpu={worker['gpu_id']}:rc={worker['process'].returncode}:"
                        f"task={task_id}:log={worker['log_path']}"
                    )
                path = item["result_path"]
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                payload.setdefault("worker_gpu_id", worker["gpu_id"])
                payload["pool_task_id"] = task_id
                payload["pool_signature"] = self._pool_signature
                _atomic_write_json(path, payload)
                resolved[task_id] = payload
            if len(resolved) < len(pending):
                time.sleep(self.poll_interval_seconds)
        if len(resolved) != len(pending):
            missing = [
                str(item["task_id"])
                for item in pending
                if str(item["task_id"]) not in resolved
            ]
            raise TimeoutError(f"stage2_worker_task_timeout:{missing}")
        return [resolved[str(item["task_id"])] for item in pending]

    def close(self) -> None:
        closed_workers: list[dict[str, Any]] = []
        for worker in self._workers:
            Path(worker["stop_path"]).touch(exist_ok=True)
        for worker in self._workers:
            process = worker["process"]
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            worker["log_handle"].close()
            closed_workers.append(
                {
                    "gpu_id": worker["gpu_id"],
                    "pid": process.pid,
                    "returncode": process.returncode,
                    "log_path": worker["log_path"],
                    "init_path": worker["init_path"],
                }
            )
        manifest_path = self.root / "pool_manifest.json"
        if closed_workers or manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {
                    "gpu_ids": self.gpu_ids,
                    "parallelism": self.parallelism,
                }
            manifest.update(
                {
                    "status": "stopped",
                    "stopped_epoch_seconds": time.time(),
                    "workers": closed_workers,
                }
            )
            _atomic_write_json(manifest_path, manifest)
        self._workers.clear()
        self._started = False

    def __enter__(self) -> "PersistentStage2ProcessPool":
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
