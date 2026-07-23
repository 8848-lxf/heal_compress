"""Artifact-gated scheduler state for the Transformer d_h experiment.

No process is killed by this module.  Stale locks are recognized from the
recorded PID/state/command and only the lock artifact is replaced.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, Iterator
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _process(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    try:
        state = (proc / "stat").read_text(encoding="utf-8").split()[2]
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        owner = (proc / "status").stat().st_uid
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return {"exists": False, "pid": pid}
    return {
        "exists": True,
        "pid": pid,
        "state": state,
        "command": command,
        "uid": owner,
        "active": state not in {"T", "t", "Z", "X"},
    }


def lock_is_stale(lock_payload: dict[str, Any], *, run_root: Path) -> tuple[bool, str]:
    try:
        pid = int(lock_payload["pid"])
    except (KeyError, TypeError, ValueError):
        return True, "invalid_or_missing_pid"
    process = _process(pid)
    if not process.get("exists"):
        return True, "pid_exited"
    if int(process.get("uid", -1)) != os.getuid():
        return False, "foreign_owner_active_lock"
    if not process.get("active"):
        return True, f"pid_not_active_state_{process.get('state')}"
    command = str(process.get("command", ""))
    if str(run_root.resolve()) not in command:
        return True, "pid_reused_or_command_not_for_run"
    return False, "active_valid_task"


class ExperimentState:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.resolve()
        self.root = self.run_root / "scheduler"
        self.heartbeats = self.root / "heartbeat"
        self.locks = self.root / "locks"
        self.failures = self.root / "failures"
        for path in (self.heartbeats, self.locks, self.failures):
            path.mkdir(parents=True, exist_ok=True)

    def update_phase(self, phase: str, **values: Any) -> None:
        path = self.root / "phase_state.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        payload[phase] = {**payload.get(phase, {}), **values, "updated_at": _now()}
        _atomic_json(path, payload)

    def append_attempt(self, row: dict[str, Any]) -> None:
        path = self.root / "attempt_ledger.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def heartbeat(self, attempt_id: str, **values: Any) -> None:
        _atomic_json(
            self.heartbeats / f"{attempt_id}.json",
            {"attempt_id": attempt_id, "timestamp": _now(), **values},
        )

    @contextmanager
    def attempt(
        self,
        *,
        phase: str,
        candidate_id: str,
        artifact_output_path: Path,
        retry_count: int = 0,
    ) -> Iterator[str]:
        attempt_id = f"{phase}-{candidate_id}-{uuid.uuid4().hex[:12]}"
        lock_path = self.locks / f"{phase}-{candidate_id}.json"
        if lock_path.is_file():
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            stale, reason = lock_is_stale(existing, run_root=self.run_root)
            if not stale:
                raise RuntimeError(f"active_scheduler_lock:{lock_path}:{reason}")
            existing["stale_detected_at"] = _now()
            existing["stale_reason"] = reason
            _atomic_json(self.failures / f"stale_lock_{attempt_id}.json", existing)
            lock_path.unlink(missing_ok=True)
        parent_pid = os.getppid()
        payload = {
            "phase": phase,
            "candidate_id": candidate_id,
            "attempt_id": attempt_id,
            "start_time": _now(),
            "finish_time": None,
            "exit_code": None,
            "failure_reason": None,
            "heartbeat": str(self.heartbeats / f"{attempt_id}.json"),
            "parent_pid": parent_pid,
            "child_pid": os.getpid(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "artifact_output_path": str(artifact_output_path),
            "retry_count": int(retry_count),
        }
        _atomic_json(lock_path, payload)
        self.append_attempt({**payload, "event": "started"})
        self.heartbeat(attempt_id, phase=phase, candidate_id=candidate_id, status="running")
        try:
            yield attempt_id
        except BaseException as exc:
            finish = {
                **payload,
                "event": "finished",
                "finish_time": _now(),
                "exit_code": 1,
                "failure_reason": repr(exc),
            }
            self.append_attempt(finish)
            self.heartbeat(attempt_id, phase=phase, candidate_id=candidate_id, status="failed")
            _atomic_json(self.failures / f"{attempt_id}.json", finish)
            lock_path.unlink(missing_ok=True)
            raise
        else:
            self.append_attempt(
                {
                    **payload,
                    "event": "finished",
                    "finish_time": _now(),
                    "exit_code": 0,
                    "failure_reason": None,
                }
            )
            self.heartbeat(attempt_id, phase=phase, candidate_id=candidate_id, status="complete")
            lock_path.unlink(missing_ok=True)


def require_phase_a_certificate(run_root: Path) -> dict[str, Any]:
    path = run_root / "phase_a_completion_certificate.json"
    if not path.is_file():
        raise RuntimeError("phase_a_completion_certificate_missing")
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate.get("status") != "accepted_from_existing_artifacts":
        raise RuntimeError("phase_a_completion_certificate_not_accepted")
    if not certificate.get("eligible_for_phase_b"):
        raise RuntimeError("phase_a_completion_certificate_not_eligible")
    return certificate


def require_phase_b_fixed500(run_root: Path) -> dict[str, Any]:
    path = run_root / "phase_b_accuracy_boundary.json"
    if not path.is_file():
        raise RuntimeError("phase_b_fixed500_certificate_missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("fixed500_complete"):
        raise RuntimeError("phase_b_fixed500_not_complete")
    return value
