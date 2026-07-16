"""Layered artifact cache index."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class ArtifactCache:
    """Tracks reusable physical, ONNX, Q/DQ ONNX, and engine artifacts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._rows[(str(row["kind"]), str(row["key"]))] = dict(row["value"])

    def _put(self, kind: str, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._rows[(kind, key)] = dict(value)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "key": key, "value": value}, sort_keys=True, ensure_ascii=True) + "\n")

    def _get(self, kind: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get((kind, key))
            return dict(row) if row is not None else None

    def put_physical(self, physical_hash: str, value: dict[str, Any]) -> None:
        self._put("physical", str(physical_hash), value)

    def get_physical(self, physical_hash: str) -> dict[str, Any] | None:
        return self._get("physical", str(physical_hash))

    def put_onnx(self, physical_hash: str, value: dict[str, Any]) -> None:
        self._put("onnx", str(physical_hash), value)

    def get_onnx(self, physical_hash: str) -> dict[str, Any] | None:
        return self._get("onnx", str(physical_hash))

    def put_qdq_onnx(self, candidate_hash: str, value: dict[str, Any]) -> None:
        self._put("qdq_onnx", str(candidate_hash), value)

    def get_qdq_onnx(self, candidate_hash: str) -> dict[str, Any] | None:
        return self._get("qdq_onnx", str(candidate_hash))

    def put_engine(self, candidate_hash: str, engine_environment_hash: str, value: dict[str, Any]) -> None:
        self._put("engine", f"{candidate_hash}:{engine_environment_hash}", value)

    def get_engine(self, candidate_hash: str, engine_environment_hash: str) -> dict[str, Any] | None:
        return self._get("engine", f"{candidate_hash}:{engine_environment_hash}")
