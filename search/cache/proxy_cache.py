"""JSONL proxy-evaluation cache."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable


class JsonlKeyValueCache:
    def __init__(self, path: str | Path, key_field: str = "candidate_hash") -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._rows: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row[self.key_field])
                payload = dict(row)
                payload.pop(self.key_field, None)
                self._rows[key] = payload

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get(str(key))
            return dict(row) if row is not None else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        key = str(key)
        payload = dict(value)
        with self._lock:
            self._rows[key] = payload
            row = {self.key_field: key, **payload}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    def get_or_evaluate(self, key: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        cached = self.get(key)
        if cached is not None:
            return cached
        result = dict(fn())
        self.put(key, result)
        return result


class ProxyCache(JsonlKeyValueCache):
    """Cache Stage-1 proxy metrics by candidate hash."""

    pass
