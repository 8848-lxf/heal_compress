"""Atomic JSON serialization for formal trace artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .exceptions import TraceSerializationError
from .hashing import to_stable_primitive
from .types import TraceResult


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace ``path`` with deterministic, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                to_stable_primitive(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def read_json(path: str | Path) -> Any:
    """Load JSON without object hooks or executable deserialization."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceSerializationError(f"cannot load trace JSON {path}: {exc}") from exc


def serialize_trace_result(result: TraceResult, path: str | Path) -> Path:
    """Write a typed trace result using an atomic file replacement."""

    return atomic_write_json(path, result.to_dict())


def load_trace_result(path: str | Path) -> TraceResult:
    """Load and schema-check a formal trace result."""

    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise TraceSerializationError("trace JSON root must be an object")
    version = str(payload.get("graph_schema_version", ""))
    if not version.startswith("trace-result-"):
        raise TraceSerializationError(f"unsupported graph schema version: {version or '<missing>'}")
    try:
        return TraceResult.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise TraceSerializationError(f"invalid trace result schema: {exc}") from exc

