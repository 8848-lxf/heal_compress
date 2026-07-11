"""Atomic artifact I/O used by the formal deployment packages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    """Hash a file with SHA256, returning an empty string when absent."""

    source = Path(path)
    if not source.is_file():
        return ""
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically replace a binary artifact and fsync its directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Write a deterministic, human-readable JSON artifact atomically."""

    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n"
    return atomic_write_bytes(path, raw)


def load_json(path: str | Path, default: Any = None) -> Any:
    """Load JSON without silently accepting malformed existing files."""

    source = Path(path)
    if not source.is_file():
        return default
    return json.loads(source.read_text(encoding="utf-8"))
