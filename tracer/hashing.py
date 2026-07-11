"""Deterministic JSON and SHA256 helpers for trace identities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_stable_primitive(value: Any) -> Any:
    """Convert ``value`` into a deterministic JSON-compatible structure."""

    if dataclasses.is_dataclass(value):
        return {
            field.name: to_stable_primitive(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): to_stable_primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [to_stable_primitive(item) for item in value]
    if isinstance(value, set):
        converted = [to_stable_primitive(item) for item in value]
        return sorted(converted, key=stable_json_dumps)
    if isinstance(value, float):
        # JSON has no portable representation for NaN or infinities. Trace
        # metadata must remain hashable across Python versions and platforms.
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    return value


def stable_json_dumps(value: Any) -> str:
    """Serialize using the single canonical JSON representation for tracer."""

    return json.dumps(
        to_stable_primitive(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    """Return the lowercase SHA256 digest of canonical JSON ``value``."""

    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any, *, length: int = 24) -> str:
    """Return a readable deterministic ID backed by a SHA256 digest."""

    return f"{prefix}_{stable_hash(value)[: int(length)]}"

