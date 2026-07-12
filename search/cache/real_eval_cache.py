"""JSONL real-evaluation cache."""

from __future__ import annotations

from pathlib import Path

from .proxy_cache import JsonlKeyValueCache


class RealEvalCache(JsonlKeyValueCache):
    """Cache Stage-2 real metrics by strict evaluation cache key."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, key_field="cache_key")
