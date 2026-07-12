"""Resume helpers for archive-backed runs."""

from __future__ import annotations

from pathlib import Path


def archives_exist(output_root: str | Path) -> bool:
    root = Path(output_root) / "archives"
    return (root / "proxy_archive.jsonl").exists() or (root / "real_eval_archive.jsonl").exists()
