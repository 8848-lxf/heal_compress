from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "bytes": p.stat().st_size,
        "size_MB": round(p.stat().st_size / (1024 * 1024), 4),
        "sha256": file_sha256(p) if p.is_file() else None,
    }
