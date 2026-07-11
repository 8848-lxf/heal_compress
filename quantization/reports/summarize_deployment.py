"""Summarize formal deployment artifacts without invoking external tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts.io import atomic_write_json, file_sha256


def summarize_deployment(root: str | Path) -> dict[str, Any]:
    """Inventory ONNX, engine, mapping, profile, and provenance artifacts."""

    directory = Path(root)
    patterns = {
        "onnx": "*.onnx",
        "engine": "*.engine",
        "profile": "*precision_profile*.json",
        "canonical_mapping": "*canonical*mapping*.json",
        "provenance": "*provenance*.json",
    }
    artifacts = {
        kind: [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in sorted(directory.rglob(pattern))
            if path.is_file()
        ]
        for kind, pattern in patterns.items()
    }
    return {
        "schema_version": "formal-deployment-summary-v1",
        "root": str(directory),
        "artifacts": artifacts,
        "artifact_counts": {key: len(value) for key, value in artifacts.items()},
    }


def write_deployment_summary(root: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = summarize_deployment(root)
    atomic_write_json(output_path, report)
    return report
