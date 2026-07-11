from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_repo_parent_to_sys_path() -> None:
    """Deprecated no-op; callers must install/import the package normally."""

    return None


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_markdown(lines: list[str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def empty_trace_summary(error: str | None = None) -> dict[str, Any]:
    return {
        "success": False if error else True,
        "total_modules": 0,
        "traced_modules": 0,
        "conv_layers": 0,
        "bn_layers": 0,
        "residual_edges": 0,
        "concat_edges": 0,
        "fusion_edges": 0,
        "detection_head_edges": 0,
        "num_dependency_groups": 0,
        "num_coupled_channel_groups": 0,
        "invalid_or_unsupported_ops": [],
        "dynamic_paths_detected": [],
        "staticized_paths": [],
        "duplicate_coupled_layers_removed": 0,
        "min_channel_constraints_detected": [],
        "error": error,
    }
