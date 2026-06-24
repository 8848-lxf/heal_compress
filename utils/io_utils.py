"""File IO utilities for JSON, YAML, NPZ, and checkpoint files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create directory and all parents if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load YAML from {path}: {exc}") from exc


def save_yaml(data: Any, path: str | os.PathLike[str]) -> None:
    """Save data to a YAML file.

    Args:
        data: Data to serialize.
        path: Output file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save YAML to {path}: {exc}") from exc


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load a JSON file and return its contents.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON contents.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load JSON from {path}: {exc}") from exc


def save_json(data: Any, path: str | os.PathLike[str], indent: int = 2) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to serialize.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    ensure_dir(path.parent)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    except Exception as exc:
        raise RuntimeError(f"Failed to save JSON to {path}: {exc}") from exc


def save_text(text: str, path: str | os.PathLike[str]) -> None:
    """Save plain text to a file.

    Args:
        text: Text content.
        path: Output file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_csv(rows: list[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    """Save a list of dicts as CSV.

    Args:
        rows: List of row dicts.
        path: Output CSV file path.
    """
    import csv

    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
