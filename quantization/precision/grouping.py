"""Precision-group helpers."""

from __future__ import annotations

from typing import Sequence


def stable_precision_groups(module_paths: Sequence[str]) -> dict[str, list[str]]:
    """Create one deterministic precision group per weighted module."""

    return {f"pg::{name}": [name] for name in sorted({str(value) for value in module_paths})}
