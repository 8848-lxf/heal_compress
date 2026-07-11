"""Formal selectors plus lazy access to the pre-formal compatibility API."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

from .budgeted import ratio_to_budget
from .constrained import filter_selectable_units
from .global_ranking import select_global_units
from .grouped_conv import select_grouped_conv_channels
from .local import rank_within_scope

_LEGACY_NAMES = {"SelectionConfig", "PruningPlan", "build_pruning_plan", "_grouped_keep_per_group"}


def _load_legacy() -> Any:
    package = __name__.rsplit(".", 1)[0]
    module_name = f"{package}._legacy_selection"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().parent.parent / "selection.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load compatibility selector from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def __getattr__(name: str) -> Any:
    if name in _LEGACY_NAMES:
        warnings.warn(
            f"pruning.selection.{name} is deprecated; use pruning.api.select_pruning_request",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_load_legacy(), name)
    raise AttributeError(name)


__all__ = [
    "PruningPlan",
    "SelectionConfig",
    "build_pruning_plan",
    "filter_selectable_units",
    "rank_within_scope",
    "ratio_to_budget",
    "select_global_units",
    "select_grouped_conv_channels",
]
