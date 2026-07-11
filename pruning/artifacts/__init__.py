"""Formal physical artifacts and deprecated v10.8 compatibility exports."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

from .hashing import compute_physical_hashes, sha256_bytes, stable_json_bytes
from .io import atomic_write_binary, atomic_write_json, load_json
from .provenance import PhysicalArtifactProvenance
from .schemas import HASH_SCHEMA_VERSION, LEDGER_SCHEMA_VERSION, SNAPSHOT_SCHEMA_VERSION
from .snapshot import build_physical_structure_snapshot

_LEGACY_NAMES = {"save_v108_model_artifacts", "load_v108_model_object", "smoke_v108_model"}


def _load_legacy() -> Any:
    package = __name__.rsplit(".", 1)[0]
    module_name = f"{package}._legacy_artifacts"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().parent.parent / "artifacts.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load compatibility artifacts from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def __getattr__(name: str) -> Any:
    if name in _LEGACY_NAMES:
        warnings.warn(
            f"pruning.artifacts.{name} is deprecated; use the formal artifact API",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_load_legacy(), name)
    raise AttributeError(name)


__all__ = [
    "HASH_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "PhysicalArtifactProvenance",
    "SNAPSHOT_SCHEMA_VERSION",
    "atomic_write_binary",
    "atomic_write_json",
    "build_physical_structure_snapshot",
    "compute_physical_hashes",
    "load_json",
    "sha256_bytes",
    "stable_json_bytes",
]
