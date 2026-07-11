"""Explicit TensorRT plugin loading."""

from __future__ import annotations

import ctypes
from pathlib import Path

from ..exceptions import TensorRTConfigurationError


def load_plugin(plugin_path: str | Path | None) -> str:
    """Load a caller-selected plugin library before engine deserialization."""

    if plugin_path is None:
        return ""
    path = Path(plugin_path).expanduser().resolve()
    if not path.is_file():
        raise TensorRTConfigurationError(f"TensorRT plugin does not exist: {path}")
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return str(path)
