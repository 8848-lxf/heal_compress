from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any


def load_tensorrt_plugin(plugin_path: str | Path | None) -> dict[str, Any]:
    if not plugin_path:
        return {"plugin_loaded": False, "plugin_path": "", "failure_reason": ""}
    path = Path(plugin_path).expanduser()
    if not path.is_file():
        raise RuntimeError(f"TensorRT plugin not found: {path}")
    resolved = path.resolve()
    ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)
    return {"plugin_loaded": True, "plugin_path": str(resolved), "failure_reason": ""}
