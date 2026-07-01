from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def find_trtexec(trt_root: str | None = None, explicit_trtexec: str | None = None) -> dict[str, Any]:
    checked: list[str] = []
    if explicit_trtexec:
        path = Path(explicit_trtexec).expanduser()
        checked.append(str(path))
        if path.is_file():
            return {"trtexec_found": True, "trtexec_path": str(path), "checked_paths": checked}
    if trt_root:
        root = Path(trt_root).expanduser()
        for rel in ("bin/trtexec", "targets/x86_64-linux-gnu/bin/trtexec"):
            path = root / rel
            checked.append(str(path))
            if path.is_file():
                return {"trtexec_found": True, "trtexec_path": str(path), "checked_paths": checked}
    checked.append("PATH")
    path = shutil.which("trtexec")
    return {"trtexec_found": bool(path), "trtexec_path": path, "checked_paths": checked}


def env_report(trt_root: str | None = None, explicit_trtexec: str | None = None) -> dict[str, Any]:
    report = find_trtexec(trt_root=trt_root, explicit_trtexec=explicit_trtexec)
    try:
        import tensorrt as trt

        report["tensorrt_available"] = True
        report["tensorrt_version"] = getattr(trt, "__version__", "unknown")
    except Exception as exc:
        report["tensorrt_available"] = False
        report["tensorrt_error"] = repr(exc)
    return report
