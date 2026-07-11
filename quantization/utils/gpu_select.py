"""Compatibility helper requiring explicit device selection."""

from __future__ import annotations

from typing import Any


def select_idle_gpu(*_args: Any, **_kwargs: Any) -> Any:
    """Reject implicit GPU scheduling; callers must provide a device."""

    raise RuntimeError("automatic GPU scheduling is not part of the formal package; pass an explicit device")
