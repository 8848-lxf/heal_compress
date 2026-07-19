"""Local compatibility module for the missing HEAL DiscoNet fusion layer."""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional


DISCONET_MODULE = "opencood.models.fuse_modules.disco_fuse"
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")


class PixelWeightLayer(nn.Module):
    """Checkpoint-compatible DiscoNet pixel weight network."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1)
        self.bn1_1 = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1)
        self.bn1_3 = nn.BatchNorm2d(8)
        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(
            -1, value.size(-3), value.size(-2), value.size(-1)
        )
        value = functional.relu(self.bn1_1(self.conv1_1(value)))
        value = functional.relu(self.bn1_2(self.conv1_2(value)))
        value = functional.relu(self.bn1_3(self.conv1_3(value)))
        return functional.relu(self.conv1_4(value))


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def install_disconet_compat_module(
    heal_root: str | Path = DEFAULT_HEAL_ROOT,
) -> dict[str, Any]:
    """Install a local module only when HEAL has no native implementation."""

    root = Path(heal_root).expanduser().resolve()
    for path in (root.parent, root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    existing = sys.modules.get(DISCONET_MODULE)
    if existing is not None:
        implementation = (
            "local_compat"
            if getattr(existing, "__heal_compress_compat__", False)
            else "native"
        )
        return {
            "module_name": DISCONET_MODULE,
            "implementation": implementation,
            "source": str(getattr(existing, "__file__", "")),
            "source_sha256": _source_sha256()
            if implementation == "local_compat"
            else "native_not_hashed",
        }
    try:
        native = importlib.import_module(DISCONET_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != DISCONET_MODULE:
            raise
    else:
        return {
            "module_name": DISCONET_MODULE,
            "implementation": "native",
            "source": str(getattr(native, "__file__", "")),
            "source_sha256": "native_not_hashed",
        }

    importlib.import_module("opencood.models.fuse_modules")
    module = types.ModuleType(DISCONET_MODULE)
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "opencood.models.fuse_modules"
    module.__heal_compress_compat__ = True
    module.PixelWeightLayer = PixelWeightLayer
    sys.modules[DISCONET_MODULE] = module
    return {
        "module_name": DISCONET_MODULE,
        "implementation": "local_compat",
        "source": str(Path(__file__).resolve()),
        "source_sha256": _source_sha256(),
    }


__all__ = [
    "DISCONET_MODULE",
    "PixelWeightLayer",
    "install_disconet_compat_module",
]

