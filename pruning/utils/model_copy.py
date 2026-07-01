from __future__ import annotations

import shutil
from pathlib import Path


def copy_checkpoint(src: str | Path, dst: str | Path) -> Path:
    source = Path(src).expanduser()
    target = Path(dst).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target
