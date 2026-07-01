from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _add_parent() -> None:
    parent = Path(__file__).resolve().parents[3]
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


def prune_model(*args: Any, **kwargs: Any) -> Any:
    _add_parent()
    from heal_compress.pruning.general_pruner import prune_model as _prune_model

    return _prune_model(*args, **kwargs)


def GeneralPruner(*args: Any, **kwargs: Any) -> Any:
    _add_parent()
    from heal_compress.pruning.general_pruner import GeneralPruner as _GeneralPruner

    return _GeneralPruner(*args, **kwargs)
