from __future__ import annotations

from typing import Any


def prune_model(*args: Any, **kwargs: Any) -> Any:
    try:
        from ..general_pruner import prune_model as _prune_model
    except ImportError:
        from pruning.general_pruner import prune_model as _prune_model

    return _prune_model(*args, **kwargs)


def GeneralPruner(*args: Any, **kwargs: Any) -> Any:
    try:
        from ..general_pruner import GeneralPruner as _GeneralPruner
    except ImportError:
        from pruning.general_pruner import GeneralPruner as _GeneralPruner

    return _GeneralPruner(*args, **kwargs)
