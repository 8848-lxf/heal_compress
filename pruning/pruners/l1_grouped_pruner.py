from __future__ import annotations

from typing import Any

from pruning.pruners.general_pruner import prune_model


def l1_grouped_prune(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("importance_scores", None)
    return prune_model(*args, **kwargs)
