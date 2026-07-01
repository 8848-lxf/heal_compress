from __future__ import annotations

from typing import Any

from pruning.pruners.general_pruner import prune_model


def grouped_channel_prune(*args: Any, **kwargs: Any) -> Any:
    return prune_model(*args, **kwargs)
