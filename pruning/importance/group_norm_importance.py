from __future__ import annotations

from typing import Any


def group_norm_scores(groups: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for group in groups:
        channels = group.get("channel_indices") or []
        scores[str(group.get("group_id"))] = float(len(channels))
    return scores
