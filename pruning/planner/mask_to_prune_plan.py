from __future__ import annotations

from typing import Any


def mask_to_keep_indices(mask: list[bool] | list[int]) -> list[int]:
    return [idx for idx, value in enumerate(mask) if bool(value)]


def mask_record_to_plan_row(record: dict[str, Any]) -> dict[str, Any]:
    keep = mask_to_keep_indices(record.get("mask", []))
    total = len(record.get("mask", []))
    return {
        "group_id": str(record.get("group_id")),
        "source_modules": list(record.get("source_modules") or []),
        "dependent_modules": list(record.get("dependent_modules") or []),
        "num_channels": total,
        "keep_indices": keep,
        "prune_indices": [idx for idx in range(total) if idx not in set(keep)],
        "keep_count": len(keep),
        "prune_count": total - len(keep),
    }
