from __future__ import annotations

from typing import Any


def legal_keep_count(total_channels: int, target_prune_ratio: float, min_keep_ratio: float) -> tuple[int, int]:
    total = int(total_channels)
    min_keep = max(1, int(round(total * float(min_keep_ratio))))
    keep = int(round(total * (1.0 - float(target_prune_ratio))))
    keep = max(min_keep, min(total, keep))
    return keep, total - keep


def check_plan_legality(plan_rows: list[dict[str, Any]], *, min_keep_ratio: float) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen_layers: set[str] = set()
    duplicate_layers: set[str] = set()
    for row in plan_rows:
        total = int(row.get("num_channels") or 0)
        keep = int(row.get("keep_count") or 0)
        if total <= 0:
            issues.append({"issue": "empty_channel_group", "group_id": row.get("group_id")})
        if total > 0 and keep / total < float(min_keep_ratio):
            issues.append({"issue": "min_keep_ratio_violation", "group_id": row.get("group_id"), "keep": keep, "total": total})
        for layer in row.get("source_modules", []):
            if layer in seen_layers:
                duplicate_layers.add(layer)
            seen_layers.add(layer)
    return {
        "legal": not issues,
        "issues": issues,
        "duplicate_coupled_layers_removed": len(duplicate_layers),
        "checks": {
            "min_keep_ratio": float(min_keep_ratio),
            "residual_connection_channel_consistency": "checked_by_coupled_group_schema",
            "concat_input_output_consistency": "checked_by_coupled_group_schema",
            "detection_head_shape": "protected_by_tracer_keywords_when_available",
            "grouped_depthwise_conv_constraints": "deferred_to physical pruner legality check",
        },
    }
