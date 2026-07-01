from __future__ import annotations

from typing import Any


def normalize_coupled_group(group: dict[str, Any]) -> dict[str, Any]:
    """Return the stable formal coupled-channel-group JSON schema."""
    source_modules = list(dict.fromkeys(group.get("source_modules") or group.get("layers") or []))
    dependent_modules = list(dict.fromkeys(group.get("dependent_modules") or []))
    channel_indices = [int(v) for v in group.get("channel_indices", [])]
    return {
        "group_id": str(group.get("group_id") or group.get("id") or "group::unknown"),
        "group_type": str(group.get("group_type") or "conv_block"),
        "source_modules": source_modules,
        "dependent_modules": dependent_modules,
        "channel_indices": channel_indices,
        "input_channel_dependencies": dict(group.get("input_channel_dependencies") or {}),
        "output_channel_dependencies": dict(group.get("output_channel_dependencies") or {}),
        "is_prunable": bool(group.get("is_prunable", True)),
        "is_protected": bool(group.get("is_protected", False)),
        "protected_reason": str(group.get("protected_reason") or ""),
        "dynamic_branch_source": list(group.get("dynamic_branch_source") or ["default"]),
        "successor_mapping": dict(group.get("successor_mapping") or {}),
        "hardware_alignment_constraint": dict(group.get("hardware_alignment_constraint") or {}),
    }


def normalize_group_collection(groups: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_coupled_group(group) for group in groups]
    seen_layers: set[str] = set()
    duplicates = 0
    for group in normalized:
        unique_sources = []
        for layer in group["source_modules"]:
            if layer in seen_layers:
                duplicates += 1
                continue
            seen_layers.add(layer)
            unique_sources.append(layer)
        group["source_modules"] = unique_sources
    return {
        "schema_version": 1,
        "groups": normalized,
        "num_coupled_channel_groups": len(normalized),
        "duplicate_coupled_layers_removed": duplicates,
    }
