"""Validation for repaired phenotype to physical pruning plan consistency."""

from __future__ import annotations

from typing import Any, Mapping


def _canon_map(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canon_map(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return sorted(_canon_map(item) for item in value)
    return value


def _request_prune_by_axis(request: Any) -> dict[str, list[int]]:
    result: dict[str, set[int]] = {}
    for entry in getattr(request, "entries", []) or []:
        key = f"{getattr(entry, 'module_path', '')}::{getattr(entry, 'axis', '')}"
        result.setdefault(key, set()).update(int(value) for value in getattr(entry, "prune_indices", []) or [])
    return {key: sorted(values) for key, values in sorted(result.items())}


def _plan_prune_by_axis(plan: Any) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for entry in getattr(plan, "entries", []) or []:
        key = f"{getattr(entry, 'module_path', '')}::{getattr(entry, 'axis', '')}"
        result[key] = sorted({int(value) for value in getattr(entry, "prune_indices", []) or []})
    return {key: result[key] for key in sorted(result)}


def _request_group_maps(request: Any, attribute: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in getattr(request, "entries", []) or []:
        mapping = getattr(entry, attribute, {}) or {}
        if mapping:
            key = f"{getattr(entry, 'module_path', '')}::{getattr(entry, 'axis', '')}"
            result[key] = _canon_map(mapping)
    return result


def _plan_group_maps(plan: Any, attribute: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in getattr(plan, "entries", []) or []:
        mapping = getattr(entry, attribute, {}) or {}
        if mapping:
            key = f"{getattr(entry, 'module_path', '')}::{getattr(entry, 'axis', '')}"
            result[key] = _canon_map(mapping)
    return result


def _phenotype_group_maps_used_by_request(phenotype: Any, request: Any, attribute: str) -> dict[str, Any]:
    metadata = dict(getattr(phenotype, "metadata", {}) or {})
    by_scope = {
        str(scope): _canon_map(mapping)
        for scope, mapping in dict(metadata.get(f"{attribute}_by_scope") or {}).items()
    }
    result: dict[str, Any] = {}
    for entry in getattr(request, "entries", []) or []:
        mapping = getattr(entry, attribute, {}) or {}
        if not mapping:
            continue
        scope = str(getattr(entry, "scope_id", ""))
        key = f"{getattr(entry, 'module_path', '')}::{getattr(entry, 'axis', '')}"
        result[key] = by_scope.get(scope, _canon_map(mapping))
    return result


def validate_repaired_physical_plan(phenotype: Any, request: Any, plan: Any) -> dict[str, Any]:
    """Fail closed if Stage-2 changed repaired unit IDs, indices, or group maps."""

    phenotype_units = sorted(str(value) for value in getattr(phenotype, "pruned_unit_ids", []) or [])
    request_units = sorted(str(value) for value in getattr(request, "selected_atomic_unit_ids", []) or [])
    source_request = getattr(plan, "source_request", None)
    plan_units = sorted(str(value) for value in getattr(source_request, "selected_atomic_unit_ids", request_units) or [])
    request_indices = _request_prune_by_axis(request)
    plan_indices = _plan_prune_by_axis(plan)
    request_keep_maps = _request_group_maps(request, "group_keep_map")
    plan_keep_maps = _plan_group_maps(plan, "group_keep_map")
    request_prune_maps = _request_group_maps(request, "group_prune_map")
    plan_prune_maps = _plan_group_maps(plan, "group_prune_map")
    phenotype_keep_maps = _phenotype_group_maps_used_by_request(phenotype, request, "group_keep_map")
    phenotype_prune_maps = _phenotype_group_maps_used_by_request(phenotype, request, "group_prune_map")
    unit_equal = phenotype_units == request_units == plan_units
    indices_equal = request_indices == plan_indices
    keep_equal = request_keep_maps == plan_keep_maps and phenotype_keep_maps == request_keep_maps
    prune_equal = request_prune_maps == plan_prune_maps and phenotype_prune_maps == request_prune_maps
    issues: list[str] = []
    if not unit_equal or not indices_equal or not keep_equal or not prune_equal:
        issues.append("repaired_physical_plan_mismatch")
    return {
        "passed": not issues,
        "issues": issues,
        "phenotype_unit_ids": phenotype_units,
        "request_unit_ids": request_units,
        "physical_plan_unit_ids": plan_units,
        "repaired_mask_to_request_verified": phenotype_units == request_units,
        "repaired_mask_to_physical_plan_verified": unit_equal and indices_equal,
        "group_keep_map_frozen_verified": keep_equal,
        "group_prune_map_frozen_verified": prune_equal,
        "request_vs_plan_prune_indices_equal": indices_equal,
        "request_prune_indices_by_module_axis": request_indices,
        "physical_plan_prune_indices_by_module_axis": plan_indices,
        "request_group_keep_map_by_module_axis": request_keep_maps,
        "physical_plan_group_keep_map_by_module_axis": plan_keep_maps,
        "request_group_prune_map_by_module_axis": request_prune_maps,
        "physical_plan_group_prune_map_by_module_axis": plan_prune_maps,
    }

