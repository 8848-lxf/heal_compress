"""Fail-closed audit for pruning and quantization group ownership."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .quantization_space.types import QuantizationSearchGroup


def _pruning_modules(metadata: Mapping[str, Any]) -> set[str]:
    modules = {str(metadata.get("root_module_path", ""))}
    for row in metadata.get("closure_entries", []) or []:
        if isinstance(row, Mapping):
            modules.add(str(row.get("module_path", "")))
    return {module for module in modules if module}


def _inherits_precision_from_pruning_scope(group: QuantizationSearchGroup) -> bool:
    metadata = dict(group.metadata)
    source = str(metadata.get("precision_group_source", "")).lower()
    if source in {"pruning_scope", "pruning_domain", "pruning_dependency_group"}:
        return True
    return any(
        "pruning" in str(key).lower() and "scope" in str(key).lower()
        for key in metadata
    )


def audit_pruning_quantization_groups(
    *,
    pruning_group_ids: Sequence[str],
    pruning_group_metadata: Mapping[str, Mapping[str, Any]],
    quantization_groups: Sequence[QuantizationSearchGroup],
    canonical_mapping: Any,
) -> dict[str, Any]:
    """Audit disjoint group namespaces and their canonical-layer crosswalk."""

    pruning_ids = sorted({str(value) for value in pruning_group_ids})
    quantization_ids = sorted(str(group.group_id) for group in quantization_groups)
    collisions = sorted(set(pruning_ids) & set(quantization_ids))

    group_by_module: dict[str, list[str]] = {}
    for group in quantization_groups:
        for module_path in group.module_paths:
            group_by_module.setdefault(str(module_path), []).append(str(group.group_id))

    entries = list(getattr(canonical_mapping, "entries", []) or [])
    weighted_entries = [entry for entry in entries if str(getattr(entry, "weight_initializer", ""))]
    weighted_modules = {str(entry.module_path) for entry in weighted_entries}
    unmapped = sorted(module for module in weighted_modules if not group_by_module.get(module))

    duplicate_module_count = sum(
        max(0, len(set(group_ids)) - 1)
        for group_ids in group_by_module.values()
    )
    canonical_name_counts = Counter(str(entry.canonical_node_name) for entry in entries)
    duplicate_name_count = sum(max(0, count - 1) for count in canonical_name_counts.values())

    canonical_modules = {str(entry.module_path) for entry in entries}
    crosswalk = []
    for pruning_id in pruning_ids:
        for module_path in sorted(
            _pruning_modules(pruning_group_metadata.get(pruning_id, {})) & canonical_modules
        ):
            crosswalk.append(
                {
                    "pruning_group_id": pruning_id,
                    "canonical_module_path": module_path,
                    "quantization_group_ids": sorted(set(group_by_module.get(module_path, []))),
                }
            )

    inherited = sorted(
        str(group.group_id)
        for group in quantization_groups
        if _inherits_precision_from_pruning_scope(group)
    )
    duplicate_count = duplicate_module_count + duplicate_name_count
    failure_reasons = []
    if collisions:
        failure_reasons.append("pruning_quant_group_collision")
    if unmapped:
        failure_reasons.append("unmapped_weighted_layer")
    if duplicate_count:
        failure_reasons.append("duplicate_canonical_mapping")
    if inherited:
        failure_reasons.append("precision_inherited_from_pruning_scope")

    return {
        "passed": not failure_reasons,
        "status": "passed" if not failure_reasons else "failed",
        "pruning_group_count": len(pruning_ids),
        "quantization_group_count": len(quantization_ids),
        "pruning_group_ids": pruning_ids,
        "quantization_group_ids": quantization_ids,
        "crosswalk_mapping_count": len(crosswalk),
        "crosswalk_mappings": crosswalk,
        "id_collision_count": len(collisions),
        "id_collisions": collisions,
        "unmapped_weighted_layer_count": len(unmapped),
        "unmapped_weighted_layers": unmapped,
        "duplicate_canonical_mapping_count": duplicate_count,
        "precision_inherited_from_pruning_scope_count": len(inherited),
        "precision_inherited_from_pruning_scope_group_ids": inherited,
        "failure_reasons": failure_reasons,
    }


def write_group_separation_audit(
    path: str | Path,
    *,
    pruning_group_ids: Sequence[str],
    pruning_group_metadata: Mapping[str, Mapping[str, Any]],
    quantization_groups: Sequence[QuantizationSearchGroup],
    canonical_mapping: Any,
) -> dict[str, Any]:
    """Persist the group audit and stop deployment when it does not pass."""

    report = audit_pruning_quantization_groups(
        pruning_group_ids=pruning_group_ids,
        pruning_group_metadata=pruning_group_metadata,
        quantization_groups=quantization_groups,
        canonical_mapping=canonical_mapping,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    if not report["passed"]:
        raise RuntimeError(
            "pruning_quant_group_audit_failed:" + ",".join(report["failure_reasons"])
        )
    return report
