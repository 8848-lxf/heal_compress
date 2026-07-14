"""Join origin mapping and precision profiles without heuristic aliases."""

from __future__ import annotations

from ..config import QDQConfig
from ..exceptions import CanonicalMappingError
from ..types import (
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
    OnnxOriginMapResult,
    PrecisionProfileResult,
)
from .policies import legalize_int8_request


def build_canonical_precision_mapping(
    origin_map: OnnxOriginMapResult,
    profile: PrecisionProfileResult,
    *,
    config: QDQConfig | None = None,
) -> CanonicalPrecisionMappingResult:
    """Map every canonical weighted call to its requested deployment precision."""

    assignments = {row.module_path: row for row in profile.assignments}
    origin_modules = {row.module_path for row in origin_map.entries}
    missing_assignments = sorted(origin_modules - set(assignments))
    unknown_assignments = sorted(set(assignments) - origin_modules)
    if missing_assignments or unknown_assignments:
        raise CanonicalMappingError(
            f"origin/profile module sets differ; missing={missing_assignments}, unknown={unknown_assignments}"
        )
    names = [row.canonical_node_name for row in origin_map.entries]
    if len(names) != len(set(names)):
        raise CanonicalMappingError("origin map contains duplicate canonical node names")
    entries: list[CanonicalPrecisionEntry] = []
    for origin in sorted(origin_map.entries, key=lambda row: (row.call_index, row.graph_index)):
        assignment = assignments[origin.module_path]
        realized = assignment.requested_precision
        fallback_reason = assignment.fallback_reason
        if assignment.requested_precision == "int8":
            realized, fallback_reason = legalize_int8_request(origin, config=config)
        entries.append(
            CanonicalPrecisionEntry(
                module_path=origin.module_path,
                canonical_node_name=origin.canonical_node_name,
                original_node_name=origin.original_node_name,
                weight_initializer=origin.weight_initializer,
                onnx_op_type=origin.onnx_op_type,
                call_index=origin.call_index,
                precision_group=assignment.precision_group,
                requested_precision=assignment.requested_precision,
                realized_request_precision=realized,
                fallback_reason=fallback_reason,
                protected_precision=assignment.protected_precision,
            )
        )
    for group in sorted(origin_map.functional_compute_groups, key=lambda row: row.canonical_node_name):
        entries.append(
            CanonicalPrecisionEntry(
                module_path=group.module_path,
                canonical_node_name=group.canonical_node_name,
                original_node_name=" + ".join(group.original_node_names),
                weight_initializer="",
                onnx_op_type=group.onnx_op_type,
                call_index=len(entries),
                precision_group="protected_functional_affine_grid",
                requested_precision=group.requested_precision,
                realized_request_precision=group.protected_precision,
                realized_output_precision=group.protected_precision,
                protected_precision=group.protected_precision,
                fallback_reason=group.protection_reason,
                constraint_node_names=tuple(
                    f"{group.canonical_node_name}__member{index:02d}"
                    for index, _ in enumerate(group.graph_indices)
                ),
            )
        )
    return CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id=profile.profile_id,
        profile_hash=profile.profile_hash,
        origin_map_hash=origin_map.origin_map_hash,
    )
