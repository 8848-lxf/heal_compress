"""Physical snapshot truth gates and structure validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from ..artifacts.hashing import compute_physical_hashes
from ..artifacts.schemas import SNAPSHOT_SCHEMA_VERSION
from ..artifacts.snapshot import build_physical_structure_snapshot
from ..config import GroupedConvConfig
from ..exceptions import ArtifactSchemaError, PhysicalStructureMismatchError
from ..types import PhysicalStructureSnapshot, PhysicalValidationResult
from .invariants import validate_module_invariants


SAMPLING_ONLY_FIELDS = frozenset(
    {"before_after_shapes", "module_channel_before_after", "sampling_after", "sampling_estimate"}
)


def require_physical_snapshot_v2(
    snapshot: PhysicalStructureSnapshot | Mapping[str, Any],
) -> PhysicalStructureSnapshot:
    """Accept only snapshot-v2 physical truth; reject sampling estimates."""

    if isinstance(snapshot, PhysicalStructureSnapshot):
        result = snapshot
    else:
        payload = dict(snapshot)
        sampling = sorted(SAMPLING_ONLY_FIELDS & set(payload))
        if sampling:
            raise ArtifactSchemaError(
                "sampling estimate cannot be used as physical truth: " + ", ".join(sampling)
            )
        if payload.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ArtifactSchemaError(
                f"expected {SNAPSHOT_SCHEMA_VERSION}, got {payload.get('snapshot_schema_version')!r}"
            )
        result = PhysicalStructureSnapshot.from_dict(payload)
    if result.snapshot_schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ArtifactSchemaError(f"invalid physical snapshot schema: {result.snapshot_schema_version}")
    if result.generated_from != "live_model_and_state_dict":
        raise ArtifactSchemaError("physical snapshot must be generated from live_model_and_state_dict")
    return result


def validate_physical_model(
    model: nn.Module,
    *,
    state_dict: Mapping[str, Any] | None = None,
    expected_snapshot: PhysicalStructureSnapshot | Mapping[str, Any] | None = None,
    grouped_config: GroupedConvConfig | None = None,
    fixed_output_contracts: Mapping[str, int] | None = None,
    example_inputs: Any | None = None,
) -> PhysicalValidationResult:
    """Validate live/state/snapshot shapes and optional CPU forward contract."""

    snapshot = build_physical_structure_snapshot(model, state_dict=state_dict)
    issues: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if name:
            issues.extend(validate_module_invariants(name, module, grouped_config=grouped_config))
    if expected_snapshot is not None:
        expected = require_physical_snapshot_v2(expected_snapshot)
        actual_hash = compute_physical_hashes(snapshot)
        expected_hash = compute_physical_hashes(expected)
        if actual_hash.shape_hash_v2 != expected_hash.shape_hash_v2:
            issues.append(
                {
                    "module_path": "",
                    "code": "snapshot_shape_hash_mismatch",
                    "detail": f"{actual_hash.shape_hash_v2} != {expected_hash.shape_hash_v2}",
                }
            )
    module_rows = {row.canonical_module_name: row for row in snapshot.modules}
    for module_path, expected_output in (fixed_output_contracts or {}).items():
        row = module_rows.get(module_path)
        actual = row.out_channels if row is not None else None
        if actual != int(expected_output):
            issues.append(
                {
                    "module_path": module_path,
                    "code": "fixed_output_contract_mismatch",
                    "detail": f"expected {expected_output}, got {actual}",
                }
            )
    forward_checked = False
    if example_inputs is not None:
        from .forward import run_forward_invariant

        try:
            run_forward_invariant(model, example_inputs)
            forward_checked = True
        except Exception as exc:  # noqa: BLE001 - converted to validation evidence
            issues.append(
                {
                    "module_path": "",
                    "code": "forward_invariant_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return PhysicalValidationResult(
        passed=not issues,
        issues=issues,
        snapshot=snapshot,
        hashes=compute_physical_hashes(snapshot),
        forward_checked=forward_checked,
        output_contract_checked=bool(fixed_output_contracts),
    )


__all__ = ["SAMPLING_ONLY_FIELDS", "require_physical_snapshot_v2", "validate_physical_model"]
