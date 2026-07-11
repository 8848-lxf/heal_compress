"""One canonical SHA256 implementation for physical artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ..types import PhysicalHashes, PhysicalStructureSnapshot, stable_json_hash
from .schemas import HASH_SCHEMA_VERSION, SHAPE_HASH_FIELDS, STRUCTURE_HASH_FIELDS


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _rows(snapshot: Mapping[str, Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    result = [
        {field: row.get(field) for field in fields}
        for row in snapshot.get("modules", [])
        if isinstance(row, Mapping)
    ]
    return sorted(
        result,
        key=lambda row: (int(row.get("canonical_order", 0)), str(row.get("canonical_module_name", ""))),
    )


def compute_physical_hashes(
    snapshot: PhysicalStructureSnapshot | Mapping[str, Any],
    *,
    model_hash: str = "",
    config_hash: str = "",
) -> PhysicalHashes:
    """Hash structure, shapes, and the canonical snapshot independently."""

    payload = snapshot.to_dict() if isinstance(snapshot, PhysicalStructureSnapshot) else dict(snapshot)
    structure = {"schema": HASH_SCHEMA_VERSION, "structure": _rows(payload, STRUCTURE_HASH_FIELDS)}
    shape = {"schema": HASH_SCHEMA_VERSION, "shape": _rows(payload, SHAPE_HASH_FIELDS)}
    snapshot_core = {
        "snapshot_schema_version": payload.get("snapshot_schema_version"),
        "generated_from": payload.get("generated_from"),
        "modules": payload.get("modules", []),
    }
    return PhysicalHashes(
        structure_hash_v2=sha256_bytes(stable_json_bytes(structure)),
        shape_hash_v2=sha256_bytes(stable_json_bytes(shape)),
        snapshot_hash=sha256_bytes(stable_json_bytes(snapshot_core)),
        hash_schema_version=HASH_SCHEMA_VERSION,
        model_hash=str(model_hash),
        config_hash=str(config_hash),
    )


__all__ = ["compute_physical_hashes", "sha256_bytes", "stable_json_bytes"]
