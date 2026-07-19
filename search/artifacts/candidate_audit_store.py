"""Content-addressed storage for completed Stage-2 audit artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_ROLE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_PLAN_ALIASES = (
    "physical_pruning_plan.json",
    "physical_plan.json",
    "legalized_plan.json",
)
_REQUEST_ALIASES = (
    "pruning_request.json",
    "sampling_pruning_request.json",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _plain(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _plain(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        _plain(payload), indent=2, ensure_ascii=True, sort_keys=True
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class ArtifactReference:
    role: str
    sha256: str
    relative_path: str
    uncompressed_size: int
    compressed_size: int
    encoding: str = "canonical-json+gzip"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateAuditStore:
    """Store canonical JSON once per role and content hash."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def put_json(self, role: str, payload: Any) -> ArtifactReference:
        normalized_role = str(role).strip().lower()
        if not _ROLE_PATTERN.fullmatch(normalized_role):
            raise ValueError(f"invalid_audit_artifact_role:{role}")
        raw = _canonical_json_bytes(payload)
        digest = _sha256_bytes(raw)
        destination = (
            self.root
            / normalized_role
            / digest[:2]
            / f"{digest}.json.gz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            stored = gzip.decompress(destination.read_bytes())
            if _sha256_bytes(stored) != digest or stored != raw:
                raise RuntimeError(
                    f"audit_store_hash_collision_or_corruption:{destination}"
                )
        else:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=handle, mtime=0
                ) as compressed:
                    compressed.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        return ArtifactReference(
            role=normalized_role,
            sha256=digest,
            relative_path=str(destination.relative_to(self.root)),
            uncompressed_size=len(raw),
            compressed_size=int(destination.stat().st_size),
        )


def _coerce_reference(
    reference: ArtifactReference | Mapping[str, Any],
) -> ArtifactReference:
    if isinstance(reference, ArtifactReference):
        return reference
    return ArtifactReference(
        role=str(reference["role"]),
        sha256=str(reference["sha256"]),
        relative_path=str(reference["relative_path"]),
        uncompressed_size=int(reference["uncompressed_size"]),
        compressed_size=int(reference["compressed_size"]),
        encoding=str(reference.get("encoding", "canonical-json+gzip")),
    )


def resolve_artifact_reference(
    store_root: str | Path,
    reference: ArtifactReference | Mapping[str, Any],
) -> Any:
    """Load and verify a referenced canonical JSON artifact."""

    ref = _coerce_reference(reference)
    root = Path(store_root).expanduser().resolve()
    path = (root / ref.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"audit_reference_escapes_store:{path}") from exc
    if not path.is_file():
        raise RuntimeError(f"audit_reference_missing:{path}")
    raw = gzip.decompress(path.read_bytes())
    if len(raw) != ref.uncompressed_size:
        raise RuntimeError(f"audit_reference_size_mismatch:{path}")
    if _sha256_bytes(raw) != ref.sha256:
        raise RuntimeError(f"audit_reference_hash_mismatch:{path}")
    return json.loads(raw.decode("utf-8"))


def _load_consistent_alias_group(
    candidate_dir: Path,
    aliases: Sequence[str],
    *,
    role: str,
) -> tuple[Any, list[Path]]:
    rows: list[tuple[Path, Any, str]] = []
    for name in aliases:
        path = candidate_dir / name
        if not path.is_file() or path.is_symlink():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path, payload, _sha256_bytes(_canonical_json_bytes(payload))))
    if not rows:
        raise RuntimeError(f"audit_artifact_group_missing:{role}:{candidate_dir}")
    hashes = {row[2] for row in rows}
    if len(hashes) != 1:
        detail = ",".join(f"{row[0].name}:{row[2]}" for row in rows)
        raise RuntimeError(f"artifact_alias_mismatch:{role}:{detail}")
    return rows[0][1], [row[0] for row in rows]


def _entry_index_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    return None


def _structure_summary(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    plan_reference: ArtifactReference,
    request_reference: ArtifactReference,
) -> dict[str, Any]:
    plan_entries = [
        dict(row) for row in plan.get("entries", []) if isinstance(row, Mapping)
    ]
    request_entries = [
        dict(row)
        for row in request.get("entries", [])
        if isinstance(row, Mapping)
    ]
    modules = sorted(
        {
            str(row.get("module_path", ""))
            for row in [*plan_entries, *request_entries]
            if str(row.get("module_path", ""))
        }
    )
    widths = []
    for row in plan_entries:
        original = row.get("original_axis_size")
        keep_count = _entry_index_count(row.get("keep_indices"))
        prune_count = _entry_index_count(row.get("prune_indices"))
        widths.append(
            {
                "module_path": str(row.get("module_path", "")),
                "axis": row.get("axis"),
                "original_axis_size": original,
                "keep_count": keep_count,
                "prune_count": prune_count,
                "repaired": bool(row.get("repaired", False)),
                "repair_reason": str(row.get("repair_reason", "")),
            }
        )
    selected = [str(value) for value in request.get("selected_atomic_unit_ids", [])]
    return {
        "schema_version": "candidate-structure-plan-summary-v1",
        "physical_plan_sha256": plan_reference.sha256,
        "pruning_request_sha256": request_reference.sha256,
        "plan_schema_version": str(plan.get("schema_version", "")),
        "request_schema_version": str(request.get("schema_version", "")),
        "indices_frozen_before_materialization": bool(
            plan.get("indices_frozen_before_materialization", False)
        ),
        "plan_entry_count": len(plan_entries),
        "request_entry_count": len(request_entries),
        "selected_atomic_unit_count": len(selected),
        "selected_atomic_unit_ids": selected,
        "requested_channel_cost": request.get("requested_channel_cost"),
        "requested_parameter_cost": request.get("requested_parameter_cost"),
        "conflict_count": len(plan.get("conflicts", []) or []),
        "modules": modules,
        "widths": widths,
    }


def finalize_completed_candidate(
    candidate_dir: str | Path,
    *,
    store: CandidateAuditStore,
    completion_marker: str | Path,
) -> dict[str, Any]:
    """Replace redundant completed-candidate plans with verified references."""

    candidate = Path(candidate_dir).expanduser().resolve()
    marker = Path(completion_marker).expanduser().resolve()
    if not marker.is_file():
        raise RuntimeError(f"completion_marker_missing:{marker}")
    expected_stage2 = marker.parent / "stage2"
    try:
        candidate.relative_to(expected_stage2)
    except ValueError as exc:
        raise RuntimeError(
            f"candidate_not_owned_by_completion_marker:{candidate}:{marker}"
        ) from exc
    manifest_path = candidate / "candidate_audit_manifest.json"
    if manifest_path.is_file() and not any(
        (candidate / name).is_file()
        for name in (*_PLAN_ALIASES, *_REQUEST_ALIASES)
    ):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "status": "already_compacted",
            "candidate_dir": str(candidate),
            "removed_file_count": 0,
            "manifest": existing,
        }

    plan, plan_paths = _load_consistent_alias_group(
        candidate, _PLAN_ALIASES, role="physical_plan"
    )
    request, request_paths = _load_consistent_alias_group(
        candidate, _REQUEST_ALIASES, role="pruning_request"
    )
    plan_reference = store.put_json("physical_plan", plan)
    request_reference = store.put_json("pruning_request", request)
    for reference in (plan_reference, request_reference):
        resolve_artifact_reference(store.root, reference)

    summary = _structure_summary(
        plan,
        request,
        plan_reference=plan_reference,
        request_reference=request_reference,
    )
    manifest = {
        "schema_version": "candidate-audit-manifest-v1",
        "candidate_id": candidate.name,
        "store_root_relative": os.path.relpath(store.root, candidate),
        "artifacts": {
            "physical_plan": plan_reference.to_dict(),
            "pruning_request": request_reference.to_dict(),
        },
        "structure_plan_summary": "structure_plan_summary.json",
        "source_files": sorted(path.name for path in (*plan_paths, *request_paths)),
    }
    _atomic_write_json(candidate / "structure_plan_summary.json", summary)
    _atomic_write_json(manifest_path, manifest)

    removed = []
    for path in (*plan_paths, *request_paths):
        size = int(path.stat().st_size)
        path.unlink()
        removed.append({"name": path.name, "size_bytes": size})
    return {
        "status": "compacted",
        "candidate_dir": str(candidate),
        "removed_file_count": len(removed),
        "removed_bytes": sum(int(row["size_bytes"]) for row in removed),
        "removed_files": removed,
        "manifest": manifest,
    }


__all__ = [
    "ArtifactReference",
    "CandidateAuditStore",
    "finalize_completed_candidate",
    "resolve_artifact_reference",
]

