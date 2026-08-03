"""Cross-process frozen pruning-domain manifest.

The physical coupling graph is produced by the normal tracer/domain adapter.
This module freezes the *ranking result* after that graph is built. Every
controller and worker then decodes exactly the same legal width masks instead
of recollecting floating-point Taylor scores independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from search.pruning_space.local_domains import (
    LocalPruningDomain,
    local_pruning_domain_from_dict,
)


SCHEMA_VERSION = "v2xvit-frozen-domain-manifest-v1"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def domain_payloads(domains: Sequence[LocalPruningDomain]) -> list[dict[str, Any]]:
    return [domain.to_dict() for domain in sorted(domains, key=lambda row: row.domain_id)]


def domain_table_hash(domains: Sequence[LocalPruningDomain]) -> str:
    return stable_hash(domain_payloads(domains))


def build_manifest(
    domains: Sequence[LocalPruningDomain],
    *,
    anchor_phenotypes: Sequence[Mapping[str, Any]],
    source_root: str,
    calibration_hash: str,
    trace_hash: str = "",
) -> dict[str, Any]:
    rows = domain_payloads(domains)
    anchors = []
    for anchor in anchor_phenotypes:
        phenotype = dict(anchor.get("phenotype", anchor))
        metadata = dict(phenotype.get("metadata", {}))
        anchors.append({
            "candidate_hash": str(anchor.get("candidate_hash", "")),
            "physical_phenotype_hash": str(
                anchor.get("physical_phenotype_hash", "")
            ),
            "phenotype_hash": stable_hash(phenotype),
            "domain_metadata_hash": stable_hash(metadata.get("domains", {})),
        })
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(source_root),
        "calibration_hash": str(calibration_hash),
        "trace_hash": str(trace_hash),
        "domain_count": len(rows),
        "domains": rows,
        "domain_table_hash": stable_hash(rows),
        "anchors": anchors,
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> tuple[dict[str, Any], tuple[LocalPruningDomain, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"frozen_domain_manifest_schema_mismatch:{payload.get('schema_version')}"
        )
    expected = dict(payload)
    actual_hash = str(expected.pop("manifest_hash", ""))
    if not actual_hash or stable_hash(expected) != actual_hash:
        raise RuntimeError("frozen_domain_manifest_hash_mismatch")
    rows = payload.get("domains", [])
    if int(payload.get("domain_count", -1)) != len(rows):
        raise RuntimeError("frozen_domain_manifest_domain_count_mismatch")
    if stable_hash(rows) != str(payload.get("domain_table_hash", "")):
        raise RuntimeError("frozen_domain_manifest_domain_table_hash_mismatch")
    domains = tuple(local_pruning_domain_from_dict(row) for row in rows)
    ids = [domain.domain_id for domain in domains]
    if len(ids) != len(set(ids)):
        raise RuntimeError("frozen_domain_manifest_duplicate_domain_id")
    return payload, domains


def validate_against_replay(
    frozen: Sequence[LocalPruningDomain],
    replay: Sequence[LocalPruningDomain],
) -> None:
    """Check that the frozen table has the same traced physical schema."""

    by_id = {domain.domain_id: domain for domain in replay}
    if set(by_id) != {domain.domain_id for domain in frozen}:
        raise RuntimeError("frozen_domain_manifest_domain_id_mismatch")
    for domain in frozen:
        current = by_id[domain.domain_id]
        fields = (
            "root_module_path", "root_axis", "scope_id", "kind", "domain_type",
            "original_width", "total_original_width", "legal_widths", "groups",
            "unit_root_indices", "dependency_members", "constraints",
        )
        for field in fields:
            if getattr(domain, field) != getattr(current, field):
                raise RuntimeError(
                    f"frozen_domain_manifest_schema_drift:{domain.domain_id}:{field}"
                )


def verify_anchor_phenotype(
    phenotype: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if stable_hash(phenotype) != str(expected.get("phenotype_hash", "")):
        raise RuntimeError("frozen_domain_manifest_anchor_phenotype_hash_mismatch")
