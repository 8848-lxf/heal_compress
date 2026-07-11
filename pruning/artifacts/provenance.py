"""Pruning artifact provenance records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PhysicalArtifactProvenance:
    physical_structure_hash: str
    shape_hash: str
    snapshot_hash: str
    model_hash: str = ""
    config_hash: str = ""
    policy_version: str = "physical-artifact-v2"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


__all__ = ["PhysicalArtifactProvenance"]
