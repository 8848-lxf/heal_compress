"""Typed exceptions for the formal pruning package."""

from __future__ import annotations


class PruningError(RuntimeError):
    """Base class for formal pruning failures."""


class PruningPlanError(PruningError):
    """A physical plan is incomplete, contradictory, or not executable."""


class PruningLegalityError(PruningError):
    """A requested channel transformation violates a structural contract."""


class GroupedConvLegalityError(PruningLegalityError):
    """Grouped-convolution channels, groups, or mappings are illegal."""


class MissingGroupKeepMapError(GroupedConvLegalityError):
    """Independent grouped replay lacks its exact per-group keep map."""


class ArtifactSchemaError(PruningError):
    """A pruning artifact is missing or uses the wrong schema/semantics."""


class PhysicalStructureMismatchError(PruningError):
    """Live module attributes, tensors, state dict, or snapshot disagree."""


class ModelLoadError(PruningError):
    """Model construction or checkpoint loading failed."""


class ProvenanceError(PruningError):
    """Required pruning provenance is absent or inconsistent."""


__all__ = [
    "ArtifactSchemaError",
    "GroupedConvLegalityError",
    "MissingGroupKeepMapError",
    "ModelLoadError",
    "PhysicalStructureMismatchError",
    "ProvenanceError",
    "PruningError",
    "PruningLegalityError",
    "PruningPlanError",
]
