"""Search artifact storage and audit helpers."""

from .candidate_audit_store import (
    ArtifactReference,
    CandidateAuditStore,
    finalize_completed_candidate,
    resolve_artifact_reference,
)

__all__ = [
    "ArtifactReference",
    "CandidateAuditStore",
    "finalize_completed_candidate",
    "resolve_artifact_reference",
]

