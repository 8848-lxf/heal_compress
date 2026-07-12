"""Search package.

The formal two-stage joint search lives in the new modules under this package.
Legacy search helpers remain importable by their direct module paths.
"""

from .candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision
from .hashing import candidate_hash, canonical_json_hash

try:  # Legacy public API compatibility.
    from .search_space import SearchSpaceEncoder
    from .proxy_objective import ProxyObjectiveEvaluator
    from .genetic_search import GeneticSearchEngine
except Exception:  # pragma: no cover - keep new package importable if legacy deps fail.
    SearchSpaceEncoder = None  # type: ignore[assignment]
    ProxyObjectiveEvaluator = None  # type: ignore[assignment]
    GeneticSearchEngine = None  # type: ignore[assignment]

__all__ = [
    "CandidateGenotype",
    "CandidatePhenotype",
    "GeneticSearchEngine",
    "PrecisionDecision",
    "ProxyObjectiveEvaluator",
    "SearchSpaceEncoder",
    "candidate_hash",
    "canonical_json_hash",
]
