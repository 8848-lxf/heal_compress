"""Constraint-first GA ranking with Taylor-primary epsilon tie breaking."""

from __future__ import annotations

import math
from typing import Any, Iterable

from ..candidate import CandidateGenotype


ScoredCandidate = tuple[CandidateGenotype, float, dict[str, Any]]


def _taylor(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get(
            "L_joint_weight_taylor",
            metrics.get("proxy_score_raw", metrics.get("F1", float("inf"))),
        )
    )


def rank_constraint_first(
    rows: Iterable[ScoredCandidate],
    *,
    taylor_relative_epsilon: float = 0.05,
    taylor_absolute_epsilon: float = 1.0e-8,
) -> list[ScoredCandidate]:
    """Rank infeasible rows by band distance and feasible rows by epsilon-Taylor.

    Taylor remains the primary objective.  Physical parameter retention is
    allowed to decide only inside the near-optimal Taylor set, avoiding an
    unstable additive weighting between quantities with very different scales.
    """

    candidates = list(rows)
    feasible_taylor = [
        _taylor(metrics)
        for _candidate, _score, metrics in candidates
        if bool(metrics.get("bops_feasible", True)) and math.isfinite(_taylor(metrics))
    ]
    best_taylor = min(feasible_taylor) if feasible_taylor else float("inf")
    near_limit = (
        best_taylor * (1.0 + max(0.0, float(taylor_relative_epsilon)))
        + max(0.0, float(taylor_absolute_epsilon))
    )

    def key(row: ScoredCandidate) -> tuple[Any, ...]:
        candidate, _score, metrics = row
        taylor = _taylor(metrics)
        parameter_retention = float(metrics.get("R_parameter_retention", 1.0))
        identity = str(candidate.to_dict())
        feasible = bool(metrics.get("bops_feasible", True))
        violation = max(0.0, float(metrics.get("bops_violation", 0.0)))
        if not feasible:
            return (1, violation, taylor, parameter_retention, identity)
        if taylor <= near_limit:
            return (0, 0, parameter_retention, taylor, identity)
        return (0, 1, taylor, parameter_retention, identity)

    return sorted(candidates, key=key)

