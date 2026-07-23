"""Constraint-first GA ranking with Taylor-primary epsilon tie breaking."""

from __future__ import annotations

import math
import hashlib
import json
from typing import Any, Iterable

from ..candidate import CandidateGenotype


ScoredCandidate = tuple[CandidateGenotype, float, dict[str, Any]]


def _taylor(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get(
            "L_joint_weight_activation_taylor",
            metrics.get(
                "L_joint_weight_taylor",
                metrics.get("proxy_score_raw", metrics.get("F1", float("inf"))),
            ),
        )
    )


def _finite_metric(metrics: dict[str, Any], *names: str) -> float:
    for name in names:
        raw = metrics.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("inf")


def _identity(candidate: CandidateGenotype, metrics: dict[str, Any]) -> str:
    explicit = str(metrics.get("candidate_hash", ""))
    if explicit:
        return explicit
    payload = json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rank_constraint_first(
    rows: Iterable[ScoredCandidate],
    *,
    taylor_relative_epsilon: float = 0.05,
    taylor_absolute_epsilon: float = 1.0e-8,
) -> list[ScoredCandidate]:
    """Rank infeasible rows by band distance and feasible rows by epsilon-Taylor.

    Taylor remains the primary objective.  Within its near-optimal set, the
    ordering is latency proxy, physical parameter retention, mixed weight size,
    and stable candidate identity. No differently-scaled term is added to the
    task-loss proxy.
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
        parameter_retention = _finite_metric(metrics, "R_parameter_retention")
        latency = _finite_metric(metrics, "latency_proxy_ms", "R_latency_proxy")
        mixed_weight_size = _finite_metric(
            metrics, "mixed_weight_size_bytes", "R_size_vs_fp32"
        )
        identity = _identity(candidate, metrics)
        feasible = bool(metrics.get("bops_feasible", True))
        violation = max(0.0, float(metrics.get("bops_violation", 0.0)))
        if not feasible:
            return (
                1,
                violation,
                taylor,
                latency,
                parameter_retention,
                mixed_weight_size,
                identity,
            )
        if taylor <= near_limit:
            return (
                0,
                0,
                latency,
                parameter_retention,
                mixed_weight_size,
                taylor,
                identity,
            )
        return (
            0,
            1,
            taylor,
            latency,
            parameter_retention,
            mixed_weight_size,
            identity,
        )

    return sorted(candidates, key=key)
