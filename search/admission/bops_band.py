"""Auditable primary and conditionally expanded BOPS admission bands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_EPSILON = 1.0e-12


@dataclass(frozen=True)
class BopsBandPolicy:
    target: float
    primary_tolerance: float = 0.005
    expanded_tolerance: float = 0.0075
    adjacent_targets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        target = float(self.target)
        primary = float(self.primary_tolerance)
        expanded = float(self.expanded_tolerance)
        adjacent = tuple(
            sorted(
                {
                    float(value)
                    for value in self.adjacent_targets
                    if not math.isclose(float(value), target, abs_tol=_EPSILON)
                }
            )
        )
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("bops_target_must_be_finite_unit_interval")
        if not math.isfinite(primary) or primary < 0.0:
            raise ValueError("bops_primary_tolerance_must_be_finite_nonnegative")
        if not math.isfinite(expanded) or expanded < primary:
            raise ValueError("bops_expanded_tolerance_must_cover_primary")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in adjacent):
            raise ValueError("bops_adjacent_targets_must_be_finite_unit_interval")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "primary_tolerance", primary)
        object.__setattr__(self, "expanded_tolerance", expanded)
        object.__setattr__(self, "adjacent_targets", adjacent)


def classify_bops_value(
    retention: Any,
    *,
    policy: BopsBandPolicy,
) -> dict[str, Any]:
    try:
        value = float(retention)
    except (TypeError, ValueError):
        value = float("nan")
    target = float(policy.target)
    primary = float(policy.primary_tolerance)
    expanded = float(policy.expanded_tolerance)
    valid = math.isfinite(value) and 0.0 <= value <= 1.0
    if not valid:
        return {
            "passed": False,
            "classification": "invalid",
            "retention": value,
            "target": target,
            "signed_error": float("nan"),
            "absolute_error": float("inf"),
            "primary_tolerance": primary,
            "expanded_tolerance": expanded,
            "effective_tolerance": None,
            "primary_interval": [target - primary, target + primary],
            "expanded_interval": [target - expanded, target + expanded],
            "nearest_adjacent_target": None,
            "in_adjacent_primary_band": False,
            "eligible_for_expanded": False,
            "failure_reason": "bops_retention_must_be_finite_unit_interval",
        }

    signed_error = float(value - target)
    absolute_error = abs(signed_error)
    nearest_adjacent = (
        min(policy.adjacent_targets, key=lambda row: (abs(value - row), row))
        if policy.adjacent_targets
        else None
    )
    in_adjacent_primary = any(
        abs(value - adjacent) <= primary + _EPSILON
        for adjacent in policy.adjacent_targets
    )
    if absolute_error <= primary + _EPSILON:
        classification = "primary"
        effective_tolerance: float | None = primary
    elif absolute_error <= expanded + _EPSILON:
        classification = "expanded_only"
        effective_tolerance = expanded
    elif signed_error < 0.0:
        classification = "below"
        effective_tolerance = None
    else:
        classification = "above"
        effective_tolerance = None
    eligible_for_expanded = (
        classification == "expanded_only" and not in_adjacent_primary
    )
    return {
        # Expanded-only values still require proof that primary supply is
        # exhausted, which is available only to ``select_bops_candidates``.
        "passed": classification == "primary",
        "classification": classification,
        "retention": value,
        "target": target,
        "signed_error": signed_error,
        "absolute_error": absolute_error,
        "primary_tolerance": primary,
        "expanded_tolerance": expanded,
        "effective_tolerance": effective_tolerance,
        "primary_interval": [target - primary, target + primary],
        "expanded_interval": [target - expanded, target + expanded],
        "nearest_adjacent_target": nearest_adjacent,
        "in_adjacent_primary_band": in_adjacent_primary,
        "eligible_for_expanded": eligible_for_expanded,
        "failure_reason": (
            "adjacent_budget_primary_band"
            if classification == "expanded_only" and in_adjacent_primary
            else ""
        ),
    }


def _identity(row: Mapping[str, Any]) -> str:
    for key in ("phenotype_hash", "candidate_hash", "id", "genotype_hash"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _admission_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    error = round(float(row["absolute_error"]), 12)
    j1 = float(row.get("J1", float("nan")))
    if math.isfinite(j1):
        objective = (0, -j1)
    else:
        loss = float(row.get("L_joint_raw", float("inf")))
        objective = (1, loss if math.isfinite(loss) else float("inf"))
    return (error, *objective, _identity(row))


def select_bops_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: BopsBandPolicy,
    retention_key: str = "R_BOPS",
) -> dict[str, Any]:
    annotated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        classification = classify_bops_value(
            row.get(retention_key, float("nan")), policy=policy
        )
        annotated.append({**row, **classification})

    primary = [row for row in annotated if row["classification"] == "primary"]
    expanded = [
        row
        for row in annotated
        if row["classification"] == "expanded_only"
        and row["eligible_for_expanded"]
    ]
    if primary:
        admitted = primary
        admission_mode = "primary_bops_tolerance"
        reason = "within_primary_bops_tolerance"
        effective_tolerance = policy.primary_tolerance
    elif expanded:
        admitted = expanded
        admission_mode = "expanded_bops_tolerance"
        reason = "primary_supply_exhausted_nearest_within_expanded_tolerance"
        effective_tolerance = policy.expanded_tolerance
    else:
        admitted = []
        admission_mode = "no_bops_candidate"
        reason = ""
        effective_tolerance = None

    admitted_rows = []
    for row in sorted(admitted, key=_admission_sort_key):
        admitted_rows.append(
            {
                **row,
                "bops_admission_mode": admission_mode,
                "effective_tolerance": effective_tolerance,
                "admission_reason": reason,
                "target_bops": policy.target,
                "actual_bops": float(row["retention"]),
                "signed_bops_error": float(row["signed_error"]),
                "nearest_adjacent_budget": row["nearest_adjacent_target"],
                "reason": (
                    "no_candidate_in_primary_interval"
                    if admission_mode == "expanded_bops_tolerance"
                    else "within_primary_bops_tolerance"
                ),
            }
        )

    nearest_misses: list[dict[str, Any]] = []
    if not admitted_rows:
        valid = [row for row in annotated if row["classification"] != "invalid"]
        below = [row for row in valid if float(row["signed_error"]) < 0.0]
        above = [row for row in valid if float(row["signed_error"]) >= 0.0]
        if below:
            nearest_misses.append(min(below, key=_admission_sort_key))
        if above:
            nearest_misses.append(min(above, key=_admission_sort_key))

    funnel = {
        "input_count": len(annotated),
        "raw_count": len(annotated),
        "invalid_count": sum(
            row["classification"] == "invalid" for row in annotated
        ),
        "bops_below_count": sum(
            row["classification"] == "below" for row in annotated
        ),
        "primary_count": len(primary),
        "bops_primary_count": len(primary),
        "expanded_only_count": sum(
            row["classification"] == "expanded_only" for row in annotated
        ),
        "bops_expanded_only_count": sum(
            row["classification"] == "expanded_only" for row in annotated
        ),
        "adjacent_primary_excluded_count": sum(
            row["classification"] == "expanded_only"
            and row["in_adjacent_primary_band"]
            for row in annotated
        ),
        "bops_above_count": sum(
            row["classification"] == "above" for row in annotated
        ),
        "admitted_count": len(admitted_rows),
    }
    return {
        "policy": {
            "target": policy.target,
            "primary_tolerance": policy.primary_tolerance,
            "expanded_tolerance": policy.expanded_tolerance,
            "adjacent_targets": list(policy.adjacent_targets),
        },
        "admission_mode": admission_mode,
        "admitted": admitted_rows,
        "annotated": annotated,
        "nearest_misses": nearest_misses,
        "funnel": funnel,
    }
