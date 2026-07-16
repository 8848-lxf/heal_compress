from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_primary_bops_candidates_suppress_expanded_candidates() -> None:
    from search.admission.bops_band import BopsBandPolicy, select_bops_candidates

    result = select_bops_candidates(
        [
            {"id": "expanded", "R_BOPS": 0.206, "J1": 2.0},
            {"id": "primary", "R_BOPS": 0.204, "J1": 1.0},
        ],
        policy=BopsBandPolicy(target=0.20, adjacent_targets=(0.15, 0.25)),
    )

    assert result["admission_mode"] == "primary_bops_tolerance"
    assert [row["id"] for row in result["admitted"]] == ["primary"]
    assert result["funnel"]["primary_count"] == 1
    assert result["funnel"]["expanded_only_count"] == 1
    assert result["funnel"]["admitted_count"] == 1


def test_expanded_bops_is_used_only_when_primary_is_empty() -> None:
    from search.admission.bops_band import BopsBandPolicy, select_bops_candidates

    result = select_bops_candidates(
        [{"id": "near", "R_BOPS": 0.206}],
        policy=BopsBandPolicy(target=0.20, adjacent_targets=(0.15, 0.25)),
    )

    assert result["admission_mode"] == "expanded_bops_tolerance"
    assert result["annotated"][0]["passed"] is False
    assert result["annotated"][0]["eligible_for_expanded"] is True
    assert result["admitted"][0]["effective_tolerance"] == pytest.approx(0.0075)
    assert result["admitted"][0]["admission_reason"] == (
        "primary_supply_exhausted_nearest_within_expanded_tolerance"
    )
    assert result["admitted"][0]["target_bops"] == pytest.approx(0.20)
    assert result["admitted"][0]["actual_bops"] == pytest.approx(0.206)
    assert result["admitted"][0]["signed_bops_error"] == pytest.approx(0.006)
    assert result["admitted"][0]["nearest_adjacent_budget"] == pytest.approx(0.25)
    assert result["admitted"][0]["reason"] == "no_candidate_in_primary_interval"


def test_expanded_candidate_in_adjacent_primary_band_is_rejected() -> None:
    from search.admission.bops_band import BopsBandPolicy, select_bops_candidates

    result = select_bops_candidates(
        [{"id": "belongs-next", "R_BOPS": 0.206}],
        policy=BopsBandPolicy(target=0.20, adjacent_targets=(0.21,)),
    )

    assert result["admitted"] == []
    assert result["admission_mode"] == "no_bops_candidate"
    assert result["funnel"]["adjacent_primary_excluded_count"] == 1
    assert result["annotated"][0]["nearest_adjacent_target"] == pytest.approx(0.21)


def test_outside_expanded_bops_reports_nearest_misses() -> None:
    from search.admission.bops_band import BopsBandPolicy, select_bops_candidates

    result = select_bops_candidates(
        [{"id": "low", "R_BOPS": 0.18}, {"id": "high", "R_BOPS": 0.22}],
        policy=BopsBandPolicy(target=0.20),
    )

    assert result["admitted"] == []
    assert result["funnel"]["bops_below_count"] == 1
    assert result["funnel"]["bops_above_count"] == 1
    assert result["funnel"]["raw_count"] == 2
    assert result["funnel"]["bops_primary_count"] == 0
    assert result["funnel"]["bops_expanded_only_count"] == 0
    assert {row["id"] for row in result["nearest_misses"]} == {"low", "high"}


def test_bops_admission_sort_is_error_then_j1_then_identity() -> None:
    from search.admission.bops_band import BopsBandPolicy, select_bops_candidates

    result = select_bops_candidates(
        [
            {"candidate_hash": "c", "R_BOPS": 0.202, "J1": 1.0},
            {"candidate_hash": "b", "R_BOPS": 0.198, "J1": 3.0},
            {"candidate_hash": "a", "R_BOPS": 0.198, "J1": 3.0},
        ],
        policy=BopsBandPolicy(target=0.20),
    )

    assert [row["candidate_hash"] for row in result["admitted"]] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -0.1, 1.1, None, "not-a-number"]
)
def test_bops_policy_fails_closed_for_invalid_retention(value: object) -> None:
    from search.admission.bops_band import BopsBandPolicy, classify_bops_value

    row = classify_bops_value(value, policy=BopsBandPolicy(target=0.20))

    assert row["passed"] is False
    assert row["classification"] == "invalid"
