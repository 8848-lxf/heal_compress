import math
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.run_global_budgeted_alignment_eval_v101 import (
    BudgetCandidate,
    build_estimate_actual_gap_report_v102,
    select_budget_candidates_with_repair_v102,
    sort_budget_candidates_by_score_v102,
)


def _candidate(name: str, importance: float, saving: int, domain: Optional[str] = None) -> BudgetCandidate:
    return BudgetCandidate(
        candidate_id=name,
        domain_id=domain or name,
        prune_indices=[0],
        keep_indices=[1],
        source_coupled_units=[name],
        importance_score=importance,
        param_saving_if_removed=saving,
        affected_modules=[name],
        strategy_policy="A1",
        alignment_status="aligned",
        legality_status="legal",
    )


def test_score_sorting_is_strictly_ascending():
    rows = [
        _candidate("A", importance=1.0, saving=100),
        _candidate("B", importance=1.0, saving=1000),
        _candidate("C", importance=10.0, saving=1000),
    ]

    sorted_rows = sort_budget_candidates_by_score_v102(rows)

    assert [row.candidate_id for row in sorted_rows] == ["B", "A", "C"]
    assert [round(row.score, 6) for row in sorted_rows] == [0.001, 0.01, 0.01]


def test_best_near_target_avoids_large_overshoot_when_smaller_combo_fits_window():
    candidates = [
        _candidate("large", importance=0.0, saving=176),
        _candidate("small30", importance=0.1, saving=30),
        _candidate("small40", importance=0.1, saving=40),
        _candidate("small50", importance=0.1, saving=50),
    ]

    selected, rejected, report = select_budget_candidates_with_repair_v102(
        candidates,
        target_param_saving=100,
        baseline_total_params=1000,
        total_candidate_units=4,
        budget_acceptance_mode="lower_bound_with_max_overshoot",
        max_budget_overshoot=0.03,
        budget_repair_mode="best_near_target",
    )

    assert "large" not in {row.candidate_id for row in selected}
    ratio = report["estimated_param_prune_ratio_selected"]
    assert 0.10 <= ratio <= 0.13
    assert report["budget_status"] == "in_budget_window"
    assert any(row["reject_reason_if_any"] in {"overshoot_budget_window", "not_selected_by_budget_repair"} for row in rejected)


def test_prefix_repair_keeps_selected_sets_monotonic_for_same_candidates():
    candidates = [
        _candidate("u1", importance=1.0, saving=50),
        _candidate("u2", importance=2.0, saving=50),
        _candidate("u3", importance=3.0, saving=50),
        _candidate("u4", importance=4.0, saving=50),
    ]
    selected_sets = []
    for target in (0.10, 0.15, 0.20):
        selected, _rejected, report = select_budget_candidates_with_repair_v102(
            candidates,
            target_param_saving=1000 * target,
            baseline_total_params=1000,
            total_candidate_units=4,
            budget_acceptance_mode="lower_bound_with_max_overshoot",
            max_budget_overshoot=0.03,
            budget_repair_mode="best_near_target",
        )
        assert report["budget_status"] == "in_budget_window"
        selected_sets.append({row.candidate_id for row in selected})

    assert selected_sets[0] <= selected_sets[1] <= selected_sets[2]


def test_estimate_actual_gap_report_marks_large_mismatch():
    report = build_estimate_actual_gap_report_v102(
        estimated_param_saving=100,
        actual_param_saving=176,
        baseline_total_params=1000,
        per_module_estimated_saving={"conv": 100},
        per_module_actual_saving={"conv": 176},
    )

    assert math.isclose(report["estimated_param_prune_ratio_before_physical"], 0.10)
    assert math.isclose(report["actual_param_prune_ratio_after_physical"], 0.176)
    assert report["estimate_actual_mismatch"] is True
    assert report["modules_with_large_estimate_error"][0]["module_name"] == "conv"


def test_non_positive_param_saving_rejected_by_default():
    selected, rejected, report = select_budget_candidates_with_repair_v102(
        [_candidate("zero", importance=0.0, saving=0)],
        target_param_saving=10,
        baseline_total_params=1000,
        total_candidate_units=1,
        budget_acceptance_mode="lower_bound_with_max_overshoot",
        max_budget_overshoot=0.03,
        budget_repair_mode="best_near_target",
    )

    assert selected == []
    assert rejected[0]["reject_reason_if_any"] == "non_positive_param_saving"
    assert report["budget_status"] == "no_feasible_candidate_in_window"
