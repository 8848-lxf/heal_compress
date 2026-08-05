from __future__ import annotations

import pytest


def _row(
    candidate_hash: str,
    mode: str,
    struct: float,
    weight: float,
    activation: float,
    bops: float,
) -> dict[str, float | str]:
    # The synthetic loss follows a fixed non-negative model exactly.
    target = 2.0 * struct + 0.5 * weight + 3.0 * activation + weight * activation
    return {
        "candidate_hash": candidate_hash,
        "candidate_mode": mode,
        "J_struct_gate": struct,
        "J_WQ": weight,
        "J_AQ": activation,
        "delta_task_loss": target,
        "R_bops_vs_fp32": bops,
    }


def test_huber_nnls_calibration_is_nonnegative_frozen_and_rank_audited() -> None:
    from search.proxy.calibrated_taylor_objective import (
        FEATURE_KEYS,
        fit_frozen_taylor_objective,
    )

    fit = [
        _row("f0", "pruning", 1.0, 0.0, 0.0, 0.90),
        _row("f1", "pruning", 2.0, 0.0, 0.0, 0.75),
        _row("f2", "weight_quant", 0.0, 1.0, 0.0, 0.80),
        _row("f3", "weight_quant", 0.0, 2.0, 0.0, 0.65),
        _row("f4", "activation_quant", 0.0, 0.0, 1.0, 0.70),
        _row("f5", "activation_quant", 0.0, 0.0, 2.0, 0.55),
        _row("f6", "mixed", 1.0, 1.0, 1.0, 0.40),
        _row("f7", "mixed", 2.0, 2.0, 1.0, 0.25),
    ]
    validation = [
        _row("v0", "pruning", 1.5, 0.0, 0.0, 0.85),
        _row("v1", "weight_quant", 0.0, 1.5, 0.0, 0.70),
        _row("v2", "activation_quant", 0.0, 0.0, 1.5, 0.55),
        _row("v3", "mixed", 1.5, 1.5, 1.5, 0.35),
    ]
    calibration = fit_frozen_taylor_objective(
        fit, validation, calibration_data_hash="train-calibration-prefix"
    )

    assert set(calibration.coefficients) == set(FEATURE_KEYS)
    assert min(calibration.coefficients.values()) >= 0.0
    assert calibration.validation_metrics["spearman"] == pytest.approx(1.0)
    assert calibration.validation_metrics["top_k_good_candidate_recall"] == 1.0
    assert calibration.validation_metrics["structural_candidate_count"] == 2
    assert calibration.validation_metrics["pure_quantization_candidate_count"] == 2
    assert calibration.raw_sum_validation_metrics["candidate_count"] == 4
    assert calibration.raw_sum_validation_metrics["candidate_mode_coverage"] == {
        "activation_quant": 1,
        "mixed": 1,
        "pruning": 1,
        "weight_quant": 1,
    }
    assert calibration.to_dict()["calibration_hash"]


def test_fit_rejects_missing_candidate_modes() -> None:
    from search.proxy.calibrated_taylor_objective import fit_frozen_taylor_objective

    rows = [_row(f"f{index}", "pruning", float(index + 1), 0.0, 0.0, 0.9) for index in range(4)]
    with pytest.raises(ValueError, match="candidate_modes_missing"):
        fit_frozen_taylor_objective(rows, rows, calibration_data_hash="x")


def test_greedy_audit_uses_unified_stage1_activation_flag() -> None:
    from search.candidate import CandidateGenotype
    from search.greedy.engine import GreedySearchResult

    result = GreedySearchResult(
        initial_candidate=CandidateGenotype(),
        initial_metrics={
            "activation_taylor_included": True,
            "objective_calibrated": True,
        },
        steps=(),
        budget_candidates={},
        budget_metrics={},
        nearest_budget_candidates={},
        nearest_budget_metrics={},
        unreachable_targets=(),
        termination_reason="test",
        evaluated_neighbor_count=0,
        budget_recovery_evaluated_neighbor_count=0,
        budget_recovery_reports={},
        bops_tolerance_abs=0.005,
        run_to_exhaustion=False,
        budget_recovery_enabled=True,
    )

    semantics = result.to_dict()["search_semantics"]
    assert semantics["activation_taylor_included"] is True
    assert semantics["objective_calibrated"] is True
