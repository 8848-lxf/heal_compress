from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_linear_joint_j1_has_expected_direction_and_no_tau_fields() -> None:
    from search.proxy.task_score import compute_linear_joint_j1

    low = compute_linear_joint_j1(
        l_joint=2.0,
        l_scale=10.0,
        original_params=100,
        candidate_params=80,
    )
    high = compute_linear_joint_j1(
        l_joint=4.0,
        l_scale=10.0,
        original_params=100,
        candidate_params=80,
    )
    more_pruned = compute_linear_joint_j1(
        l_joint=2.0,
        l_scale=10.0,
        original_params=100,
        candidate_params=70,
    )

    assert low["task_score_mapping"] == "linear_fixed_scale"
    assert low["normalized_joint_loss"] == pytest.approx(0.2)
    assert low["R_prune"] == pytest.approx(0.2)
    assert low["J1"] > high["J1"]
    assert more_pruned["J1"] > low["J1"]
    assert low["J1"] == pytest.approx(-0.8 * 0.2 + 0.2 * 0.2)
    assert low["F1"] == pytest.approx(-low["J1"])
    assert low["sqnr_main_objective_contribution"] == 0.0
    for obsolete in (
        "tau",
        "exponent_value",
        "S_task",
        "task_score_saturated",
    ):
        assert obsolete not in low


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"l_joint": -1.0}, "joint_taylor_loss_must_be_finite_nonnegative"),
        ({"l_joint": float("inf")}, "joint_taylor_loss_must_be_finite_nonnegative"),
        ({"l_scale": 0.0}, "joint_loss_scale_must_be_finite_positive"),
        ({"l_scale": float("nan")}, "joint_loss_scale_must_be_finite_positive"),
        ({"candidate_params": 101}, "invalid_structural_parameter_counts"),
        ({"task_weight": -0.1}, "joint_score_weights_must_be_finite_nonnegative"),
    ],
)
def test_linear_joint_j1_fails_closed(overrides: dict[str, float], reason: str) -> None:
    from search.proxy.task_score import compute_linear_joint_j1

    arguments = {
        "l_joint": 2.0,
        "l_scale": 10.0,
        "original_params": 100,
        "candidate_params": 80,
        **overrides,
    }
    with pytest.raises(ValueError, match=reason):
        compute_linear_joint_j1(**arguments)


def test_raw_joint_loss_mapping_is_calibration_only() -> None:
    from search.candidate import CandidatePhenotype
    from search.proxy.joint_taylor import JointTaylorResult
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig

    class JointStub:
        def evaluate(self, phenotype: CandidatePhenotype) -> JointTaylorResult:
            del phenotype
            return JointTaylorResult(
                first_order_sum=2.0,
                second_order_fisher_sum=3.0,
                total_importance=5.0,
                unique_parameter_count=10,
                duplicate_slice_count=0,
                importance_mode="joint_taylor_second_order_fisher_diag",
            )

    class SizeStub:
        def structural_parameter_counts(
            self, phenotype: CandidatePhenotype
        ) -> tuple[int, int]:
            del phenotype
            return 100, 80

    class BopsStub:
        def evaluate_breakdown(self, phenotype: CandidatePhenotype) -> dict[str, float]:
            del phenotype
            return {
                "R_bops_vs_fp32": 0.21,
                "R_bops_vs_fp16_deploy": 0.84,
                "int8_macs_ratio": 0.0,
                "int8_macs_share_full": 0.0,
                "R_MAC": 0.84,
                "bops_fp16_baseline": 25.0,
                "bops_fp32_baseline": 100.0,
            }

    objective = ProxyObjective(
        joint=JointStub(),
        size=SizeStub(),
        bops=BopsStub(),
        config=ProxyObjectiveConfig(
            proxy_mode="joint_taylor_second_order_fisher_diag",
            task_score_mapping="raw_joint_loss",
            bops_threshold=None,
        ),
    )

    row = objective.evaluate(CandidatePhenotype())

    assert row["task_score_mapping"] == "raw_joint_loss"
    assert row["L_joint_raw"] == pytest.approx(5.0)
    assert row["L_joint_first_order"] == pytest.approx(2.0)
    assert row["L_joint_second_order"] == pytest.approx(3.0)
    assert row["R_prune"] == pytest.approx(0.2)
    assert row["F1"] == pytest.approx(5.0)
    assert "J1" not in row
    assert "L_scale" not in row
    assert "tau" not in row
