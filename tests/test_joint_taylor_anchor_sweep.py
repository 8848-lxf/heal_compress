from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _units():
    from search.anchors.joint_taylor_sweep import AnchorPruningUnit

    return [
        AnchorPruningUnit(
            unit_id=f"a{i}",
            prune_domain_id="a",
            importance=float(i),
            root_module="conv_a",
            parameter_cost=5,
        )
        for i in range(8)
    ] + [
        AnchorPruningUnit(
            unit_id=f"b{i}",
            prune_domain_id="b",
            importance=float(8 + i),
            root_module="conv_b",
            parameter_cost=5,
        )
        for i in range(8)
    ]


def _params(mask: dict[str, int]) -> int:
    return 100 - 5 * sum(1 for keep in mask.values() if keep == 0)


def test_anchor_grid_uses_actual_parameter_rate_and_domain_cap() -> None:
    from search.anchors.joint_taylor_sweep import plan_anchor_structures

    plan = plan_anchor_structures(
        _units(),
        requested_prune_rates=(0.0, 0.1, 0.3, 0.7),
        original_params=100,
        parameter_count_fn=_params,
        dense_alignment_by_domain={"a": 4, "b": 4},
        minimum_width_by_domain={"a": 1, "b": 1},
        per_domain_max_prune_rate=0.8,
    )

    assert [row.requested_prune_rate for row in plan.structures] == [0.0, 0.1, 0.3, 0.7]
    assert plan.structures[0].realized_prune_rate == pytest.approx(0.0)
    assert plan.structures[1].realized_prune_rate == pytest.approx(0.0)
    assert plan.structures[2].realized_prune_rate == pytest.approx(0.2)
    assert plan.structures[3].realized_prune_rate == pytest.approx(0.4)
    assert plan.structures[3].infeasible_under_domain_cap is True
    for structure in plan.structures:
        assert max(structure.domain_prune_rates.values(), default=0.0) <= 0.8


def test_anchor_nearest_legal_tie_prefers_not_exceeding_requested_rate() -> None:
    from search.anchors.joint_taylor_sweep import plan_anchor_structures

    plan = plan_anchor_structures(
        _units(),
        requested_prune_rates=(0.1,),
        original_params=100,
        parameter_count_fn=_params,
        dense_alignment_by_domain={"a": 4, "b": 4},
        minimum_width_by_domain={"a": 1, "b": 1},
        per_domain_max_prune_rate=0.8,
    )

    assert plan.structures[0].realized_prune_rate == pytest.approx(0.0)


def test_anchor_global_ranking_is_raw_sum_without_normalization() -> None:
    from search.anchors.joint_taylor_sweep import global_group_ranking

    rows = global_group_ranking(_units())

    assert [row["unit_id"] for row in rows[:3]] == ["a0", "a1", "a2"]
    assert all(row["normalization"] == "none" for row in rows)
    assert all(row["importance"] == row["total_importance"] for row in rows)


def test_engine_matrix_variants_share_each_structure_physical_hash() -> None:
    from search.anchors.joint_taylor_sweep import build_engine_matrix_requests

    matrix = build_engine_matrix_requests(
        [
            {"anchor_id": "r000", "physical_hash": "p0"},
            {"anchor_id": "r100", "physical_hash": "p1"},
        ]
    )

    assert len(matrix) == 6
    assert {row["precision_variant"] for row in matrix} == {
        "strict_fp32",
        "strict_fp16",
        "maximal_legal_int8",
    }
    for physical_hash in ("p0", "p1"):
        assert {
            row["physical_hash"] for row in matrix if row["physical_hash"] == physical_hash
        } == {physical_hash}
        assert sum(row["physical_hash"] == physical_hash for row in matrix) == 3


def test_formal_latency_isolation_rejects_workers_and_same_gpu_builds() -> None:
    from search.anchors.joint_taylor_sweep import assert_formal_latency_isolation

    assert_formal_latency_isolation(
        active_process_commands=[],
        selected_gpu_uuid="gpu",
        gpu_processes=[],
    )
    with pytest.raises(RuntimeError, match="formal_latency_parallel_worker_active"):
        assert_formal_latency_isolation(
            active_process_commands=["python -m search.stage2.candidate_worker"],
            selected_gpu_uuid="gpu",
            gpu_processes=[],
        )
    with pytest.raises(RuntimeError, match="formal_latency_selected_gpu_busy"):
        assert_formal_latency_isolation(
            active_process_commands=[],
            selected_gpu_uuid="gpu",
            gpu_processes=[
                {"gpu_uuid": "gpu", "process_name": "trtexec", "pid": 123}
            ],
        )


def test_tau_boundary_bisection_proposes_at_most_three_legal_rates() -> None:
    from search.anchors.joint_taylor_sweep import propose_boundary_bisections

    rates = propose_boundary_bisections(
        [
            {"realized_prune_rate": 0.4, "delta_mAP": 0.08, "valid_for_tau": True},
            {"realized_prune_rate": 0.6, "delta_mAP": 0.12, "valid_for_tau": True},
        ],
        max_rounds=3,
    )

    assert rates == pytest.approx((0.5,))
    assert len(rates) <= 3
