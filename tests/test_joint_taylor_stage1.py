from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_exponential_j1_has_correct_direction_and_structural_prune_rate() -> None:
    from search.proxy.task_score import compute_exponential_j1

    low_loss = compute_exponential_j1(
        l_joint=1.0,
        tau=2.0,
        original_params=100,
        candidate_params=80,
        proxy_mode="joint_taylor_second_order_fisher_diag",
    )
    high_loss = compute_exponential_j1(
        l_joint=2.0,
        tau=2.0,
        original_params=100,
        candidate_params=80,
        proxy_mode="joint_taylor_second_order_fisher_diag",
    )
    more_pruned = compute_exponential_j1(
        l_joint=1.0,
        tau=2.0,
        original_params=100,
        candidate_params=70,
        proxy_mode="joint_taylor_second_order_fisher_diag",
    )

    assert low_loss["R_prune"] == pytest.approx(0.2)
    assert low_loss["J1"] > high_loss["J1"]
    assert more_pruned["J1"] > low_loss["J1"]
    assert low_loss["F1"] == pytest.approx(-low_loss["J1"])


def test_exponential_task_score_landmarks_and_saturation_are_explicit() -> None:
    from search.proxy.task_score import compute_exponential_j1

    l_safe = 3.0
    tau = l_safe / math.log(2.0)
    zero = compute_exponential_j1(l_joint=0.0, tau=tau, original_params=10, candidate_params=10)
    safe = compute_exponential_j1(l_joint=l_safe, tau=tau, original_params=10, candidate_params=10)
    double = compute_exponential_j1(l_joint=2.0 * l_safe, tau=tau, original_params=10, candidate_params=10)
    saturated = compute_exponential_j1(l_joint=1000.0 * tau, tau=tau, original_params=10, candidate_params=10)

    assert zero["S_task"] == pytest.approx(1.0)
    assert safe["S_task"] == pytest.approx(0.5)
    assert double["S_task"] == pytest.approx(0.25)
    assert saturated["exponent_value"] == pytest.approx(80.0)
    assert saturated["task_score_saturated"] is True


def test_dense_conditional_repair_floors_requested_prune_count_without_extra_pruning() -> None:
    from search.stage1.conditional_repair import conditional_dense_floor_repair

    raw = {f"u{i}": 0 if i < 5 else 1 for i in range(8)}
    costs = {f"u{i}": float(7 - i) for i in range(8)}

    result = conditional_dense_floor_repair(
        raw,
        conditional_costs=costs,
        alignment=4,
        minimum_width=1,
    )

    assert result.status == "ok"
    assert result.raw_prune_count == 5
    assert result.legal_prune_count == 4
    assert sum(1 for keep in result.repaired_mask.values() if keep == 0) == 4
    assert {unit for unit, keep in result.repaired_mask.items() if keep == 0} == {
        "u4",
        "u5",
        "u6",
        "u7",
    }
    assert result.legal_prune_count <= result.raw_prune_count


def test_dense_conditional_repair_does_not_prune_below_one_alignment_step() -> None:
    from search.stage1.conditional_repair import conditional_dense_floor_repair

    raw = {f"u{i}": 0 if i < 3 else 1 for i in range(8)}
    result = conditional_dense_floor_repair(
        raw,
        conditional_costs={unit: float(i) for i, unit in enumerate(raw)},
        alignment=4,
        minimum_width=1,
    )

    assert result.legal_prune_count == 0
    assert all(keep == 1 for keep in result.repaired_mask.values())


def test_grouped_conditional_repair_ranks_each_physical_group_independently() -> None:
    from search.stage1.conditional_repair import conditional_grouped_floor_repair

    groups = {
        0: tuple(f"g0_{i}" for i in range(8)),
        1: tuple(f"g1_{i}" for i in range(8)),
    }
    raw = {
        unit: (0 if int(unit.rsplit("_", 1)[1]) < 4 else 1)
        for units in groups.values()
        for unit in units
    }
    costs = {
        **{f"g0_{i}": float(i) for i in range(8)},
        **{f"g1_{i}": float(7 - i) for i in range(8)},
    }

    result = conditional_grouped_floor_repair(
        raw,
        physical_groups=groups,
        local_indices={group: {unit: i for i, unit in enumerate(units)} for group, units in groups.items()},
        conditional_costs=costs,
        allowed_channels_per_group=(4, 8),
    )

    assert result.status == "ok"
    assert result.group_prune_map[0] == [0, 1, 2, 3]
    assert result.group_prune_map[1] == [4, 5, 6, 7]
    assert result.group_prune_map[0] != result.group_prune_map[1]
    assert result.shared_position_strategy is False


def test_conditional_repair_preserves_precision_profile() -> None:
    from search.candidate import CandidateGenotype
    from search.stage1.conditional_repair import repair_candidate_domains

    raw = CandidateGenotype(
        pruning_genes={f"u{i}": 0 if i < 5 else 1 for i in range(8)},
        precision_genes={"pg0": "INT8", "pg1": "FP16"},
    )
    repaired, report = repair_candidate_domains(
        raw,
        dense_domains={"d": tuple(raw.pruning_genes)},
        grouped_domains={},
        conditional_costs={unit: float(i) for i, unit in enumerate(raw.pruning_genes)},
        dense_alignment=4,
        minimum_width_by_domain={"d": 1},
    )

    assert repaired.precision_genes == raw.precision_genes
    assert report["raw_precision_gene_hash"] == report["repaired_precision_gene_hash"]
    assert report["precision_profile_modified"] is False
