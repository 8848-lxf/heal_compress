from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_independent_solver_maximizes_each_domain_without_greedy_order() -> None:
    from search.audits.prune_rate_reachability import solve_independent_domain_max

    result = solve_independent_domain_max(
        {
            "dense": {
                "unit_ids": [f"d{i}" for i in range(16)],
                "kind": "dense",
                "alignment": 4,
                "minimum_width": 4,
            },
            "grouped": {
                "unit_ids": [f"g{i}" for i in range(16)],
                "kind": "grouped",
                "physical_groups": {
                    0: [f"g{i}" for i in range(8)],
                    1: [f"g{i}" for i in range(8, 16)],
                },
                "allowed_channels_per_group": [4, 8],
            },
        },
        per_domain_cap=0.8,
    )

    assert result["domain_pruned_counts"] == {"dense": 12, "grouped": 8}
    assert result["maximum_domain_prune_rates"]["dense"] == 0.75
    assert result["maximum_domain_prune_rates"]["grouped"] == 0.5
    assert len(result["pruned_unit_ids"]) == 20


def test_solver_and_greedy_difference_is_explicit() -> None:
    from search.audits.prune_rate_reachability import compare_reachability_masks

    result = compare_reachability_masks(
        planner_pruned_ids={"u0"},
        independent_pruned_ids={"u0", "u1"},
        planner_predicted_rate=0.2,
        independent_predicted_rate=0.3,
    )

    assert result["mask_equal"] is False
    assert result["independent_minus_planner_rate"] == 0.1
    assert result["difference_reason"] == "planner_selection_or_projection_under_reaches_legal_space"


def test_grouped_floor_repair_uses_common_conservative_width() -> None:
    from search.stage1.conditional_repair import conditional_grouped_floor_repair

    groups = {
        0: [f"g0_{index}" for index in range(8)],
        1: [f"g1_{index}" for index in range(8)],
    }
    raw = {
        unit_id: (0 if unit_id.startswith("g0_") and int(unit_id[-1]) < 4 else 1)
        for rows in groups.values()
        for unit_id in rows
    }
    result = conditional_grouped_floor_repair(
        raw,
        physical_groups=groups,
        local_indices={
            group: {unit_id: index for index, unit_id in enumerate(rows)}
            for group, rows in groups.items()
        },
        conditional_costs={unit_id: float(index) for index, unit_id in enumerate(raw)},
        allowed_channels_per_group=(4, 8),
    )

    assert result.status == "ok"
    assert result.legal_prune_count == 0
    assert result.legal_prune_count <= result.raw_prune_count
    assert all(keep == 1 for keep in result.repaired_mask.values())


def test_anchor_planner_records_grouped_max_even_when_importance_is_skewed() -> None:
    from search.anchors.joint_taylor_sweep import (
        AnchorPruningUnit,
        plan_anchor_structures,
    )

    units = [
        AnchorPruningUnit(
            unit_id=f"g{group}_{index}",
            prune_domain_id="grouped",
            importance=float(group * 100 + index),
            root_module="grouped",
        )
        for group in range(2)
        for index in range(8)
    ]
    unit_ids = [unit.unit_id for unit in units]
    plan = plan_anchor_structures(
        units,
        requested_prune_rates=(0.5,),
        original_params=160,
        parameter_count_fn=lambda mask: 160
        - 10 * sum(1 for keep in mask.values() if keep == 0),
        dense_alignment_by_domain={},
        minimum_width_by_domain={},
        per_domain_max_prune_rate=0.5,
        grouped_domain_specs={
            "grouped": {
                "physical_groups": {
                    0: unit_ids[:8],
                    1: unit_ids[8:],
                },
                "local_indices": {
                    0: {unit_id: index for index, unit_id in enumerate(unit_ids[:8])},
                    1: {unit_id: index for index, unit_id in enumerate(unit_ids[8:])},
                },
                "allowed_channels_per_group": (4, 8),
            }
        },
    )

    assert plan.maximum_realized_prune_rate == 0.5
    assert plan.structures[0].realized_prune_rate == 0.5
    assert sum(keep == 0 for keep in plan.structures[0].group_mask.values()) == 8
