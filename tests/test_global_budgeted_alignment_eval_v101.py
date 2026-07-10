import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.run_global_budgeted_alignment_eval_v101 import (
    BudgetCandidate,
    build_v101_budget_report,
    estimate_param_saving_for_scope,
    global_budgeted_coupled_unit_selector,
    parse_strategy_spec,
    write_summary,
)
from tracer.pruning_group import PruningGroup
from pruning.pruning_fns import prune_bn, prune_conv_out


def _toy_scope() -> PruningGroup:
    conv = nn.Conv2d(8, 8, 1, bias=False)
    bn = nn.BatchNorm2d(8)
    scope = PruningGroup(group_id="toy::conv", num_channels=8)
    scope.add_dep(
        "conv",
        conv,
        prune_conv_out,
        "out",
        idxs=list(range(8)),
        reason="root",
    )
    scope.add_dep(
        "bn",
        bn,
        prune_bn,
        "out",
        idxs=list(range(8)),
        reason="bn",
    )
    return scope


def test_strategy_spec_parses_alignment_variants():
    assert parse_strategy_spec("A1").policy_key == "A"
    assert parse_strategy_spec("A1").a_total_cout_align == 8
    assert parse_strategy_spec("A2").a_total_cout_align == 4
    assert parse_strategy_spec("B3").b_per_group_align == 8
    assert parse_strategy_spec("C2").c_groups_after_align == 8


def test_estimate_param_saving_for_scope_counts_conv_and_bn_channels():
    scope = _toy_scope()

    saving = estimate_param_saving_for_scope(scope, [1, 3])

    # Conv2d(8,8,1,bias=False): two output filters save 16 weights.
    # BatchNorm2d has affine weight+bias, so two channels save 4 params.
    assert saving == 20


def test_global_budgeted_selector_prefers_low_importance_per_param_saving():
    candidates = [
        BudgetCandidate(
            candidate_id="expensive_low_loss",
            domain_id="d0",
            prune_indices=[0, 1],
            keep_indices=[2, 3],
            source_coupled_units=["u0", "u1"],
            importance_score=2.0,
            param_saving_if_removed=100,
            affected_modules=["m0"],
            strategy_policy="A1",
            alignment_status="aligned",
            legality_status="legal",
        ),
        BudgetCandidate(
            candidate_id="cheap_higher_loss",
            domain_id="d1",
            prune_indices=[0],
            keep_indices=[1],
            source_coupled_units=["u2"],
            importance_score=1.0,
            param_saving_if_removed=10,
            affected_modules=["m1"],
            strategy_policy="A1",
            alignment_status="aligned",
            legality_status="legal",
        ),
    ]

    selected, rejected, report = global_budgeted_coupled_unit_selector(
        candidates,
        target_param_saving=50,
        total_candidate_units=3,
    )

    assert [item.candidate_id for item in selected] == ["expensive_low_loss"]
    assert rejected[0]["reject_reason_if_any"] == "budget_reached"
    assert report["selector"] == "global_budgeted_coupled_unit_selector"
    assert report["score_formula"] == "importance_score / max(param_saving_if_removed, eps)"


def test_v101_budget_report_includes_channel_ratio_and_taylor_metadata():
    report = build_v101_budget_report(
        strategy="A1",
        target_param_prune_ratio_full_model=0.2,
        actual_param_prune_ratio_full_model=0.18,
        target_channel_prune_ratio=0.2,
        actual_channel_prune_ratio=0.19,
        grouped_conv_param_prune_ratio=0.3,
        non_grouped_conv_param_prune_ratio=0.2,
        total_conv_param_prune_ratio=0.22,
        baseline_total_params=1000,
        pruned_total_params=820,
        importance_report={
            "importance_mode_used": "first_order_taylor",
            "calibration_batches": 1,
            "gradients_successfully_collected": True,
        },
    )

    assert report["importance_mode_used"] == "first_order_taylor"
    assert math.isclose(report["actual_channel_prune_ratio"], 0.19)
    assert report["why_not_reached"] == "selector_budget_unreachable"


def test_summary_verdict_contains_concrete_alignment_comparisons(tmp_path):
    baseline = {
        "latency_report": {"p90_ms": 20.0, "p95_ms": 25.0},
    }
    common = {
        "target_param_prune_ratio_full_model": 0.10,
        "target_channel_prune_ratio": 0.10,
        "actual_grouped_conv_param_prune_ratio": 0.0,
        "actual_non_grouped_conv_param_prune_ratio": 0.1,
        "actual_total_conv_param_prune_ratio": 0.1,
        "AP_baseline": 0.8,
        "AP_pruned": 0.8,
        "AP_drop_rel": 0.0,
        "latency_baseline_p50_ms": 10.0,
        "latency_baseline_mean_ms": 12.0,
        "shape_simulator_passed": True,
        "physical_prune_passed": True,
        "synthetic_forward_passed": True,
        "eval_forward_passed": True,
        "latency_passed": True,
        "importance_mode_used": "first_order_taylor",
        "selector": "global_budgeted_coupled_unit_selector",
        "num_candidate_units": 10,
        "num_selected_units": 2,
        "num_rejected_units": 8,
        "num_unaligned_grouped_conv_shapes": 0,
        "failure_status": "success",
        "failure_category": "",
        "failure_reason": "",
    }
    results = [
        {
            **common,
            "experiment": "A1_010",
            "strategy": "A1",
            "policy_name": "A1",
            "target_ratio": 0.10,
            "actual_param_prune_ratio_full_model": 0.11,
            "actual_channel_prune_ratio": 0.02,
            "AP_drop_abs": 0.001,
            "latency_pruned_p50_ms": 8.0,
            "latency_pruned_mean_ms": 10.0,
            "speedup_p50": 1.25,
            "speedup_mean": 1.2,
            "latency_pruned_p90_ms": 18.0,
            "latency_pruned_p95_ms": 22.0,
        },
        {
            **common,
            "experiment": "A2_010",
            "strategy": "A2",
            "policy_name": "A2",
            "target_ratio": 0.10,
            "actual_param_prune_ratio_full_model": 0.11,
            "actual_channel_prune_ratio": 0.02,
            "AP_drop_abs": 0.0,
            "latency_pruned_p50_ms": 11.0,
            "latency_pruned_mean_ms": 13.0,
            "speedup_p50": 0.9,
            "speedup_mean": 0.92,
            "latency_pruned_p90_ms": 21.0,
            "latency_pruned_p95_ms": 26.0,
        },
    ]

    write_summary(tmp_path, baseline, results)

    text = (tmp_path / "summary" / "v101_global_budgeted_alignment_verdict.md").read_text()
    assert "A1 vs A2: A1" in text
    assert "p50/mean/p90/p95 all improved: A1_010" in text
