import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.run_abcd_small_eval_v100 import (
    build_model_args_for_strategy,
    build_target_budget_report,
    choose_d_compact_frontfill_keep,
    compute_param_inventory,
    compute_shape_alignment_report,
    summarize_latency,
)


def test_param_inventory_uses_full_model_ratio_as_primary_metric():
    base = nn.Sequential(
        nn.Conv2d(3, 8, 1),
        nn.Conv2d(8, 16, 3, padding=1, groups=4),
        nn.BatchNorm2d(16),
        nn.Linear(16, 4),
    )
    pruned = nn.Sequential(
        nn.Conv2d(3, 6, 1),
        nn.Conv2d(8, 8, 3, padding=1, groups=4),
        nn.BatchNorm2d(8),
        nn.Linear(16, 4),
    )

    before = compute_param_inventory(base)
    after = compute_param_inventory(pruned)
    report = build_target_budget_report(
        strategy="A",
        target_param_prune_ratio_full_model=0.20,
        target_domain_unit_prune_ratio=0.20,
        before=before,
        after=after,
        supported_surface_params_before=100,
        supported_surface_params_after=80,
        total_units=50,
        pruned_units=10,
    )

    expected_full = 1.0 - after["total_params"] / before["total_params"]
    assert math.isclose(report["actual_param_prune_ratio_full_model"], expected_full)
    assert report["target_param_prune_ratio_full_model"] == 0.20
    assert math.isclose(report["actual_param_prune_ratio_supported_surface"], 0.20)
    assert "actual_grouped_conv_param_prune_ratio" in report
    assert report["why_not_reached"] in {"", "domain unit ratio does not match parameter ratio"}


def test_d_compact_frontfill_choice_allows_semantic_mismatch_and_records_flags():
    conv = nn.Conv2d(32, 32, 3, padding=1, groups=8, bias=False)
    preferred_keep = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 31, 30]

    keep, metadata = choose_d_compact_frontfill_keep(conv, preferred_keep)

    assert keep == preferred_keep
    assert metadata["groups_old"] == 8
    assert metadata["groups_new"] == 4
    assert metadata["old_output_keep_indices"] == preferred_keep
    assert metadata["old_input_keep_indices"] == list(range(32))
    assert metadata["semantic_preserved"] is False
    assert metadata["compact_first"] is True
    assert metadata["frontfill_weight_transplant"] is True
    assert metadata["new_connections_zero_initialized"] is True
    assert metadata["requires_recovery_finetune"] is True
    assert metadata["ordered_keep_indices"] == preferred_keep


def test_shape_alignment_report_marks_unfriendly_grouped_shapes():
    model = nn.Sequential(
        nn.Conv2d(3, 10, 1),
        nn.Conv2d(10, 14, 3, padding=1, groups=2),
    )

    report = compute_shape_alignment_report(model, policy="B")

    assert report["num_conv2d"] == 2
    assert report["num_grouped_conv2d"] == 1
    grouped = report["grouped_conv2d"][0]
    assert grouped["policy"] == "B"
    assert grouped["in_per_group"] == 5
    assert grouped["out_per_group"] == 7
    assert grouped["is_likely_hardware_friendly"] is False
    assert report["num_unaligned_grouped_conv_shapes"] == 1


def test_summarize_latency_reports_mean_p50_p90_p95_and_speedup():
    baseline = summarize_latency([10.0, 12.0, 14.0, 16.0], baseline=None)
    pruned = summarize_latency([8.0, 9.0, 10.0, 11.0], baseline=baseline)

    assert baseline["mean_ms"] == 13.0
    assert baseline["p50_ms"] == 13.0
    assert pruned["p50_ms"] == 9.5
    assert pruned["p90_ms"] >= pruned["p50_ms"]
    assert pruned["p95_ms"] >= pruned["p90_ms"]
    assert math.isclose(pruned["speedup_p50"], baseline["p50_ms"] / pruned["p50_ms"], rel_tol=1e-6)
    assert math.isclose(pruned["speedup_mean"], baseline["mean_ms"] / pruned["mean_ms"], rel_tol=1e-6)


def test_strategy_model_args_include_importance_fields():
    class Args:
        checkpoint = "ckpt.pth"
        model_config = "config.yaml"
        heal_root = "/heal"
        device = "cuda:5"
        extra_protected_prefix = ["head"]
        num_calib_batches = 3
        importance_mode = "first_order_taylor"

    model_args = build_model_args_for_strategy(Args())

    assert model_args.importance_mode == "first_order_taylor"
    assert model_args.num_calib_batches == 3
    assert model_args.extra_protected_prefix == ["head"]
