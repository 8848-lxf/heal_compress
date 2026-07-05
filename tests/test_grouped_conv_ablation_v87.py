from __future__ import annotations

import torch

from tools.latency_lut.run_grouped_conv_ablation_v87 import (
    compute_reinterpretation,
    make_grouped_conv_candidate,
    ordinary_grouped_conv,
)


def test_depthwise_conv_is_excluded_from_ordinary_grouped_conv():
    depthwise = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16)
    regular = torch.nn.Conv2d(64, 128, 3, padding=1, groups=32)

    assert ordinary_grouped_conv(depthwise) is False
    assert ordinary_grouped_conv(regular) is True


def test_flat_policy_can_have_local_input_reinterpretation():
    old_to_new = {0: 0, 2: 1, 3: 2, 7: 3}
    stats = compute_reinterpretation(old_to_new, c_out_before=8, c_out_after=4, groups=2)

    assert stats["reinterpretation_count"] > 0
    assert stats["reinterpretation_ratio"] > 0


def test_group_balanced_policy_preserves_old_group_assignment():
    conv = torch.nn.Conv2d(16, 16, 1, groups=4, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel(), dtype=torch.float32).view_as(conv.weight))

    cand = make_grouped_conv_candidate(
        module_name="gconv",
        module=conv,
        policy="group_balanced_output_groups_fixed",
        score_mode="l1",
        target_prune_ratio=0.5,
    )

    assert cand["legality_status"] == "legal"
    assert cand["group_balanced_pass"] is True
    assert cand["reinterpretation_ratio"] == 0
    assert len(set(cand["old_group_keep_count"].values())) == 1


def test_group_coarsening_zero_padded_reblock_has_no_weight_truncation_and_tracks_flops():
    conv = torch.nn.Conv2d(16, 16, 1, groups=4, bias=False)
    cand = make_grouped_conv_candidate(
        module_name="gconv",
        module=conv,
        policy="group_coarsening_zero_padded_reblock",
        score_mode="l1",
        target_prune_ratio=0.5,
    )

    assert cand["legality_status"] == "legal"
    assert cand["groups_after"] < cand["groups_before"]
    assert cand["weight_copy_status"] == "copy_plus_zero_pad"
    assert cand["weight_truncation_count"] == 0
    assert cand["dense_flops_after"] >= 0

