from __future__ import annotations


def test_v941_dense_conv_flops_proxy_reduces_with_channels():
    from tools.latency_lut.finalize_v941_reports import conv_flops_proxy

    before = conv_flops_proxy(c_in=128, c_out=128, groups=32, kh=3, kw=3)
    after = conv_flops_proxy(c_in=128, c_out=96, groups=32, kh=3, kw=3)

    assert after < before
    assert before == 128 * (128 // 32) * 3 * 3


def test_v941_gap_causes_include_group_legality_and_noop():
    from tools.latency_lut.finalize_v941_reports import classify_gap_causes

    causes = classify_gap_causes(
        policy="B",
        target=0.05,
        actual=0.0,
        num_operations=0,
        friendly_ratio=1.0,
    )

    assert "group_balanced_discretization_no_valid_prune_at_low_ratio" in causes
    assert "noop_global_physical_plan" in causes
