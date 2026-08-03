from __future__ import annotations

import pytest

from search.reporting.v2xvit_six_budget import (
    classify_ap_drop,
    classify_ga_admission,
    compression_metrics,
)


@pytest.mark.parametrize(
    ("candidate", "label", "catastrophic"),
    ((0.655, "SAFE", False), (0.64, "MILD_DROP", False), (0.60, "SIGNIFICANT_DROP", False), (0.50, "SEVERE_COLLAPSE", False), (0.30, "SEVERE_COLLAPSE", True)),
)
def test_ap_collapse_boundaries(candidate, label, catastrophic):
    row = classify_ap_drop(0.66, candidate)
    assert row["classification"] == label
    assert row["catastrophic_collapse"] is catastrophic


def test_ga_admission_is_fail_closed():
    common = dict(
        budget_reached=True,
        s32_drop=0.001,
        jmix_drop=0.002,
        jmix_engine_built=True,
        requested_realized_exact=True,
        precision_conflict_count=0,
        fallback_count=0,
        latency_batch_valid=True,
        jmix_speedup=1.2,
        taylor_convergence_passed=True,
        framework_tests_passed=True,
    )
    assert classify_ga_admission(**common) == "GA_ADMISSIBLE"
    assert classify_ga_admission(**{**common, "fallback_count": 1}) == "GA_DEPLOYMENT_INVALID"
    assert classify_ga_admission(**{**common, "s32_drop": 0.02}) == "GA_UNSAFE"
    assert classify_ga_admission(**{**common, "taylor_convergence_passed": False}) == "GA_PROXY_UNRELIABLE"


def test_compression_metrics_do_not_confuse_retention_and_ratio():
    row = compression_metrics(bops_retention=0.25, parameter_retention=0.5, mixed_weight_retention=0.125)
    assert row["BOPS_compression"] == 4.0
    assert row["parameter_compression"] == 2.0
    assert row["mixed_weight_compression"] == 8.0
