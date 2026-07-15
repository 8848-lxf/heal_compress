from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_full_model_physical_prune_rate_uses_actual_parameter_counts() -> None:
    from search.audits.prune_rate_reachability import prune_rate_metrics

    row = prune_rate_metrics(
        requested_prune_rate=0.60,
        predicted_original_params=1000,
        predicted_candidate_params=420,
        physical_original_params=1000,
        physical_candidate_params=400,
        prunable_original_params=800,
        atomic_unit_count=100,
        pruned_atomic_unit_count=30,
        channel_count=200,
        pruned_channel_count=50,
    )

    assert row["predicted_full_param_prune_rate"] == pytest.approx(0.58)
    assert row["physical_full_param_prune_rate"] == pytest.approx(0.60)
    assert row["physical_prunable_param_prune_rate"] == pytest.approx(0.75)
    assert row["atomic_unit_prune_ratio"] == pytest.approx(0.30)
    assert row["channel_prune_ratio"] == pytest.approx(0.25)
    assert row["requested_prune_rate"] == pytest.approx(0.60)


def test_requested_and_realized_rates_are_never_aliased() -> None:
    from search.audits.prune_rate_reachability import prune_rate_metrics

    row = prune_rate_metrics(
        requested_prune_rate=0.7,
        predicted_original_params=100,
        predicted_candidate_params=80,
        physical_original_params=100,
        physical_candidate_params=79,
        prunable_original_params=80,
        atomic_unit_count=10,
        pruned_atomic_unit_count=4,
        channel_count=10,
        pruned_channel_count=4,
    )

    assert row["requested_prune_rate"] == pytest.approx(0.7)
    assert row["physical_full_param_prune_rate"] == pytest.approx(0.21)
    assert row["requested_prune_rate"] != row["physical_full_param_prune_rate"]
