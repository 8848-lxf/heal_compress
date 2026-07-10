from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.real_val500_stage512_reload_eval_v1071 import (  # noqa: E402
    build_ap_report,
    build_latency_report_row,
    count_skip_reasons,
)


def test_count_skip_reasons_ignores_successful_rows() -> None:
    rows = [
        {"success": True, "skip_reason": ""},
        {"success": False, "skip_reason": "empty_batch"},
        {"success": False, "skip_reason": "empty_batch"},
        {"success": False, "skip_reason": "RuntimeError: bad frame"},
    ]

    assert count_skip_reasons(rows) == {"empty_batch": 2, "RuntimeError: bad frame": 1}


def test_latency_report_row_schema() -> None:
    row = build_latency_report_row(
        variant="pruned",
        summary={
            "actual_frames": 500,
            "missing_frames": 2,
            "forward_time_p50_ms": 8.0,
            "forward_time_mean_ms": 8.5,
            "total_time_p50_ms": 10.0,
            "total_time_mean_ms": 11.0,
        },
        rows=[{"success": False, "skip_reason": "x"}],
        baseline=None,
        requested_frames=500,
    )

    assert row["variant"] == "pruned"
    assert row["num_frames"] == 500
    assert row["evaluated_frames"] == 500
    assert row["skipped_frames"] == 2
    assert row["skip_reason_counts"] == {"x": 1}
    assert row["forward_latency_p50"] == 8.0
    assert row["total_latency_mean"] == 11.0


def test_ap_report_computes_drop_vs_baseline() -> None:
    baseline = build_ap_report(
        variant="baseline",
        summary={"actual_frames": 500, "AP_0_30": 0.8, "AP_0_50": 0.7, "AP_0_70": 0.5},
        baseline_ap30=None,
        requested_frames=500,
        failure_reason="",
    )
    pruned = build_ap_report(
        variant="pruned",
        summary={"actual_frames": 500, "AP_0_30": 0.65, "AP_0_50": 0.55, "AP_0_70": 0.35},
        baseline_ap30=baseline["AP@0.30"],
        requested_frames=500,
        failure_reason="",
    )

    assert baseline["metric_helper_used"] == "test_prune_and_eval.evaluate_one_model"
    assert pruned["AP_drop_vs_baseline"] == 0.15
    assert pruned["mAP"] == (0.65 + 0.55 + 0.35) / 3.0
