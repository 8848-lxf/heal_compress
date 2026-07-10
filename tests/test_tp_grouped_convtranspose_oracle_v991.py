from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


torch_pruning = pytest.importorskip("torch_pruning")


def test_tp_grouped_convtranspose_oracle_writes_in_and_out_reports(tmp_path: Path) -> None:
    from tools.latency_lut.tp_grouped_convtranspose_oracle_v991 import run_oracle

    out = tmp_path / "oracle"
    summary = run_oracle(out)

    out_report = json.loads((out / "tp_grouped_convtranspose_out_pruning_report.json").read_text())
    in_report = json.loads((out / "tp_grouped_convtranspose_in_pruning_report.json").read_text())
    recommendation = json.loads((out / "tp_grouped_convtranspose_migration_recommendation.json").read_text())
    analysis = (out / "tp_grouped_convtranspose_strategy_analysis.md").read_text()

    assert summary["num_out_cases"] == 2
    assert summary["num_in_cases"] == 2
    assert {case["case_name"] for case in out_report["cases"]} == {"out_global_unbalanced", "out_local_balanced_expanded"}
    assert {case["case_name"] for case in in_report["cases"]} == {"in_global_unbalanced", "in_group_balanced"}
    assert all("build_group_ok" in case for case in out_report["cases"] + in_report["cases"])
    assert all("forward_passed" in case for case in out_report["cases"] + in_report["cases"])
    assert recommendation["recommendation"] in {
        "do_not_migrate_tp_unsupported",
        "migrate_group_balanced_input_output_resolver",
        "migrate_true_group_block_only",
        "investigate_further_due_to_ambiguous_tp_behavior",
    }
    assert "TP 是否支持 ordinary grouped ConvTranspose2d" in analysis
