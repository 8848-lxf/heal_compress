from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class ResidualConcatAuditToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch_a = nn.Conv2d(4, 4, 1)
        self.branch_b = nn.Conv2d(4, 4, 1)
        self.consumer = nn.Conv2d(8, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        _plain_channel_concat = self.consumer(torch.cat([a, b], dim=1))
        non_channel = torch.cat([a, b], dim=2)
        return non_channel + 1.0


def test_residual_concat_fail_root_cause_classifies_non_channel_concat(tmp_path: Path) -> None:
    from tools.latency_lut.audit_dependency_graph_v99 import run_toy_residual_concat_fail_root_cause_audit

    out = tmp_path / "v99"
    report = run_toy_residual_concat_fail_root_cause_audit(ResidualConcatAuditToy().eval(), torch.randn(1, 4, 8, 8), out)

    csv_path = out / "residual_concat_fail_root_cause_report.csv"
    matrix_path = out / "residual_concat_fixability_matrix.csv"
    update_path = out / "residual_concat_supported_surface_update.json"
    assert csv_path.exists()
    assert matrix_path.exists()
    assert update_path.exists()

    rows = list(csv.DictReader(csv_path.open()))
    assert rows
    assert any("non_channelwise_add_or_cat" in row["failure_category"] for row in rows)
    assert all(row["failure_category"] for row in rows)
    assert all(row["recommended_action"] for row in rows)
    assert all(row["can_enter_prunable_surface"] in {"True", "False"} for row in rows)

    matrix = list(csv.DictReader(matrix_path.open()))
    assert any(row["fixability"] == "should_remain_protected" for row in matrix)
    update = json.loads(update_path.read_text())
    assert update["dynamic_branch_enumeration_enabled"] is False
    assert report["num_fail_rows"] == len(rows)
