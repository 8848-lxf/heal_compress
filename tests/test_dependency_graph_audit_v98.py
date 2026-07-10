from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heal_compress.pruning.pruning_fns import prune_conv_in, prune_conv_out
from heal_compress.pruning.units import expand_coupled_channel_units
from heal_compress.tracer.pruning_group import PruningGroup


def test_coupled_channel_units_export_v98_member_proof_fields() -> None:
    conv1 = nn.Conv2d(8, 16, 1, bias=False)
    conv2 = nn.Conv2d(16, 24, 1, bias=False)
    scope = PruningGroup(
        "group::conv1",
        num_channels=16,
        meta={"group_type": "plain", "roots": ["conv1"]},
    )
    scope.add_dep("conv1", conv1, prune_conv_out, "out", reason="root_out")
    scope.add_dep("conv2", conv2, prune_conv_in, "in", reason="downstream_in")

    unit = expand_coupled_channel_units(scope)[0]

    assert isinstance(unit.is_minimal_proven, bool)
    assert isinstance(unit.proof_edges, list)
    assert unit.unsupported_reason == ""
    required = {
        "module_name",
        "module_type",
        "axis",
        "local_index",
        "dependency_type",
        "producer_tensor",
        "consumer_tensor",
        "branch_id",
        "concat_offset",
        "residual_add_id",
        "grouped_conv_role",
        "transpose_conv_role",
    }
    assert required.issubset(unit.members[0])
    assert unit.members[0]["module_name"] == "conv1"
    assert unit.members[0]["local_index"] == 0


class AuditToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.branch_a = nn.Conv2d(8, 8, 1, bias=False)
        self.branch_b = nn.Conv2d(8, 8, 1, bias=False)
        self.concat_consumer = nn.Conv2d(16, 8, 1, bias=False)
        self.grouped = nn.Conv2d(8, 8, 3, padding=1, groups=4, bias=False)
        self.deblock = nn.ConvTranspose2d(8, 8, 2, stride=2, bias=False)
        self.cls_head = nn.Conv2d(8, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(self.stem(x))
        a = self.branch_a(y)
        b = self.branch_b(y)
        c = self.concat_consumer(torch.cat([a, b], dim=1))
        g = self.grouped(c)
        return self.cls_head(self.deblock(g + c))


def test_dependency_graph_audit_v98_writes_required_toy_artifacts(tmp_path) -> None:
    from tools.latency_lut.audit_dependency_graph_v98 import run_audit_for_model

    out = tmp_path / "audit"
    report = run_audit_for_model(
        AuditToy().eval(),
        torch.randn(1, 3, 8, 8),
        out,
        forward_fn=None,
        protected_layers=["cls_head", "deblock"],
        group_conv_policy="A",
        align=4,
        group_conv_align=1,
        run_tp_oracle=False,
    )

    expected = {
        "trace_graph_coverage_report.json",
        "coupled_channel_unit_completeness_report.json",
        "coupled_channel_units_full_model.json",
        "operator_dependency_coverage_matrix.csv",
        "residual_concat_full_model_proof.csv",
        "grouped_conv_dependency_proof.csv",
        "convtranspose_dependency_proof.csv",
        "tp_oracle_sampling_diff_report.json",
        "mask0_physical_removal_dryrun_report.json",
    }
    assert expected.issubset({p.name for p in out.iterdir()})

    coverage = json.loads((out / "trace_graph_coverage_report.json").read_text())
    assert coverage["dynamic_branch_enumeration_enabled"] is False
    assert coverage["num_forward_paths_traced"] == 1
    assert "stem" in coverage["traced_modules"]
    assert coverage["dynamic_paths_not_covered"]

    units = json.loads((out / "coupled_channel_units_full_model.json").read_text())
    assert units
    assert {"unit_id", "root_node", "members", "is_minimal_proven", "proof_edges", "unsupported_reason"}.issubset(units[0])

    completeness = json.loads((out / "coupled_channel_unit_completeness_report.json").read_text())
    assert completeness["num_units"] == len(units)
    assert completeness["all_units_have_required_member_fields"] is True
    assert report["output_dir"] == str(out)
