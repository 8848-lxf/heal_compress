from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from heal_compress.pruning.group_checker import check_pruning_group
from heal_compress.pruning.grouped_conv import grouped_conv_pruning_fn
from heal_compress.pruning.pruning_fns import prune_bn, prune_conv_in
from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan
from heal_compress.tracer.pruning_group import PruningGroup


class FlatGroupedToy(nn.Module):
    def __init__(self, groups: int = 4, in_per_group: int = 8, out_per_group: int = 8):
        super().__init__()
        c_in = groups * in_per_group
        c_out = groups * out_per_group
        self.grouped = nn.Conv2d(c_in, c_out, 3, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.head = nn.Conv2d(c_out, 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(self.grouped(x)))


def _flat_scope(model: FlatGroupedToy) -> PruningGroup:
    scope = PruningGroup(
        group_id="toy.flat_grouped_out",
        num_channels=model.grouped.out_channels,
        meta={"group_type": "plain"},
    )
    scope.add_dep(
        "grouped",
        model.grouped,
        grouped_conv_pruning_fn("flat_output_groups_fixed"),
        "out",
        reason="grouped_conv:flat_output_groups_fixed",
    )
    scope.add_dep("bn", model.bn, prune_bn, "out", reason="bn")
    scope.add_dep("head", model.head, prune_conv_in, "in", reason="downstream_in")
    return scope


def test_flat_output_groups_fixed_allows_uneven_old_group_keep_pattern():
    model = FlatGroupedToy(groups=4, in_per_group=8, out_per_group=8).eval()
    scope = _flat_scope(model)
    # Low scores are all in old output group 0, so A intentionally creates an
    # uneven old-group keep pattern. This is a risk metric, not a rejection.
    importance = torch.arange(32, dtype=torch.float32)

    plan = build_pruning_plan(
        [scope],
        {scope.group_id: importance},
        SelectionConfig(
            prune_ratio=0.25,
            selection_mode="local_scope",
            group_conv_selection_mode="flat_output_groups_fixed",
            align=4,
            # The requested 24-output shape has six channels per group.  Use
            # align=2 so the test exercises uneven old-group selection without
            # contradicting the deployment per-group alignment contract.
            group_conv_align=2,
            min_channels=4,
        ),
    )

    concrete = plan.concrete_groups[0]
    report = plan.grouped_conv_reports[0]

    assert concrete.prune_indices == list(range(8))
    assert concrete.keep_indices == list(range(8, 32))
    assert report["grouped_keep_pattern_mismatch"] is True
    assert report["reinterpretation_ratio"] > 0
    assert report["structure_legal"] is True

    check = check_pruning_group(scope, concrete.keep_indices, group_conv_align=2)
    assert check["legal"], check["issues"]
    result = scope.prune(concrete.keep_indices)
    assert result["applied"]
    assert model.grouped.groups == 4
    assert model.grouped.in_channels == 32
    assert model.grouped.out_channels == 24
    assert tuple(model.grouped.weight.shape) == (24, 8, 3, 3)
    assert model.bn.num_features == 24
    assert model.head.in_channels == 24
    assert model(torch.randn(2, 32, 8, 8)).shape == (2, 2, 8, 8)


def test_v93_output_validator_rejects_tp_as_a_model_generation_path(tmp_path: Path):
    from tools.latency_lut.run_grouped_conv_ablation_v93_pruner_native_abcd import validate_v93_output_bundle

    out = tmp_path
    (out / "a_project_pruner_implementation_report.json").write_text(
        json.dumps({"uses_torch_pruning_for_model_generation": True}),
        encoding="utf-8",
    )
    (out / "a_project_pruner_vs_tp_oracle_audit.json").write_text(
        json.dumps({"num_layers_checked": 6, "all_shape_equivalent": True}),
        encoding="utf-8",
    )
    with (out / "a_project_pruner_full_model_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target_prune_ratio", "forward_smoke_status"])
        writer.writeheader()
        writer.writerow({"target_prune_ratio": "0.05", "forward_smoke_status": "forward_failed"})
        writer.writerow({"target_prune_ratio": "0.10", "forward_smoke_status": "forward_failed"})
    with (out / "d_full_model_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target_prune_ratio", "attempted_full_model_rewrite"])
        writer.writeheader()
        writer.writerow({"target_prune_ratio": "0.05", "attempted_full_model_rewrite": "true"})
        writer.writerow({"target_prune_ratio": "0.10", "attempted_full_model_rewrite": "true"})

    result = validate_v93_output_bundle(out)
    assert result["valid"] is False
    assert "a_uses_torch_pruning_for_model_generation" in result["errors"]


def test_v93_output_validator_requires_d_full_model_attempts(tmp_path: Path):
    from tools.latency_lut.run_grouped_conv_ablation_v93_pruner_native_abcd import validate_v93_output_bundle

    (tmp_path / "a_project_pruner_implementation_report.json").write_text(
        json.dumps({"uses_torch_pruning_for_model_generation": False}),
        encoding="utf-8",
    )
    (tmp_path / "a_project_pruner_vs_tp_oracle_audit.json").write_text(
        json.dumps({"num_layers_checked": 6, "all_shape_equivalent": True}),
        encoding="utf-8",
    )
    with (tmp_path / "a_project_pruner_full_model_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target_prune_ratio", "forward_smoke_status"])
        writer.writeheader()
        writer.writerow({"target_prune_ratio": "0.05", "forward_smoke_status": "forward_failed"})
        writer.writerow({"target_prune_ratio": "0.10", "forward_smoke_status": "forward_failed"})
    with (tmp_path / "d_full_model_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target_prune_ratio", "attempted_full_model_rewrite"])
        writer.writeheader()
        writer.writerow({"target_prune_ratio": "0.05", "attempted_full_model_rewrite": "false"})

    result = validate_v93_output_bundle(tmp_path)
    assert result["valid"] is False
    assert "d_full_model_attempt_missing_or_false:0.05" in result["errors"]
    assert "d_full_model_attempt_missing_or_false:0.10" in result["errors"]
