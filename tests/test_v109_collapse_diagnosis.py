from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.latency_lut.run_v109_collapse_diagnosis import (  # noqa: E402
    boundary_stage_summary,
    diff_selected_units,
    final_report_has_required_sections,
    grouped_conv_diff_rows,
    is_stage1_grouped_output_domain,
    protect_stage1_grouped_output_domains,
    remove_stage1_grouped_output_from_selection,
    require_shape_invariant_passed,
)


class _Unit:
    def __init__(self, domain: str, idx: int, *, grouped: bool = False) -> None:
        self.pruning_domain_id = domain
        self.root_module_name = domain.removeprefix("group::")
        self.coupled_unit_id = f"{domain}::idx{idx}"
        self.root_channel_index = idx
        self.is_grouped_conv = grouped
        self.importance_normalized = float(idx)
        self.importance_raw = float(idx)
        self.group_index = idx // 8 if grouped else None
        self.local_channel_index = idx % 8 if grouped else None
        self.grouped_local_unit_id = f"g{self.group_index}:l{self.local_channel_index}" if grouped else None
        self.protected_reason = ""
        self.skipped_reason = ""

    def to_report_row(self) -> dict[str, object]:
        return dict(self.__dict__)


class _Domain:
    def __init__(self, root: str, *, grouped: bool = False, channels: int = 256, groups: int = 32) -> None:
        self.pruning_domain_id = f"group::{root}"
        self.root_module_name = root
        self.is_grouped_conv = grouped
        self.num_root_channels = channels
        self.groups = groups
        self.per_group = channels // groups if grouped else None
        self.protected_reason = ""
        self.skipped_reason = ""
        self.units = [_Unit(self.pruning_domain_id, idx, grouped=grouped) for idx in range(channels)]


class _Plan:
    def __init__(self, domain: str, prune_indices: list[int]) -> None:
        self.pruning_domain_id = domain
        self.prune_indices = list(prune_indices)
        self.keep_indices = [idx for idx in range(256) if idx not in set(prune_indices)]
        self.final_n_pruned = len(prune_indices)
        self.raw_selected_count = len(prune_indices)
        self.grouped_decision = object()
        self.skipped_reason = ""


class _Selection:
    def __init__(self, domain: str, prune_indices: list[int]) -> None:
        self.domain_plans = {domain: _Plan(domain, prune_indices)}
        self.selected_units = [_Unit(domain, idx, grouped=True) for idx in prune_indices]
        self.skipped_units = []
        self.grouped_shape_rows = [
            {
                "module_name": "pyramid_backbone.resnet.layer1.0.conv2",
                "stage_guess": "stage1_like",
                "out_per_group_before": 8,
                "out_per_group_after": 4,
                "output_pruned": True,
                "input_pruned": True,
            }
        ]
        self.actual_channel_prune_ratio_on_searchable_surface = 0.5
        self.predicted_param_prune_ratio = 0.7


def test_selected_units_diff_identifies_newly_selected_in_070() -> None:
    m60 = {"selected_units": [{"coupled_unit_id": "u1", "pruning_domain_id": "d", "root_module_name": "m", "root_channel_index": 1}]}
    m70 = {
        "selected_units": [
            {"coupled_unit_id": "u1", "pruning_domain_id": "d", "root_module_name": "m", "root_channel_index": 1},
            {"coupled_unit_id": "u2", "pruning_domain_id": "d", "root_module_name": "m", "root_channel_index": 2},
        ]
    }

    rows = diff_selected_units(m60, m70)

    by_id = {row["unit_id"]: row for row in rows}
    assert by_id["u1"]["newly_selected_in_070"] is False
    assert by_id["u2"]["newly_selected_in_070"] is True


def test_grouped_conv_diff_detects_stage1_output_8_to4() -> None:
    rows = grouped_conv_diff_rows(
        [
            {
                "module_name": "pyramid_backbone.resnet.layer1.0.conv2",
                "stage_guess": "stage1_like",
                "groups": 32,
                "out_per_group_before": 8,
                "out_per_group_after": 8,
                "in_per_group_before": 8,
                "in_per_group_after": 8,
                "output_pruned": False,
                "input_pruned": False,
                "legality_passed": True,
            }
        ],
        [
            {
                "module_name": "pyramid_backbone.resnet.layer1.0.conv2",
                "stage_guess": "stage1_like",
                "groups": 32,
                "out_per_group_before": 8,
                "out_per_group_after": 4,
                "in_per_group_before": 8,
                "in_per_group_after": 4,
                "output_pruned": True,
                "input_pruned": True,
                "legality_passed": True,
            }
        ],
    )

    assert rows[0]["newly_pruned_output_in_070"] is True
    assert rows[0]["out_per_group_060"] == 8
    assert rows[0]["out_per_group_070"] == 4


def test_experiment_a_policy_protects_stage1_grouped_output_but_not_input() -> None:
    stage1 = _Domain("pyramid_backbone.resnet.layer1.0.conv2", grouped=True)
    protected = protect_stage1_grouped_output_domains([stage1])

    assert protected[0]["root_module_name"] == "pyramid_backbone.resnet.layer1.0.conv2"
    assert stage1.protected_reason == "diagnostic_protect_stage1_grouped_output"
    assert is_stage1_grouped_output_domain(stage1) is True
    assert protected[0]["input_direction_protected"] is False


def test_experiment_b_removes_stage1_output_prune_units_from_selection() -> None:
    domain = _Domain("pyramid_backbone.resnet.layer1.0.conv2", grouped=True)
    selection = _Selection(domain.pruning_domain_id, list(range(128)))

    removed = remove_stage1_grouped_output_from_selection(selection, [domain])

    assert len(removed) == 128
    assert selection.domain_plans[domain.pruning_domain_id].prune_indices == []
    assert selection.domain_plans[domain.pruning_domain_id].grouped_decision is None
    assert not any(unit.pruning_domain_id == domain.pruning_domain_id for unit in selection.selected_units)


def test_boundary_stage_summary_parses_per_group_changes() -> None:
    row = {
        "target_pruning_ratio": "0.68",
        "stage1_out_per_group_before_after": "[8, 4]",
        "stage2_out_per_group_before_after": "[16, 12]",
    }

    parsed = boundary_stage_summary(row)

    assert parsed["target"] == 0.68
    assert parsed["stage1_out_per_group_before_after"] == [8, 4]
    assert parsed["stage2_out_per_group_before_after"] == [16, 12]


def test_shape_invariant_gate_blocks_val500_on_violation() -> None:
    with pytest.raises(RuntimeError, match="shape_invariant_failed"):
        require_shape_invariant_passed({"passed": False, "non_channel_shape_violation_count": 2})


def test_shape_invariant_gate_allows_clean_report() -> None:
    require_shape_invariant_passed({"passed": True, "non_channel_shape_violation_count": 0})


def test_final_report_contains_required_abc_sections() -> None:
    text = """
    # report
    ## evidence
    A result
    ## diagnosis
    B result
    ## recommended_policy_change
    ## next_experiments
    """

    assert final_report_has_required_sections(text) is True
