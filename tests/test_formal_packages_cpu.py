from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


def _toy_units():
    from tracer.types import CoupledChannelUnit, DependencyMember

    return [
        CoupledChannelUnit(
            scope_id="scope::conv1",
            root_module_path="conv1",
            root_axis="out",
            root_channel_index=index,
            members=[
                DependencyMember("conv1", "out", [index], "root_output"),
                DependencyMember("bn1", "channel", [index], "conv_bn"),
                DependencyMember("conv2", "in", [index], "downstream_input"),
            ],
        )
        for index in range(2)
    ]


class ResidualConcatModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.left = nn.Conv2d(8, 8, 1)
        self.right = nn.Conv2d(8, 8, 1)
        self.tail = nn.Conv2d(16, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        added = self.left(x) + self.right(x)
        return self.tail(torch.cat([added, x], dim=1))


class DirectionalProtectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(8, 8, 1)
        self.deblocks = nn.ModuleList([nn.ConvTranspose2d(8, 8, 2, stride=2)])
        self.fpn_out = nn.Conv2d(8, 8, 1)
        self.cls_head = nn.Conv2d(8, 6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.deblocks[0](x)
        x = self.fpn_out(x)
        return self.cls_head(x)


def test_formal_defaults_are_frozen_and_round_trip() -> None:
    from pruning.config import PruningConfig
    from quantization.config import QDQConfig

    config = PruningConfig()
    assert config.importance.mode.value == "first_order_taylor"
    assert config.importance.normalization.strategy.value == "coupled_dependency_mean_then_scope_mean_v1"
    assert config.selection.strategy.value == "global_one_shot"
    assert config.alignment.dense_conv_channel_alignment == 4
    assert config.grouped_conv.allowed_channels_per_group == (4, 8, 16, 32, 64, 128, 256, 512)
    assert config.grouped_conv.selection_policy.value == "independent_group_topk"
    assert PruningConfig.from_dict(config.to_dict()) == config
    assert QDQConfig().allowed_precisions == ("fp16", "int8")


def test_formal_configs_round_trip_through_safe_yaml() -> None:
    from pruning.config import PruningConfig
    from quantization.config import QuantizationConfig
    from tracer.config import TraceConfig

    for config_type, config in (
        (TraceConfig, TraceConfig()),
        (PruningConfig, PruningConfig()),
        (QuantizationConfig, QuantizationConfig()),
    ):
        assert config_type.from_yaml(config.to_yaml()).to_dict() == config.to_dict()


def test_trace_result_serialization_hash_and_stable_unit_id(tmp_path: Path) -> None:
    from tracer.api import load_trace_result, serialize_trace_result, trace_model

    model = ResidualConcatModel().eval()
    first = trace_model(model, torch.randn(1, 3, 4, 4))
    second = trace_model(model, torch.randn(1, 3, 4, 4))
    assert first.trace_hash == second.trace_hash
    assert [unit.stable_id for unit in first.coupled_channel_units] == [
        unit.stable_id for unit in second.coupled_channel_units
    ]
    path = tmp_path / "trace.json"
    serialize_trace_result(first, path)
    loaded = load_trace_result(path)
    assert loaded.to_dict() == first.to_dict()


def test_trace_residual_concat_offsets_and_module_calls() -> None:
    from tracer.api import trace_model

    result = trace_model(ResidualConcatModel().eval(), torch.randn(1, 3, 4, 4))
    edge_types = {edge.dependency_type for edge in result.dependency_edges}
    assert "residual_add" in edge_types
    assert "concat" in edge_types
    concat_edges = [edge for edge in result.dependency_edges if edge.dependency_type == "concat"]
    assert sorted(edge.channel_offset for edge in concat_edges) == [0, 8]
    assert [row.call_index for row in result.module_call_trace] == list(range(len(result.module_call_trace)))
    assert result.trace_coverage.weighted_module_coverage == 1.0


def test_convtranspose_and_directional_protection_are_distinct() -> None:
    from tracer.api import trace_model

    result = trace_model(DirectionalProtectionModel().eval(), torch.randn(1, 8, 4, 4))
    policies = {row.module_path: row for row in result.protection_policies}
    for name in ("deblocks.0", "fpn_out", "cls_head"):
        assert policies[name].root_pruning_allowed is False
        assert policies[name].input_dependency_pruning_allowed is True
        assert policies[name].fixed_output_contract is True
    assert any(row.module_type == "ConvTranspose2d" for row in result.module_inventory)


def test_unresolved_channel_changing_operation_fails_closed() -> None:
    from tracer.api import trace_model
    from tracer.exceptions import UnsupportedOperationError

    class Unsupported(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.einsum("nchw->nhwc", self.conv(x))

    with pytest.raises(UnsupportedOperationError):
        trace_model(Unsupported().eval(), torch.randn(1, 4, 2, 2))


def test_runtime_trace_fallback_handles_real_tensor_assignment_and_keeps_closure() -> None:
    from tracer.api import trace_model
    from tracer.config import TraceConfig

    class RuntimeOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(4, 8, 1)
            self.bn = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 4, 1)

        def forward(self, value):
            value = value.clone()
            value[:, 0] = value[:, 0] * 1.0
            return self.conv2(self.bn(self.conv1(value)))

    result = trace_model(
        RuntimeOnly().eval(),
        torch.randn(1, 4, 3, 3),
        config=TraceConfig(fail_on_fx_trace_error=False),
    )
    assert result.config["realized_backend"] == "runtime_tensor_flow"
    scope = next(row for row in result.dependency_scopes if row.root_module_path == "conv1")
    assert {(member.module_path, member.axis) for member in scope.members} >= {
        ("conv1", "out"),
        ("bn", "channel"),
        ("conv2", "in"),
    }


def test_normalized_first_order_taylor_exact_formula() -> None:
    from pruning.api import score_pruning_units

    model = nn.Module()
    model.conv1 = nn.Conv2d(1, 2, 1, bias=False)
    model.bn1 = nn.BatchNorm2d(2)
    model.conv2 = nn.Conv2d(2, 1, 1, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]]]))
        model.bn1.weight.copy_(torch.tensor([2.0, 4.0]))
        model.bn1.bias.zero_()
        model.conv2.weight.copy_(torch.tensor([[[[3.0]], [[5.0]]]]))
    model.conv1.weight.grad = torch.tensor([[[[0.5]]], [[[1.5]]]])
    model.bn1.weight.grad = torch.tensor([1.0, 1.0])
    model.bn1.bias.grad = torch.zeros(2)
    model.conv2.weight.grad = torch.tensor([[[[2.0]], [[4.0]]]])

    result = score_pruning_units(model, _toy_units())
    raw = [(0.5 + 2.0 + 6.0) / 3.0, (3.0 + 4.0 + 20.0) / 3.0]
    mean = sum(raw) / 2.0
    assert result.raw_scores == pytest.approx({unit.stable_id: value for unit, value in zip(_toy_units(), raw)})
    assert result.normalized_scores == pytest.approx(
        {unit.stable_id: value / mean for unit, value in zip(_toy_units(), raw)}
    )
    assert result.implementation_version == "first-order-taylor-v1"


def test_taylor_importance_records_nonzero_coupled_parameter_costs() -> None:
    from pruning.api import score_pruning_units
    from tracer.types import CoupledChannelUnit, DependencyMember

    model = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8), nn.Conv2d(8, 4, 1))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    units = [
        CoupledChannelUnit(
            scope_id="scope",
            root_module_path="0",
            root_axis="out",
            root_channel_index=index,
            members=[
                DependencyMember("0", "out", [index], "root_output"),
                DependencyMember("1", "channel", [index], "conv_bn"),
                DependencyMember("2", "in", [index], "downstream_input"),
            ],
        )
        for index in range(8)
    ]
    result = score_pruning_units(model, units)
    assert set(result.unit_parameter_costs) == {unit.stable_id for unit in units}
    assert all(value > 0 for value in result.unit_parameter_costs.values())


def test_formal_model_loading_and_physical_validation_are_path_explicit(tmp_path: Path) -> None:
    from pruning.api import load_model, validate_physical_model

    source = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 4, 1)).eval()
    checkpoint = tmp_path / "state_dict.pth"
    torch.save(source.state_dict(), checkpoint)
    loaded = load_model(
        lambda _config: nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 4, 1)),
        checkpoint_path=checkpoint,
        model_config={"name": "toy"},
        device="cpu",
        training=False,
    )
    assert loaded.model.training is False
    assert loaded.provenance.checkpoint_path == str(checkpoint)
    assert len(loaded.provenance.checkpoint_sha256) == 64
    validation = validate_physical_model(
        loaded.model,
        fixed_output_contracts={"2": 4},
        example_inputs=torch.randn(1, 3, 2, 2),
    )
    assert validation.passed is True
    assert validation.forward_checked is True
    assert validation.output_contract_checked is True


def test_scope_mean_normalization_makes_different_parameter_scales_comparable() -> None:
    from pruning.importance.normalization import normalize_scope_scores

    normalized = normalize_scope_scores({"small": [1.0, 3.0], "large": [100.0, 300.0]})
    assert normalized["small"] == pytest.approx([0.5, 1.5])
    assert normalized["large"] == pytest.approx([0.5, 1.5])


def test_global_selector_is_deterministic_budgeted_and_non_mutating() -> None:
    from pruning.api import select_pruning_request
    from pruning.types import AtomicPruneUnit

    model = nn.Sequential(nn.Conv2d(4, 8, 1))
    state_before = copy.deepcopy(model.state_dict())
    units = [
        AtomicPruneUnit(
            scope_id="scope::0",
            root_module_path="0",
            root_axis="out",
            root_indices=[index],
            source_coupled_unit_ids=[f"unit::{index}"],
            normalized_score=float(index),
            parameter_cost=4,
        )
        for index in range(8)
    ]
    request = select_pruning_request(units, channel_budget=4)
    assert request.one_shot is True
    assert request.selected_atomic_unit_ids == [unit.stable_id for unit in units[:4]]
    assert request.requested_channel_cost == 4
    assert all(torch.equal(state_before[key], model.state_dict()[key]) for key in state_before)


def test_global_selector_enforces_minimum_retained_channels_per_scope() -> None:
    from pruning.api import select_pruning_request
    from pruning.types import AtomicPruneUnit

    units = [
        AtomicPruneUnit(
            scope_id="scope",
            root_module_path="conv",
            root_axis="out",
            root_indices=[index],
            source_coupled_unit_ids=[f"cu::{index}"],
            normalized_score=float(index),
            constraints={"original_channel_count": 8},
        )
        for index in range(8)
    ]
    request = select_pruning_request(units, channel_budget=8)
    assert request.requested_channel_cost == 4
    assert sorted(index for entry in request.entries for index in entry.prune_indices) == [0, 1, 2, 3]


def test_global_selector_never_selects_nonfinite_importance() -> None:
    from pruning.api import select_pruning_request
    from pruning.types import AtomicPruneUnit

    units = [
        AtomicPruneUnit(
            scope_id="scope",
            root_module_path="conv",
            root_axis="out",
            root_indices=[index],
            source_coupled_unit_ids=[f"cu::{index}"],
            normalized_score=0.1 if index == 0 else float("inf"),
            constraints={"original_channel_count": 8},
        )
        for index in range(8)
    ]
    request = select_pruning_request(units, channel_budget=4)
    assert request.requested_channel_cost == 1
    assert [entry.prune_indices for entry in request.entries] == [[0]]


def test_global_selector_bundles_grouped_channels_with_independent_maps() -> None:
    from pruning.api import select_pruning_request
    from pruning.types import AtomicPruneUnit

    units = []
    scores = {
        0: [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0],
        1: [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0],
    }
    for group in range(2):
        for local, score in enumerate(scores[group]):
            absolute = group * 8 + local
            units.append(
                AtomicPruneUnit(
                    scope_id="grouped-scope",
                    root_module_path="grouped",
                    root_axis="out",
                    root_indices=[absolute],
                    source_coupled_unit_ids=[f"cu::{absolute}"],
                    normalized_score=score,
                    channel_cost=1,
                    constraints={"grouped_conv": True, "groups": 2, "channels_per_group": 8},
                )
            )
    request = select_pruning_request(units, channel_budget=8)
    assert len(request.entries) == 1
    assert request.entries[0].group_keep_map == {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]}
    assert request.entries[0].group_prune_map == {0: [1, 3, 5, 7], 1: [0, 2, 4, 6]}
    assert request.entries[0].prune_indices == [1, 3, 5, 7, 8, 10, 12, 14]


def test_trace_atomic_units_connect_to_selector_through_importance_result() -> None:
    from pruning.api import select_pruning_request
    from pruning.types import ImportanceResult
    from tracer.types import AtomicPruneUnit

    units = [
        AtomicPruneUnit(
            scope_id="scope",
            root_module_path="conv",
            root_axis="out",
            root_indices=[index],
            source_coupled_unit_ids=[f"cu::{index}"],
        )
        for index in range(4)
    ]
    importance = ImportanceResult(
        mode="first_order_taylor",
        normalization="coupled_dependency_mean_then_scope_mean_v1",
        aggregation="coupled_unit",
        raw_scores={f"cu::{index}": float(index + 1) for index in range(4)},
        normalized_scores={f"cu::{index}": float(index + 1) / 2.5 for index in range(4)},
    )
    request = select_pruning_request(
        units,
        importance_result=importance,
        channel_budget=2,
    )
    assert [entry.prune_indices for entry in request.entries] == [[0], [1]]
    assert all(
        entry.metadata["importance_normalization"]
        == "coupled_dependency_mean_then_scope_mean_v1"
        for entry in request.entries
    )


def test_trace_atomic_selection_expands_complete_dependency_closure() -> None:
    from pruning.api import select_pruning_request
    from pruning.types import ImportanceResult
    from tracer.api import trace_model
    from tracer.config import TraceConfig

    class RuntimeOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(4, 8, 1)
            self.bn = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 4, 1)

        def forward(self, value):
            value = value.clone()
            value[:, 0] = value[:, 0]
            return self.conv2(self.bn(self.conv1(value)))

    trace = trace_model(
        RuntimeOnly().eval(),
        torch.randn(1, 4, 3, 3),
        config=TraceConfig(fail_on_fx_trace_error=False),
    )
    scope_units = [unit for unit in trace.atomic_prune_units if unit.root_module_path == "conv1"]
    result = ImportanceResult(
        mode="first_order_taylor",
        normalization="coupled_dependency_mean_then_scope_mean_v1",
        aggregation="coupled_unit",
        raw_scores={unit.source_coupled_unit_ids[0]: float(index + 1) for index, unit in enumerate(scope_units)},
        normalized_scores={unit.source_coupled_unit_ids[0]: float(index + 1) for index, unit in enumerate(scope_units)},
    )
    request = select_pruning_request(scope_units, importance_result=result, channel_budget=1)
    assert {(entry.module_path, entry.axis) for entry in request.entries} >= {
        ("conv1", "out"),
        ("bn", "channel"),
        ("conv2", "in"),
    }
    assert len({tuple(entry.prune_indices) for entry in request.entries}) == 1


def test_dense_alignment_repairs_full_scope_but_not_fixed_output() -> None:
    from pruning.materialization.legalizer import legalize_dense_keep_count

    repaired = legalize_dense_keep_count(10, requested_keep=7, alignment=4, fixed_output_contract=False)
    protected = legalize_dense_keep_count(10, requested_keep=7, alignment=4, fixed_output_contract=True)
    assert repaired.final_keep == 4
    assert repaired.repaired is True
    assert protected.final_keep == 10
    assert protected.protection_preserved is True


def test_dense_alignment_repair_is_propagated_to_the_complete_identity_closure() -> None:
    from pruning.api import build_physical_pruning_plan, legalize_pruning_plan
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    model = nn.Sequential(nn.Conv2d(3, 10, 1), nn.BatchNorm2d(10), nn.Conv2d(10, 4, 1))
    request = SamplingPruningRequest(
        entries=[
            SamplingPruningEntry("root", "scope", "0", "out", [0, 1, 2]),
            SamplingPruningEntry("bn", "scope", "1", "channel", [0, 1, 2]),
            SamplingPruningEntry("next", "scope", "2", "in", [0, 1, 2]),
        ]
    )
    legal = legalize_pruning_plan(model, build_physical_pruning_plan(model, request))
    by_axis = {(row.module_path, row.axis): row for row in legal.entries}
    expected = by_axis[("0", "out")].keep_indices
    assert len(expected) == 4
    assert by_axis[("1", "channel")].keep_indices == expected
    assert by_axis[("2", "in")].keep_indices == expected
    assert all(row.repaired for row in by_axis.values())


def test_independent_group_topk_keeps_different_local_positions_and_replays() -> None:
    from pruning.selection.grouped_conv import select_grouped_conv_channels
    from pruning.materialization.replay import replay_group_keep_map

    scores = {
        0: [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0],
        1: [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0],
    }
    decision = select_grouped_conv_channels(scores, final_channels_per_group=4)
    assert decision.group_keep_map == {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]}
    assert decision.group_prune_map == {0: [1, 3, 5, 7], 1: [0, 2, 4, 6]}
    assert replay_group_keep_map(decision.group_keep_map, groups=2, channels_per_group=8) == [0, 2, 4, 6, 9, 11, 13, 15]
    assert decision.final_channels_per_group == 4
    assert decision.legality_report["legal"] is True


@pytest.mark.parametrize("channels", [4, 8, 16, 32, 64, 128, 256, 512])
def test_grouped_allowed_channels_per_group(channels: int) -> None:
    from pruning.policies.grouped_conv import validate_grouped_conv_shape

    report = validate_grouped_conv_shape(
        in_channels=channels * 2,
        out_channels=channels * 2,
        groups=2,
    )
    assert report.legal is True
    assert report.channels_per_group == channels


def test_grouped_invalid_shape_and_missing_keep_map_fail_closed() -> None:
    from pruning.exceptions import GroupedConvLegalityError, MissingGroupKeepMapError
    from pruning.materialization.grouped_conv import validate_grouped_materialization
    from pruning.policies.grouped_conv import validate_grouped_conv_shape

    with pytest.raises(GroupedConvLegalityError):
        validate_grouped_conv_shape(in_channels=10, out_channels=12, groups=4, raise_on_error=True)
    with pytest.raises(MissingGroupKeepMapError):
        validate_grouped_materialization(groups=2, channels_per_group=8, group_keep_map=None)


def test_grouped_materialization_and_replay_use_exact_independent_maps() -> None:
    from pruning.api import build_physical_pruning_plan, legalize_pruning_plan, materialize_pruning, replay_pruning
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    model = nn.Conv2d(16, 16, 1, groups=2, bias=False)
    keep_map = {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]}
    prune_map = {0: [1, 3, 5, 7], 1: [0, 2, 4, 6]}
    absolute_prune = [1, 3, 5, 7, 8, 10, 12, 14]
    request = SamplingPruningRequest(
        entries=[
            SamplingPruningEntry("in", "scope", "", "in", absolute_prune, group_keep_map=keep_map, group_prune_map=prune_map),
            SamplingPruningEntry("out", "scope", "", "out", absolute_prune, group_keep_map=keep_map, group_prune_map=prune_map),
        ]
    )
    plan = legalize_pruning_plan(model, build_physical_pruning_plan(model, request))
    result = materialize_pruning(model, plan)
    replayed = replay_pruning(copy.deepcopy(model), plan)
    assert (result.model.in_channels, result.model.out_channels, result.model.groups) == (8, 8, 2)
    assert tuple(result.model.weight.shape) == (8, 4, 1, 1)
    assert tuple(replayed.model.weight.shape) == tuple(result.model.weight.shape)
    assert result.model(torch.randn(1, 8, 2, 2)).shape == (1, 8, 2, 2)


def test_depthwise_materialization_updates_groups_with_coupled_axes() -> None:
    from pruning.api import build_physical_pruning_plan, legalize_pruning_plan, materialize_pruning
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    model = nn.Conv2d(8, 8, 3, padding=1, groups=8)
    request = SamplingPruningRequest(
        entries=[
            SamplingPruningEntry("in", "scope", "", "in", [1, 3, 5, 7]),
            SamplingPruningEntry("out", "scope", "", "out", [1, 3, 5, 7]),
        ]
    )
    result = materialize_pruning(model, legalize_pruning_plan(model, build_physical_pruning_plan(model, request)))
    assert (result.model.in_channels, result.model.out_channels, result.model.groups) == (4, 4, 4)
    assert tuple(result.model.weight.shape) == (4, 1, 3, 3)
    assert result.model(torch.randn(1, 4, 4, 4)).shape == (1, 4, 4, 4)


def test_shared_local_mean_is_explicit_compatibility_only() -> None:
    from pruning.config import GroupedConvConfig, GroupedConvSelectionPolicy
    from pruning.selection.grouped_conv import select_grouped_conv_channels

    config = GroupedConvConfig(selection_policy=GroupedConvSelectionPolicy.SHARED_LOCAL_MEAN)
    decision = select_grouped_conv_channels(
        {0: [9, 1, 8, 2, 7, 3, 6, 4], 1: [1, 9, 2, 8, 3, 7, 4, 6]},
        final_channels_per_group=4,
        config=config,
    )
    assert decision.selection_policy == "shared_local_mean"
    assert decision.group_keep_map[0] == decision.group_keep_map[1]


def test_one_shot_plan_merges_duplicates_and_freezes_original_indices() -> None:
    from pruning.api import build_physical_pruning_plan
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    request = SamplingPruningRequest(
        entries=[
            SamplingPruningEntry("r1", "scope", "0", "out", [0, 1]),
            SamplingPruningEntry("r2", "scope", "0", "out", [1, 2]),
        ]
    )
    model = nn.Sequential(nn.Conv2d(4, 8, 1))
    plan = build_physical_pruning_plan(model, request)
    assert len(plan.entries) == 1
    assert plan.entries[0].prune_indices == [0, 1, 2]
    assert plan.entries[0].original_axis_size == 8
    assert set(plan.entries[0].source_request_ids) == {"r1", "r2"}
    assert plan.indices_frozen_before_materialization is True


def test_one_shot_materialization_and_ledger_terminal_statuses() -> None:
    from pruning.api import build_physical_pruning_plan, materialize_pruning
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    model = nn.Sequential(nn.Conv2d(4, 8, 1), nn.BatchNorm2d(8), nn.Conv2d(8, 2, 1))
    request = SamplingPruningRequest(
        entries=[
            SamplingPruningEntry("root", "scope", "0", "out", [0, 1, 2, 3]),
            SamplingPruningEntry("bn", "scope", "1", "channel", [0, 1, 2, 3]),
            SamplingPruningEntry("next", "scope", "2", "in", [0, 1, 2, 3]),
        ]
    )
    result = materialize_pruning(model, build_physical_pruning_plan(model, request))
    assert result.model[0].out_channels == result.model[1].num_features == result.model[2].in_channels == 4
    assert {row.status for row in result.ledger.entries} <= {"applied", "repaired", "merged", "skipped"}
    assert {row.request_id for row in result.ledger.entries} == {"root", "bn", "next"}


def test_plan_parameter_predictor_matches_materialized_snapshot() -> None:
    from pruning.api import (
        build_physical_pruning_plan,
        build_physical_structure_snapshot,
        estimate_physical_parameter_count,
        materialize_pruning,
    )
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    model = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8), nn.Conv2d(8, 4, 1))
    request = SamplingPruningRequest(
        entries=[
            SamplingPruningEntry("root", "scope", "0", "out", [0, 1, 2, 3]),
            SamplingPruningEntry("bn", "scope", "1", "channel", [0, 1, 2, 3]),
            SamplingPruningEntry("next", "scope", "2", "in", [0, 1, 2, 3]),
        ]
    )
    plan = build_physical_pruning_plan(model, request)
    predicted = estimate_physical_parameter_count(model, plan)
    result = materialize_pruning(model, plan)
    assert predicted == build_physical_structure_snapshot(result.model).parameter_count


@pytest.mark.parametrize(
    ("module_name", "target_factory", "fixed_output"),
    [
        ("deblock", lambda: nn.ConvTranspose2d(8, 12, 2, stride=2), 12),
        ("fpn_out", lambda: nn.Conv2d(8, 12, 1), 12),
        ("cls_head", lambda: nn.Conv2d(8, 6, 1), 6),
        ("reg_head", lambda: nn.Conv2d(8, 14, 1), 14),
        ("dir_head", lambda: nn.Conv2d(8, 4, 1), 4),
    ],
)
def test_protected_outputs_keep_contract_while_dependency_input_is_pruned(
    module_name: str,
    target_factory,
    fixed_output: int,
) -> None:
    from pruning.api import build_physical_pruning_plan, materialize_pruning
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    model = nn.Module()
    model.source = nn.Conv2d(3, 8, 1)
    setattr(model, module_name, target_factory())
    request = SamplingPruningRequest(
        entries=[
            SamplingPruningEntry("source", "scope", "source", "out", [0, 1, 2, 3]),
            SamplingPruningEntry("consumer", "scope", module_name, "in", [0, 1, 2, 3]),
        ]
    )
    result = materialize_pruning(model, build_physical_pruning_plan(model, request))
    target = getattr(result.model, module_name)
    assert target.in_channels == 4
    assert target.out_channels == fixed_output


def test_physical_snapshot_hash_and_sampling_truth_gate() -> None:
    from pruning.api import build_physical_structure_snapshot, compute_physical_hashes
    from pruning.exceptions import ArtifactSchemaError
    from pruning.validation.structure import require_physical_snapshot_v2

    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    snapshot = build_physical_structure_snapshot(model)
    first = compute_physical_hashes(snapshot)
    second = compute_physical_hashes(snapshot)
    assert first == second
    assert snapshot.snapshot_schema_version == "physical-structure-snapshot-v2"
    with pytest.raises(ArtifactSchemaError, match="sampling"):
        require_physical_snapshot_v2({"before_after_shapes": []})


def _write_weighted_onnx(path: Path, repeated: bool = False) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    initializers = [
        numpy_helper.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), "stem.weight"),
        numpy_helper.from_array(np.zeros((4,), dtype=np.float32), "stem.bias"),
    ]
    nodes = [helper.make_node("Conv", ["input", "stem.weight", "stem.bias"], ["x"], name="/stem/Conv")]
    output = "x"
    if repeated:
        nodes.append(helper.make_node("Conv", ["x", "stem.weight", "stem.bias"], ["y"], name="/stem/Conv_1"))
        output = "y"
    graph = helper.make_graph(
        nodes,
        "formal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, 4, 4, 4])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_origin_mapping_canonical_names_and_repeated_calls(tmp_path: Path) -> None:
    import onnx

    from quantization.api import apply_canonical_node_names, build_onnx_origin_map

    source = tmp_path / "source.onnx"
    named = tmp_path / "named.onnx"
    _write_weighted_onnx(source, repeated=True)
    calls = [
        {"module_path": "stem", "module_type": "Conv2d", "call_index": 0, "weight_initializer": "stem.weight"},
        {"module_path": "stem", "module_type": "Conv2d", "call_index": 1, "weight_initializer": "stem.weight"},
    ]
    mapping = build_onnx_origin_map(source, calls)
    report = apply_canonical_node_names(source, mapping, output_path=named)
    names = [node.name for node in onnx.load(str(named)).graph.node]
    assert names == ["__canonical__stem__Conv__call00000", "__canonical__stem__Conv__call00001"]
    assert report.renamed_node_count == 2
    assert len({entry.canonical_node_name for entry in mapping.entries}) == 2


def test_functional_affine_grid_matmuls_form_one_protected_canonical_compute_group(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.api import (
        apply_fp16_merge_output_contract,
        apply_canonical_node_names,
        build_canonical_precision_mapping,
        build_onnx_origin_map,
        validate_engine_structure,
        validate_precision_realization,
    )
    from quantization.config import TensorRTValidationConfig
    from quantization.types import PrecisionProfileResult

    source = tmp_path / "functional.onnx"
    named = tmp_path / "functional_named.onnx"
    nodes = []
    outputs = []
    initializers = []
    graph_inputs = []
    for index in range(3):
        suffix = "" if index == 0 else f"_{index}"
        left = f"left{suffix}"
        right = f"right{suffix}"
        product = f"product{suffix}"
        grid = f"grid{suffix}"
        feature = f"feature{suffix}"
        output = f"warped{suffix}"
        shape = f"shape{suffix}"
        graph_inputs.extend(
            [
                helper.make_tensor_value_info(left, TensorProto.FLOAT, [1, 4, 3]),
                helper.make_tensor_value_info(right, TensorProto.FLOAT, [1, 3, 2]),
                helper.make_tensor_value_info(feature, TensorProto.FLOAT, [1, 1, 2, 2]),
            ]
        )
        initializers.append(numpy_helper.from_array(np.asarray([1, 2, 2, 2], dtype=np.int64), shape))
        nodes.extend(
            [
                helper.make_node("MatMul", [left, right], [product], name=f"/MatMul{suffix}"),
                helper.make_node("Reshape", [product, shape], [grid], name=f"/Reshape{suffix}"),
                helper.make_node("GridSample", [feature, grid], [output], name=f"/GridSample{suffix}"),
            ]
        )
        outputs.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, 1, 2, 2]))
    graph = helper.make_graph(nodes, "functional-affine-grid", graph_inputs, outputs, initializers)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(source))

    origin = build_onnx_origin_map(source, [])
    assert len(origin.entries) == 0
    assert len(origin.functional_compute_groups) == 1
    group = origin.functional_compute_groups[0]
    assert group.module_path == "pyramid_backbone.functional_affine_grid_matmul"
    assert group.mapping_status == "mapped_but_protected_fp16"
    assert group.weight_initializer == ""
    assert len(group.original_node_names) == 3

    rename = apply_canonical_node_names(source, origin, output_path=named)
    assert rename.renamed_node_count == 3
    member_names = [node.name for node in onnx.load(str(named)).graph.node if node.op_type == "MatMul"]
    assert member_names == [f"{group.canonical_node_name}__member{index:02d}" for index in range(3)]

    profile = PrecisionProfileResult(
        profile_id="protected_functional",
        assignments=[],
        requested_int8_count=0,
        requested_int8_ratio=0.0,
        policy_version="test-protected-functional-v1",
    )
    mapping = build_canonical_precision_mapping(origin, profile)
    assert len(mapping.entries) == 1
    assert mapping.entries[0].requested_precision == "fp16"
    assert mapping.entries[0].protected_precision == "fp16"
    assert mapping.entries[0].constraint_node_names == tuple(member_names)
    inspector = [
        {
            "Name": " + ".join(member_names),
            "LayerType": "gemm",
            "Precision": "FP16",
            "Inputs": [],
            "Outputs": [],
        }
    ]
    assert validate_engine_structure(
        inspector,
        mapping,
        config=TensorRTValidationConfig(require_physical_snapshot_v2=False),
    ).passed is True
    realized = validate_precision_realization(inspector, mapping)
    assert realized.passed is True
    assert realized.realized_fp16_count == 1
    assert realized.unresolved_layer_count == 0


def test_fp16_merge_output_contract_marks_only_nearest_weighted_producers(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.api import (
        apply_fp16_merge_output_contract,
        build_trt_command,
        insert_explicit_qdq,
    )
    from quantization.config import QDQConfig, TensorRTBuildConfig
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    names = ["__canonical__left__Conv__call00000", "__canonical__right__Conv__call00001"]
    weights = [
        numpy_helper.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), "left.weight"),
        numpy_helper.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), "right.weight"),
    ]
    nodes = [
        helper.make_node("Conv", ["left", "left.weight"], ["left_y"], name=names[0]),
        helper.make_node("Relu", ["left_y"], ["left_relu"], name="left_relu"),
        helper.make_node("Conv", ["right", "right.weight"], ["right_y"], name=names[1]),
        helper.make_node("Add", ["left_relu", "right_y"], ["sum"], name="residual_add"),
        helper.make_node("Relu", ["sum"], ["sum_relu"], name="residual_relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "merge",
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, [1, 3, 4, 4]),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, [1, 3, 4, 4]),
        ],
        [helper.make_tensor_value_info("sum_relu", TensorProto.FLOAT16, [1, 4, 4, 4])],
        weights,
    )
    base_onnx = tmp_path / "merge.onnx"
    qdq_onnx = tmp_path / "merge_qdq.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), str(base_onnx))
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                "left",
                names[0],
                "left_group",
                "int8",
                "int8",
                weight_initializer="left.weight",
                onnx_op_type="Conv",
            ),
            CanonicalPrecisionEntry(
                "right",
                names[1],
                "right_group",
                "int8",
                "int8",
                weight_initializer="right.weight",
                onnx_op_type="Conv",
            ),
        ]
    )
    resolved, report = apply_fp16_merge_output_contract(base_onnx, mapping)
    assert {row.module_path: row.realized_output_precision for row in resolved.entries} == {
        "left": "fp16",
        "right": "fp16",
    }
    assert report["resolved_merge_count"] == 1
    assert report["fp16_output_canonical_count"] == 2
    assert report["merges"][0]["merge_op_name"] == "residual_add"
    assert resolved.auxiliary_layer_precisions["residual_add"] == "fp16"
    assert resolved.auxiliary_layer_precisions["residual_relu"] == "fp16"
    assert report["merges"][0]["post_merge_activation_nodes"] == ["residual_relu"]
    assert len(report["merges"][0]["input_cast_nodes"]) == 2

    scales = {
        module: {
            "activation_input_scale": 0.1,
            "weight_scale": [0.01] * 4,
            "weight_axis": 0,
            "activation_output_scale": 0.2,
            "insert_activation_output_qdq": False,
        }
        for module in ("left", "right")
    }
    qdq = insert_explicit_qdq(
        base_onnx,
        qdq_onnx,
        resolved,
        scales=scales,
        config=QDQConfig(),
    )
    qdq_model = onnx.load(str(qdq_onnx))
    nodes = {node.name: node for node in qdq_model.graph.node}
    add = nodes["residual_add"]
    cast_names = report["merges"][0]["input_cast_nodes"]
    assert [nodes[name].op_type for name in cast_names] == ["Cast", "Cast"]
    assert [int(nodes[name].attribute[0].i) for name in cast_names] == [TensorProto.FLOAT16] * 2
    assert list(add.input) == [nodes[name].output[0] for name in cast_names]
    assert all(row["policy"] == "A_fp16_merge" for row in qdq.calibration_metadata["merge_quantization_audit"])
    assert all(
        branch["boundary"] == "explicit_cast_to_fp16" and branch["cast_to_fp16"] is True
        for branch in qdq.calibration_metadata["merge_quantization_audit"][0]["input_branches"]
    )

    command = build_trt_command(
        qdq_onnx,
        tmp_path / "merge.plan",
        resolved,
        config=TensorRTBuildConfig(trtexec_path=Path("/opt/tensorrt/bin/trtexec")),
    )
    joined = " ".join(command.command)
    assert "residual_add:fp16" in joined
    assert "residual_relu:fp16" in joined
    assert all(f"{name}:fp16" in joined for name in cast_names)


def test_origin_mapping_ambiguity_and_functional_matmul_fail_closed(tmp_path: Path) -> None:
    from quantization.api import build_onnx_origin_map
    from quantization.exceptions import CanonicalMappingError

    source = tmp_path / "source.onnx"
    _write_weighted_onnx(source, repeated=True)
    with pytest.raises(CanonicalMappingError):
        build_onnx_origin_map(
            source,
            [{"module_path": "stem", "module_type": "Conv2d", "call_index": 0, "weight_initializer": "stem.weight"}],
        )


def test_precision_profiles_are_deterministic_and_counted() -> None:
    from quantization.api import generate_precision_profile

    modules = [f"backbone.conv{index}" for index in range(10)]
    first = generate_precision_profile(modules, profile_id="profile_002")
    second = generate_precision_profile(list(reversed(modules)), profile_id="profile_002")
    assert first.profile_hash == second.profile_hash
    assert first.requested_int8_count == 5
    assert {row.requested_precision for row in first.assignments} == {"fp16", "int8"}


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    [("strict_fp32", "fp32"), ("strict_fp16", "fp16")],
)
def test_strict_precision_profiles_assign_every_weighted_layer(profile_id: str, expected: str) -> None:
    from quantization.api import generate_precision_profile

    profile = generate_precision_profile(["backbone.conv", "cls_head"], profile_id=profile_id)
    assert {row.requested_precision for row in profile.assignments} == {expected}


def test_strict_int8_profile_keeps_explicit_policy_exemptions() -> None:
    from quantization.api import generate_precision_profile

    profile = generate_precision_profile(["backbone.conv", "cls_head"], profile_id="strict_int8")
    assignments = {row.module_path: row for row in profile.assignments}
    assert assignments["backbone.conv"].requested_precision == "int8"
    assert assignments["cls_head"].requested_precision == "fp16"
    assert assignments["cls_head"].protected_precision == "fp16"


def test_grouped_conv_int8_falls_back_and_canonical_precision_mapping_is_used() -> None:
    from quantization.api import build_canonical_precision_mapping, generate_precision_profile
    from quantization.types import CanonicalMappingEntry, OnnxOriginMapResult

    origin = OnnxOriginMapResult(
        entries=[
            CanonicalMappingEntry(
                module_path="grouped",
                module_type="Conv2d",
                call_index=0,
                onnx_op_type="Conv",
                original_node_name="/grouped/Conv",
                canonical_node_name="__canonical__grouped__Conv__call00000",
                weight_initializer="grouped.weight",
                groups=2,
                channels_per_group=6,
            )
        ]
    )
    profile = generate_precision_profile(["grouped"], profile_id="profile_003")
    mapping = build_canonical_precision_mapping(origin, profile)
    assert mapping.entries[0].requested_precision == "int8"
    assert mapping.entries[0].realized_request_precision == "fp16"
    assert mapping.entries[0].fallback_reason == "grouped_channels_per_group_not_allowed"


def test_qdq_root_trace_and_snapshot_validation(tmp_path: Path) -> None:
    from pruning.api import build_physical_structure_snapshot
    from quantization.api import (
        apply_canonical_node_names,
        build_canonical_precision_mapping,
        build_onnx_origin_map,
        generate_precision_profile,
        insert_explicit_qdq,
        trace_qdq_root_initializer,
        validate_qdq_against_physical_snapshot,
    )

    source = tmp_path / "source.onnx"
    named = tmp_path / "named.onnx"
    qdq = tmp_path / "qdq.onnx"
    _write_weighted_onnx(source)
    origin = build_onnx_origin_map(
        source,
        [{"module_path": "stem", "module_type": "Conv2d", "call_index": 0, "weight_initializer": "stem.weight"}],
    )
    apply_canonical_node_names(source, origin, output_path=named)
    profile = generate_precision_profile(["stem"], profile_id="profile_003")
    precision = build_canonical_precision_mapping(origin, profile)
    insertion = insert_explicit_qdq(named, qdq, precision, scales={"stem": 0.1})
    trace = trace_qdq_root_initializer(qdq, "__canonical__stem__Conv__call00000")
    snapshot = build_physical_structure_snapshot(nn.Sequential())
    # Match the ONNX module name explicitly for this source-grounded toy.
    model = nn.Module()
    model.stem = nn.Conv2d(3, 4, 1)
    snapshot = build_physical_structure_snapshot(model)
    validation = validate_qdq_against_physical_snapshot(qdq, snapshot, origin)
    assert insertion.inserted_layer_count == 1
    assert trace.root_initializer == "stem.weight"
    assert validation.passed is True


def test_qdq_uses_distinct_activation_weight_and_output_scales(tmp_path: Path) -> None:
    from quantization.api import (
        apply_canonical_node_names,
        build_canonical_precision_mapping,
        build_onnx_origin_map,
        generate_precision_profile,
        insert_explicit_qdq,
    )

    source = tmp_path / "source.onnx"
    named = tmp_path / "named.onnx"
    qdq = tmp_path / "qdq.onnx"
    _write_weighted_onnx(source)
    origin = build_onnx_origin_map(
        source,
        [{"module_path": "stem", "module_type": "Conv2d", "call_index": 0, "weight_initializer": "stem.weight"}],
    )
    apply_canonical_node_names(source, origin, output_path=named)
    profile = generate_precision_profile(["stem"], profile_id="strict_int8")
    mapping = build_canonical_precision_mapping(origin, profile)
    insertion = insert_explicit_qdq(
        named,
        qdq,
        mapping,
        scales={
            "stem": {
                "activation_input_scale": 0.1,
                "weight_scale": 0.02,
                "activation_output_scale": 0.3,
            }
        },
    )
    record = insertion.records[0]
    assert record.activation_input_scale == pytest.approx(0.1)
    assert record.weight_scale == pytest.approx(0.02)
    assert record.activation_output_scale == pytest.approx(0.3)
    boundary = insertion.calibration_metadata["weighted_qdq_boundary_audit"][0]
    assert boundary["qdq_placement"] == "weighted_output_before_following_ops"
    assert boundary["q_node_actual_input_tensor"] == record.activation_output_q_inputs
    assert insertion.calibration_metadata["qdq_topology_hash"]


def test_qdq_boundary_report_distinguishes_graph_input_from_engine_fusion(tmp_path: Path) -> None:
    from quantization.reports.qdq_boundary import enrich_weighted_qdq_boundary_audit

    canonical = "__canonical__shrink_conv_0__Conv__call00000"
    report = enrich_weighted_qdq_boundary_audit(
        [
            {
                "canonical_layer": "shrink_conv.layers.0.double_conv.0",
                "weighted_node": canonical,
                "weighted_output_tensor": ["conv_out"],
                "following_ops": [
                    {
                        "name": "/shrink/relu",
                        "op_type": "Relu",
                        "input_tensor": "conv_out",
                        "output_tensors": ["relu_out"],
                        "depth": 0,
                        "next_weighted_node": False,
                    }
                ],
                "q_node_actual_input_tensor": ["conv_out__before_output_qdq"],
                "qdq_placement": "weighted_output_before_following_ops",
                "activation_scale_owner": "conv_out",
                "merge_policy": "fp16_merge",
                "requested_precision": "int8",
                "legalized_precision": "int8",
            }
        ],
        [
            {
                "Name": f"weight + {canonical} + PWN(/shrink/relu)",
                "LayerType": "CaskConvolution",
                "Precision": "Int8",
                "Metadata": f"[ONNX Layer: {canonical}] [ONNX Layer: /shrink/relu]",
            }
        ],
    )

    assert report["passed"] is False
    assert report["layers"][0]["engine_effective_boundary"] == "invalid_pre_relu_qdq_even_when_engine_fused"
    assert report["layers"][0]["boundary_issues"] == ["explicit_qdq_precedes_relu_semantic_boundary"]


def test_qdq_boundary_report_ignores_kgen_with_weighted_canonical_metadata() -> None:
    from quantization.reports.qdq_boundary import enrich_weighted_qdq_boundary_audit

    canonical = "__canonical__pillar_vfe_linear__MatMul__call00000"
    report = enrich_weighted_qdq_boundary_audit(
        [
            {
                "canonical_layer": "encoder_m1.pillar_vfe.pfn_layers.0.linear",
                "weighted_node": canonical,
                "weighted_output_tensor": ["linear_out"],
                "following_ops": [],
                "q_node_actual_input_tensor": ["linear_out__before_output_qdq"],
                "qdq_placement": "weighted_output_before_following_ops",
                "activation_scale_owner": "linear_out",
                "merge_policy": "fp16_merge",
                "requested_precision": "int8",
                "legalized_precision": "int8",
            }
        ],
        [
            {
                "Name": "__myl_MoveConc",
                "LayerType": "kgen",
                "Precision": "Int8",
                "Metadata": f"[ONNX Layer: {canonical}]",
            },
            {
                "Name": f"{canonical}_myl3",
                "LayerType": "gemm",
                "Precision": "Int8",
                "Metadata": f"[ONNX Layer: {canonical}]",
            },
        ],
    )

    assert report["passed"] is True
    assert report["layers"][0]["realized_precision"] == "int8"
    assert report["layers"][0]["engine_fused_layer"] == [f"{canonical}_myl3"]


def test_activation_boundary_resolves_pfn_linear_unique_bn_relu_chain() -> None:
    import onnx
    from onnx import TensorProto, helper

    from quantization.precision.activation_boundary import resolve_activation_output_boundary

    canonical = "__canonical__pillar_vfe_linear__MatMul__call00000"
    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["linear_out"], name=canonical),
        helper.make_node("Transpose", ["linear_out"], ["transposed"], name="transpose_0"),
        helper.make_node(
            "BatchNormalization",
            ["transposed", "scale", "bias", "mean", "variance"],
            ["bn_out"],
            name="bn",
        ),
        helper.make_node("Transpose", ["bn_out"], ["restored"], name="transpose_1"),
        helper.make_node("Relu", ["restored"], ["relu_out"], name="relu"),
        helper.make_node("ReduceMax", ["relu_out"], ["pooled"], name="max"),
    ]
    graph = helper.make_graph(
        nodes,
        "pfn-boundary",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("pooled", TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(graph)

    boundary = resolve_activation_output_boundary(model, canonical)

    assert boundary["boundary_node_name"] == "relu"
    assert boundary["boundary_output_tensor"] == "relu_out"
    assert boundary["resolution"] == "post_relu_semantic_boundary_via_unique_pre_activation_chain"
    assert [row["op_type"] for row in boundary["following_ops"]] == [
        "Transpose",
        "BatchNormalization",
        "Transpose",
        "Relu",
    ]


def test_qdq_boundary_report_rejects_raw_output_before_transitive_relu() -> None:
    from quantization.reports.qdq_boundary import enrich_weighted_qdq_boundary_audit

    canonical = "__canonical__pillar_vfe_linear__MatMul__call00000"
    following = [
        {
            "name": "transpose_0",
            "op_type": "Transpose",
            "input_tensor": "linear_out",
            "output_tensors": ["transposed"],
            "depth": 0,
            "next_weighted_node": False,
        },
        {
            "name": "bn",
            "op_type": "BatchNormalization",
            "input_tensor": "transposed",
            "output_tensors": ["bn_out"],
            "depth": 1,
            "next_weighted_node": False,
        },
        {
            "name": "relu",
            "op_type": "Relu",
            "input_tensor": "bn_out",
            "output_tensors": ["relu_out"],
            "depth": 2,
            "next_weighted_node": False,
        },
    ]
    report = enrich_weighted_qdq_boundary_audit(
        [
            {
                "canonical_layer": "encoder_m1.pillar_vfe.pfn_layers.0.linear",
                "weighted_node": canonical,
                "weighted_output_tensor": ["linear_out"],
                "following_ops": following,
                "q_node_actual_input_tensor": ["linear_out__before_output_qdq"],
                "qdq_placement": "weighted_output_before_following_ops",
                "activation_scale_owner": "linear_out",
                "merge_policy": "fp16_merge",
                "requested_precision": "int8",
                "legalized_precision": "int8",
            }
        ],
        [
            {
                "Name": canonical,
                "LayerType": "gemm",
                "Precision": "Int8",
                "Metadata": f"[ONNX Layer: {canonical}]",
            }
        ],
    )

    assert report["passed"] is False
    assert "explicit_qdq_precedes_relu_semantic_boundary" in report["layers"][0]["boundary_issues"]
    assert report["layers"][0]["engine_effective_boundary"] == "raw_weighted_output_before_relu"


def test_qdq_output_is_inserted_after_relu_semantic_boundary(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.api import insert_explicit_qdq
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    canonical = "__canonical__stem__Conv__call00000"
    weight = numpy_helper.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), "stem.weight")
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "stem.weight"], ["conv_out"], name=canonical),
            helper.make_node("Relu", ["conv_out"], ["relu_out"], name="stem_relu"),
        ],
        "post-relu-qdq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("relu_out", TensorProto.FLOAT, [1, 4, 4, 4])],
        [weight],
    )
    source = tmp_path / "source.onnx"
    destination = tmp_path / "qdq.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), source)
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name=canonical,
                precision_group="pg::stem",
                requested_precision="int8",
                realized_request_precision="int8",
                weight_initializer="stem.weight",
                onnx_op_type="Conv",
            )
        ]
    )
    insertion = insert_explicit_qdq(
        source,
        destination,
        mapping,
        scales={
            "stem": {
                "activation_input_scale": 0.1,
                "weight_scale": [0.01] * 4,
                "weight_axis": 0,
                "activation_output_scale": 0.2,
                "activation_output_tensor": "relu_out",
            }
        },
    )
    model = onnx.load(str(destination))
    nodes = {node.name: node for node in model.graph.node}
    record = insertion.records[0]
    assert nodes["stem_relu"].output[0] == "relu_out__before_output_qdq"
    assert nodes[record.output_quantize_nodes[0]].input[0] == "relu_out__before_output_qdq"
    assert nodes[record.output_dequantize_nodes[0]].output[0] == "relu_out"
    boundary = insertion.calibration_metadata["weighted_qdq_boundary_audit"][0]
    assert boundary["qdq_placement"] == "post_weighted_semantic_boundary"
    assert boundary["activation_scale_owner"] == "relu_out"


def test_qdq_can_leave_concat_branch_output_fp16_without_output_qdq(tmp_path: Path) -> None:
    from dataclasses import replace

    from quantization.api import (
        apply_canonical_node_names,
        build_canonical_precision_mapping,
        build_onnx_origin_map,
        generate_precision_profile,
        insert_explicit_qdq,
    )
    from quantization.types import CanonicalPrecisionMappingResult

    source = tmp_path / "source.onnx"
    named = tmp_path / "named.onnx"
    qdq = tmp_path / "qdq.onnx"
    _write_weighted_onnx(source)
    origin = build_onnx_origin_map(
        source,
        [{"module_path": "stem", "module_type": "Conv2d", "call_index": 0, "weight_initializer": "stem.weight"}],
    )
    apply_canonical_node_names(source, origin, output_path=named)
    raw_mapping = build_canonical_precision_mapping(origin, generate_precision_profile(["stem"], profile_id="strict_int8"))
    mapping = CanonicalPrecisionMappingResult(
        entries=[replace(raw_mapping.entries[0], realized_output_precision="fp16")],
        profile_id=raw_mapping.profile_id,
        profile_hash=raw_mapping.profile_hash,
        origin_map_hash=raw_mapping.origin_map_hash,
    )
    result = insert_explicit_qdq(
        named,
        qdq,
        mapping,
        scales={
            "stem": {
                "activation_input_scale": 0.1,
                "weight_scale": 0.02,
                "activation_output_scale": 0.3,
                "activation_output_tensor": "output",
                "insert_activation_output_qdq": False,
            }
        },
    )

    assert result.records[0].output_quantize_nodes == []
    boundary = result.calibration_metadata["weighted_qdq_boundary_audit"][0]
    assert boundary["qdq_placement"] == "fp16_weighted_output_no_output_qdq"


def test_qdq_fp16_merge_policy_audits_single_quantized_branch(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.api import (
        apply_fp16_merge_output_contract,
        apply_canonical_node_names,
        build_canonical_precision_mapping,
        build_onnx_origin_map,
        insert_explicit_qdq,
    )
    from quantization.types import PrecisionAssignment, PrecisionProfileResult

    source = tmp_path / "residual.onnx"
    named = tmp_path / "residual_named.onnx"
    qdq = tmp_path / "residual_qdq.onnx"
    weight = np.ones((4, 4, 1, 1), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "left.weight"], ["left_out"], name="/left/Conv"),
            helper.make_node("Conv", ["input", "right.weight"], ["right_out"], name="/right/Conv"),
            helper.make_node("Add", ["left_out", "right_out"], ["output"], name="/residual/Add"),
        ],
        "residual",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 2, 2])],
        [
            numpy_helper.from_array(weight, "left.weight"),
            numpy_helper.from_array(weight, "right.weight"),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), str(source))
    origin = build_onnx_origin_map(
        source,
        [
            {"module_path": "left", "module_type": "Conv2d", "call_index": 0, "weight_initializer": "left.weight"},
            {"module_path": "right", "module_type": "Conv2d", "call_index": 1, "weight_initializer": "right.weight"},
        ],
    )
    apply_canonical_node_names(source, origin, output_path=named)
    profile = PrecisionProfileResult(
        profile_id="left_int8",
        assignments=[
            PrecisionAssignment("left", "pg_left", "int8", 0),
            PrecisionAssignment("right", "pg_right", "fp16", 1),
        ],
        requested_int8_count=1,
        requested_int8_ratio=0.5,
        policy_version="test",
    )
    mapping = build_canonical_precision_mapping(origin, profile)
    mapping, contract = apply_fp16_merge_output_contract(named, mapping)
    result = insert_explicit_qdq(
        named,
        qdq,
        mapping,
        scales={
            "left": {
                "activation_input_scale": 0.1,
                "weight_scale": 0.1,
                "activation_output_scale": 0.1,
                "insert_activation_output_qdq": False,
            }
        },
    )
    merge = result.calibration_metadata["merge_quantization_audit"][0]
    assert merge["merge_op_name"] == "/residual/Add"
    assert merge["policy"] == "A_fp16_merge"
    assert contract["resolved_merge_count"] == 1
    assert merge["partial_explicit_qdq_input_count"] == 0
    assert {row["boundary"] for row in merge["input_branches"]} == {"explicit_cast_to_fp16"}
    assert all(row["cast_to_fp16"] is True for row in merge["input_branches"])
    assert result.records[0].output_quantize_nodes == []


def test_qdq_supports_conv_per_channel_weight_scales_with_axis_zero(tmp_path: Path) -> None:
    import onnx
    from onnx import numpy_helper

    from quantization.api import (
        apply_canonical_node_names,
        build_canonical_precision_mapping,
        build_onnx_origin_map,
        generate_precision_profile,
        insert_explicit_qdq,
    )

    source = tmp_path / "source.onnx"
    named = tmp_path / "named.onnx"
    qdq = tmp_path / "qdq.onnx"
    _write_weighted_onnx(source)
    origin = build_onnx_origin_map(
        source,
        [{"module_path": "stem", "module_type": "Conv2d", "call_index": 0, "weight_initializer": "stem.weight"}],
    )
    apply_canonical_node_names(source, origin, output_path=named)
    profile = generate_precision_profile(["stem"], profile_id="strict_int8")
    mapping = build_canonical_precision_mapping(origin, profile)
    result = insert_explicit_qdq(
        named,
        qdq,
        mapping,
        scales={
            "stem": {
                "activation_input_scale": 0.1,
                "weight_scale": [0.01, 0.02, 0.03, 0.04],
                "weight_axis": 0,
                "weight_granularity": "per_channel",
                "activation_output_scale": 0.3,
            }
        },
    )
    graph = onnx.load(str(qdq))
    qnode = next(node for node in graph.graph.node if node.name == result.records[0].weight_quantize_node)
    dqnode = next(node for node in graph.graph.node if node.name == result.records[0].weight_dequantize_node)
    initializers = {row.name: numpy_helper.to_array(row) for row in graph.graph.initializer}
    assert {attr.name: attr.i for attr in qnode.attribute} == {"axis": 0}
    assert {attr.name: attr.i for attr in dqnode.attribute} == {"axis": 0}
    assert initializers[qnode.input[1]].shape == (4,)
    assert initializers[qnode.input[2]].shape == (4,)
    assert result.records[0].weight_granularity == "per_channel"
    assert result.records[0].weight_scale_shape == [4]
    assert result.records[0].weight_axis == 0


def test_qdq_supports_convtranspose_output_channel_axis_one(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.api import (
        apply_canonical_node_names,
        build_canonical_precision_mapping,
        build_onnx_origin_map,
        generate_precision_profile,
        insert_explicit_qdq,
    )

    source = tmp_path / "deconv.onnx"
    named = tmp_path / "deconv_named.onnx"
    qdq = tmp_path / "deconv_qdq.onnx"
    weight = np.arange(3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2) + 1.0
    graph = helper.make_graph(
        [helper.make_node("ConvTranspose", ["input", "up.weight"], ["output"], name="/up/ConvTranspose")],
        "deconv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 5, 5])],
        [numpy_helper.from_array(weight, "up.weight")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), str(source))
    origin = build_onnx_origin_map(
        source,
        [{"module_path": "up", "module_type": "ConvTranspose2d", "call_index": 0, "weight_initializer": "up.weight", "weight_shape": weight.shape}],
    )
    apply_canonical_node_names(source, origin, output_path=named)
    mapping = build_canonical_precision_mapping(origin, generate_precision_profile(["up"], profile_id="strict_int8"))
    result = insert_explicit_qdq(
        named,
        qdq,
        mapping,
        scales={
            "up": {
                "activation_input_scale": 0.1,
                "weight_scale": [0.01, 0.02, 0.03, 0.04],
                "weight_axis": 1,
                "weight_granularity": "per_channel",
                "activation_output_scale": 0.2,
            }
        },
    )
    model = onnx.load(str(qdq))
    qnode = next(node for node in model.graph.node if node.name == result.records[0].weight_quantize_node)
    assert {attr.name: attr.i for attr in qnode.attribute} == {"axis": 1}
    assert result.records[0].weight_axis == 1
    assert result.records[0].weight_scale_shape == [4]


def test_formal_calibration_collects_distinct_absmax_scales() -> None:
    from quantization.api import collect_calibration_scales
    from quantization.config import CalibrationConfig

    model = nn.Module()
    model.stem = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.stem.weight.copy_(torch.tensor([[2.0, -1.0]]))
    batches = [torch.tensor([[1.0, -4.0]]), torch.tensor([[3.0, 2.0]])]
    result = collect_calibration_scales(
        model,
        batches,
        module_paths=["stem"],
        forward_fn=lambda current, batch: current.stem(batch),
        config=CalibrationConfig(frame_count=2),
    )
    record = result.records[0]
    assert record.activation_input_scale == pytest.approx(4.0 / 127.0)
    assert record.weight_scale == pytest.approx(2.0 / 127.0)
    assert record.activation_output_scale == pytest.approx(6.0 / 127.0)
    assert result.frame_count == 2


def test_formal_heal_signal_maxk_input_preparation_uses_fixed_k_and_real_agent_count() -> None:
    from quantization.api import prepare_signal_maxk_inputs
    from quantization.config import OnnxExportConfig

    ego = {
        "inputs_m1": {
            "voxel_features": torch.randn(3, 2, 4),
            "voxel_coords": torch.tensor([[0, 0, 1, 1], [1, 0, 2, 2], [1, 0, 3, 3]], dtype=torch.int32),
            "voxel_num_points": torch.tensor([2, 2, 2], dtype=torch.int32),
        },
        "record_len": torch.tensor([2]),
        "pairwise_t_matrix": torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1),
    }
    prepared = prepare_signal_maxk_inputs(ego, config=OnnxExportConfig(fixed_k=8))
    assert prepared["voxel_features"].shape == (8, 2, 4)
    assert prepared["valid_voxel_mask"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert prepared["pairwise_t_matrix"].shape == (1, 2, 2, 4, 4)


def test_pointpillar_scatter_uses_declared_custom_onnx_domain() -> None:
    from quantization.config import OnnxExportConfig
    from quantization.export.heal_lidar_pyramid import DynamicPointPillarScatterTRT

    calls: list[tuple[str, dict[str, object]]] = []

    class Graph:
        def op(self, name: str, *_args: object, **kwargs: object) -> str:
            calls.append((name, kwargs))
            return name

    result = DynamicPointPillarScatterTRT.symbolic(Graph(), "f", "c", "m", "p", 16, 32)
    policy = OnnxExportConfig()
    assert result == "trt::PointPillarScatterTRT"
    assert calls[0][0] == "trt::PointPillarScatterTRT"
    assert policy.custom_op_domain == "trt"
    assert policy.custom_opset_version == 1


def test_trt_command_generation_uses_canonical_names_without_execution(tmp_path: Path) -> None:
    from quantization.api import build_trt_command
    from quantization.config import TensorRTBuildConfig
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name="__canonical__stem__Conv__call00000",
                precision_group="pg::stem",
                requested_precision="int8",
                realized_request_precision="int8",
            )
        ]
    )
    config = TensorRTBuildConfig(
        trtexec_path=Path("/opt/tensorrt/bin/trtexec"),
        workspace_mib=512,
        shape_profiles={
            "voxel_features": {"min": (1, 32, 4), "opt": (4, 32, 4), "max": (8, 32, 4)}
        },
    )
    result = build_trt_command(tmp_path / "model.onnx", tmp_path / "model.engine", mapping, config=config)
    joined = " ".join(result.command)
    assert "--precisionConstraints=obey" in joined
    assert "--layerPrecisions=__canonical__stem__Conv__call00000:int8" in joined
    assert "--layerOutputTypes=__canonical__stem__Conv__call00000:int8" in joined
    assert "--minShapes=voxel_features:1x32x4" in joined


def test_trt_command_separates_int8_compute_from_fp16_merge_output(tmp_path: Path) -> None:
    from quantization.api import build_trt_command
    from quantization.config import TensorRTBuildConfig
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    name = "__canonical__single_head_0__Conv__call00000"
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="single_head_0",
                canonical_node_name=name,
                precision_group="pg_merge",
                requested_precision="int8",
                realized_request_precision="int8",
                realized_output_precision="fp16",
            )
        ]
    )
    result = build_trt_command(
        tmp_path / "model.onnx",
        tmp_path / "model.engine",
        mapping,
        config=TensorRTBuildConfig(trtexec_path=Path("/opt/tensorrt/bin/trtexec")),
    )
    joined = " ".join(result.command)
    assert f"--layerPrecisions={name}:int8" in joined
    assert f"--layerOutputTypes={name}:fp16" in joined


def test_strongly_typed_trt_command_omits_weak_precision_hints(tmp_path: Path) -> None:
    from quantization.api import build_trt_command
    from quantization.config import TensorRTBuildConfig
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    name = "__canonical__stem__Conv__call00000"
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name=name,
                precision_group="group_stem",
                requested_precision="int8",
                realized_request_precision="int8",
            )
        ]
    )
    result = build_trt_command(
        tmp_path / "model.onnx",
        tmp_path / "model.plan",
        mapping,
        config=TensorRTBuildConfig(
            trtexec_path=Path("/opt/tensorrt/bin/trtexec"),
            strongly_typed=True,
        ),
    )
    command = result.command
    assert "--stronglyTyped" in command
    assert "--fp16" not in command
    assert "--int8" not in command
    assert not any(flag.startswith("--precisionConstraints=") for flag in command)
    assert not any(flag.startswith("--layerPrecisions=") for flag in command)
    assert not any(flag.startswith("--layerOutputTypes=") for flag in command)


def test_qdq_graph_encodes_fp16_parameterized_and_functional_compute(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.api import insert_explicit_qdq
    from quantization.config import QDQConfig
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    conv_name = "__canonical__stem__Conv__call00000"
    matmul_name = "__canonical__functional_affine__MatMulGroup__member00"
    weight = numpy_helper.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), "stem.weight")
    bias = numpy_helper.from_array(np.ones((4,), dtype=np.float32), "stem.bias")
    epsilon = numpy_helper.from_array(np.asarray(1.0e-4, dtype=np.float32), "epsilon")
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "stem.weight", "stem.bias"], ["conv_y"], name=conv_name),
            helper.make_node("Sigmoid", ["conv_y"], ["sigmoid_y"], name="score_sigmoid"),
            helper.make_node("Add", ["sigmoid_y", "epsilon"], ["score"], name="score_epsilon_add"),
            helper.make_node("MatMul", ["left", "right"], ["product"], name=matmul_name),
        ],
        "strong-types",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4]),
            helper.make_tensor_value_info("left", TensorProto.FLOAT, [1, 4, 3]),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, [1, 3, 2]),
        ],
        [
            helper.make_tensor_value_info("score", TensorProto.FLOAT16, [1, 4, 4, 4]),
            helper.make_tensor_value_info("product", TensorProto.FLOAT16, [1, 4, 2]),
        ],
        [weight, bias, epsilon],
    )
    source = tmp_path / "source.onnx"
    output = tmp_path / "typed.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), source)
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name=conv_name,
                precision_group="stem_group",
                requested_precision="fp16",
                realized_request_precision="fp16",
                weight_initializer="stem.weight",
                onnx_op_type="Conv",
            ),
            CanonicalPrecisionEntry(
                module_path="pyramid_backbone.functional_affine_grid_matmul",
                canonical_node_name="__canonical__functional_affine__MatMulGroup",
                precision_group="protected_functional_affine_grid",
                requested_precision="fp16",
                realized_request_precision="fp16",
                realized_output_precision="fp16",
                onnx_op_type="MatMul",
                constraint_node_names=(matmul_name,),
            ),
        ]
    )
    result = insert_explicit_qdq(
        source,
        output,
        mapping,
        scales={},
        config=QDQConfig(),
    )
    typed = onnx.load(output)
    nodes = {node.name: node for node in typed.graph.node}
    conv = nodes[conv_name]
    matmul = nodes[matmul_name]
    assert all(nodes[str(value).removesuffix("__output")].op_type == "Cast" for value in conv.input)
    assert all(nodes[str(value).removesuffix("__output")].op_type == "Cast" for value in matmul.input)
    assert len(result.calibration_metadata["fp16_compute_cast_records"]) == 5
    peer_casts = result.calibration_metadata["strong_type_compatibility_cast_records"]
    assert [(row["consumer_node"], row["source_tensor"]) for row in peer_casts] == [
        ("score_epsilon_add", "epsilon")
    ]
    assert result.calibration_metadata["strong_typing_graph_contract_hash"]
    assert {initializer.name for initializer in typed.graph.initializer} == {"stem.weight", "stem.bias", "epsilon"}


def test_precision_realization_and_provenance_validation() -> None:
    from quantization.api import validate_engine_provenance, validate_precision_realization
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name="__canonical__stem__Conv__call00000",
                precision_group="pg::stem",
                requested_precision="int8",
                realized_request_precision="int8",
            )
        ]
    )
    report = validate_precision_realization(
        [{"Name": "__canonical__stem__Conv__call00000", "LayerType": "Convolution", "Precision": "Int8"}],
        mapping,
    )
    assert report.passed is True
    assert report.realized_int8_count == 1
    provenance = {
        "physical_structure_hash": "a",
        "precision_profile_hash": "b",
        "canonical_mapping_hash": "c",
        "base_onnx_hash": "d",
        "qdq_onnx_hash": "e",
        "engine_hash": "f",
        "plugin_hash": "",
        "tensorrt_version": "10.9",
        "build_policy_version": "trt-fp16-int8-explicit-qdq-v1",
    }
    assert validate_engine_provenance(provenance).passed is True


def test_tensorrt_109_float_format_is_recognized_as_fp32() -> None:
    from quantization.tensorrt.layer_info import precision_name

    row = {
        "Name": "__canonical__stem__Conv__call00000",
        "LayerType": "CaskConvolution",
        "Inputs": [{"Format/Datatype": "Float"}],
        "Outputs": [{"Format/Datatype": "Float"}],
    }
    assert precision_name(row) == "fp32"


def test_engine_checkers_do_not_confuse_adjacent_qdq_metadata_with_compute_identity() -> None:
    from quantization.api import validate_engine_structure, validate_precision_realization
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    canonical = "__canonical__stem__Conv__call00000"
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name=canonical,
                precision_group="pg::stem",
                requested_precision="int8",
                realized_request_precision="int8",
                onnx_op_type="Conv",
            )
        ]
    )
    rows = [
        {
            "Name": "previous_compute",
            "LayerType": "CaskConvolution",
            "Metadata": f"[ONNX Layer: {canonical}__activation_input__QuantizeLinear]",
            "Outputs": [{"Format/Datatype": "Half"}],
        },
        {
            "Name": canonical,
            "LayerType": "CaskConvolution",
            "Metadata": f"[ONNX Layer: {canonical}]",
            "Outputs": [{"Format/Datatype": "Int8"}],
        },
    ]
    snapshot = {
        "snapshot_schema_version": "physical-structure-snapshot-v2",
        "modules": [{"canonical_module_name": "stem", "module_type": "Conv2d", "groups": 1, "weight_shape": [4, 3, 1, 1]}],
    }
    structure = validate_engine_structure(rows, mapping, physical_snapshot=snapshot)
    precision = validate_precision_realization(rows, mapping)
    assert structure.passed is True
    assert structure.matched_canonical_count == 1
    assert precision.passed is True
    assert precision.realized_int8_count == 1


def test_structure_checker_requires_physical_snapshot_v2() -> None:
    from quantization.api import validate_engine_structure
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name="__canonical__stem__Conv__call00000",
                precision_group="pg::stem",
                requested_precision="fp16",
                realized_request_precision="fp16",
            )
        ]
    )
    report = validate_engine_structure(
        [{"Name": "__canonical__stem__Conv__call00000", "LayerType": "Convolution", "Precision": "Half"}],
        mapping,
    )
    assert report.passed is False
    assert any(issue.code == "physical_snapshot_missing" for issue in report.issues)


def test_latency_and_detection_metric_helpers_are_cpu_pure() -> None:
    from quantization.api import compute_detection_metrics, summarize_latency

    latency = summarize_latency([1.0, 2.0, 3.0, 4.0])
    assert latency.count == 4
    assert latency.p50_ms == pytest.approx(2.5)
    metrics = compute_detection_metrics(
        predictions=[{"score": 0.9, "matched": True}, {"score": 0.2, "matched": False}],
        ground_truth_count=1,
    )
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1


def test_formal_source_has_no_tests_dependency_or_server_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for package in ("tracer", "pruning", "quantization")
        for path in (root / package).rglob("*.py")
    )
    forbidden = ("from tests", "import tests", "tests.", "runpy", "/home/lixingfeng/")
    assert not [token for token in forbidden if token in source]


def test_formal_packages_import_in_clean_subprocess() -> None:
    root = Path(__file__).resolve().parents[1]
    code = """
import tracer
import pruning
import quantization
from tracer.api import trace_model
from pruning.api import score_pruning_units, select_pruning_request, build_physical_pruning_plan, materialize_pruning
from quantization.api import export_pruned_signal_maxk_onnx, generate_precision_profile, insert_explicit_qdq, build_trt_command
print('formal package import smoke passed')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "formal package import smoke passed" in completed.stdout
