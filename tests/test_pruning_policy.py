import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from heal_compress.pruning.general_pruner import _group_aligned_keep_indices
from heal_compress.pruning.group_checker import check_pruning_group
from heal_compress.pruning.grouped_conv import grouped_conv_pruning_fn
from heal_compress.pruning.pruning_fns import prune_conv_in, prune_conv_out
from heal_compress.pruning.propagation import GroupBuilder
from heal_compress.search.importance import compute_layer_channel_importance
from heal_compress.tracer.generic_tracer import trace_model
from heal_compress.tracer.op_graph import build_op_graph
from heal_compress.tracer.pruning_group import PruningGroup

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_general_pruner import (
    build_protected_layers,
    parse_args,
    select_keep,
)


def _plain_conv_group() -> tuple[PruningGroup, nn.Conv2d]:
    conv = nn.Conv2d(3, 8, 1, bias=False)
    group = PruningGroup(group_id="group::conv", num_channels=8, meta={"group_type": "plain"})
    group.add_dep("conv", conv, prune_conv_out, "out", reason="root_out")
    return group, conv


def test_group_keep_indices_use_per_channel_importance():
    group, _conv = _plain_conv_group()
    importance = {"conv": torch.tensor([1.0, 2.0, 30.0, 4.0, 50.0, 6.0, 70.0, 8.0])}

    keep = _group_aligned_keep_indices(
        group,
        prune_ratio=0.5,
        align=1,
        min_channels=1,
        importance_scores=importance,
    )

    assert keep == [2, 4, 6, 7]


def test_select_keep_passes_layer_importance_scores():
    group, _conv = _plain_conv_group()
    args = argparse.Namespace(prune_ratio=0.5, align=1, ffn_align=8, group_conv_align=8)
    importance = {"conv": torch.tensor([1.0, 2.0, 30.0, 4.0, 50.0, 6.0, 70.0, 8.0])}

    keep = select_keep(group, args, importance_scores=importance)

    assert keep == [2, 4, 6, 7]


def test_compute_layer_channel_importance_returns_out_axis_scores():
    conv = nn.Conv2d(3, 4, 1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0].fill_(1.0)
        conv.weight[1].fill_(2.0)
        conv.weight[2].fill_(3.0)
        conv.weight[3].fill_(4.0)
    group = PruningGroup(group_id="group::conv", num_channels=4, meta={"group_type": "plain"})
    group.add_dep("conv", conv, prune_conv_out, "out", reason="root_out")

    scores = compute_layer_channel_importance([group], method="l1_norm")

    assert torch.equal(scores["conv"], torch.tensor([3.0, 6.0, 9.0, 12.0]))


def test_remove_groups_aligns_conv_group_count_not_channel_count_to_original_groups():
    conv_groups = 24
    channels_per_group = 8
    channels = conv_groups * channels_per_group
    grouped = nn.Conv2d(channels, channels, 3, padding=1, groups=conv_groups, bias=False)
    group = PruningGroup(
        group_id="group::grouped",
        num_channels=channels,
        meta={"group_type": "plain"},
    )
    group.add_dep(
        "grouped",
        grouped,
        grouped_conv_pruning_fn("remove_groups"),
        "out",
        reason="grouped_conv:remove_groups",
    )

    keep = _group_aligned_keep_indices(
        group,
        prune_ratio=0.30,
        align=16,
        min_channels=8,
        group_conv_align=8,
    )
    kept_conv_groups = len({idx // channels_per_group for idx in keep})

    assert kept_conv_groups == 16
    assert kept_conv_groups % 8 == 0
    assert channels_per_group % 8 == 0


def test_remove_groups_skips_when_channels_per_conv_group_are_not_align8():
    conv_groups = 32
    channels_per_group = 4
    channels = conv_groups * channels_per_group
    grouped = nn.Conv2d(channels, channels, 3, padding=1, groups=conv_groups, bias=False)
    group = PruningGroup(
        group_id="group::grouped",
        num_channels=channels,
        meta={"group_type": "plain"},
    )
    group.add_dep(
        "grouped",
        grouped,
        grouped_conv_pruning_fn("remove_groups"),
        "out",
        reason="grouped_conv:remove_groups",
    )

    keep = _group_aligned_keep_indices(
        group,
        prune_ratio=0.25,
        align=16,
        min_channels=8,
        group_conv_align=8,
    )

    assert keep == list(range(channels))


def test_default_parse_args_protects_neck_and_detection_head_inputs():
    args = parse_args([])

    assert args.protect_neck_and_heads is True
    assert "pyramid_backbone.deblocks" in args.extra_protected_prefix
    assert "shrink_conv" in args.extra_protected_prefix
    assert "cls_head" in args.extra_protected_prefix


def test_build_protected_layers_matches_prefixes():
    model = nn.Module()
    model.pyramid_backbone = nn.Module()
    model.pyramid_backbone.deblocks = nn.ModuleList([nn.ConvTranspose2d(8, 8, 1)])
    model.shrink_conv = nn.Conv2d(8, 8, 1)
    model.cls_head = nn.Conv2d(8, 2, 1)
    model.safe_conv = nn.Conv2d(8, 8, 1)

    protected = build_protected_layers(
        model,
        adapter_protected=[],
        extra_prefixes=["pyramid_backbone.deblocks", "shrink_conv", "cls_head"],
    )

    assert "pyramid_backbone.deblocks.0" in protected
    assert "shrink_conv" in protected
    assert "cls_head" in protected
    assert "safe_conv" not in protected


def test_residual_add_group_is_prunable_and_synchronizes_branches():
    class ResidualNet(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.proj = nn.Conv2d(3, channels, 1)
            self.head = nn.Conv2d(channels, 4, 1)

        def forward(self, x):
            identity = self.proj(x)
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.head(out + identity)

    model = ResidualNet().eval()
    sample = torch.randn(2, 3, 8, 8)
    groups = GroupBuilder(build_op_graph(trace_model(model, sample), model), align=4).build()
    add_groups = [g for g in groups if g.meta.get("group_type") == "add"]
    assert len(add_groups) == 1
    group = add_groups[0]

    assert not group.protected
    assert {(item.name, item.direction) for item in group.items} >= {
        ("conv2", "out"),
        ("bn2", "out"),
        ("proj", "out"),
        ("head", "in"),
    }

    keep = list(range(8))
    check = check_pruning_group(group, keep, group_conv_align=4)
    assert check["legal"], check["issues"]
    result = group.prune(keep)

    assert result["applied"]
    assert model.conv2.out_channels == 8
    assert model.bn2.num_features == 8
    assert model.proj.out_channels == 8
    assert model.head.in_channels == 8
    assert model(sample).shape == (2, 4, 8, 8)


def test_residual_add_can_be_explicitly_protected_for_ablation():
    class ResidualNet(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.conv = nn.Conv2d(3, channels, 1)
            self.proj = nn.Conv2d(3, channels, 1)
            self.head = nn.Conv2d(channels, 4, 1)

        def forward(self, x):
            return self.head(self.conv(x) + self.proj(x))

    model = ResidualNet().eval()
    sample = torch.randn(2, 3, 8, 8)
    groups = GroupBuilder(
        build_op_graph(trace_model(model, sample), model),
        align=4,
        protect_residual_add=True,
    ).build()
    add_groups = [g for g in groups if g.meta.get("group_type") == "add"]

    assert len(add_groups) == 1
    assert add_groups[0].protected
    assert add_groups[0].protected_reason == "residual_add_output_protected"


def test_chained_residual_adds_share_one_prunable_output_group():
    class Block(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return out + x

    class ResidualStage(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.proj = nn.Conv2d(3, channels, 1)
            self.block0 = Block(channels)
            self.block1 = Block(channels)
            self.head = nn.Conv2d(channels, 4, 1)

        def forward(self, x):
            x = self.proj(x)
            x = self.block0(x)
            x = self.block1(x)
            return self.head(x)

    model = ResidualStage().eval()
    sample = torch.randn(2, 3, 8, 8)
    groups = GroupBuilder(build_op_graph(trace_model(model, sample), model), align=4).build()
    add_groups = [g for g in groups if g.meta.get("group_type") == "add"]

    assert len(add_groups) == 1
    group = add_groups[0]
    assert not group.protected
    assert {"proj", "block0.conv2", "block1.conv2"}.issubset(set(group.meta["roots"]))

    keep = list(range(8))
    check = check_pruning_group(group, keep, group_conv_align=4)
    assert check["legal"], check["issues"]
    group.prune(keep)

    assert model.proj.out_channels == 8
    assert model.block0.conv1.in_channels == 8
    assert model.block0.conv2.out_channels == 8
    assert model.block1.conv1.in_channels == 8
    assert model.block1.conv2.out_channels == 8
    assert model.head.in_channels == 8
    assert model(sample).shape == (2, 4, 8, 8)


def test_residual_output_through_identity_and_stack_reaches_downstream_consumer():
    class ResidualStackNet(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.branch = nn.Conv2d(3, channels, 1)
            self.residual = nn.Conv2d(3, channels, 1)
            self.identity = nn.Identity()
            self.consumer = nn.Conv2d(channels, 4, 1)

        def forward(self, x):
            residual_out = self.branch(x) + self.residual(x)
            aligned = self.identity(residual_out)
            return self.consumer(torch.stack([aligned, aligned], dim=0).sum(dim=0))

    model = ResidualStackNet().eval()
    sample = torch.randn(2, 3, 8, 8)
    groups = GroupBuilder(build_op_graph(trace_model(model, sample), model), align=4).build()
    add_groups = [g for g in groups if g.meta.get("group_type") == "add"]
    assert len(add_groups) == 1
    group = add_groups[0]

    assert not group.protected
    assert ("consumer", "in") in {(item.name, item.direction) for item in group.items}

    keep = list(range(8))
    check = check_pruning_group(group, keep, group_conv_align=4)
    assert check["legal"], check["issues"]
    group.prune(keep)

    assert model.branch.out_channels == 8
    assert model.residual.out_channels == 8
    assert model.consumer.in_channels == 8
    assert model(sample).shape == (2, 4, 8, 8)


def test_residual_output_through_weighted_fuse_reaches_deblock_input():
    class WeightedFuseNet(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.branch = nn.Conv2d(3, channels, 1)
            self.residual = nn.Conv2d(3, channels, 1)
            self.score = nn.Conv2d(channels, 1, 1)
            self.deblock = nn.ConvTranspose2d(channels, 4, 1)

        def forward(self, x):
            feature = self.branch(x) + self.residual(x)
            score = torch.sigmoid(self.score(feature))
            grid = torch.zeros(
                x.shape[0],
                x.shape[2],
                x.shape[3],
                2,
                dtype=x.dtype,
                device=x.device,
            )
            warped_feature = torch.nn.functional.grid_sample(
                feature,
                grid,
                align_corners=False,
            )
            warped_score = torch.nn.functional.grid_sample(
                score,
                grid,
                align_corners=False,
            )
            fused = torch.sum(warped_feature * warped_score, dim=0)
            return self.deblock(torch.stack([fused], dim=0))

    model = WeightedFuseNet().eval()
    sample = torch.randn(2, 3, 8, 8)
    groups = GroupBuilder(build_op_graph(trace_model(model, sample), model), align=4).build()
    add_groups = [g for g in groups if g.meta.get("group_type") == "add"]
    assert len(add_groups) == 1
    group = add_groups[0]

    assert not group.protected
    assert ("deblock", "in") in {(item.name, item.direction) for item in group.items}

    keep = list(range(8))
    check = check_pruning_group(group, keep, group_conv_align=4)
    assert check["legal"], check["issues"]
    group.prune(keep)

    assert model.branch.out_channels == 8
    assert model.residual.out_channels == 8
    assert model.score.in_channels == 8
    assert model.deblock.in_channels == 8
    assert model(sample).shape == (1, 4, 8, 8)


def test_residual_output_through_tensor_split_weighted_fuse_reaches_deblock_input():
    class SplitWeightedFuseNet(nn.Module):
        def __init__(self, channels: int = 16):
            super().__init__()
            self.branch = nn.Conv2d(3, channels, 1)
            self.residual = nn.Conv2d(3, channels, 1)
            self.score = nn.Conv2d(channels, 1, 1)
            self.deblock = nn.ConvTranspose2d(channels, 4, 1)

        def forward(self, x):
            feature = self.branch(x) + self.residual(x)
            score = torch.sigmoid(self.score(feature))
            split_feature = torch.tensor_split(feature, [1], dim=0)[0]
            split_score = torch.tensor_split(score, [1], dim=0)[0]
            grid = torch.zeros(
                split_feature.shape[0],
                split_feature.shape[2],
                split_feature.shape[3],
                2,
                dtype=x.dtype,
                device=x.device,
            )
            warped_feature = torch.nn.functional.grid_sample(
                split_feature,
                grid,
                align_corners=False,
            )
            warped_score = torch.nn.functional.grid_sample(
                split_score,
                grid,
                align_corners=False,
            )
            fused = torch.sum(warped_feature * warped_score, dim=0)
            return self.deblock(torch.stack([fused], dim=0))

    model = SplitWeightedFuseNet().eval()
    sample = torch.randn(2, 3, 8, 8)
    groups = GroupBuilder(build_op_graph(trace_model(model, sample), model), align=4).build()
    add_groups = [g for g in groups if g.meta.get("group_type") == "add"]
    assert len(add_groups) == 1
    group = add_groups[0]

    assert not group.protected
    assert ("deblock", "in") in {(item.name, item.direction) for item in group.items}

    keep = list(range(8))
    check = check_pruning_group(group, keep, group_conv_align=4)
    assert check["legal"], check["issues"]
    group.prune(keep)

    assert model.branch.out_channels == 8
    assert model.residual.out_channels == 8
    assert model.score.in_channels == 8
    assert model.deblock.in_channels == 8
    assert model(sample).shape == (1, 4, 8, 8)


def test_multiscale_score_add_does_not_merge_feature_scales():
    class MultiScaleWeightedFuseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage0_branch = nn.Conv2d(3, 16, 1)
            self.stage0_residual = nn.Conv2d(3, 16, 1)
            self.stage0_score = nn.Conv2d(16, 1, 1)
            self.stage1_branch = nn.Conv2d(16, 32, 1, stride=2)
            self.stage1_residual = nn.Conv2d(16, 32, 1, stride=2)
            self.stage1_score = nn.Conv2d(32, 1, 1)
            self.deblock0 = nn.ConvTranspose2d(16, 4, 1)
            self.deblock1 = nn.ConvTranspose2d(32, 4, 1)

        def _fuse(self, feature, score):
            split_feature = torch.tensor_split(feature, [1], dim=0)[0]
            split_score = torch.tensor_split(score + 1e-4, [1], dim=0)[0]
            grid = torch.zeros(
                split_feature.shape[0],
                split_feature.shape[2],
                split_feature.shape[3],
                2,
                dtype=feature.dtype,
                device=feature.device,
            )
            warped_feature = torch.nn.functional.grid_sample(
                split_feature,
                grid,
                align_corners=False,
            )
            warped_score = torch.nn.functional.grid_sample(
                split_score,
                grid,
                align_corners=False,
            )
            fused = torch.sum(warped_feature * warped_score, dim=0)
            return torch.stack([fused], dim=0)

        def forward(self, x):
            stage0 = self.stage0_branch(x) + self.stage0_residual(x)
            stage0_score = torch.sigmoid(self.stage0_score(stage0))
            stage1 = self.stage1_branch(stage0) + self.stage1_residual(stage0)
            stage1_score = torch.sigmoid(self.stage1_score(stage1))
            return self.deblock0(self._fuse(stage0, stage0_score)), self.deblock1(self._fuse(stage1, stage1_score))

    model = MultiScaleWeightedFuseNet().eval()
    sample = torch.randn(2, 3, 8, 8)
    groups = GroupBuilder(build_op_graph(trace_model(model, sample), model), align=4).build()

    stage0_groups = [
        g for g in groups
        if g.meta.get("group_type") == "add" and "stage0_branch" in set(g.meta.get("roots", []))
    ]
    stage1_groups = [
        g for g in groups
        if g.meta.get("group_type") == "add" and "stage1_branch" in set(g.meta.get("roots", []))
    ]

    assert len(stage0_groups) == 1
    assert len(stage1_groups) == 1
    stage0_group = stage0_groups[0]
    stage1_group = stage1_groups[0]

    assert "stage1_branch" not in set(stage0_group.meta["roots"])
    assert "stage1_residual" not in set(stage0_group.meta["roots"])
    assert ("deblock0", "in") in {(item.name, item.direction) for item in stage0_group.items}
    assert ("deblock1", "in") not in {(item.name, item.direction) for item in stage0_group.items}
    assert ("deblock1", "in") in {(item.name, item.direction) for item in stage1_group.items}


def test_score_bias_add_does_not_inherit_stale_tensor_producer():
    class ScoreBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Conv2d(3, 16, 1)
            self.score = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            feature = self.feature(x)
            score = torch.sigmoid(self.score(feature))
            del feature
            return score + 1e-4

    model = ScoreBiasNet().eval()
    sample = torch.randn(2, 3, 8, 8)
    graph = build_op_graph(trace_model(model, sample), model)
    score_adds = [
        node for node in graph.nodes.values()
        if node.op_type == "Add" and node.output_shapes == [[2, 1, 8, 8]]
    ]

    assert len(score_adds) == 1
    add_node = score_adds[0]
    add_inputs = [src for src, _idx in graph.incoming(add_node.name)]

    assert "feature" not in add_inputs
    assert any(graph.nodes[src].raw_type == "torch.sigmoid" for src in add_inputs)
