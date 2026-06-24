"""Tests for tracer module: dependency graph and coupled channel groups."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.tracer.dependency_graph import (
    DependencyGraphBuilder, ChannelRef, DependencyEdge,
)
from heal_compress.tracer.coupled_channel_group import (
    CoupledChannelGroup, CoupledChannelGroupBuilder,
)


class ToyModel(nn.Module):
    """Simple Conv-BN-ReLU-Conv model for testing dependency tracing."""

    def __init__(self, in_ch=3, mid_ch=16, out_ch=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class ResidualModel(nn.Module):
    """Model with a residual connection for testing Add dependencies."""

    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        return out + identity


def _make_trace_graph(model: nn.Module) -> dict:
    """Build a simple static trace graph from a model (no runtime tracing)."""
    nodes = {}
    edges = []
    prev_name = None
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            info = {"type": module.__class__.__name__, "protected": False}
            if hasattr(module, "in_channels"):
                info["in_channels"] = module.in_channels
            if hasattr(module, "out_channels"):
                info["out_channels"] = module.out_channels
            if hasattr(module, "num_features"):
                info["num_features"] = module.num_features
            nodes[name] = info
            if prev_name is not None:
                edges.append({"src": prev_name, "dst": name, "kind": "sequential"})
                prev_type = nodes[prev_name]["type"]
                if prev_type in ("Conv2d",) and info["type"] == "BatchNorm2d":
                    edges.append({"src": prev_name, "dst": name, "kind": "conv_bn"})
            prev_name = name
    return {"nodes": nodes, "edges": edges}


class TestDependencyGraphBuilder:
    """Tests for DependencyGraphBuilder."""

    def test_basic_graph_build(self):
        """Test building a dependency graph from a simple model."""
        model = ToyModel()
        trace = _make_trace_graph(model)
        builder = DependencyGraphBuilder(trace, model)
        builder.build()

        assert "conv1" in builder.nodes
        assert "bn1" in builder.nodes
        assert "conv2" in builder.nodes

    def test_conv_bn_edges(self):
        """Test that Conv->BN edges are detected."""
        model = ToyModel()
        trace = _make_trace_graph(model)
        builder = DependencyGraphBuilder(trace, model)
        builder.build()

        conv_bn_edges = [e for e in builder.edges if e.kind == "conv_bn"]
        assert len(conv_bn_edges) >= 2

    def test_protected_layers(self):
        """Test that protected layers are marked correctly."""
        model = ToyModel()
        trace = _make_trace_graph(model)
        builder = DependencyGraphBuilder(trace, model, protected_layers=["conv1"])
        builder.build()

        assert builder.nodes["conv1"].get("protected") is True

    def test_prunable_modules(self):
        """Test listing prunable modules."""
        model = ToyModel()
        trace = _make_trace_graph(model)
        builder = DependencyGraphBuilder(trace, model, protected_layers=["conv1"])
        builder.build()

        prunable = builder.get_prunable_modules()
        assert "conv1" not in prunable
        assert "conv2" in prunable

    def test_to_dict_roundtrip(self):
        """Test serialization to dict."""
        model = ToyModel()
        trace = _make_trace_graph(model)
        builder = DependencyGraphBuilder(trace, model)
        builder.build()
        d = builder.to_dict()
        assert "nodes" in d
        assert "edges" in d


class TestCoupledChannelGroups:
    """Tests for CoupledChannelGroupBuilder."""

    def test_basic_grouping(self):
        """Test basic group generation from a simple model."""
        model = ToyModel()
        trace = _make_trace_graph(model)
        builder = DependencyGraphBuilder(trace, model)
        builder.build()

        group_builder = CoupledChannelGroupBuilder(builder.to_dict(), model)
        groups = group_builder.build()

        assert len(groups) > 0
        for g in groups:
            assert g.group_id
            assert len(g.channel_indices) > 0

    def test_protected_groups(self):
        """Test that protected layers produce protected groups."""
        model = ToyModel()
        trace = _make_trace_graph(model)
        builder = DependencyGraphBuilder(trace, model, protected_layers=["conv1"])
        builder.build()

        group_builder = CoupledChannelGroupBuilder(builder.to_dict(), model)
        groups = group_builder.build()

        protected_groups = [g for g in groups if g.is_protected]
        assert len(protected_groups) > 0

    def test_group_serialization(self):
        """Test CoupledChannelGroup to_dict/from_dict."""
        group = CoupledChannelGroup(
            group_id="test_group",
            group_type="conv_block",
            source_modules=["conv1"],
            dependent_modules=["conv2"],
            channel_indices=[0, 1, 2, 3],
            is_prunable=True,
        )
        d = group.to_dict()
        restored = CoupledChannelGroup.from_dict(d)
        assert restored.group_id == "test_group"
        assert restored.channel_indices == [0, 1, 2, 3]
