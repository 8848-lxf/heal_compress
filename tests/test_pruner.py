"""Tests for pruning module: physical pruner, alignment, and legality."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.pruning.physical_pruner import PhysicalPruner
from heal_compress.pruning.alignment_checker import ChannelAlignmentChecker
from heal_compress.pruning.legality_checker import StructureLegalityChecker
from heal_compress.quantization.pseudo_quant import pseudo_quantize_weight


class SimpleModel(nn.Module):
    """Test model with Conv-BN-Conv chain."""

    def __init__(self, in_ch=3, mid_ch=32, out_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))


class TestPhysicalPruner:
    """Tests for PhysicalPruner channel slicing operations."""

    def test_slice_output_channels(self):
        """Test slicing Conv2d output channels."""
        model = SimpleModel()
        pruner = PhysicalPruner(model)

        keep = [0, 2, 4, 6, 8, 10, 12, 14]
        op = pruner._slice_output_channels("conv1", model.conv1, keep)

        assert op is not None
        assert model.conv1.out_channels == 8
        assert model.conv1.weight.shape[0] == 8

    def test_slice_input_channels(self):
        """Test slicing Conv2d input channels."""
        model = SimpleModel()
        pruner = PhysicalPruner(model)

        keep = [0, 2, 4, 6, 8, 10, 12, 14]
        pruner._slice_output_channels("conv1", model.conv1, keep)
        op = pruner._slice_input_channels("conv2", model.conv2, keep)

        assert op is not None
        assert model.conv2.in_channels == 8
        assert model.conv2.weight.shape[1] == 8

    def test_slice_batchnorm(self):
        """Test slicing BatchNorm parameters."""
        model = SimpleModel()
        pruner = PhysicalPruner(model)

        keep = [0, 2, 4, 6, 8, 10, 12, 14]
        op = pruner._slice_batchnorm("bn1", model.bn1, keep)

        assert op is not None
        assert model.bn1.num_features == 8
        assert model.bn1.weight.shape[0] == 8
        assert model.bn1.running_mean.shape[0] == 8


class TestAlignmentChecker:
    """Tests for ChannelAlignmentChecker."""

    def test_aligned_model(self):
        """Test a model with aligned channels passes."""
        model = SimpleModel(in_ch=3, mid_ch=32, out_ch=16)
        checker = ChannelAlignmentChecker(default_align=8)
        report = checker.check(model)
        assert report["all_aligned"]

    def test_misaligned_model(self):
        """Test a model with misaligned channels fails."""
        model = SimpleModel(in_ch=3, mid_ch=17, out_ch=13)
        checker = ChannelAlignmentChecker(default_align=8)
        report = checker.check(model)
        assert not report["all_aligned"]

    def test_align_value(self):
        """Test value alignment utility."""
        assert ChannelAlignmentChecker.align_value(30, 8) == 24
        assert ChannelAlignmentChecker.align_value(3, 8) == 8
        assert ChannelAlignmentChecker.align_value(16, 8) == 16


class TestLegalityChecker:
    """Tests for StructureLegalityChecker."""

    def test_valid_model(self):
        """Test that a valid model passes legality check."""
        model = SimpleModel()
        checker = StructureLegalityChecker(model)
        report = checker.check("/tmp/test_legality")
        assert report["legal"]

    def test_invalid_channels(self):
        """Test detection of invalid channel counts."""
        model = SimpleModel()
        model.conv1.in_channels = 0  # Force invalid
        checker = StructureLegalityChecker(model)
        report = checker.check("/tmp/test_legality")
        assert not report["legal"]


class TestPseudoQuant:
    """Tests for pseudo-quantization."""

    def test_fp16_roundtrip(self):
        """Test FP16 pseudo-quantization preserves approximate values."""
        w = torch.randn(16, 8, 3, 3)
        qw = pseudo_quantize_weight(w, "FP16")
        assert qw.shape == w.shape
        assert torch.allclose(w, qw, atol=1e-3)

    def test_int8_quantization(self):
        """Test INT8 pseudo-quantization changes values."""
        w = torch.randn(16, 8, 3, 3)
        qw = pseudo_quantize_weight(w, "INT8")
        assert qw.shape == w.shape
        # Quantized values should differ from original
        assert not torch.equal(w, qw)

    def test_int4_quantization(self):
        """Test INT4 pseudo-quantization produces coarser values."""
        w = torch.randn(16, 8, 3, 3)
        qw4 = pseudo_quantize_weight(w, "INT4")
        qw8 = pseudo_quantize_weight(w, "INT8")
        # INT4 should have larger quantization error than INT8
        err4 = (w - qw4).abs().mean()
        err8 = (w - qw8).abs().mean()
        assert err4 > err8

    def test_zero_weight(self):
        """Test quantization of zero weights."""
        w = torch.zeros(8, 4, 1, 1)
        qw = pseudo_quantize_weight(w, "INT8")
        assert torch.equal(qw, w)
