"""Unit tests for the general TP-style pruner (core propagation cases)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.pruning.general_pruner import prune_model
from heal_compress.pruning.group_checker import check_model_legality


class TestGeneralPruner:
    """Test the general pruner on synthetic models covering propagation rules."""

    def test_residual_add(self):
        """Residual Add: c2.out ↔ proj.out ↔ downstream must prune together."""
        class ResNet(nn.Module):
            def __init__(self, c=16):
                super().__init__()
                self.c1 = nn.Conv2d(3, c, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(c)
                self.c2 = nn.Conv2d(c, c, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(c)
                self.proj = nn.Conv2d(3, c, 1)
                self.head = nn.Conv2d(c, 4, 1)

            def forward(self, x):
                identity = self.proj(x)
                out = torch.relu(self.bn1(self.c1(x)))
                out = self.bn2(self.c2(out))
                return self.head(out + identity)

        m = ResNet(c=16).eval()
        x = torch.randn(2, 3, 8, 8)
        ref_out = m(x)

        pruned, report = prune_model(m, x, prune_ratio=0.5, align=8, min_channels=4)

        assert report["legality"]["legal"], f"Model illegal: {report['legality']['issues']}"
        assert report["num_groups_applied"] >= 1, "At least one group should be pruned"

        # The residual group (c2, proj, bn2, head.in) should all sync.
        assert m.c2.out_channels == m.proj.out_channels, "Residual branches misaligned"
        assert m.c2.out_channels == m.bn2.num_features
        assert m.head.in_channels == m.c2.out_channels

        # Forward still works.
        out = m(x)
        assert out.shape == ref_out.shape, f"Output shape changed: {out.shape} vs {ref_out.shape}"

    def test_concat_shrink(self):
        """Cat: branch offsets correctly map to shrink conv input."""
        class CatNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch_a = nn.Conv2d(3, 8, 3, padding=1)
                self.branch_b = nn.Conv2d(3, 8, 3, padding=1)
                self.shrink = nn.Conv2d(16, 16, 1)

            def forward(self, x):
                return self.shrink(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1))

        m = CatNet().eval()
        x = torch.randn(2, 3, 8, 8)

        pruned, report = prune_model(m, x, prune_ratio=0.5, align=4, min_channels=2)

        assert report["legality"]["legal"]
        # Cat group: branch_a + branch_b outputs are sliced with offsets,
        # shrink.in_channels must match sum.
        assert m.shrink.in_channels == m.branch_a.out_channels + m.branch_b.out_channels
        out = m(x)
        assert out.shape[1] > 0, "Output channels zeroed"

    def test_depthwise_separable(self):
        """Depthwise conv: pw1.out ↔ dw.in/out/groups ↔ pw2.in synced."""
        class DepthwiseSep(nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = nn.Conv2d(3, 16, 1)
                self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
                self.bn = nn.BatchNorm2d(16)
                self.pw2 = nn.Conv2d(16, 16, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.bn(self.dw(self.pw1(x)))))

        m = DepthwiseSep().eval()
        x = torch.randn(2, 3, 8, 8)

        pruned, report = prune_model(m, x, prune_ratio=0.5, align=8, min_channels=4)

        assert report["legality"]["legal"]
        # pw1.out drives the whole chain.
        assert m.pw1.out_channels == m.dw.in_channels
        assert m.dw.in_channels == m.dw.out_channels == m.dw.groups
        assert m.bn.num_features == m.dw.out_channels
        assert m.pw2.in_channels == m.dw.out_channels
        out = m(x)
        assert out.shape[1] > 0

    def test_grouped_conv_keep_groups(self):
        """Grouped conv (non-depthwise): producer ↔ grouped conv merged."""
        class GroupedNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.pw1 = nn.Conv2d(3, 32, 1)
                self.grouped = nn.Conv2d(32, 32, 3, padding=1, groups=4)
                self.bn = nn.BatchNorm2d(32)
                self.pw2 = nn.Conv2d(32, 16, 1)

            def forward(self, x):
                return self.pw2(torch.relu(self.bn(self.grouped(self.pw1(x)))))

        m = GroupedNet().eval()
        x = torch.randn(2, 3, 8, 8)

        pruned, report = prune_model(m, x, prune_ratio=0.5, align=8, min_channels=8, grouped_conv_mode="keep_groups")

        assert report["legality"]["legal"]
        # pw1.out == grouped.in == grouped.out (merged group).
        assert m.pw1.out_channels == m.grouped.in_channels == m.grouped.out_channels
        # groups unchanged.
        assert m.grouped.groups == 4
        out = m(x)
        assert out.shape[1] > 0

    def test_convtranspose(self):
        """ConvTranspose2d: proper axis handling (out on axis 1 of weight)."""
        class UpsampleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.up = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
                self.bn = nn.BatchNorm2d(8)
                self.out = nn.Conv2d(8, 4, 1)

            def forward(self, x):
                return self.out(torch.relu(self.bn(self.up(self.conv(x)))))

        m = UpsampleNet().eval()
        x = torch.randn(2, 3, 8, 8)

        pruned, report = prune_model(m, x, prune_ratio=0.5, align=4, min_channels=2)

        assert report["legality"]["legal"]
        # conv.out → up.in, up.out → bn, bn → out.in.
        assert m.up.in_channels == m.conv.out_channels
        assert m.bn.num_features == m.up.out_channels
        assert m.out.in_channels == m.up.out_channels
        out = m(x)
        # Head output is an independent group, so it may be pruned separately.
        # Just check dimensions are consistent and model is legal.
        assert out.shape[2:] == (16, 16), f"Spatial dims wrong: {out.shape}"
        assert out.shape[1] == m.out.out_channels

    def test_protected_layer(self):
        """Protected layers should not be pruned."""
        class Simple(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3, 16, 3, padding=1)
                self.c2 = nn.Conv2d(16, 8, 3, padding=1)

            def forward(self, x):
                return self.c2(self.c1(x))

        m = Simple().eval()
        x = torch.randn(2, 3, 8, 8)

        pruned, report = prune_model(
            m, x, prune_ratio=0.5, protected_layers=["c1"], align=4, min_channels=2
        )

        assert report["legality"]["legal"]
        assert m.c1.out_channels == 16, "Protected layer c1 was pruned"
        # c2 can still be pruned (its output is independent).
        assert m.c2.out_channels < 8 or report["num_groups_applied"] == 0

    def test_atomic_skip_unsupported_op(self):
        """A group with a protected grouped conv should be skipped."""
        # Use a grouped conv that will be classified as "protected" due to
        # alignment constraints.
        class ModelWithProtectedGrouped(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3, 18, 3, padding=1)
                # groups=3, not 8/16-aligned, per-group width not regular.
                self.gc = nn.Conv2d(18, 18, 3, padding=1, groups=3)
                self.c2 = nn.Conv2d(18, 8, 3, padding=1)

            def forward(self, x):
                return self.c2(self.gc(self.c1(x)))

        m = ModelWithProtectedGrouped().eval()
        x = torch.randn(2, 3, 8, 8)

        # The grouped conv should be classified as protected (not 8-aligned).
        pruned, report = prune_model(m, x, prune_ratio=0.5, align=8, min_channels=4)

        # The group with c1→gc merged should be protected.
        assert report["num_groups_skipped"] >= 1, "Protected grouped conv group should be skipped"
        # c1 remains unpruned since it's merged with gc.
        assert m.c1.out_channels == 18, "c1 should remain unpruned (merged with protected gc)"
