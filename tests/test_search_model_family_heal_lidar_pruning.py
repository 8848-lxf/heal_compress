from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BaseBEVBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            self._block(64, 64, count=4, stride=2),
            self._block(64, 128, count=6, stride=2),
            self._block(128, 256, count=9, stride=2),
        ])
        self.deblocks = nn.ModuleList([
            self._deblock(64, stride=1),
            self._deblock(128, stride=2),
            self._deblock(256, stride=4),
        ])

    @staticmethod
    def _block(in_channels: int, out_channels: int, *, count: int, stride: int) -> nn.Sequential:
        rows: list[nn.Module] = [
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        for _ in range(count - 1):
            rows.extend((
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
        return nn.Sequential(*rows)

    @staticmethod
    def _deblock(in_channels: int, *, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, stride, stride=stride, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = data["spatial_features"]
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return {"spatial_features_2d": torch.cat(ups, dim=1)}


class DoubleConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.double_conv(value)


class DownsampleConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DoubleConv()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class MaxFusion(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.amax(dim=0, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.sqrt_dim = math.sqrt(float(width))


class AttFusion(nn.Module):
    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.att = ScaledDotProductAttention(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:1]


class PixelWeightLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(512, 128, 1)
        self.bn1_1 = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 32, 1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 8, 1)
        self.bn1_3 = nn.BatchNorm2d(8)
        self.conv1_4 = nn.Conv2d(8, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.bn1_1(self.conv1_1(value)))
        value = torch.relu(self.bn1_2(self.conv1_2(value)))
        value = torch.relu(self.bn1_3(self.conv1_3(value)))
        return self.conv1_4(value)


class DiscoFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixel_weight_layer = PixelWeightLayer()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        agents = value.shape[0]
        ego = value[:1].expand(agents, -1, -1, -1)
        weights = torch.softmax(self.pixel_weight_layer(torch.cat((value, ego), dim=1)), dim=0)
        return torch.sum(weights * value, dim=0, keepdim=True)


class HeterModelBaseline(nn.Module):
    def __init__(self, fusion: nn.Module) -> None:
        super().__init__()
        self.backbone_m1 = BaseBEVBackbone()
        self.shrinker_m1 = DownsampleConv()
        self.fusion_net = fusion
        self.cls_head = nn.Conv2d(256, 2, 1)
        self.reg_head = nn.Conv2d(256, 14, 1)
        self.dir_head = nn.Conv2d(256, 4, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        value = self.backbone_m1({"spatial_features": value})["spatial_features_2d"]
        value = self.shrinker_m1(value)
        value = self.fusion_net(value)
        return self.cls_head(value), self.reg_head(value), self.dir_head(value)


def _feature_units(model: nn.Module, family_id: str):
    from search.model_family import build_heal_lidar_baseline_atomic_units

    units = build_heal_lidar_baseline_atomic_units(model, family_id)
    root = "shrinker_m1.layers.0.double_conv.2"
    return units, [unit for unit in units if unit.root_module_path == root]


def _request_for_units(units, selected) -> object:
    from search.adapters.pruning_adapter import FormalPruningAdapter
    from search.candidate import CandidatePhenotype

    phenotype = CandidatePhenotype(
        pruned_unit_ids=[unit.stable_id for unit in selected],
        precision_profile={},
    )
    return FormalPruningAdapter().request_from_phenotype(phenotype, units)


def test_fcooper_topology_and_atomic_domains_cover_exact_backbone() -> None:
    from search.model_family import (
        build_heal_lidar_baseline_atomic_units,
        validate_heal_lidar_baseline_pruning_topology,
    )

    model = HeterModelBaseline(MaxFusion())
    topology = validate_heal_lidar_baseline_pruning_topology(model, "heal_lidar_fcooper")
    units = build_heal_lidar_baseline_atomic_units(model, "heal_lidar_fcooper")

    assert [len(paths) for paths in topology.backbone_conv_paths] == [4, 6, 9]
    assert topology.concat_width == 384
    assert topology.feature_width == 256
    assert len(topology.production_domain_specs) == 21
    assert len(units) == 3840
    assert all(path in topology.fixed_output_contracts for path in topology.deblock_conv_paths)
    assert topology.fixed_output_contracts["cls_head"] == 2


def test_disconet_feature_closure_uses_two_original_concat_halves() -> None:
    from search.model_family import build_heal_lidar_baseline_atomic_units

    units = build_heal_lidar_baseline_atomic_units(
        HeterModelBaseline(DiscoFusion()), "heal_lidar_disco"
    )
    feature = next(
        unit
        for unit in units
        if unit.root_module_path == "shrinker_m1.layers.0.double_conv.2"
        and unit.root_indices == [3]
    )
    mapped = next(
        member
        for member in feature.members
        if member.module_path == "fusion_net.pixel_weight_layer.conv1_1"
    )

    assert mapped.indices == [3, 259]
    assert len(mapped.index_map) == 256
    assert mapped.index_map[3] == [3, 259]
    assert mapped.index_map[255] == [255, 511]
    assert {unit.root_module_path for unit in units} >= {
        "fusion_net.pixel_weight_layer.conv1_1",
        "fusion_net.pixel_weight_layer.conv1_2",
    }
    assert not any(
        unit.root_module_path == "fusion_net.pixel_weight_layer.conv1_3"
        for unit in units
    )


def test_local_domains_use_aligned_widths_and_protect_disco_prelogit() -> None:
    from search.model_family import (
        build_heal_lidar_baseline_atomic_units,
        validate_heal_lidar_baseline_pruning_topology,
    )
    from search.pruning_space.local_domains import build_local_pruning_domains

    model = HeterModelBaseline(DiscoFusion())
    topology = validate_heal_lidar_baseline_pruning_topology(model, "heal_lidar_disco")
    units = build_heal_lidar_baseline_atomic_units(model, "heal_lidar_disco")
    domains = build_local_pruning_domains(
        units,
        minimum_retained_ratio=0.10,
        dense_alignment=4,
    )
    by_root = {row.root_module_path: row for row in domains}

    assert len(domains) == 23
    assert by_root["shrinker_m1.layers.0.double_conv.2"].legal_widths[0] == 28
    assert by_root["fusion_net.pixel_weight_layer.conv1_2"].legal_widths[0] == 4
    protected = next(
        row for row in topology.domain_specs
        if row.root_module_path == "fusion_net.pixel_weight_layer.conv1_3"
    )
    assert protected.protected is True
    assert topology.fixed_output_contracts["fusion_net.pixel_weight_layer.conv1_4"] == 1


@pytest.mark.parametrize(
    ("family_id", "fusion"),
    (("heal_lidar_fcooper", MaxFusion()), ("heal_lidar_disco", DiscoFusion())),
)
def test_feature_width_materialization_closes_fusion_heads_and_strict_replay(
    family_id: str,
    fusion: nn.Module,
) -> None:
    from search.model_family import materialize_heal_lidar_baseline

    torch.manual_seed(7)
    model = HeterModelBaseline(fusion).eval()
    units, feature_units = _feature_units(model, family_id)
    request = _request_for_units(units, feature_units[:4])
    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    example = torch.randn(2, 64, 16, 16)

    result = materialize_heal_lidar_baseline(
        model,
        request,
        family=family_id,
        example_inputs=example,
    )
    physical = result["model"]

    assert result["strict_replay_reload_verified"] is True
    assert result["validation"].forward_checked is True
    assert result["physical_topology"].feature_width == 252
    assert physical.cls_head.in_channels == 252
    assert physical.reg_head.in_channels == 252
    assert physical.dir_head.in_channels == 252
    assert sum(parameter.numel() for parameter in physical.parameters()) < original_parameters
    assert result["ledger_summary"]["status_counts"]["repaired"] == 0
    assert result["ledger_summary"]["status_counts"]["skipped"] == 0
    if family_id == "heal_lidar_disco":
        assert physical.fusion_net.pixel_weight_layer.conv1_1.in_channels == 504


def test_attfusion_feature_width_materialization_updates_parameter_free_scale() -> None:
    from search.model_family import materialize_heal_lidar_baseline

    model = HeterModelBaseline(AttFusion()).eval()
    units, feature_units = _feature_units(model, "heal_lidar_attfusion")
    request = _request_for_units(units, feature_units[:4])

    result = materialize_heal_lidar_baseline(
        model,
        request,
        family="heal_lidar_attfusion",
        example_inputs=torch.randn(2, 64, 16, 16),
    )

    assert result["physical_topology"].feature_width == 252
    assert result["model"].fusion_net.att.sqrt_dim == pytest.approx(math.sqrt(252))
    assert result["attfusion_scale_audit"] == {
        "feature_width": 252,
        "observed_sqrt_dim": pytest.approx(math.sqrt(252)),
        "expected_sqrt_dim": pytest.approx(math.sqrt(252)),
        "passed": True,
        "before_sqrt_dim": pytest.approx(16.0),
        "updated": True,
    }


def test_attfusion_topology_rejects_stale_parameter_free_scale() -> None:
    from search.model_family import validate_heal_lidar_baseline_pruning_topology

    model = HeterModelBaseline(AttFusion()).eval()
    model.fusion_net.att.sqrt_dim = math.sqrt(128.0)
    with pytest.raises(RuntimeError, match="attfusion_sqrt_dim_mismatch"):
        validate_heal_lidar_baseline_pruning_topology(
            model, "heal_lidar_attfusion"
        )


def test_disconet_alignment_repair_uses_complete_double_half_map() -> None:
    from search.model_family import materialize_heal_lidar_baseline

    model = HeterModelBaseline(DiscoFusion()).eval()
    units, feature_units = _feature_units(model, "heal_lidar_disco")
    request = _request_for_units(units, feature_units[:1])

    result = materialize_heal_lidar_baseline(
        model,
        request,
        family="heal_lidar_disco",
        example_inputs=torch.randn(2, 64, 16, 16),
        require_zero_alignment_repair=False,
    )

    assert result["physical_topology"].feature_width == 252
    assert result["model"].fusion_net.pixel_weight_layer.conv1_1.in_channels == 504
    assert result["ledger_summary"]["status_counts"]["repaired"] > 0


def test_disconet_hidden_materialization_keeps_scalar_logit_contract() -> None:
    from search.model_family import (
        build_heal_lidar_baseline_atomic_units,
        materialize_heal_lidar_baseline,
    )

    model = HeterModelBaseline(DiscoFusion()).eval()
    units = build_heal_lidar_baseline_atomic_units(model, "heal_lidar_disco")
    hidden = [
        unit for unit in units
        if unit.root_module_path == "fusion_net.pixel_weight_layer.conv1_1"
    ][:4]
    request = _request_for_units(units, hidden)
    result = materialize_heal_lidar_baseline(
        model,
        request,
        family="heal_lidar_disco",
        example_inputs=torch.randn(2, 64, 16, 16),
    )
    pixel = result["model"].fusion_net.pixel_weight_layer

    assert pixel.conv1_1.out_channels == 124
    assert pixel.bn1_1.num_features == 124
    assert pixel.conv1_2.in_channels == 124
    assert pixel.conv1_4.out_channels == 1


def test_topology_rejects_prunable_deblock_output_contract() -> None:
    from search.model_family import validate_heal_lidar_baseline_pruning_topology

    model = HeterModelBaseline(MaxFusion())
    model.backbone_m1.deblocks[1][0] = nn.ConvTranspose2d(128, 124, 2, stride=2, bias=False)

    with pytest.raises(RuntimeError, match="deblock_contract"):
        validate_heal_lidar_baseline_pruning_topology(model, "heal_lidar_fcooper")
