from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _PillarVFE(nn.Module):
    def forward(self, data):
        features = data["voxel_features"][:, 0, :4]
        return {"pillar_features": features}


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pillar_vfe = _PillarVFE()
        self.scatter = SimpleNamespace(nx=4, ny=4)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())])
        self.deblocks = nn.ModuleList([nn.Identity()])

    def forward(self, data):
        value = self.blocks[0](data["spatial_features"])
        data["spatial_features_2d"] = self.deblocks[0](value)
        return data


class _DiscoFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from search.integration.disconet_compat import PixelWeightLayer

        self.pixel_weight_layer = PixelWeightLayer(4)


class _SoftModel(nn.Module):
    def __init__(self, fusion_kind: str) -> None:
        super().__init__()
        self.encoder_m1 = _Encoder()
        self.backbone_m1 = _Backbone()
        self.shrinker_m1 = nn.Identity()
        self.fusion_net = _DiscoFusion() if fusion_kind == "disconet" else nn.Identity()
        self.cls_head = nn.Conv2d(4, 2, 1)
        self.reg_head = nn.Conv2d(4, 14, 1)
        self.dir_head = nn.Conv2d(4, 4, 1)
        self.ego_modality = "m1"
        self.H = 4.0
        self.W = 4.0
        self.fake_voxel_size = 1.0
        self.compress = False
        self.shrink_flag = False


def _inputs():
    voxel_features = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        ]
    )
    voxel_coords = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 2, 2], [1, 0, 3, 3]],
        dtype=torch.int32,
    )
    voxel_num_points = torch.ones(4, dtype=torch.int32)
    pairwise = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    valid = torch.ones(4)
    return voxel_features, voxel_coords, voxel_num_points, pairwise, valid


def test_family_export_factory_preserves_pyramid_and_selects_softfusion() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.integration.trt_compatible_export import (
        SearchTensorRTCompatibleLidarPyramid,
        build_family_trt_export_module,
    )
    from search.integration.softfusion_trt_export import (
        SearchTensorRTCompatibleSoftFusion,
    )

    pyramid = object.__new__(nn.Module)
    nn.Module.__init__(pyramid)
    pyramid_wrapper = build_family_trt_export_module(
        pyramid,
        family=get_lidar_family_spec("lidar_pyramid"),
        output_names=("cls_preds",),
        fixed_k=8,
    )
    disco_wrapper = build_family_trt_export_module(
        _SoftModel("disconet").eval(),
        family=get_lidar_family_spec("lidar_disco"),
        output_names=("cls_preds", "reg_preds", "dir_preds"),
        fixed_k=4,
    )

    assert isinstance(pyramid_wrapper, SearchTensorRTCompatibleLidarPyramid)
    assert isinstance(disco_wrapper, SearchTensorRTCompatibleSoftFusion)
    assert disco_wrapper.execution_contract == (
        "pillar_vfe",
        "scatter_plugin",
        "base_bev_backbone",
        "modality_shrinker",
        "disconet_fusion",
        "heads",
    )


def test_max_softfusion_wrapper_runs_and_keeps_output_contract() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.integration.softfusion_trt_export import (
        SearchTensorRTCompatibleSoftFusion,
    )

    wrapper = SearchTensorRTCompatibleSoftFusion(
        _SoftModel("max").eval(),
        family=get_lidar_family_spec("lidar_fcooper"),
        output_names=("cls_preds", "reg_preds", "dir_preds"),
        fixed_k=4,
    ).eval()
    outputs = wrapper(*_inputs())

    assert [tuple(value.shape) for value in outputs] == [
        (1, 2, 4, 4),
        (1, 14, 4, 4),
        (1, 4, 4, 4),
    ]
    assert all(torch.isfinite(value).all() for value in outputs)
    assert wrapper.execution_contract[4] == "max_fusion"


def test_disco_softfusion_wrapper_runs_weighted_agent_softmax() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.integration.softfusion_trt_export import (
        SearchTensorRTCompatibleSoftFusion,
    )

    wrapper = SearchTensorRTCompatibleSoftFusion(
        _SoftModel("disconet").eval(),
        family=get_lidar_family_spec("lidar_disco"),
        output_names=("cls_preds", "reg_preds", "dir_preds"),
        fixed_k=4,
    ).eval()
    outputs = wrapper(*_inputs())

    assert [tuple(value.shape) for value in outputs] == [
        (1, 2, 4, 4),
        (1, 14, 4, 4),
        (1, 4, 4, 4),
    ]
    assert all(torch.isfinite(value).all() for value in outputs)
    audit = wrapper.last_fusion_audit
    assert audit["fusion_kind"] == "disconet"
    assert audit["agent_count"] == 2
    assert audit["softmax_axis"] == 0


def test_disco_export_bypasses_redundant_dynamic_pixel_weight_reshape() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.integration.softfusion_trt_export import (
        SearchTensorRTCompatibleSoftFusion,
    )

    wrapper = SearchTensorRTCompatibleSoftFusion(
        _SoftModel("disconet").eval(),
        family=get_lidar_family_spec("lidar_disco"),
        output_names=("cls_preds", "reg_preds", "dir_preds"),
        fixed_k=4,
    ).eval()
    traced = torch.jit.trace(wrapper, _inputs(), strict=False, check_trace=False)
    graph = str(traced.inlined_graph)

    pixel_weight_size_nodes = [
        line
        for line in graph.splitlines()
        if "aten::size" in line and "pixel_weight_layer" in line
    ]
    assert pixel_weight_size_nodes == []


def test_softfusion_factory_rejects_family_model_mismatch() -> None:
    import pytest

    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.integration.softfusion_trt_export import (
        SearchTensorRTCompatibleSoftFusion,
    )

    model = _SoftModel("max").eval()
    with pytest.raises(RuntimeError, match="disconet_pixel_weight_layer_missing"):
        SearchTensorRTCompatibleSoftFusion(
            model,
            family=get_lidar_family_spec("lidar_disco"),
            output_names=("cls_preds",),
            fixed_k=4,
        )


def test_base_bev_backbone_static_matches_module_forward() -> None:
    from search.integration.softfusion_trt_export import _base_bev_backbone_static

    torch.manual_seed(7)
    backbone = _Backbone().eval()
    value = torch.randn(2, 4, 4, 4)
    expected = backbone({"spatial_features": value.clone()})[
        "spatial_features_2d"
    ]
    actual = _base_bev_backbone_static(backbone, value)

    torch.testing.assert_close(actual, expected)
