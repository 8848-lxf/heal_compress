from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
from torch import nn

from deploy.post_scatter import (
    POST_SCATTER_CONTRACT,
    audit_post_scatter_onnx,
    export_post_scatter_onnx,
    filter_post_scatter_module_paths,
    filter_post_scatter_pruning_units,
    filter_post_scatter_quantization_groups,
    post_scatter_shape_profiles,
)
from search.proxy.bops_proxy import BOPSProxy
from search.proxy.size_proxy import SizeProxy


class _PostScatterToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone_m1 = nn.Conv2d(4, 3, 1)

    def forward(self, spatial_features, pairwise_t_matrix):
        value = self.backbone_m1(spatial_features[:1])
        keep_pairwise = pairwise_t_matrix.sum() * 0.0
        return value + keep_pairwise, value, value


def test_formal_post_scatter_export_has_no_fixed_k_or_plugin(tmp_path: Path) -> None:
    destination = tmp_path / "candidate.onnx"
    result = export_post_scatter_onnx(
        _PostScatterToy().eval(),
        {
            "spatial_features": torch.randn(2, 4, 8, 8),
            "pairwise_t_matrix": torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(
                1, 2, 2, 1, 1
            ),
        },
        destination,
    )

    audit = audit_post_scatter_onnx(destination)
    graph = onnx.load(str(destination))
    metadata = {row.key: row.value for row in graph.metadata_props}
    assert result.fixed_k == 0
    assert audit["passed"] is True
    assert audit["runtime_max_k_dependency"] is False
    assert audit["scatter_node_count"] == 0
    assert set(audit["input_names"]) == {
        "spatial_features",
        "pairwise_t_matrix",
    }
    assert metadata["heal.engine_contract"] == POST_SCATTER_CONTRACT


def test_post_scatter_search_space_excludes_pfn_and_rejects_crossing_group() -> None:
    frontend = SimpleNamespace(
        group_id="frontend",
        module_paths=("encoder_m1.pillar_vfe.pfn_layers.0.linear",),
    )
    backbone = SimpleNamespace(
        group_id="backbone", module_paths=("backbone_m1.blocks.0.1",)
    )
    assert filter_post_scatter_quantization_groups([frontend, backbone]) == [
        backbone
    ]
    crossing = SimpleNamespace(
        group_id="crossing",
        module_paths=(
            "encoder_m1.pillar_vfe.pfn_layers.0.linear",
            "backbone_m1.blocks.0.1",
        ),
    )
    with pytest.raises(
        RuntimeError, match="precision_group_crosses_post_scatter_boundary"
    ):
        filter_post_scatter_quantization_groups([crossing])


def test_post_scatter_pruning_space_excludes_pfn_and_rejects_crossing_unit() -> None:
    frontend = SimpleNamespace(
        stable_id="frontend",
        root_module_path="encoder_m1.pillar_vfe.pfn_layers.0.linear",
        members=(),
    )
    backbone = SimpleNamespace(
        stable_id="backbone",
        root_module_path="backbone_m1.blocks.0.1",
        members=(),
    )
    assert filter_post_scatter_pruning_units([frontend, backbone]) == [backbone]
    crossing = SimpleNamespace(
        stable_id="crossing",
        root_module_path="backbone_m1.blocks.0.1",
        members=(
            SimpleNamespace(
                module_path="encoder_m1.pillar_vfe.pfn_layers.0.linear"
            ),
        ),
    )
    with pytest.raises(
        RuntimeError, match="pruning_unit_crosses_post_scatter_boundary"
    ):
        filter_post_scatter_pruning_units([crossing])


def test_post_scatter_cost_scope_excludes_external_frontend() -> None:
    paths = filter_post_scatter_module_paths(
        [
            "encoder_m1.pillar_vfe.pfn_layers.0.linear",
            "backbone_m1.blocks.0.1",
        ]
    )
    assert paths == ["backbone_m1.blocks.0.1"]
    phenotype = SimpleNamespace(
        pruned_unit_ids=(), realized_precision_profile={}
    )
    bops = BOPSProxy(
        layer_ops={
            "encoder_m1.pillar_vfe.pfn_layers.0.linear": 100,
            "backbone_m1.blocks.0.1": 200,
        },
        include_module_paths=paths,
        default_precision="FP32",
    ).evaluate_breakdown(phenotype)
    size = SizeProxy(
        layer_parameter_counts={
            "encoder_m1.pillar_vfe.pfn_layers.0.linear": 10,
            "backbone_m1.blocks.0.1": 20,
        },
        include_module_paths=paths,
        default_precision="FP32",
    ).evaluate_breakdown(phenotype)
    assert bops["bops_total"] == 200 * 32 * 32
    assert bops["bops_fp32_baseline"] == 200 * 32 * 32
    assert size["size_bits_total"] == 20 * 32
    assert size["parameter_count_base"] == 20


def test_post_scatter_profiles_have_only_bev_inputs() -> None:
    profiles = post_scatter_shape_profiles()
    assert set(profiles) == {"spatial_features", "pairwise_t_matrix"}
    assert profiles["spatial_features"]["min"] == (1, 64, 256, 512)
    assert profiles["spatial_features"]["max"] == (2, 64, 256, 512)
    assert all("voxel" not in name for name in profiles)
