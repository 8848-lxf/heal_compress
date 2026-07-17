from __future__ import annotations

import pytest

from quantization.types import (
    CanonicalFunctionalComputeGroup,
    CanonicalMappingEntry,
    OnnxOriginMapResult,
)


def _origin_map() -> OnnxOriginMapResult:
    entries = [
        CanonicalMappingEntry(
            module_path="backbone_m1.blocks.0.1",
            module_type="Conv2d",
            call_index=0,
            onnx_op_type="Conv",
            original_node_name="conv",
            canonical_node_name="__canonical__backbone_conv",
            weight_initializer="conv.weight",
        ),
        CanonicalMappingEntry(
            module_path="fusion_net.layers.0.window_attention.fn.to_qkv",
            module_type="Linear",
            call_index=1,
            onnx_op_type="MatMul",
            original_node_name="qkv",
            canonical_node_name="__canonical__qkv",
            weight_initializer="qkv.weight",
        ),
        CanonicalMappingEntry(
            module_path="cls_head",
            module_type="Conv2d",
            call_index=2,
            onnx_op_type="Conv",
            original_node_name="head",
            canonical_node_name="__canonical__cls_head",
            weight_initializer="head.weight",
        ),
    ]
    functional = CanonicalFunctionalComputeGroup(
        module_path="lidar_cobevt.functional_affine_grid_matmul",
        module_type="functional_bmm",
        canonical_node_name="__canonical__cobevt_grid",
        original_node_names=("grid",),
        graph_indices=(10,),
        source_call="quantization.export.heal_lidar_cobevt._warp_agents",
    )
    return OnnxOriginMapResult(entries=entries, functional_compute_groups=[functional])


def test_precision_actions_are_generated_from_verified_deployment_capability():
    from search.model_families.lidar_cobevt.quantization_recipe import (
        CobevtQuantizationRecipe,
    )

    recipe = CobevtQuantizationRecipe(
        verified_int8_modules={
            "backbone_m1.blocks.0.1",
            "fusion_net.layers.0.window_attention.fn.to_qkv",
        }
    )
    capability = recipe.build_capability(_origin_map())
    by_module = {row.module_path: row for row in capability.weighted_entries}

    assert by_module["backbone_m1.blocks.0.1"].actions == ("FP32", "FP16", "INT8")
    assert by_module[
        "fusion_net.layers.0.window_attention.fn.to_qkv"
    ].actions == ("FP32", "FP16", "INT8")
    assert by_module["cls_head"].actions == ("FP32", "FP16")
    assert capability.functional_entries[0].actions == ("FP16",)
    assert capability.duplicate_precision_group_count == 0


def test_unverified_int8_request_fails_closed():
    from search.model_families.lidar_cobevt.quantization_recipe import (
        CobevtQuantizationRecipe,
    )

    recipe = CobevtQuantizationRecipe(verified_int8_modules=set())
    capability = recipe.build_capability(_origin_map())

    with pytest.raises(RuntimeError, match="precision_action_not_deployable"):
        recipe.build_profile(
            capability,
            {
                "backbone_m1.blocks.0.1": "INT8",
                "fusion_net.layers.0.window_attention.fn.to_qkv": "FP16",
                "cls_head": "FP32",
            },
            profile_id="illegal",
        )


def test_scatter_and_layernorm_are_not_precision_genes():
    from search.model_families.lidar_cobevt.quantization_recipe import (
        CobevtQuantizationRecipe,
    )

    capability = CobevtQuantizationRecipe().build_capability(_origin_map())
    paths = {row.module_path for row in capability.weighted_entries}

    assert "PointPillarScatterTRT" not in paths
    assert not any("norm" in path.lower() for path in paths)

