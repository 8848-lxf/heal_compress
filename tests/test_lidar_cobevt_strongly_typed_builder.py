from __future__ import annotations

from pathlib import Path

from quantization.types import (
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
)


def _mapping() -> CanonicalPrecisionMappingResult:
    return CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="encoder_m1.pillar_vfe.pfn_layers.0.linear",
                canonical_node_name="__canonical__cobevt_linear",
                precision_group="cobevt_pg::encoder_m1.pillar_vfe.pfn_layers.0.linear",
                requested_precision="fp16",
                realized_request_precision="fp16",
                weight_initializer="linear.weight",
                onnx_op_type="MatMul",
            )
        ]
    )


def test_cobevt_builder_command_is_production_strongly_typed(tmp_path: Path) -> None:
    from search.model_families.lidar_cobevt.deployment_recipe import (
        CobevtDeploymentRecipe,
    )

    recipe = CobevtDeploymentRecipe(
        tensorrt_root=tmp_path / "TensorRT-10.9",
        trtexec_path=tmp_path / "TensorRT-10.9/bin/trtexec",
        plugin_path=tmp_path / "scatter.so",
        fixed_k=64,
    )

    result = recipe.builder_command(
        onnx_path=tmp_path / "typed.onnx",
        engine_path=tmp_path / "engine.plan",
        mapping=_mapping(),
    )

    command = result.command
    assert "--stronglyTyped" in command
    assert "--noTF32" in command
    assert "--skipInference" in command
    assert not any(
        token.startswith(
            ("--fp16", "--int8", "--precisionConstraints", "--layerPrecisions", "--layerOutputTypes")
        )
        for token in command
    )


def test_cobevt_builder_fixes_scatter_boundary_to_fp32(tmp_path: Path) -> None:
    from search.model_families.lidar_cobevt.deployment_recipe import (
        CobevtDeploymentRecipe,
    )

    recipe = CobevtDeploymentRecipe(
        tensorrt_root=tmp_path,
        trtexec_path=tmp_path / "trtexec",
        plugin_path=tmp_path / "scatter.so",
        fixed_k=64,
    )

    assert recipe.plugin_boundary_dtype == "fp32"
    assert recipe.model_family == "lidar_cobevt"

