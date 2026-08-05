from pathlib import Path

from carla_integration.post_scatter_deploy import (
    build_command,
    post_scatter_shape_profiles,
)


def test_post_scatter_profiles_have_dynamic_agents_without_max_k():
    profiles = post_scatter_shape_profiles()
    assert set(profiles) == {"spatial_features", "pairwise_t_matrix"}
    assert profiles["spatial_features"]["min"][0] == 1
    assert profiles["spatial_features"]["max"][0] == 2
    assert all("voxel" not in name for name in profiles)


def test_post_scatter_trtexec_command_is_strongly_typed_and_plugin_free():
    command = build_command(
        trtexec=Path("../TensorRT/bin/trtexec"),
        onnx_path=Path("../outputs/candidate/post_scatter_qdq.onnx"),
        engine_path=Path("../outputs/candidate/post_scatter.plan"),
        layer_info_path=Path("../outputs/candidate/engine_layer_info.json"),
    )
    assert "--stronglyTyped" in command
    assert "--skipInference" in command
    assert not any("Plugins=" in item for item in command)
    assert not any("voxel" in item for item in command)
    assert any(item.startswith("--minShapes=") for item in command)
    assert any(item.startswith("--maxShapes=") for item in command)
