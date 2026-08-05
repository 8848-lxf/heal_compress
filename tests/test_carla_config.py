from pathlib import Path

from carla_integration.config import load_collection_config


def test_dair_aligned_collection_config_contract():
    path = Path(__file__).resolve().parents[1] / "configs/carla/dair_v2x_aligned.yaml"
    config = load_collection_config(str(path))
    assert len(config["scenes"]) == 5
    assert config["model_contract"]["point_frontend"].endswith("outside_tensorrt")
    vehicle = config["_profiles"]["vehicle"]
    infrastructure = config["_profiles"]["infrastructure"]
    assert vehicle.channels == 40
    assert vehicle.horizontal_fov == 360.0
    assert infrastructure.channels == 300
    assert infrastructure.horizontal_fov == 100.0
    assert infrastructure.points_per_second == 2052000
    assert infrastructure.physical_height_m > infrastructure.canonical_height_m
    assert (
        config["carla_0_9_10_compatibility"]["sensor_tick"]
        == "every_fixed_0_1_second_world_step"
    )
    assert (
        config["carla_0_9_10_compatibility"][
            "infrastructure_effective_points_per_second_after_crop"
        ]
        == 570000
    )
