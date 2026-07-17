from __future__ import annotations

from pathlib import Path

import yaml


def test_cobevt_smoke_scale_is_exact() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "search/configs/lidar_cobevt_4090_model_family_smoke.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["model"]["family"] == "lidar_cobevt"
    assert config["ga"] == {
        "population_size": 16,
        "initial_population_size": 16,
        "offspring_size": 16,
        "generations": 3,
        "independent_seeds": 1,
        "seed": 4090,
        "topk_per_generation": 2,
    }
    assert config["evaluation"]["smoke_frames"] == 10
    assert config["evaluation"]["screening_frames"] == 50
    assert config["evaluation"]["num_workers"] == 8
    assert config["evaluation"]["ap_iou_backend"] == "gpu"
    assert config["greedy"]["frontier_size"] == 1
    assert config["cache"] == {
        "fresh_run": True,
        "reuse_external_cache": False,
        "resume": False,
    }


def test_cobevt_smoke_production_builder_is_strongly_typed() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "search/configs/lidar_cobevt_4090_model_family_smoke.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["deployment"]["strongly_typed"] is True
    assert config["deployment"]["no_tf32"] is True
    assert config["deployment"]["plugin_boundary_dtype"] == "fp32"
    assert config["deployment"]["weak_precision_fallback"] is False
