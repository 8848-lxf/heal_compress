from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_disco_family_declares_softmax_output_contract() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec

    disco = get_lidar_family_spec("lidar_disco")
    pyramid = get_lidar_family_spec("lidar_pyramid")

    assert disco.functional_fp16_output_modules == (
        "fusion_net.pixel_weight_layer.conv1_4",
    )
    assert disco.required_merge_contract_names == ()
    assert "encoder_m1.pillar_vfe.pfn_layers.0.linear" in (
        disco.protected_precision_modules
    )
    assert pyramid.required_merge_contract_names == ("/Concat_9",)


def test_functional_boundary_is_family_specific() -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.integration.lidar_pyramid_context import (
        _functional_fp16_output_boundary,
    )

    disco = get_lidar_family_spec("lidar_disco")
    pyramid = get_lidar_family_spec("lidar_pyramid")
    disco_boundary = _functional_fp16_output_boundary(
        "fusion_net.pixel_weight_layer.conv1_4", family=disco
    )

    assert disco_boundary is not None
    assert disco_boundary["merge_kind"] == "functional_agent_softmax_weighted_sum"
    assert disco_boundary["following_ops"] == ["Relu", "Softmax", "Mul", "ReduceSum"]
    assert (
        _functional_fp16_output_boundary(
            "fusion_net.pixel_weight_layer.conv1_4", family=pyramid
        )
        is None
    )


def test_family_context_facade_forwards_model_family(monkeypatch) -> None:
    from search.integration import lidar_family_context as module

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return "context"

    monkeypatch.setattr(module, "_build_lidar_context", fake_build)

    assert (
        module.build_lidar_family_context(
            model_family="lidar_disco",
            checkpoint_path="checkpoint.pth",
            output_dir="output",
            plugin_boundary_dtype="fp32",
        )
        == "context"
    )
    assert captured["model_family"] == "lidar_disco"


def test_candidate_worker_context_includes_configured_family() -> None:
    from search.stage2.candidate_worker import _context_kwargs

    kwargs = _context_kwargs(
        {
            "gpu_id": 7,
            "checkpoint": "checkpoint.pth",
            "worker_dir": "worker",
            "config": {
                "model": {
                    "family": "lidar_disco",
                    "config": "config.yaml",
                },
                "runtime": {"plugin_boundary_dtype": "fp32"},
            },
        }
    )

    assert kwargs["model_family"] == "lidar_disco"

