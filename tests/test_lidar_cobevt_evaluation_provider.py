from __future__ import annotations

from pathlib import Path


def test_cobevt_evaluation_request_is_gpu_only_and_family_specific(tmp_path: Path) -> None:
    from search.integration.lidar_cobevt_evaluation_provider import (
        build_cobevt_evaluation_request,
    )

    request = build_cobevt_evaluation_request(
        engine_path=tmp_path / "engine.plan",
        checkpoint=tmp_path / "model.pth",
        model_config=tmp_path / "config.yaml",
        heal_root=tmp_path / "HEAL",
        device="cuda:7",
        output_path=tmp_path / "evaluation.json",
        plugin_path=tmp_path / "scatter.so",
        fixed_k=24064,
        num_frames=50,
        warmup_frames=20,
        eval_manifest_path=tmp_path / "fixed50.json",
        num_workers=8,
        ap_iou_backend="gpu",
    )

    assert request["model_family"] == "lidar_cobevt"
    assert request["num_workers"] == 8
    assert request["ap_iou_backend"] == "gpu"
    assert request["strict_gpu_ap_iou"] is True
    assert request["fixed_k"] == 24064
    assert request["device"] == "cuda:0"
    assert request["physical_device"] == "cuda:7"


def test_cobevt_evaluation_provider_uses_separate_worker_module() -> None:
    from search.integration.lidar_cobevt_evaluation_provider import worker_module

    assert worker_module() == "search.integration.lidar_cobevt_evaluation_worker"
