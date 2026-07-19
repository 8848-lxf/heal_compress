from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_search_runner_imports_family_context_and_evaluator() -> None:
    from search.orchestration import lidar_pyramid_search as module

    assert callable(module.build_lidar_family_context)
    assert module.LidarFamilyRealEvaluator.__name__ == "LidarFamilyRealEvaluator"


def test_supported_six_budget_orchestration_names() -> None:
    from search.orchestration.lidar_pyramid_search import (
        SIX_BUDGET_ORCHESTRATION_NAMES,
    )

    assert "three_seed_six_budget_joint_ga" in SIX_BUDGET_ORCHESTRATION_NAMES
    assert "single_seed_six_budget_joint_ga" in SIX_BUDGET_ORCHESTRATION_NAMES


def test_baseline_only_does_not_require_stage1_joint_loss_scale(
    monkeypatch, tmp_path: Path
) -> None:
    from search.orchestration import lidar_pyramid_search as module

    context = SimpleNamespace(
        gpu_selection=SimpleNamespace(to_dict=lambda: {"physical_gpu_id": 7}),
        tensorrt=SimpleNamespace(to_dict=lambda: {"version": "10.9"}),
        physical_gpu_id=7,
    )

    class FakeEvaluator:
        def __init__(self, **_kwargs) -> None:
            pass

        def evaluate_original_baselines(self, precisions):
            return {str(value): {"status": "ok"} for value in precisions}

    monkeypatch.setattr(module, "build_lidar_family_context", lambda **_kwargs: context)
    monkeypatch.setattr(module, "require_gpu_isolation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "LidarFamilyRealEvaluator", FakeEvaluator)

    runner = module.LidarPyramidTwoStageSearch(
        config={
            "model": {"family": "lidar_disco"},
            "proxy": {
                "proxy_mode": "joint_taylor_second_order_fisher_diag",
                "task_score_mapping": "linear_fixed_scale",
                "joint_loss_scale_path": None,
            },
            "runtime": {"gpu_id": "7"},
            "stage2": {
                "num_frames": 10,
                "warmup_frames": 1,
                "num_workers": 8,
                "ap_iou_backend": "gpu",
            },
            "baselines": {"precisions": ["strict_fp32"]},
        },
        checkpoint=tmp_path / "model.pth",
        output_root=tmp_path / "outputs",
    )

    result = runner.run(baseline_only=True)

    assert result["baseline_only"] is True
    assert result["baselines"]["strict_fp32"]["status"] == "ok"
