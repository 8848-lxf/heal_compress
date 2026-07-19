from __future__ import annotations

import sys
from pathlib import Path

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

