from __future__ import annotations

from pathlib import Path

import pytest


def test_band_selection_is_deterministic() -> None:
    from search.orchestration.lidar_cobevt_smoke import select_smoke_band

    counts = {0.10: 2, 0.15: 8, 0.20: 8, 0.25: 4, 0.30: 1}
    assert select_smoke_band(counts) == 0.20


def test_band_selection_fails_when_no_budget_is_reachable() -> None:
    from search.orchestration.lidar_cobevt_smoke import select_smoke_band

    with pytest.raises(RuntimeError, match="cobevt_no_reachable_smoke_budget"):
        select_smoke_band({0.10: 0, 0.20: 0})


def test_runner_uses_generic_greedy_and_ga_with_frozen_scale(
    monkeypatch, tmp_path: Path
) -> None:
    from search.orchestration import lidar_cobevt_smoke as module

    calls = []

    def greedy(context, proxy, run_dir, config):
        calls.append(("greedy", dict(config)))
        return {"endpoints": [{"actual_bops": 0.201}]}

    def ga(*, context, proxy, run_dir, search_config):
        calls.append(("ga", dict(search_config)))
        return {"archive_summary": {"feasible_phenotype_count": 3}}

    monkeypatch.setattr(module, "run_six_budget_greedy", greedy)
    monkeypatch.setattr(module, "run_legal_width_stage1_seeds", ga)

    result = module.run_bounded_cobevt_stage1(
        context=object(),
        proxy=object(),
        run_dir=tmp_path,
        config={
            "reachable_budget_counts": {0.15: 2, 0.20: 4, 0.25: 4},
            "ga": {
                "population_size": 16,
                "initial_population_size": 16,
                "offspring_size": 16,
                "generations": 3,
                "independent_seeds": 1,
                "seed": 4090,
                "topk_per_generation": 2,
            },
            "budget": {"tolerance": 0.0075},
        },
    )

    assert result["selected_budget"] == 0.20
    assert [row[0] for row in calls] == ["greedy", "ga"]
    assert calls[0][1]["targets"] == [0.20]
    assert calls[1][1]["target_bops_retention"] == 0.20
    assert calls[1][1]["bops_tolerance"] == 0.0075


def test_runner_rejects_resume_for_fresh_smoke(tmp_path: Path) -> None:
    from search.orchestration.lidar_cobevt_smoke import LidarCobevtSmokeSearch

    with pytest.raises(ValueError, match="cobevt_smoke_resume_forbidden"):
        LidarCobevtSmokeSearch(
            config={"model": {"family": "lidar_cobevt"}},
            checkpoint=tmp_path / "model.pth",
            output_root=tmp_path,
            resume=tmp_path / "old-run",
        )

