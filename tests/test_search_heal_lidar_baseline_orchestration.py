from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CONFIG_NAMES = (
    "heal_lidar_fcooper_h800_domain_width_joint_ga.yaml",
    "heal_lidar_fcooper_h800_domain_width_joint_greedy.yaml",
    "heal_lidar_disco_h800_domain_width_joint_ga.yaml",
    "heal_lidar_disco_h800_domain_width_joint_greedy.yaml",
)


def _configs() -> list[dict]:
    root = Path(__file__).resolve().parents[1] / "search/configs"
    return [yaml.safe_load((root / name).read_text(encoding="utf-8")) for name in CONFIG_NAMES]


def test_formal_baseline_configs_preserve_six_budget_and_h800_protocol() -> None:
    for config in _configs():
        assert config["model"]["family_id"] in {"heal_lidar_fcooper", "heal_lidar_disco"}
        assert config["model"]["fixed_k"] == 29696
        assert config["model"]["max_agents"] == 2
        # Both families run serially on the dedicated pool selected for this rerun.
        expected_gpu_ids = [5, 6, 7]
        assert config["runtime"]["stage1_gpu_ids"] == expected_gpu_ids
        assert config["runtime"]["stage2_gpu_ids"] == expected_gpu_ids
        assert config["runtime"]["stage1_minimum_workers"] == 3
        assert config["runtime"]["stage2_minimum_workers"] == 3
        assert config["runtime"]["parallel_gpu_min_free_mib"] == 60000
        assert config["runtime"]["parallel_gpu_max_utilization_pct"] == 10
        assert config["proxy"]["bops_targets"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        assert config["proxy"]["bops_tolerance_abs"] == 0.005
        assert config["stage2"]["num_frames"] == 500
        assert config["stage2"]["warmup_frames"] == 200
        assert config["stage2"]["latency_rounds"] == 3
        assert config["full_validation"]["num_frames"] == 1789
        assert config["full_validation"]["warmup_frames"] == 200
        assert config["full_validation"]["latency_rounds"] == 3
        if config["search"]["method"] == "ga":
            assert config["search"]["initial_population_size"] == 1024
            assert config["search"]["population_size"] == 512
            assert config["search"]["offspring_size"] == 512
            assert config["search"]["generations_per_round"] == 15
            assert config["search"]["topk_stage2"] == 5
        else:
            assert config["search"]["maximum_steps"] == 10000


def test_baseline_gpu_pool_fails_closed_when_requested_pool_is_not_idle(monkeypatch) -> None:
    from search.orchestration import heal_lidar_baseline_search as module

    monkeypatch.setattr(module, "query_gpus", lambda: [
        {"index": 1, "memory_free_mib": 61000, "utilization_gpu_pct": 0},
        {"index": 3, "memory_free_mib": 59000, "utilization_gpu_pct": 0},
    ])
    runtime = {
        "stage2_gpu_ids": [1, 3],
        "parallel_gpu_min_free_mib": 60000,
        "parallel_gpu_max_utilization_pct": 10,
        "stage2_minimum_workers": 2,
    }

    with pytest.raises(RuntimeError, match="gpu_pool_insufficient"):
        module.HealLidarBaselineTwoStageSearch._strict_idle_gpu_pool(
            runtime,
            role="stage2",
            primary_gpu_id=1,
        )


def test_baseline_ga_uses_distinct_screening_and_full_validation_worker_pools(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from search.orchestration import heal_lidar_baseline_search as module

    class FakeEvaluator:
        def __init__(self, marker: str) -> None:
            self.marker = marker
            self._reference_baseline_override = None

        def _stage2_reference_baseline(self):
            return {"mAP": 1.0, "forward_p50_ms": 1.0}

    monkeypatch.setattr(
        module,
        "query_gpus",
        lambda: [
            {
                "index": 5,
                "memory_free_mib": 80000,
                "utilization_gpu_pct": 0,
            }
        ],
    )
    runner = object.__new__(module.HealLidarBaselineTwoStageSearch)
    runner._stage2_gpu_ids = [5]
    runner._runtime_config = {"stage2_minimum_workers": 1}
    context = SimpleNamespace(physical_gpu_id=5)
    screening = FakeEvaluator("screening")
    full = FakeEvaluator("full")

    screening_pool = runner._ga_stage2_evaluator_pool(
        context,
        screening,
        tmp_path,
        pool_kind="screening_500",
    )
    full_pool = runner._ga_stage2_evaluator_pool(
        context,
        full,
        tmp_path,
        pool_kind="full_validation",
    )

    assert screening_pool == [(5, screening)]
    assert full_pool == [(5, full)]
    assert runner._ga_stage2_worker_pool == screening_pool
    assert runner._ga_full_validation_worker_pool == full_pool
    assert (tmp_path / "stage2_workers/worker_pool_manifest.json").is_file()
    assert (
        tmp_path / "full_validation_workers/worker_pool_manifest.json"
    ).is_file()


def test_cli_routes_baseline_family_to_baseline_orchestrator(tmp_path: Path, monkeypatch) -> None:
    from search import cli

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "model": {
            "family_id": "heal_lidar_fcooper",
            "checkpoint": str(checkpoint),
            "config": "/tmp/model.yaml",
        },
        "search": {"method": "greedy"},
    }), encoding="utf-8")
    calls = {}

    class FakeBaselineRunner:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run(self, **kwargs):
            calls["run"] = kwargs
            run_dir = tmp_path / "run"
            run_dir.mkdir()
            return {"run_dir": str(run_dir), "status": "ok"}

    class WrongRunner:
        def __init__(self, **kwargs):
            raise AssertionError("pyramid runner must not be selected")

    monkeypatch.setattr(cli, "HealLidarBaselineTwoStageSearch", FakeBaselineRunner)
    monkeypatch.setattr(cli, "LidarPyramidTwoStageSearch", WrongRunner)

    result = cli.main([
        "--config", str(config_path),
        "--output-root", str(tmp_path / "outputs"),
        "--stage1-only",
    ])

    assert result == 0
    assert calls["run"]["stage1_only"] is True
    assert calls["init"]["checkpoint"] == str(checkpoint)


def test_stage2_candidate_loader_accepts_stage1_budget_wrapper(tmp_path: Path) -> None:
    import json

    from search.candidate import CandidateGenotype, CandidatePhenotype
    from search.orchestration.lidar_pyramid_search import _load_candidate

    genotype = CandidateGenotype(
        pruning_genes={},
        pruning_width_genes={"domain": 4},
        precision_genes={"qg": "INT8"},
    )
    genotype_path = tmp_path / "genotype_wrapper.json"
    genotype_path.write_text(
        json.dumps({"target": 0.1, "genotype": genotype.to_dict()}),
        encoding="utf-8",
    )
    loaded_genotype = _load_candidate(genotype_path)
    assert isinstance(loaded_genotype, CandidateGenotype)
    assert loaded_genotype.pruning_width_genes == {"domain": 4}

    phenotype = CandidatePhenotype(
        pruned_unit_ids=["unit"],
        precision_profile={},
        pruning_policy_version="p",
        precision_policy_version="q",
    )
    phenotype_path = tmp_path / "phenotype_wrapper.json"
    phenotype_path.write_text(
        json.dumps({"target": 0.1, "phenotype": phenotype.to_dict()}),
        encoding="utf-8",
    )
    loaded_phenotype = _load_candidate(phenotype_path)
    assert isinstance(loaded_phenotype, CandidatePhenotype)
    assert loaded_phenotype.pruned_unit_ids == ["unit"]
