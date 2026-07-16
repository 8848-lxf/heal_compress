from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _record(
    candidate_hash: str,
    phenotype_hash: str,
    *,
    j1: float,
    seed_index: int | None = None,
):
    from search.stage1.topk_selector import ProxyCandidateRecord

    metrics = {"J1": float(j1), "R_BOPS": 0.20}
    if seed_index is not None:
        metrics["seed_index"] = int(seed_index)
    return ProxyCandidateRecord(
        candidate_hash=candidate_hash,
        genotype=SimpleNamespace(genotype_hash=f"g-{candidate_hash}"),
        phenotype=SimpleNamespace(
            metadata={"phenotype_hash": phenotype_hash}
        ),
        F1=-float(j1),
        metrics=metrics,
    )


def test_merge_three_seed_generation_records_deduplicates_and_sorts_j1() -> None:
    from search.orchestration.legal_width_six_budget_ga import (
        merge_seed_generation_records,
    )

    merged = merge_seed_generation_records(
        {
            0: [_record("a", "duplicate", j1=0.2), _record("b", "p0", j1=0.7)],
            1: [_record("c", "duplicate", j1=0.8), _record("d", "p1", j1=0.6)],
            2: [_record("e", "p2", j1=0.5)],
        },
        generation=3,
    )

    assert [row.metrics["J1"] for row in merged] == [0.8, 0.7, 0.6, 0.5]
    assert [row.candidate_hash for row in merged] == ["c", "b", "d", "e"]
    assert len(
        {row.phenotype.metadata["phenotype_hash"] for row in merged}
    ) == len(merged)
    assert {row.metrics["seed_index"] for row in merged} == {0, 1, 2}
    assert all(row.metrics["generation"] == 3 for row in merged)


def test_six_budget_orchestrator_calls_stage2_once_per_generation(
    tmp_path: Path, monkeypatch
) -> None:
    from search.orchestration import legal_width_six_budget_ga as module

    stage1_calls: list[float] = []
    stage2_calls: list[tuple[float, int, int]] = []

    def fake_stage1(*, search_config, **_kwargs):
        target = float(search_config["target_bops_retention"])
        stage1_calls.append(target)
        return {
            "generation_records": {
                generation: {
                    seed: [
                        _record(
                            f"b{target:.2f}-g{generation}-s{seed}",
                            f"p{target:.2f}-g{generation}-s{seed}",
                            j1=1.0 - seed * 0.1,
                            seed_index=seed,
                        )
                    ]
                    for seed in range(3)
                }
                for generation in range(20)
            },
            "total_proxy_evaluations": 64 * 20 * 3,
            "repair_report": {
                "normal_candidate_count": 64 * 20 * 3,
                "repair_invocation_count": 0,
                "repair_invocation_rate": 0.0,
            },
            "generation_summaries": [],
        }

    def fake_stage2(ranked_records, *, generation_index, policy, **_kwargs):
        stage2_calls.append(
            (float(policy.target), int(generation_index), len(ranked_records))
        )
        return {
            "status": "no_deployable_candidates",
            "generation": generation_index + 1,
            "selected_count": 0,
            "failure_records": [],
        }

    monkeypatch.setattr(module, "run_legal_width_stage1_seeds", fake_stage1)
    monkeypatch.setattr(module, "run_generation_stage2", fake_stage2)

    report = module.run_six_budget_joint_ga(
        context=object(),
        proxy=object(),
        stage2_pool=object(),
        run_dir=tmp_path,
        config={
            "targets": [0.20, 0.25],
            "primary_bops_tolerance": 0.005,
            "expanded_bops_tolerance": 0.0075,
            "independent_seeds": 3,
            "initial_population_size": 64,
            "population_size": 64,
            "offspring_size": 64,
            "generations": 20,
            "topk_stage2": 5,
        },
    )

    assert stage1_calls == [0.20, 0.25]
    assert len(stage2_calls) == 40
    assert {(target, generation) for target, generation, _ in stage2_calls} == {
        (target, generation)
        for target in (0.20, 0.25)
        for generation in range(20)
    }
    assert all(merged_count == 3 for _, _, merged_count in stage2_calls)
    assert report["generation_stage2_call_count"] == 40
    assert report["generation_winner_count"] == 0


def test_formal_six_budget_config_contract() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "search"
        / "configs"
        / "lidar_pyramid_4090_joint_six_budget_ga.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["search"]["targets"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert config["search"]["independent_seeds"] == 3
    assert config["search"]["population_size"] == 64
    assert config["search"]["offspring_size"] == 64
    assert config["search"]["generations"] == 20
    assert config["proxy"]["task_score_mapping"] == "linear_fixed_scale"
    assert config["proxy"]["joint_loss_scale_path"] is None
    assert config["proxy"]["include_activation_taylor"] is False
    assert config["proxy"]["sqnr_main_objective_weight"] == 0.0
    assert config["stage2"]["score_mode"] == "map_minus_latency_ratio"
    assert not ({"min_map", "min_ap07", "max_map_drop"} & set(config["stage2"]))
    assert config["stage2"]["num_frames"] == 500
    assert config["stage2"]["num_workers"] == 8
    assert config["runtime"]["stage2_gpu_ids"] == "auto"
    assert config["runtime"]["max_stage2_gpu_memory_fraction"] == 0.50
