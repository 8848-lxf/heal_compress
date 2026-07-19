from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _record(candidate_hash: str, generation: int):
    from search.stage1.topk_selector import ProxyCandidateRecord

    return ProxyCandidateRecord(
        candidate_hash=candidate_hash,
        genotype=SimpleNamespace(genotype_hash=f"g-{candidate_hash}"),
        phenotype=SimpleNamespace(
            metadata={"phenotype_hash": f"p-{candidate_hash}"}
        ),
        F1=-1.0,
        metrics={"J1": 1.0, "R_BOPS": 0.20, "generation": generation},
    )


def test_single_seed_fifteen_generation_orchestrator(monkeypatch, tmp_path: Path) -> None:
    from search.orchestration import legal_width_six_budget_ga as module

    stage2_calls = []

    def fake_stage1(*, search_config, **_kwargs):
        generations = int(search_config["generations"])
        assert int(search_config["independent_seeds"]) == 1
        return {
            "generation_records": {
                generation: {
                    0: [_record(f"g{generation}", generation)]
                }
                for generation in range(generations)
            },
            "total_proxy_evaluations": 64 * generations,
            "repair_report": {
                "normal_candidate_count": 64 * generations,
                "repair_invocation_count": 0,
                "repair_invocation_rate": 0.0,
            },
            "generation_summaries": [],
        }

    def fake_stage2(records, *, generation_index, topk, policy, **_kwargs):
        stage2_calls.append(
            {
                "generation": generation_index,
                "target": float(policy.target),
                "record_count": len(records),
                "topk": int(topk),
            }
        )
        return {
            "status": "no_deployable_candidates",
            "generation": generation_index + 1,
            "selected_count": 0,
            "failure_records": [],
        }

    monkeypatch.setattr(module, "run_legal_width_stage1_seeds", fake_stage1)
    monkeypatch.setattr(module, "run_generation_stage2", fake_stage2)

    result = module.run_six_budget_joint_ga(
        context=object(),
        proxy=object(),
        stage2_pool=object(),
        run_dir=tmp_path,
        config={
            "targets": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "primary_bops_tolerance": 0.005,
            "expanded_bops_tolerance": 0.0075,
            "independent_seeds": 1,
            "initial_population_size": 64,
            "population_size": 64,
            "offspring_size": 64,
            "generations": 15,
            "topk_stage2": 5,
        },
    )

    assert result["generation_stage2_call_count"] == 6 * 15
    assert result["total_proxy_evaluations"] == 6 * 64 * 15
    assert len(stage2_calls) == 90
    assert all(row["record_count"] == 1 for row in stage2_calls)
    assert all(row["topk"] == 5 for row in stage2_calls)


def test_disco_ga_config_is_one_seed_fifteen_generations() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "search/configs/lidar_disco_4090_joint_six_budget_ga.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["model"]["family"] == "lidar_disco"
    assert config["search"]["targets"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert config["search"]["independent_seeds"] == 1
    assert config["search"]["initial_population_size"] == 64
    assert config["search"]["population_size"] == 64
    assert config["search"]["offspring_size"] == 64
    assert config["search"]["generations"] == 15
    assert config["search"]["topk_stage2"] == 5
    assert config["search"]["per_generation_stage2"] is True
    retention = config["search"]["stage2_artifact_retention"]
    assert retention["content_addressed_audit"] is True
    assert config["stage2"]["num_frames"] == 500
    assert config["full_validation"]["num_frames"] == 1789
    assert config["baselines"]["precisions"] == [
        "strict_fp32",
        "strict_fp16",
        "maximal_legal_int8",
    ]


def test_disco_greedy_config_has_same_six_budgets() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "search/configs/lidar_disco_4090_greedy_six_budget.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["model"]["family"] == "lidar_disco"
    assert config["greedy"]["targets"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert config["greedy"]["deploy_final_only"] is True
    assert config["stage2"]["num_frames"] == 500
    assert config["baselines"]["precisions"] == [
        "strict_fp32",
        "strict_fp16",
        "maximal_legal_int8",
    ]
