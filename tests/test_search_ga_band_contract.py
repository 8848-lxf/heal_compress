from __future__ import annotations

import random


def test_epsilon_lexicographic_ranking_keeps_taylor_primary() -> None:
    from search.candidate import CandidateGenotype
    from search.ga.ranking import rank_constraint_first

    def row(name: str, taylor: float, retention: float, *, feasible: bool = True):
        candidate = CandidateGenotype(
            pruning_genes={name: 1}, precision_genes={"q": "FP16"}
        )
        metrics = {
            "F1": taylor,
            "L_joint_weight_taylor": taylor,
            "R_parameter_retention": retention,
            "bops_feasible": feasible,
            "bops_violation": 0.0 if feasible else 0.001,
        }
        return candidate, taylor, metrics

    ranked = rank_constraint_first(
        [
            row("best_taylor", 1.00, 0.90),
            row("near_lower_params", 1.04, 0.60),
            row("outside_epsilon", 1.10, 0.20),
            row("infeasible", 0.01, 0.10, feasible=False),
        ],
        taylor_relative_epsilon=0.05,
        taylor_absolute_epsilon=0.0,
    )

    assert list(ranked[0][0].pruning_genes) == ["near_lower_params"]
    assert list(ranked[1][0].pruning_genes) == ["best_taylor"]
    assert list(ranked[2][0].pruning_genes) == ["outside_epsilon"]
    assert list(ranked[3][0].pruning_genes) == ["infeasible"]


def test_action_count_mutation_changes_only_requested_number_of_genes() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.ga.mutation import mutate_candidate

    space = SearchSpaceSpec(
        pruning_unit_ids=["u0", "u1", "u2", "u3"],
        precision_layer_ids=["q0", "q1", "q2"],
        default_precision="FP32",
    )
    candidate = CandidateGenotype(
        pruning_genes={key: 1 for key in space.pruning_unit_ids},
        precision_genes={key: "FP32" for key in space.precision_gene_ids},
    )

    mutated = mutate_candidate(
        candidate,
        space,
        random.Random(9),
        prune_mutation_rate=1.0,
        precision_mutation_rate=1.0,
        action_count=2,
        adjacent_precision=True,
    )

    changed = sum(
        mutated.pruning_genes[key] != candidate.pruning_genes[key]
        for key in space.pruning_unit_ids
    ) + sum(
        mutated.precision_genes[key] != candidate.precision_genes[key]
        for key in space.precision_gene_ids
    )
    assert changed == 2
    assert all(
        value in {"FP32", "FP16"} for value in mutated.precision_genes.values()
    )


def test_formal_h800_ga_config_uses_band_and_fp32_reference() -> None:
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        Path("search/configs/lidar_pyramid_h800_domain_width_joint_ga.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["proxy"]["bops_constraint_mode"] == "hard_band_feasibility"
    assert config["proxy"]["bops_tolerance_abs"] == 0.005
    assert config["search"]["generations_per_round"] == 15
    assert config["search"]["mutation_action_max"] == 2
    assert config["search"]["independent_budget_rounds"] is True
    assert config["search"]["greedy_frontier_warm_start"] is True
    assert config["stage2"]["latency_reference"] == "original_strict_fp32"


def test_resume_reconstructs_stage1_topk_without_rerunning_ga(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.orchestration.lidar_pyramid_search import (
        LidarPyramidTwoStageSearch,
    )

    space = SearchSpaceSpec(
        pruning_unit_ids=["u0", "u1"],
        precision_layer_ids=["q0"],
        default_precision="FP32",
    )
    phenotype = canonicalize_candidate(
        CandidateGenotype(
            pruning_genes={"u0": 0, "u1": 1},
            precision_genes={"q0": "FP16"},
        ),
        space,
    )
    (tmp_path / "stage1_topk.json").write_text(
        json.dumps(
            [
                {
                    "role": "repaired",
                    "candidate_hash": "candidate",
                    "F1": 0.1,
                    "phenotype": phenotype.to_dict(),
                }
            ]
        ),
        encoding="utf-8",
    )

    selected = LidarPyramidTwoStageSearch._load_round_stage1_selections(
        tmp_path, SimpleNamespace(search_space=space)
    )

    assert len(selected) == 1
    assert selected[0].record.candidate_hash == "candidate"
    assert selected[0].record.genotype.pruning_genes == {"u0": 0, "u1": 1}
    assert selected[0].record.genotype.precision_genes == {"q0": "FP16"}
