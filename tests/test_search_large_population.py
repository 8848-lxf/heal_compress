from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_ga_config_supports_initial_population_and_offspring_size() -> None:
    from search.ga.engine import GAConfig

    config = GAConfig(initial_population_size=1024, population_size=512, offspring_size=512)

    assert config.initial_population_size == 1024
    assert config.population_size == 512
    assert config.offspring_size == 512


def test_large_population_yaml_contains_required_counts() -> None:
    import yaml

    path = Path("search/configs/lidar_pyramid_joint_search_large_population.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert data["search"]["initial_population_size"] == 1024
    assert data["search"]["population_size"] == 512
    assert data["search"]["offspring_size"] == 512
    assert data["pruning"]["grouped_conv"]["position_mode"] == "independent_group_topk"


def test_ga_evaluates_full_initial_population_before_active_population() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.ga.engine import GAConfig, GeneticSearchEngine

    calls: list[int] = []
    space = SearchSpaceSpec(pruning_unit_ids=["a"], precision_layer_ids=["m"], default_precision="FP16")
    engine = GeneticSearchEngine(
        space,
        GAConfig(
            initial_population_size=8,
            population_size=4,
            offspring_size=4,
            num_generations=2,
            random_seed=3,
        ),
    )

    def evaluator(candidate: CandidateGenotype, generation: int) -> dict[str, float]:
        calls.append(generation)
        return {"F1": float(candidate.pruning_genes.get("a", 1))}

    engine.run(evaluator)

    assert calls.count(0) == 8
    assert calls.count(1) == 4
