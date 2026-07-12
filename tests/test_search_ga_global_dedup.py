from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_ga_skips_seen_raw_genotypes() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.ga.engine import GAConfig, GeneticSearchEngine
    from search.hashing import canonical_json_hash

    space = SearchSpaceSpec(
        pruning_unit_ids=["u0", "u1", "u2"],
        precision_layer_ids=["layer"],
        default_precision="FP16",
    )
    seen = CandidateGenotype(
        pruning_genes={"u0": 1, "u1": 1, "u2": 1},
        precision_genes={"layer": "FP16"},
    )
    seen_key = canonical_json_hash(seen.to_dict())
    evaluated: list[str] = []

    def key_fn(genotype: CandidateGenotype) -> str:
        return canonical_json_hash(genotype.to_dict())

    def evaluator(genotype: CandidateGenotype, _generation: int) -> dict[str, float]:
        evaluated.append(key_fn(genotype))
        return {"F1": 1.0}

    engine = GeneticSearchEngine(
        space,
        GAConfig(initial_population_size=8, population_size=6, offspring_size=6, num_generations=1, random_seed=0),
    )
    engine.run(evaluator, previous_best=seen, seen_candidate_keys={seen_key}, candidate_key_fn=key_fn)

    assert seen_key not in evaluated
