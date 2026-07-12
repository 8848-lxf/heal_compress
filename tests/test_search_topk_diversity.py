from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision
from search.stage1.topk_selector import ProxyCandidateRecord, TopKConfig, select_stage1_topk
from search.ga.diversity import average_hamming_distance
from search.ga.immigrants import immigrant_ratio_for_generation


def _record(idx: int, score: float, genes: dict[str, int]) -> ProxyCandidateRecord:
    return ProxyCandidateRecord(
        candidate_hash=f"h{idx}",
        genotype=CandidateGenotype(pruning_genes=genes, precision_genes={"l": "FP16"}),
        phenotype=CandidatePhenotype(
            pruned_unit_ids=sorted(k for k, v in genes.items() if v == 0),
            precision_profile={"l": PrecisionDecision("FP16", "FP16", "")},
        ),
        F1=score,
        metrics={"complete": True},
    )


def test_top5_is_exploitation_diversity_exploration() -> None:
    records = [
        _record(0, 0.10, {"a": 1, "b": 1, "c": 1}),
        _record(1, 0.11, {"a": 1, "b": 1, "c": 0}),
        _record(2, 0.12, {"a": 1, "b": 0, "c": 1}),
        _record(3, 0.13, {"a": 0, "b": 0, "c": 0}),
        _record(4, 0.14, {"a": 0, "b": 1, "c": 0}),
        _record(5, 0.15, {"a": 0, "b": 0, "c": 1}),
    ]
    selected = select_stage1_topk(
        records,
        real_eval_hashes=set(),
        archive_genotypes=[records[0].genotype],
        config=TopKConfig(topk_real=5, exploitation_count=3, diversity_count=1, exploration_count=1),
    )

    assert [row.role for row in selected] == ["exploitation", "exploitation", "exploitation", "diversity", "exploration"]
    assert len({row.record.candidate_hash for row in selected}) == 5


def test_topk_skips_already_evaluated_and_backfills() -> None:
    records = [_record(i, float(i), {"a": i % 2, "b": (i + 1) % 2}) for i in range(6)]
    selected = select_stage1_topk(
        records,
        real_eval_hashes={"h0", "h1"},
        archive_genotypes=[],
        config=TopKConfig(topk_real=3, exploitation_count=2, diversity_count=1, exploration_count=0),
    )

    assert all(row.record.candidate_hash not in {"h0", "h1"} for row in selected)
    assert len(selected) == 3


def test_random_immigrant_ratio_increases_when_stagnant() -> None:
    assert immigrant_ratio_for_generation(2, base_ratio=0.08, stagnation_generations=8, stagnant_ratio=0.25) == 0.08
    assert immigrant_ratio_for_generation(8, base_ratio=0.08, stagnation_generations=8, stagnant_ratio=0.25) == 0.25


def test_average_hamming_distance_detects_collapsed_population() -> None:
    diverse = [
        CandidateGenotype({"a": 1, "b": 1}, {"l": "FP16"}),
        CandidateGenotype({"a": 0, "b": 0}, {"l": "INT8"}),
    ]
    collapsed = [
        CandidateGenotype({"a": 1, "b": 1}, {"l": "FP16"}),
        CandidateGenotype({"a": 1, "b": 1}, {"l": "FP16"}),
    ]

    assert average_hamming_distance(diverse) > average_hamming_distance(collapsed)
