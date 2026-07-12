"""Population diversity helpers."""

from __future__ import annotations

from itertools import combinations

from ..candidate import CandidateGenotype


def hamming_distance(a: CandidateGenotype, b: CandidateGenotype) -> int:
    prune_keys = set(a.pruning_genes) | set(b.pruning_genes)
    precision_keys = set(a.precision_genes) | set(b.precision_genes)
    return sum(a.pruning_genes.get(key, 1) != b.pruning_genes.get(key, 1) for key in prune_keys) + sum(
        a.precision_genes.get(key, "FP16") != b.precision_genes.get(key, "FP16")
        for key in precision_keys
    )


def average_hamming_distance(population: list[CandidateGenotype]) -> float:
    if len(population) < 2:
        return 0.0
    distances = [hamming_distance(left, right) for left, right in combinations(population, 2)]
    return float(sum(distances) / len(distances))


def min_distance_to_archive(candidate: CandidateGenotype, archive: list[CandidateGenotype]) -> int:
    if not archive:
        return 10**9
    return min(hamming_distance(candidate, row) for row in archive)
