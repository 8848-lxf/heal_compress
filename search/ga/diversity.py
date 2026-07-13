"""Population diversity helpers."""

from __future__ import annotations

from collections import Counter

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
    size = len(population)
    pair_count = size * (size - 1) // 2
    total_distance = 0
    pruning_keys = set().union(*(candidate.pruning_genes for candidate in population))
    precision_keys = set().union(*(candidate.precision_genes for candidate in population))
    for key in pruning_keys:
        ones = sum(int(candidate.pruning_genes.get(key, 1)) != 0 for candidate in population)
        total_distance += ones * (size - ones)
    for key in precision_keys:
        counts = Counter(candidate.precision_genes.get(key, "FP16") for candidate in population)
        equal_pairs = sum(count * (count - 1) // 2 for count in counts.values())
        total_distance += pair_count - equal_pairs
    return float(total_distance / pair_count)


def min_distance_to_archive(candidate: CandidateGenotype, archive: list[CandidateGenotype]) -> int:
    if not archive:
        return 10**9
    return min(hamming_distance(candidate, row) for row in archive)
