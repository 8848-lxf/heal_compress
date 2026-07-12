"""Population-level helpers."""

from __future__ import annotations

from ..candidate import CandidateGenotype


def dedupe_population(population: list[CandidateGenotype]) -> list[CandidateGenotype]:
    seen: set[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, str], ...]]] = set()
    result = []
    for candidate in population:
        key = (tuple(sorted(candidate.pruning_genes.items())), tuple(sorted(candidate.precision_genes.items())))
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
