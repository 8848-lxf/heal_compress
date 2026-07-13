"""Semantic block crossover."""

from __future__ import annotations

import random
from functools import lru_cache

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, repair_genotype
from .immigrants import _repairable_grouped_seed_mask


def _stage_key(name: str) -> str:
    parts = str(name).replace("/", ".").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


@lru_cache(maxsize=32)
def _stage_groups(names: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(_stage_key(name), []).append(name)
    return tuple((stage, tuple(grouped[stage])) for stage in sorted(grouped))


def block_crossover(
    left: CandidateGenotype,
    right: CandidateGenotype,
    space: SearchSpaceSpec,
    rng: random.Random,
) -> CandidateGenotype:
    pruning = dict(left.pruning_genes)
    for _stage, unit_ids in _stage_groups(tuple(space.pruning_unit_ids)):
        if rng.random() < 0.5:
            for unit_id in unit_ids:
                pruning[unit_id] = right.pruning_genes.get(unit_id, 1)
    pruning = _repairable_grouped_seed_mask(space, pruning, rng)
    precision = dict(left.precision_genes)
    for _stage, layer_ids in _stage_groups(tuple(space.precision_gene_ids)):
        if rng.random() < 0.5:
            for layer_id in layer_ids:
                precision[layer_id] = right.precision_genes.get(layer_id, space.default_precision)
    return repair_genotype(CandidateGenotype(pruning, precision, {"created_by": "block_crossover"}), space)
