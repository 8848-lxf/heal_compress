"""Semantic block crossover."""

from __future__ import annotations

import random

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, repair_genotype
from .immigrants import _repairable_grouped_seed_mask


def _stage_key(name: str) -> str:
    parts = str(name).replace("/", ".").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def block_crossover(
    left: CandidateGenotype,
    right: CandidateGenotype,
    space: SearchSpaceSpec,
    rng: random.Random,
) -> CandidateGenotype:
    pruning = dict(left.pruning_genes)
    for stage in sorted({_stage_key(name) for name in space.pruning_unit_ids}):
        if rng.random() < 0.5:
            for unit_id in space.pruning_unit_ids:
                if _stage_key(unit_id) == stage:
                    pruning[unit_id] = right.pruning_genes.get(unit_id, 1)
    pruning = _repairable_grouped_seed_mask(space, pruning, rng)
    precision = dict(left.precision_genes)
    for stage in sorted({_stage_key(name) for name in space.precision_gene_ids}):
        if rng.random() < 0.5:
            for layer_id in space.precision_gene_ids:
                if _stage_key(layer_id) == stage:
                    precision[layer_id] = right.precision_genes.get(layer_id, space.default_precision)
    return repair_genotype(CandidateGenotype(pruning, precision, {"created_by": "block_crossover"}), space)
