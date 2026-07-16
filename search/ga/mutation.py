"""Adaptive mutation helpers."""

from __future__ import annotations

import random

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, repair_genotype
from .immigrants import _repairable_grouped_seed_mask


def mutate_candidate(
    candidate: CandidateGenotype,
    space: SearchSpaceSpec,
    rng: random.Random,
    *,
    prune_mutation_rate: float,
    precision_mutation_rate: float,
) -> CandidateGenotype:
    pruning = dict(candidate.pruning_genes)
    width_genes = dict(candidate.pruning_width_genes)
    if space.pruning_domains:
        pruning = {}
        for domain in space.pruning_domains:
            current = int(width_genes.get(domain.domain_id, domain.original_width))
            if rng.random() >= prune_mutation_rate or len(domain.legal_widths) <= 1:
                width_genes[domain.domain_id] = current
                continue
            index = domain.legal_widths.index(current)
            neighbors = [
                domain.legal_widths[position]
                for position in (index - 1, index + 1)
                if 0 <= position < len(domain.legal_widths)
            ]
            width_genes[domain.domain_id] = int(rng.choice(neighbors))
    else:
        for unit_id in space.pruning_unit_ids:
            if unit_id in space.protected_pruning_unit_ids:
                pruning[unit_id] = 1
                continue
            if rng.random() < prune_mutation_rate:
                pruning[unit_id] = 1 - int(pruning.get(unit_id, 1))
        pruning = _repairable_grouped_seed_mask(space, pruning, rng)
    precision_values = ["FP32", "FP16", "INT8"]
    precision = dict(candidate.precision_genes)
    for layer_id in space.precision_gene_ids:
        if rng.random() < precision_mutation_rate:
            current = precision.get(layer_id, space.default_precision)
            choices = [value for value in precision_values if value != current]
            precision[layer_id] = rng.choice(choices)
    return repair_genotype(
        CandidateGenotype(
            pruning,
            precision,
            {"created_by": "mutation"},
            pruning_width_genes=width_genes,
        ),
        space,
    )


def adapt_mutation_rate(base_rate: float, diversity: float, gene_count: int) -> float:
    if gene_count <= 0:
        return float(base_rate)
    normalized = diversity / max(float(gene_count), 1.0)
    if normalized < 0.10:
        return min(0.50, base_rate * 2.0)
    if normalized < 0.20:
        return min(0.35, base_rate * 1.5)
    return float(base_rate)
