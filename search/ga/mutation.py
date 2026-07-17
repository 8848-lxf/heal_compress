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
    action_count: int | None = None,
    adjacent_precision: bool = False,
) -> CandidateGenotype:
    pruning = dict(candidate.pruning_genes)
    width_genes = dict(candidate.pruning_width_genes)
    selected_pruning: set[str] | None = None
    selected_precision: set[str] | None = None
    if action_count is not None:
        actions: list[tuple[str, str]] = [
            ("pruning", gene_id) for gene_id in space.pruning_gene_ids
        ] + [("precision", gene_id) for gene_id in space.precision_gene_ids]
        weights = [
            max(float(prune_mutation_rate), 1.0e-12)
            if kind == "pruning"
            else max(float(precision_mutation_rate), 1.0e-12)
            for kind, _gene_id in actions
        ]
        chosen: list[tuple[str, str]] = []
        for _ in range(min(max(0, int(action_count)), len(actions))):
            position = rng.choices(range(len(actions)), weights=weights, k=1)[0]
            chosen.append(actions.pop(position))
            weights.pop(position)
        selected_pruning = {gene_id for kind, gene_id in chosen if kind == "pruning"}
        selected_precision = {gene_id for kind, gene_id in chosen if kind == "precision"}

    if space.pruning_domains:
        pruning = {}
        for domain in space.pruning_domains:
            current = int(width_genes.get(domain.domain_id, domain.original_width))
            mutate = (
                domain.domain_id in selected_pruning
                if selected_pruning is not None
                else rng.random() < prune_mutation_rate
            )
            if not mutate or len(domain.legal_widths) <= 1:
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
            mutate = (
                unit_id in selected_pruning
                if selected_pruning is not None
                else rng.random() < prune_mutation_rate
            )
            if mutate:
                pruning[unit_id] = 1 - int(pruning.get(unit_id, 1))
        pruning = _repairable_grouped_seed_mask(space, pruning, rng)
    precision_values = ["FP32", "FP16", "INT8"]
    allowed_by_group = {
        group.group_id: tuple(str(value).upper() for value in group.allowed_precisions)
        for group in space.quantization_groups
    }
    precision = dict(candidate.precision_genes)
    for layer_id in space.precision_gene_ids:
        mutate = (
            layer_id in selected_precision
            if selected_precision is not None
            else rng.random() < precision_mutation_rate
        )
        if mutate:
            current = precision.get(layer_id, space.default_precision)
            allowed = [
                value
                for value in precision_values
                if value in allowed_by_group.get(layer_id, tuple(precision_values))
            ]
            if adjacent_precision and current in allowed:
                index = allowed.index(current)
                choices = [
                    allowed[position]
                    for position in (index - 1, index + 1)
                    if 0 <= position < len(allowed)
                ]
            else:
                choices = [value for value in allowed if value != current]
            if not choices:
                continue
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
