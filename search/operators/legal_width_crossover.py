"""Whole-domain structure and whole-group precision crossover."""

from __future__ import annotations

import random

from ..encoding.legal_width_genotype import LegalWidthGenotype


def _validate_parent_keys(
    left: LegalWidthGenotype, right: LegalWidthGenotype
) -> None:
    if set(left.width_genes) != set(right.width_genes):
        raise ValueError("width_crossover_domain_mismatch")
    if set(left.precision_genes) != set(right.precision_genes):
        raise ValueError("precision_crossover_group_mismatch")


def crossover_width_only(
    left: LegalWidthGenotype,
    right: LegalWidthGenotype,
    *,
    rng: random.Random,
) -> LegalWidthGenotype:
    _validate_parent_keys(left, right)
    return LegalWidthGenotype(
        width_genes={
            domain_id: (
                left.width_genes[domain_id]
                if rng.random() < 0.5
                else right.width_genes[domain_id]
            )
            for domain_id in sorted(left.width_genes)
        },
        precision_genes=dict(left.precision_genes),
        meta={**left.meta, "created_by": "width_domain_crossover"},
    )


def crossover_precision_only(
    left: LegalWidthGenotype,
    right: LegalWidthGenotype,
    *,
    rng: random.Random,
) -> LegalWidthGenotype:
    _validate_parent_keys(left, right)
    return LegalWidthGenotype(
        width_genes=dict(left.width_genes),
        precision_genes={
            group_id: (
                left.precision_genes[group_id]
                if rng.random() < 0.5
                else right.precision_genes[group_id]
            )
            for group_id in sorted(left.precision_genes)
        },
        meta={**left.meta, "created_by": "precision_group_crossover"},
    )


def crossover_legal_width(
    left: LegalWidthGenotype,
    right: LegalWidthGenotype,
    *,
    rng: random.Random,
) -> LegalWidthGenotype:
    width_child = crossover_width_only(left, right, rng=rng)
    return crossover_precision_only(width_child, right, rng=rng)
