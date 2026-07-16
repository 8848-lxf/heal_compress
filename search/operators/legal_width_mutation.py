"""Independent legal-width and precision mutation operators."""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from ..candidate import normalize_precision
from ..encoding.legal_width_genotype import LegalWidthGenotype
from ..space.legal_width_inventory import LegalWidthInventory


def mutate_width_only(
    genotype: LegalWidthGenotype,
    *,
    inventory: LegalWidthInventory,
    rng: random.Random,
    mutation_rate: float,
    direction: str = "auto",
) -> LegalWidthGenotype:
    """Move selected domains by one legal-width index only."""

    if direction not in {"auto", "compress", "restore"}:
        raise ValueError(f"unsupported_width_mutation_direction:{direction}")
    widths = dict(genotype.width_genes)
    for domain in inventory.domains:
        if rng.random() >= float(mutation_rate):
            continue
        current = int(widths[domain.domain_id])
        neighbors = [
            index
            for index in (current - 1, current + 1)
            if 0 <= index < len(domain.legal_keep_widths)
        ]
        if direction == "compress":
            neighbors = [index for index in neighbors if index < current]
        elif direction == "restore":
            neighbors = [index for index in neighbors if index > current]
        if neighbors:
            widths[domain.domain_id] = rng.choice(neighbors)
    return LegalWidthGenotype(
        width_genes=widths,
        precision_genes=dict(genotype.precision_genes),
        meta={
            **genotype.meta,
            "created_by": "width_only_mutation",
            "width_mutation_direction": direction,
        },
    )


def mutate_precision_only(
    genotype: LegalWidthGenotype,
    *,
    precision_actions: Mapping[str, Sequence[str]],
    rng: random.Random,
    mutation_rate: float,
) -> LegalWidthGenotype:
    precision = dict(genotype.precision_genes)
    for group_id, actions in sorted(precision_actions.items()):
        if rng.random() >= float(mutation_rate):
            continue
        allowed = tuple(normalize_precision(value) for value in actions)
        current = precision[str(group_id)]
        choices = tuple(value for value in allowed if value != current)
        if choices:
            precision[str(group_id)] = rng.choice(choices)
    return LegalWidthGenotype(
        width_genes=dict(genotype.width_genes),
        precision_genes=precision,
        meta={**genotype.meta, "created_by": "precision_only_mutation"},
    )
