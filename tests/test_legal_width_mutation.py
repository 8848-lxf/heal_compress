from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _setup():
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], float(index),
            _stable_id=f"u{index}",
        )
        for index in range(16)
    ]
    inventory = build_legal_width_inventory(units, dense_alignment=4)
    domain_id = inventory.domain_ids[0]
    return inventory, LegalWidthGenotype({domain_id: 2}, {"pg": "FP16"}), domain_id


def test_width_mutation_moves_only_to_adjacent_legal_index() -> None:
    from search.operators.legal_width_mutation import mutate_width_only

    inventory, genotype, domain_id = _setup()
    mutated = mutate_width_only(
        genotype,
        inventory=inventory,
        rng=random.Random(3),
        mutation_rate=1.0,
    )

    assert abs(mutated.width_genes[domain_id] - genotype.width_genes[domain_id]) == 1
    assert mutated.precision_genes == genotype.precision_genes
    mutated.validate(inventory, {"pg": ("FP16", "INT8")})


def test_budget_direction_can_force_smaller_or_larger_width() -> None:
    from search.operators.legal_width_mutation import mutate_width_only

    inventory, genotype, domain_id = _setup()
    smaller = mutate_width_only(
        genotype,
        inventory=inventory,
        rng=random.Random(4),
        mutation_rate=1.0,
        direction="compress",
    )
    larger = mutate_width_only(
        genotype,
        inventory=inventory,
        rng=random.Random(4),
        mutation_rate=1.0,
        direction="restore",
    )

    assert smaller.width_genes[domain_id] == genotype.width_genes[domain_id] - 1
    assert larger.width_genes[domain_id] == genotype.width_genes[domain_id] + 1

