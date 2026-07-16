from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def test_precision_only_change_keeps_structure_identity() -> None:
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.operators.legal_width_mutation import mutate_precision_only
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], float(index),
            _stable_id=f"u{index}",
        )
        for index in range(8)
    ]
    inventory = build_legal_width_inventory(units, dense_alignment=4)
    domain_id = inventory.domain_ids[0]
    decoder = FixedTaylorWidthDecoder(
        inventory,
        [
            {
                "domain_id": domain_id,
                "physical_group_id": 0,
                "atomic_unit_id": f"u{index}",
                "first_order_score": float(index),
                "second_order_score": float(index),
            }
            for index in range(8)
        ],
    )
    before = LegalWidthGenotype({domain_id: 0}, {"pg": "FP16"})
    after = mutate_precision_only(
        before,
        precision_actions={"pg": ("FP16", "INT8")},
        rng=random.Random(7),
        mutation_rate=1.0,
    )
    before_decoded = decoder.decode(before.width_genes)
    after_decoded = decoder.decode(after.width_genes)

    assert before.width_genes == after.width_genes
    assert before_decoded.pruned_unit_ids == after_decoded.pruned_unit_ids
    assert before_decoded.group_mask == after_decoded.group_mask
    assert before_decoded.structure_hash == after_decoded.structure_hash
    assert before.precision_hash != after.precision_hash
