from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def test_more_aggressive_legal_width_mask_contains_less_aggressive_mask() -> None:
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
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
            for index in range(16)
        ],
    )

    decoded = [decoder.decode({domain_id: index}) for index in range(4)]
    pruned = [set(row.pruned_unit_ids) for row in decoded]
    assert pruned[0] >= pruned[1] >= pruned[2] >= pruned[3]
