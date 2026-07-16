from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _decoder():
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope",
            "conv",
            "out",
            [index],
            [f"c{index}"],
            float(7 - index),
            _stable_id=f"u{index}",
        )
        for index in range(8)
    ]
    inventory = build_legal_width_inventory(units, dense_alignment=4)
    ranking = [
        {
            "domain_id": inventory.domain_ids[0],
            "physical_group_id": 0,
            "atomic_unit_id": f"u{index}",
            "first_order_score": float(index + 1),
            "second_order_score": float(index + 10),
        }
        for index in range(8)
    ]
    return FixedTaylorWidthDecoder(inventory, ranking), inventory.domain_ids[0]


def test_same_width_vector_decodes_same_mask_and_structure_hash() -> None:
    decoder, domain_id = _decoder()
    # Legal widths are (4, 8); index 0 retains four.
    first = decoder.decode({domain_id: 0})
    second = decoder.decode({domain_id: 0})

    assert first.pruned_unit_ids == ("u0", "u1", "u2", "u3")
    assert first.group_mask == second.group_mask
    assert first.structure_hash == second.structure_hash
    assert first.physical_plan_hash == second.physical_plan_hash


def test_decoder_api_has_no_precision_argument() -> None:
    import inspect

    decoder, _domain_id = _decoder()
    assert list(inspect.signature(decoder.decode).parameters) == ["width_genes"]


def test_structure_hash_describes_mask_not_ranking_lineage() -> None:
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], 0.0,
            _stable_id=f"u{index}",
        )
        for index in range(8)
    ]
    inventory = build_legal_width_inventory(units, dense_alignment=4)
    domain_id = inventory.domain_ids[0]
    first = FixedTaylorWidthDecoder(
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
    reversed_ranking = FixedTaylorWidthDecoder(
        inventory,
        [
            {
                "domain_id": domain_id,
                "physical_group_id": 0,
                "atomic_unit_id": f"u{index}",
                "first_order_score": float(8 - index),
                "second_order_score": float(8 - index),
            }
            for index in range(8)
        ],
    )

    full_first = first.decode({domain_id: 1})
    full_reversed = reversed_ranking.decode({domain_id: 1})

    assert first.ranking_hash != reversed_ranking.ranking_hash
    assert full_first.pruned_unit_ids == full_reversed.pruned_unit_ids == ()
    assert full_first.structure_hash == full_reversed.structure_hash
    assert full_first.physical_plan_hash == full_reversed.physical_plan_hash
