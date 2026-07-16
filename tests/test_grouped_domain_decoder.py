from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def test_grouped_decoder_keeps_equal_counts_without_shared_positions() -> None:
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = []
    for absolute in range(16):
        units.append(
            AtomicPruneUnit(
                "grouped_scope",
                "gconv",
                "out",
                [absolute],
                [f"c{absolute}"],
                float(absolute),
                constraints={
                    "grouped_conv": True,
                    "depthwise": False,
                    "groups": 2,
                    "channels_per_group": 8,
                },
                _stable_id=f"g{absolute}",
            )
        )
    inventory = build_legal_width_inventory(
        units,
        grouped_allowed_channels_per_group=(4, 8),
    )
    domain_id = inventory.domain_ids[0]
    domain = inventory.domains_by_id[domain_id]
    assert domain.domain_kind == "grouped"
    assert domain.legal_keep_widths == (4, 8)
    ranking = []
    group_orders = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [12, 13, 14, 15, 8, 9, 10, 11],
    }
    for physical_group, order in group_orders.items():
        ranking.extend(
            {
                "domain_id": domain_id,
                "physical_group_id": physical_group,
                "atomic_unit_id": f"g{absolute}",
                "first_order_score": float(rank),
                "second_order_score": float(rank),
            }
            for rank, absolute in enumerate(order)
        )
    decoded = FixedTaylorWidthDecoder(inventory, ranking).decode({domain_id: 0})

    assert decoded.group_prune_map == {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)}
    assert all(len(values) == 4 for values in decoded.group_keep_map.values())
    assert decoded.group_keep_map[0] != decoded.group_keep_map[1]
    assert decoded.group_keep_map_by_scope == {
        "grouped_scope": decoded.group_keep_map
    }
    assert decoded.group_prune_map_by_scope == {
        "grouped_scope": decoded.group_prune_map
    }
