from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _dense_units(width: int) -> list[AtomicPruneUnit]:
    return [
        AtomicPruneUnit(
            scope_id="dense_scope",
            root_module_path="backbone.conv",
            root_axis="out",
            root_indices=[index],
            source_coupled_unit_ids=[f"coupled_{index}"],
            normalized_score=float(index),
            _stable_id=f"dense_{index}",
        )
        for index in range(width)
    ]


def test_inventory_contains_only_prevalidated_dense_widths() -> None:
    from search.space.legal_width_inventory import build_legal_width_inventory

    inventory = build_legal_width_inventory(
        _dense_units(18),
        minimum_retained_ratio=0.10,
        minimum_retained_channels=4,
        dense_alignment=4,
        per_domain_max_prune_rate=0.80,
    )

    domain = inventory.domains_by_id["backbone.conv::out::dense_scope"]
    assert domain.domain_kind == "dense"
    assert domain.original_width == 18
    assert domain.legal_keep_widths == (4, 8, 12, 16, 18)
    assert all(
        width == 18 or width % 4 == 0 for width in domain.legal_keep_widths
    )
    assert max((18 - width) / 18 for width in domain.legal_keep_widths) <= 0.80
    assert inventory.width_space_hash


def test_protected_domain_has_only_original_width() -> None:
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = _dense_units(8)
    for unit in units:
        unit.protected = True
        unit.protection_reason = "head_contract"
    inventory = build_legal_width_inventory(units)
    domain = next(iter(inventory.domains))

    assert domain.protected is True
    assert domain.prunable is False
    assert domain.legal_keep_widths == (8,)
    assert domain.exclusion_reason == "head_contract"
