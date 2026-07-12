from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_local_pruning_domains_are_keyed_by_root_module_and_axis() -> None:
    from pruning.types import AtomicPruneUnit
    from search.pruning_space.local_domains import build_local_pruning_domains

    units = [
        AtomicPruneUnit("scope_a", "block.conv", "out", [0], ["cu0"], 0.2),
        AtomicPruneUnit("scope_a", "block.conv", "out", [1], ["cu1"], 0.1),
        AtomicPruneUnit("scope_b", "block.other", "in", [0], ["cu2"], 0.3),
    ]

    domains = build_local_pruning_domains(units)

    assert [domain.domain_id for domain in domains] == ["block.conv::out", "block.other::in"]
    assert domains[0].ordered_unit_ids == ("apu_",) or domains[0].ordered_unit_ids[0].startswith("apu_")
    assert [row.root_module_path for row in domains] == ["block.conv", "block.other"]


def test_local_domain_width_choices_are_dense_aligned_to_four() -> None:
    from search.pruning_space.local_domains import legal_dense_widths

    assert legal_dense_widths(original_width=18, minimum_retained_ratio=0.10, alignment=4) == (4, 8, 12, 16, 18)
