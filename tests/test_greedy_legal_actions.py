from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _space():
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope",
            root,
            "out",
            [index],
            [f"{root}.c{index}"],
            float(index),
            _stable_id=f"{root}.u{index}",
        )
        for root in ("conv0", "conv1")
        for index in range(8)
    ]
    inventory = build_legal_width_inventory(
        units,
        dense_alignment=2,
        minimum_retained_channels=2,
        per_domain_max_prune_rate=0.75,
    )
    precision_actions = {
        "fixed": ("FP16",),
        "pg0": ("INT8", "FP32", "FP16"),
    }
    genotype = LegalWidthGenotype(
        {
            domain.domain_id: len(domain.legal_keep_widths) - 1
            for domain in inventory.domains
        },
        {"fixed": "FP16", "pg0": "FP32"},
    )
    genotype.validate(inventory, precision_actions)
    return inventory, precision_actions, genotype


def test_greedy_width_action_moves_one_adjacent_index_only() -> None:
    from search.greedy.legal_actions import enumerate_legal_actions

    inventory, precision_actions, parent = _space()
    actions = enumerate_legal_actions(
        parent,
        inventory=inventory,
        precision_actions=precision_actions,
    )
    width_action, child = next(row for row in actions if row[0].kind == "width")

    changed = [
        key
        for key in parent.width_genes
        if parent.width_genes[key] != child.width_genes[key]
    ]
    assert changed == [width_action.gene_id]
    assert child.width_genes[changed[0]] == parent.width_genes[changed[0]] - 1
    assert child.precision_genes == parent.precision_genes
    child.validate(inventory, precision_actions)


def test_greedy_precision_action_preserves_width_vector() -> None:
    from search.greedy.legal_actions import enumerate_legal_actions

    inventory, precision_actions, parent = _space()
    action, child = next(
        row
        for row in enumerate_legal_actions(
            parent,
            inventory=inventory,
            precision_actions=precision_actions,
        )
        if row[0].kind == "precision"
    )

    assert child.width_genes == parent.width_genes
    assert child.width_vector_hash == parent.width_vector_hash
    assert action.gene_id == "pg0"
    assert action.from_value == "FP32"
    assert action.to_value == "FP16"
    assert child.precision_genes["fixed"] == "FP16"
    child.validate(inventory, precision_actions)


def test_greedy_precision_action_uses_one_deployable_level_per_step() -> None:
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.greedy.legal_actions import enumerate_legal_actions

    inventory, precision_actions, parent = _space()
    fp16_parent = LegalWidthGenotype(
        parent.width_genes,
        {"fixed": "FP16", "pg0": "FP16"},
    )
    precision_rows = [
        row
        for row in enumerate_legal_actions(
            fp16_parent,
            inventory=inventory,
            precision_actions=precision_actions,
        )
        if row[0].kind == "precision"
    ]

    assert len(precision_rows) == 1
    action, child = precision_rows[0]
    assert (action.from_value, action.to_value) == ("FP16", "INT8")
    assert child.precision_genes["pg0"] == "INT8"
    assert child.width_vector_hash == fp16_parent.width_vector_hash


def test_greedy_actions_are_deterministic_and_stably_sorted() -> None:
    from search.greedy.legal_actions import enumerate_legal_actions

    inventory, precision_actions, parent = _space()
    first = enumerate_legal_actions(
        parent,
        inventory=inventory,
        precision_actions=precision_actions,
    )
    second = enumerate_legal_actions(
        parent,
        inventory=inventory,
        precision_actions=precision_actions,
    )

    assert [row[0].action_id for row in first] == sorted(
        row[0].action_id for row in first
    )
    assert [row[0].action_id for row in first] == [row[0].action_id for row in second]
    assert [row[1].genotype_hash for row in first] == [
        row[1].genotype_hash for row in second
    ]
