from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _inventory():
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], float(index),
            _stable_id=f"u{index}",
        )
        for index in range(8)
    ]
    return build_legal_width_inventory(units, dense_alignment=4)


def test_normal_candidate_is_validated_without_repair_invocation() -> None:
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.stage1.exception_repair import ExceptionOnlyRepairMonitor

    inventory = _inventory()
    candidate = LegalWidthGenotype(
        {inventory.domain_ids[0]: 0},
        {"pg": "FP16"},
    )
    monitor = ExceptionOnlyRepairMonitor()
    monitor.observe_normal_candidate(
        candidate,
        inventory=inventory,
        precision_actions={"pg": ("FP16", "INT8")},
    )

    assert monitor.report() == {
        "normal_candidate_count": 1,
        "repair_invocation_count": 0,
        "repair_invocation_rate": 0.0,
        "repair_reason_histogram": {},
        "repair_changed_structure_count": 0,
        "repair_changed_precision_count": 0,
    }


def test_legacy_mask_repair_never_adds_requested_outside_pruning() -> None:
    from search.stage1.exception_repair import (
        ExceptionOnlyRepairMonitor,
        repair_legacy_mask_to_legal_width,
    )

    inventory = _inventory()
    domain_id = inventory.domain_ids[0]
    raw = {f"u{index}": 0 if index in {0, 2, 4, 6, 7} else 1 for index in range(8)}
    monitor = ExceptionOnlyRepairMonitor()
    result = repair_legacy_mask_to_legal_width(
        raw,
        inventory=inventory,
        conditional_costs={f"u{index}": float(index) for index in range(8)},
        precision_genes={"pg": "INT8"},
        monitor=monitor,
        reason="legacy_mask_import",
    )

    requested_pruned = {key for key, keep in raw.items() if keep == 0}
    repaired_pruned = {key for key, keep in result.group_mask.items() if keep == 0}
    assert repaired_pruned <= requested_pruned
    assert result.keep_widths[domain_id] >= sum(raw.values())
    assert result.precision_genes == {"pg": "INT8"}
    assert monitor.report()["repair_invocation_count"] == 1
    assert monitor.report()["repair_changed_precision_count"] == 0


def test_legal_width_ga_normal_candidates_never_invoke_repair() -> None:
    from search.canonicalization import SearchSpaceSpec
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.ga.engine import GAConfig, GeneticSearchEngine
    from search.stage1.exception_repair import ExceptionOnlyRepairMonitor

    inventory = _inventory()
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
    actions = {"pg": ("FP16", "INT8")}
    space = SearchSpaceSpec(
        pruning_unit_ids=list(inventory.unit_ids),
        precision_layer_ids=["pg"],
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space=actions,
    )
    monitor = ExceptionOnlyRepairMonitor()
    evaluated = []

    def evaluator(candidate, _generation):
        monitor.observe_normal_candidate(
            candidate, inventory=inventory, precision_actions=actions
        )
        evaluated.append(candidate.genotype_hash)
        return {"F1": float(candidate.width_genes[domain_id])}

    GeneticSearchEngine(
        space,
        GAConfig(
            initial_population_size=8,
            population_size=6,
            offspring_size=6,
            num_generations=3,
            random_seed=11,
        ),
    ).run(evaluator)

    assert len(evaluated) == 20
    assert monitor.report()["normal_candidate_count"] == 20
    assert monitor.report()["repair_invocation_count"] == 0
