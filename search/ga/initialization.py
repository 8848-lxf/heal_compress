"""Population initialization for the joint discrete GA."""

from __future__ import annotations

import random

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, repair_genotype
from ..encoding.legal_width_genotype import (
    LegalWidthGenotype,
    random_legal_width_genotype,
)
from ..quantization_space.codec import forced_int8_group_genotype
from .immigrants import random_immigrant


def baseline_candidate(space: SearchSpaceSpec) -> CandidateGenotype:
    return CandidateGenotype(
        pruning_genes={unit_id: 1 for unit_id in space.pruning_unit_ids},
        precision_genes={gene_id: "FP32" for gene_id in space.precision_gene_ids},
        meta={"created_by": "baseline_full_fp32"},
    )


def compressed_seed(space: SearchSpaceSpec, keep_probability: float, precision: str) -> CandidateGenotype:
    keep_count = max(1, int(round(len(space.pruning_unit_ids) * keep_probability)))
    keep = set(space.pruning_unit_ids[:keep_count]) | set(space.protected_pruning_unit_ids)
    return repair_genotype(
        CandidateGenotype(
            pruning_genes={unit_id: 1 if unit_id in keep else 0 for unit_id in space.pruning_unit_ids},
            precision_genes={gene_id: precision for gene_id in space.precision_gene_ids},
            meta={"created_by": f"compressed_seed_{keep_probability:.2f}_{precision}"},
        ),
        space,
    )


def initialize_population(
    space: SearchSpaceSpec,
    population_size: int,
    rng: random.Random,
    *,
    previous_elite: list[CandidateGenotype] | None = None,
    previous_best: CandidateGenotype | None = None,
) -> list[CandidateGenotype]:
    population: list[CandidateGenotype] = [repair_genotype(baseline_candidate(space), space)]
    population.append(
        repair_genotype(
            CandidateGenotype(
                pruning_genes={unit_id: 1 for unit_id in space.pruning_unit_ids},
                precision_genes={gene_id: "FP16" for gene_id in space.precision_gene_ids},
                meta={"created_by": "baseline_fp16_deploy"},
            ),
            space,
        )
    )
    if space.quantization_groups:
        population.extend(
            [
                repair_genotype(forced_int8_group_genotype(space.quantization_groups, minimum_int8_macs_ratio=0.10), space),
                repair_genotype(forced_int8_group_genotype(space.quantization_groups, minimum_int8_macs_ratio=0.20), space),
            ]
        )
    population.extend(
        [
            compressed_seed(space, 0.75, "FP16"),
            compressed_seed(space, 0.50, "FP16"),
            compressed_seed(space, 0.50, "INT8"),
        ]
    )
    population.extend(repair_genotype(row, space) for row in (previous_elite or []))
    if previous_best is not None:
        population.append(repair_genotype(previous_best, space))
    while len(population) < population_size:
        population.append(random_immigrant(space, rng))
    return population[:population_size]


def initialize_legal_width_population(
    space: SearchSpaceSpec,
    population_size: int,
    rng: random.Random,
    *,
    previous_elite: list[LegalWidthGenotype] | None = None,
    previous_best: LegalWidthGenotype | None = None,
) -> list[LegalWidthGenotype]:
    if space.structure_gene_type != "legal_keep_width":
        raise ValueError("initialize_legal_width_population_requires_legal_width_space")
    inventory = space.legal_width_inventory
    full_width = {
        domain.domain_id: len(domain.legal_keep_widths) - 1
        for domain in inventory.domains
    }
    default_precision = {
        group_id: (
            space.default_precision
            if space.default_precision in actions
            else actions[0]
        )
        for group_id, actions in space.precision_action_space.items()
    }
    population: list[LegalWidthGenotype] = [
        LegalWidthGenotype(
            full_width,
            default_precision,
            {"created_by": "legal_width_original"},
        )
    ]
    population.extend(previous_elite or [])
    if previous_best is not None:
        population.append(previous_best)
    while len(population) < int(population_size):
        population.append(
            random_legal_width_genotype(
                inventory,
                precision_actions=space.precision_action_space,
                rng=rng,
            )
        )
    return population[: int(population_size)]
