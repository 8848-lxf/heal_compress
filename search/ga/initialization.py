"""Population initialization for the joint discrete GA."""

from __future__ import annotations

import random

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, repair_genotype
from ..quantization_space.codec import forced_int8_group_genotype
from .immigrants import random_immigrant
from .mutation import mutate_candidate


def baseline_candidate(space: SearchSpaceSpec) -> CandidateGenotype:
    return CandidateGenotype(
        pruning_genes=(
            {} if space.pruning_domains
            else {unit_id: 1 for unit_id in space.pruning_unit_ids}
        ),
        precision_genes={gene_id: "FP32" for gene_id in space.precision_gene_ids},
        meta={"created_by": "baseline_full_fp32"},
        pruning_width_genes={domain.domain_id: domain.original_width for domain in space.pruning_domains},
    )


def compressed_seed(space: SearchSpaceSpec, keep_probability: float, precision: str) -> CandidateGenotype:
    if space.pruning_domains:
        width_genes = {}
        for domain in space.pruning_domains:
            target = float(domain.original_width) * float(keep_probability)
            width_genes[domain.domain_id] = min(
                domain.legal_widths,
                key=lambda width: (abs(float(width) - target), -int(width)),
            )
        return repair_genotype(
            CandidateGenotype(
                pruning_genes={},
                precision_genes={gene_id: precision for gene_id in space.precision_gene_ids},
                meta={"created_by": f"domain_width_seed_{keep_probability:.2f}_{precision}"},
                pruning_width_genes=width_genes,
            ),
            space,
        )
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
    seed_candidates: list[CandidateGenotype] | None = None,
    seeded_population_ratio: float = 0.90,
) -> list[CandidateGenotype]:
    population: list[CandidateGenotype] = [repair_genotype(baseline_candidate(space), space)]
    population.append(
        repair_genotype(
            CandidateGenotype(
                pruning_genes=(
                    {} if space.pruning_domains
                    else {unit_id: 1 for unit_id in space.pruning_unit_ids}
                ),
                precision_genes={gene_id: "FP16" for gene_id in space.precision_gene_ids},
                meta={"created_by": "baseline_fp16_deploy"},
                pruning_width_genes={
                    domain.domain_id: domain.original_width for domain in space.pruning_domains
                },
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
    seeds = [repair_genotype(row, space) for row in (seed_candidates or [])]
    population.extend(seeds)
    if previous_best is not None:
        population.append(repair_genotype(previous_best, space))
    seeded_index = 0
    seeded_target = min(
        population_size,
        max(
            len(population),
            int(round(population_size * float(seeded_population_ratio))),
        ),
    )
    while seeds and len(population) < seeded_target:
        base = seeds[seeded_index % len(seeds)]
        action_count = 1 + ((seeded_index // max(len(seeds), 1)) % 4)
        population.append(
            mutate_candidate(
                base,
                space,
                rng,
                prune_mutation_rate=1.0,
                precision_mutation_rate=1.0,
                action_count=action_count,
                adjacent_precision=True,
            )
        )
        seeded_index += 1
    while len(population) < population_size:
        population.append(random_immigrant(space, rng))
    return population[:population_size]
