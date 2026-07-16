"""Constraint-aware random immigrant utilities."""

from __future__ import annotations

import random

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, repair_genotype


def _repairable_grouped_seed_mask(space: SearchSpaceSpec, pruning: dict[str, int], rng: random.Random) -> dict[str, int]:
    grouped: dict[str, dict[int, list[str]]] = {}
    for unit_id, metadata in space.pruning_unit_metadata.items():
        constraints = dict(metadata.get("constraints") or {})
        if not (constraints.get("grouped_conv") and not constraints.get("depthwise")):
            continue
        root_indices = list(metadata.get("root_indices") or [])
        if not root_indices:
            continue
        width = int(constraints.get("channels_per_group") or constraints.get("channels_per_group_before") or 0)
        if width <= 0:
            continue
        scope = str(metadata.get("scope_id", ""))
        group, _local = divmod(int(root_indices[0]), width)
        grouped.setdefault(scope, {}).setdefault(group, []).append(unit_id)
    for groups in grouped.values():
        for unit_ids in groups.values():
            kept = [unit_id for unit_id in unit_ids if int(pruning.get(unit_id, 1)) == 1]
            if len(kept) >= 4:
                continue
            candidates = [unit_id for unit_id in unit_ids if int(pruning.get(unit_id, 1)) == 0]
            rng.shuffle(candidates)
            for unit_id in candidates[: max(0, 4 - len(kept))]:
                pruning[unit_id] = 1
    return pruning


def immigrant_ratio_for_generation(
    stagnant_generations: int,
    *,
    base_ratio: float,
    stagnation_generations: int,
    stagnant_ratio: float,
) -> float:
    return float(stagnant_ratio if stagnant_generations >= stagnation_generations else base_ratio)


def random_immigrant(space: SearchSpaceSpec, rng: random.Random, *, keep_probability: float | None = None) -> CandidateGenotype:
    keep_p = 0.5 if keep_probability is None else float(keep_probability)
    precision_values = ["FP32", "FP16", "INT8"]
    if space.pruning_domains:
        pruning = {unit_id: 1 for unit_id in space.pruning_unit_ids}
        width_genes = {}
        for domain in space.pruning_domains:
            target = float(domain.original_width) * keep_p
            ranked = sorted(
                domain.legal_widths,
                key=lambda width: (abs(float(width) - target), rng.random()),
            )
            # Mix the nearest choices so immigrants remain diverse while every
            # genotype is legal without a later alignment repair.
            pool = ranked[: min(3, len(ranked))]
            width_genes[domain.domain_id] = int(rng.choice(pool))
    else:
        pruning = {
            unit_id: 1 if rng.random() < keep_p else 0
            for unit_id in space.pruning_unit_ids
        }
        pruning = _repairable_grouped_seed_mask(space, pruning, rng)
        width_genes = {}
    genotype = CandidateGenotype(
        pruning_genes=pruning,
        precision_genes={
            layer_id: rng.choice(precision_values)
            for layer_id in space.precision_gene_ids
        },
        meta={"created_by": "random_immigrant"},
        pruning_width_genes=width_genes,
    )
    return repair_genotype(genotype, space)


def make_immigrants(space: SearchSpaceSpec, count: int, rng: random.Random) -> list[CandidateGenotype]:
    result = []
    for idx in range(max(0, int(count))):
        keep_probability = 0.25 + 0.5 * ((idx % 5) / 4.0)
        result.append(random_immigrant(space, rng, keep_probability=keep_probability))
    return result
