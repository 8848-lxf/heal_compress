"""Deterministic C-centered population construction for constrained Stage A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec
from ..hashing import canonical_json_hash
from .policy import (
    ConstrainedStageAPolicy,
    constrained_resource_admission,
    precision_repair_identity,
    validate_precision_genes,
)


SEED_FAMILY_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("c_neighborhood", 0.35),
    ("hybrid", 0.35),
    ("b_derived", 0.20),
    ("constrained_fresh", 0.10),
)


class ConstrainedPopulationSupplyError(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(
            "constrained_population_supply_exhausted:"
            f"{self.report.get('accepted_seed_count', 0)}<"
            f"{self.report.get('requested_seed_count', 0)}"
        )


def allocate_seed_family_counts(population_size: int) -> dict[str, int]:
    requested = max(0, int(population_size))
    raw = [(name, requested * fraction) for name, fraction in SEED_FAMILY_FRACTIONS]
    counts = {name: int(math.floor(value)) for name, value in raw}
    remaining = requested - sum(counts.values())
    order = sorted(
        enumerate(raw),
        key=lambda item: (-(item[1][1] - math.floor(item[1][1])), item[0]),
    )
    for index in range(remaining):
        counts[order[index][1][0]] += 1
    return counts


def restore_anchor_b_mask(
    anchor_b_fisher_order: Sequence[str],
    *,
    r_mac_by_pruned_count: Mapping[int, float],
    r_mac_floor: float,
    alignment: int,
) -> dict[str, Any]:
    order = [str(value) for value in anchor_b_fisher_order]
    step = max(1, int(alignment))
    eligible = [
        count
        for count, r_mac in r_mac_by_pruned_count.items()
        if int(count) % step == 0
        and 0 <= int(count) <= len(order)
        and float(r_mac) >= float(r_mac_floor)
    ]
    if not eligible:
        raise ValueError("anchor_b_restoration_has_no_R_MAC_eligible_width")
    pruned_count = max(int(value) for value in eligible)
    return {
        "pruned_unit_ids": order[:pruned_count],
        "pruned_unit_count": pruned_count,
        "restored_unit_count": len(order) - pruned_count,
        "R_MAC": float(r_mac_by_pruned_count[pruned_count]),
    }


def genotype_identity_hash(candidate: CandidateGenotype) -> str:
    return canonical_json_hash(
        {
            "pruning_genes": {
                str(key): int(value)
                for key, value in sorted(candidate.pruning_genes.items())
            },
            "precision_genes": {
                str(key): str(value).upper()
                for key, value in sorted(candidate.precision_genes.items())
            },
        }
    )


def pruning_plan_hash(candidate: CandidateGenotype) -> str:
    return canonical_json_hash(
        sorted(
            str(unit_id)
            for unit_id, keep in candidate.pruning_genes.items()
            if int(keep) == 0
        )
    )


@dataclass
class ConstrainedSeedFactory:
    space: SearchSpaceSpec
    policy: ConstrainedStageAPolicy
    int8_allowlist: Sequence[str]
    anchor_c_group_id: str
    local_fisher_order: Sequence[str]
    anchor_b_fisher_order: Sequence[str]
    repair_fn: Callable[[CandidateGenotype], tuple[CandidateGenotype | None, dict[str, Any]]]
    metrics_fn: Callable[[list[CandidateGenotype]], list[dict[str, Any]]]
    alignment: int = 4
    random_seed: int = 42
    proposal_multiplier: int = 20
    max_light_pruned_units: int | None = None

    def _max_pruned_steps(self, order_length: int) -> int:
        step = max(1, int(self.alignment))
        available = max(1, int(order_length) // step)
        if self.max_light_pruned_units is None:
            return available
        return max(
            1,
            min(available, int(self.max_light_pruned_units) // step),
        )

    def _precision_profile(self, subset_code: int) -> dict[str, str]:
        allowlist = [str(value) for value in self.int8_allowlist]
        anchor = str(self.anchor_c_group_id)
        if anchor not in allowlist:
            raise ValueError(f"anchor_C_group_not_allowlisted:{anchor}")
        optional = [value for value in allowlist if value != anchor]
        selected = {anchor}
        for index, group_id in enumerate(optional):
            if int(subset_code) & (1 << index):
                selected.add(group_id)
        return {
            group_id: "INT8" if group_id in selected else "FP16"
            for group_id in self.space.precision_gene_ids
        }

    def _pruned_prefix(self, family: str, attempt: int) -> list[str]:
        step = max(1, int(self.alignment))
        if family == "c_neighborhood":
            return []
        if family == "b_derived":
            order = [str(value) for value in self.anchor_b_fisher_order]
            max_steps = self._max_pruned_steps(len(order))
            pruned_steps = max_steps - (int(attempt) % (max_steps + 1))
            return order[: pruned_steps * step]
        order = [str(value) for value in self.local_fisher_order]
        max_steps = self._max_pruned_steps(len(order))
        if family == "constrained_fresh":
            pruned_steps = 1 + ((int(attempt) * 7) % max_steps)
        else:
            pruned_steps = 1 + (int(attempt) % max_steps)
        pruned_count = pruned_steps * step
        selected = list(order[:pruned_count])
        variant = int(attempt) // max_steps
        if variant <= 0 or not selected or pruned_count >= len(order):
            return selected
        boundary = list(order[pruned_count : min(len(order), pruned_count + 16)])
        if not boundary:
            return selected
        requested_swaps = 1 if family == "hybrid" else 2
        swaps = min(requested_swaps, len(selected), len(boundary))
        offset = (variant * swaps) % len(boundary)
        replacements = [
            boundary[(offset + index) % len(boundary)] for index in range(swaps)
        ]
        return [*selected[:-swaps], *replacements]

    def _subset_code(self, family: str, attempt: int) -> int:
        optional_count = max(0, len(self.int8_allowlist) - 1)
        code_space = max(1, 1 << optional_count)
        if family == "c_neighborhood":
            return int(attempt) % code_space
        order_length = (
            len(self.anchor_b_fisher_order)
            if family == "b_derived"
            else len(self.local_fisher_order)
        )
        prune_variants = self._max_pruned_steps(order_length)
        if family == "b_derived":
            prune_variants += 1
        family_offset = {
            "hybrid": 0,
            "b_derived": 1,
            "constrained_fresh": 2,
        }[family]
        return (3 * (int(attempt) // prune_variants) + family_offset) % code_space

    def _proposal(self, family: str, attempt: int) -> CandidateGenotype:
        pruned = set(self._pruned_prefix(family, attempt))
        return CandidateGenotype(
            pruning_genes={
                unit_id: 0 if unit_id in pruned else 1
                for unit_id in self.space.pruning_unit_ids
            },
            precision_genes=self._precision_profile(self._subset_code(family, attempt)),
            meta={
                "created_by": "constrained_stage_a_seed",
                "seed_family": family,
                "seed_attempt": int(attempt),
                "local_pruning_ranking": "fisher_low_to_high_prefix",
                "local_pruning_variant": (
                    "exact_prefix"
                    if family in {"c_neighborhood", "b_derived"}
                    else "low_sensitivity_boundary_swap"
                ),
            },
        )

    def build(self, population_size: int) -> tuple[list[CandidateGenotype], dict[str, Any]]:
        requested_counts = allocate_seed_family_counts(population_size)
        accepted: list[CandidateGenotype] = []
        seen: set[str] = set()
        proposed: set[str] = set()
        repaired_proposed: set[str] = set()
        counters = {
            "generated_unique_genotype_count": 0,
            "repair_legal_count": 0,
            "R_MAC_eligible_count": 0,
            "INT8_MAC_share_eligible_count": 0,
            "legalized_BOPS_eligible_count": 0,
            "duplicate_rejection_count": 0,
            "infeasible_rejection_count": 0,
            "repair_failure_count": 0,
            "precision_illegal_count": 0,
            "precision_repair_identity_failure_count": 0,
        }
        family_accepted: dict[str, int] = {name: 0 for name, _fraction in SEED_FAMILY_FRACTIONS}
        int8_allowlist = {str(value) for value in self.int8_allowlist}
        for family, _fraction in SEED_FAMILY_FRACTIONS:
            quota = int(requested_counts[family])
            max_attempts = max(quota + 1, quota * max(1, int(self.proposal_multiplier)))
            attempt = 0
            proposal_batch_size = max(64, min(max_attempts, max(256, quota * 2)))
            while attempt < max_attempts and family_accepted[family] < quota:
                proposals: list[CandidateGenotype] = []
                stop = min(max_attempts, attempt + proposal_batch_size)
                while attempt < stop:
                    raw = self._proposal(family, attempt)
                    attempt += 1
                    raw_key = genotype_identity_hash(raw)
                    if raw_key in seen or raw_key in proposed:
                        counters["duplicate_rejection_count"] += 1
                        continue
                    proposed.add(raw_key)
                    counters["generated_unique_genotype_count"] += 1
                    precision_audit = validate_precision_genes(
                        raw.precision_genes,
                        int8_allowlist=int8_allowlist,
                        policy=self.policy,
                    )
                    if not precision_audit["passed"]:
                        counters["precision_illegal_count"] += 1
                        continue
                    repaired, repair_report = self.repair_fn(raw)
                    if repaired is None or str(repair_report.get("status", "")) != "ok":
                        counters["repair_failure_count"] += 1
                        continue
                    counters["repair_legal_count"] += 1
                    identity = precision_repair_identity(raw, repaired)
                    if not identity["passed"]:
                        counters["precision_repair_identity_failure_count"] += 1
                        continue
                    repaired_key = genotype_identity_hash(repaired)
                    if repaired_key in seen or repaired_key in repaired_proposed:
                        counters["duplicate_rejection_count"] += 1
                        continue
                    repaired_proposed.add(repaired_key)
                    proposals.append(
                        CandidateGenotype(
                            dict(repaired.pruning_genes),
                            dict(repaired.precision_genes),
                            {**dict(repaired.meta), "seed_family": family},
                        )
                    )
                metrics = [dict(value) for value in self.metrics_fn(proposals)]
                if len(metrics) != len(proposals):
                    raise RuntimeError(
                        "constrained_seed_metrics_length_mismatch:"
                        f"{len(metrics)}!={len(proposals)}"
                    )
                for candidate, row in zip(proposals, metrics):
                    r_mac = float(row.get("R_MAC", float("-inf")))
                    int8_share = float(row.get("int8_macs_share_full", float("inf")))
                    bops = float(row.get("R_bops_vs_fp32", float("inf")))
                    if r_mac >= float(self.policy.r_mac_floor):
                        counters["R_MAC_eligible_count"] += 1
                    if float(self.policy.int8_mac_share_min) <= int8_share <= float(self.policy.int8_mac_share_max):
                        counters["INT8_MAC_share_eligible_count"] += 1
                    lower, upper = self.policy.bops_interval
                    if lower <= bops <= upper:
                        counters["legalized_BOPS_eligible_count"] += 1
                    admission = constrained_resource_admission(row, self.policy)
                    if not admission["passed"]:
                        counters["infeasible_rejection_count"] += 1
                        continue
                    key = genotype_identity_hash(candidate)
                    if key in seen:
                        counters["duplicate_rejection_count"] += 1
                        continue
                    seen.add(key)
                    accepted.append(candidate)
                    family_accepted[family] += 1
                    if family_accepted[family] >= quota:
                        break

        exact_anchor_c_count = sum(
            not any(int(value) == 0 for value in candidate.pruning_genes.values())
            and {
                group_id
                for group_id, precision in candidate.precision_genes.items()
                if str(precision).upper() == "INT8"
            }
            == {str(self.anchor_c_group_id)}
            for candidate in accepted
        )
        report = {
            "status": "passed" if len(accepted) == int(population_size) else "constrained_population_supply_exhausted",
            "requested_seed_count": int(population_size),
            "accepted_seed_count": len(accepted),
            "requested_family_counts": requested_counts,
            "family_accepted_counts": dict(sorted(family_accepted.items())),
            "exact_anchor_c_count": int(exact_anchor_c_count),
            "unique_genotype_count": len(seen),
            "unique_pruning_plan_count": len({pruning_plan_hash(candidate) for candidate in accepted}),
            "initial_physical_hash_note": (
                "no-prune C-neighborhood intentionally shares one physical plan; "
                "actual physical/deployment uniqueness is enforced before Top-5 admission"
            ),
            **counters,
        }
        if len(accepted) != int(population_size) or any(
            family_accepted[name] != requested_counts[name]
            for name in requested_counts
        ):
            raise ConstrainedPopulationSupplyError(report)
        if exact_anchor_c_count != 1:
            report["status"] = "exact_anchor_C_multiplicity_failure"
            raise ConstrainedPopulationSupplyError(report)
        return accepted, report
