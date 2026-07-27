"""Strict Stage-1/Stage-2 GA with V1/V2/V3 real anchors.

This module is the deployment-closed formal path.  It intentionally does not
call the historical repair operators: every genotype has the complete mutable
schema, mutation moves by one legal state, crossover copies only the same
locus, and candidates outside the exact BOPS band are rejected unchanged.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..hashing import candidate_hash


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StrictGAConfig:
    target_bops_retention: float
    tolerance_abs: float = 0.005
    population_size: int = 64
    offspring_size: int = 64
    generations: int = 10
    stage2_new_candidate_quota: int = 5
    random_seed: int = 0
    stage2_accuracy_tolerance: float = 0.005
    generation_contract: str = "formal_gen10"

    def __post_init__(self) -> None:
        contracts = {"formal_gen10": 10, "formal_gen5": 5}
        expected = contracts.get(str(self.generation_contract))
        if expected is None:
            raise ValueError(
                f"unsupported_formal_ga_generation_contract:{self.generation_contract}"
            )
        if self.generations != expected:
            if self.generation_contract == "formal_gen10":
                raise ValueError(
                    f"formal_ga_generations_must_equal_10:{self.generations}"
                )
            raise ValueError(
                "formal_ga_generations_contract_mismatch:"
                f"contract={self.generation_contract}:"
                f"expected={expected}:actual={self.generations}"
            )
        if self.population_size != 64 or self.offspring_size != 64:
            raise ValueError("formal_ga_population_and_offspring_must_equal_64")
        if self.stage2_new_candidate_quota != 5:
            raise ValueError("formal_ga_stage2_quota_must_equal_5")


def validate_genotype_schema(
    genotype: CandidateGenotype,
    space: SearchSpaceSpec,
) -> None:
    """Fail closed on missing, fixed, unknown, or illegal loci."""

    expected_widths = set(space.pruning_gene_ids)
    actual_widths = set(genotype.pruning_width_genes)
    if actual_widths != expected_widths:
        raise ValueError(
            f"ga_width_schema_mismatch:missing={sorted(expected_widths-actual_widths)}:"
            f"extra={sorted(actual_widths-expected_widths)}"
        )
    if genotype.pruning_genes:
        raise ValueError("ga_atomic_pruning_mask_forbidden_with_domain_widths")
    domains = {str(row.domain_id): row for row in space.pruning_domains}
    for locus, width in genotype.pruning_width_genes.items():
        if int(width) not in {int(value) for value in domains[locus].legal_widths}:
            raise ValueError(f"ga_illegal_width:{locus}:{width}")

    expected_precision = set(space.precision_gene_ids)
    actual_precision = set(genotype.precision_genes)
    if actual_precision != expected_precision:
        raise ValueError(
            f"ga_precision_schema_mismatch:missing={sorted(expected_precision-actual_precision)}:"
            f"extra={sorted(actual_precision-expected_precision)}"
        )
    groups = {str(row.group_id): row for row in space.quantization_groups}
    fixed = set(space.constant_precision_group_ids)
    if actual_precision & fixed:
        raise ValueError(f"ga_fixed_precision_locus_present:{sorted(actual_precision & fixed)}")
    for locus, precision in genotype.precision_genes.items():
        if str(precision) not in set(groups[locus].allowed_precisions):
            raise ValueError(f"ga_illegal_precision:{locus}:{precision}")


def phenotype_identity(
    genotype: CandidateGenotype,
    space: SearchSpaceSpec,
) -> dict[str, str]:
    validate_genotype_schema(genotype, space)
    phenotype = canonicalize_candidate(genotype, space)
    return {
        "canonical_structure_hash": _stable_hash(genotype.pruning_width_genes),
        "physical_structure_hash": _stable_hash(
            {
                "widths": genotype.pruning_width_genes,
                "pruned_units": phenotype.pruned_unit_ids,
            }
        ),
        "precision_map_hash": _stable_hash({
            "mutable": genotype.precision_genes,
            "derived": phenotype.metadata.get("derived_precision_group_profile", {}),
        }),
        "complete_phenotype_hash": candidate_hash(phenotype, space),
    }


def adjacent_mutation(
    genotype: CandidateGenotype,
    space: SearchSpaceSpec,
    rng: random.Random,
) -> CandidateGenotype:
    """Move exactly one mutable locus to one neighboring legal state."""

    validate_genotype_schema(genotype, space)
    width = dict(genotype.pruning_width_genes)
    precision = dict(genotype.precision_genes)
    choices: list[tuple[str, str, tuple[Any, ...]]] = []
    for domain in space.pruning_domains:
        states = tuple(int(value) for value in domain.legal_widths)
        if len(states) > 1:
            choices.append(("width", str(domain.domain_id), states))
    for group in space.quantization_groups:
        if group.group_id in space.precision_gene_ids:
            choices.append(
                ("precision", str(group.group_id), tuple(group.allowed_precisions))
            )
    rng.shuffle(choices)
    for kind, locus, states in choices:
        current = width[locus] if kind == "width" else precision[locus]
        index = states.index(current)
        neighbors = [
            states[position]
            for position in (index - 1, index + 1)
            if 0 <= position < len(states)
        ]
        if not neighbors:
            continue
        value = rng.choice(neighbors)
        if kind == "width":
            width[locus] = int(value)
        else:
            precision[locus] = str(value)
        child = CandidateGenotype(
            pruning_width_genes=width,
            precision_genes=precision,
            meta={"created_by": "strict_adjacent_mutation", "repair_count": 0},
        )
        validate_genotype_schema(child, space)
        return child
    raise RuntimeError("ga_no_mutable_locus")


def same_locus_crossover(
    left: CandidateGenotype,
    right: CandidateGenotype,
    space: SearchSpaceSpec,
    rng: random.Random,
) -> CandidateGenotype:
    """Copy parent values only at the identical domain/precision locus."""

    validate_genotype_schema(left, space)
    validate_genotype_schema(right, space)
    widths = {
        locus: (
            left.pruning_width_genes[locus]
            if rng.random() < 0.5
            else right.pruning_width_genes[locus]
        )
        for locus in space.pruning_gene_ids
    }
    precision = {
        locus: (
            left.precision_genes[locus]
            if rng.random() < 0.5
            else right.precision_genes[locus]
        )
        for locus in space.precision_gene_ids
    }
    child = CandidateGenotype(
        pruning_width_genes=widths,
        precision_genes=precision,
        meta={"created_by": "strict_same_locus_crossover", "repair_count": 0},
    )
    validate_genotype_schema(child, space)
    return child


class UnifiedTaylorStage1Evaluator:
    """Exact BOPS hard gate plus conservative gate/WQ/AQ candidate score."""

    def __init__(
        self,
        space: SearchSpaceSpec,
        *,
        baseline: CandidateGenotype,
        structure_proxy: Any,
        weight_proxy: Any,
        activation_cache: Any,
        bops_evaluator: Callable[[Any], Mapping[str, Any]],
        size_evaluator: Callable[[Any], Mapping[str, Any]],
        target: float,
        tolerance_abs: float = 0.005,
        enforce_bops_hard_gate: bool = True,
        activation_taylor_fitness_weight: float = 1.0,
    ) -> None:
        validate_genotype_schema(baseline, space)
        self.space = space
        self.baseline = baseline
        self.structure_proxy = structure_proxy
        self.weight_proxy = weight_proxy
        self.activation_cache = activation_cache
        self.bops_evaluator = bops_evaluator
        self.size_evaluator = size_evaluator
        self.target = float(target)
        self.tolerance_abs = float(tolerance_abs)
        self.enforce_bops_hard_gate = bool(enforce_bops_hard_gate)
        self.activation_taylor_fitness_weight = float(
            activation_taylor_fitness_weight
        )
        if (
            not math.isfinite(self.activation_taylor_fitness_weight)
            or self.activation_taylor_fitness_weight < 0.0
        ):
            raise ValueError("ga_activation_taylor_fitness_weight_invalid")
        self._baseline_phenotype = canonicalize_candidate(baseline, space)
        self._groups = {str(row.group_id): row for row in space.quantization_groups}
        self._cache: dict[str, dict[str, Any]] = {}

    def __call__(self, genotype: CandidateGenotype) -> dict[str, Any]:
        validate_genotype_schema(genotype, self.space)
        identity = phenotype_identity(genotype, self.space)
        cache_key = str(identity["complete_phenotype_hash"])
        cached = self._cache.get(cache_key)
        if cached is not None:
            # The immutable numerical fields are shared, while the genotype is
            # restored explicitly so callers never observe stale metadata.
            return {**cached, "genotype": genotype, "stage1_cache_hit": True}
        candidate_phenotype = canonicalize_candidate(genotype, self.space)
        bops = dict(self.bops_evaluator(candidate_phenotype))
        retention = float(bops["R_bops_vs_fp32"])
        deviation = abs(retention - self.target)
        feasible = deviation <= self.tolerance_abs
        if not feasible and self.enforce_bops_hard_gate:
            result = {
                **identity,
                "J_struct_gate": None,
                "J_WQ": None,
                "J_AQ": None,
                "J_AQ_fitness_contribution": None,
                "J_total": float("inf"),
                "F1": float("inf"),
                "R_bops_vs_fp32": retention,
                "bops_deviation": deviation,
                "bops_feasible": False,
                "bops_hard_gate_passed": False,
                "R_parameter_retention": None,
                "mixed_weight_retention": None,
                "structural_repair_count": 0,
                "precision_repair_count": 0,
                "budget_repair_count": 0,
                "joint_taylor_used_for_fitness": False,
                "cross_residual_used_for_fitness": False,
                "legacy_weight_taylor_used_for_fitness": False,
                "activation_taylor_fitness_weight": (
                    self.activation_taylor_fitness_weight
                ),
                "activation_taylor_used_for_fitness": bool(
                    self.activation_taylor_fitness_weight != 0.0
                ),
                "genotype": genotype,
                "stage1_cache_hit": False,
                "taylor_evaluated_after_bops_hard_gate": False,
            }
            self._cache[cache_key] = {
                key: value for key, value in result.items() if key != "genotype"
            }
            return result

        structure_state = CandidateGenotype(
            pruning_width_genes=dict(genotype.pruning_width_genes),
            precision_genes=dict(self.baseline.precision_genes),
            meta={"created_by": "stage1_structure_state"},
        )
        structure_phenotype = canonicalize_candidate(structure_state, self.space)
        structural = self.structure_proxy.pruning_action_breakdown(
            self._baseline_phenotype, structure_phenotype
        )
        j_struct = float(structural["delta_J_prune"])
        j_wq = 0.0
        j_aq = 0.0
        current = structure_state
        current_phenotype = structure_phenotype
        for locus in self.space.precision_gene_ids:
            target_precision = genotype.precision_genes[locus]
            states = tuple(self._groups[locus].allowed_precisions)
            current_precision = current.precision_genes[locus]
            start = states.index(current_precision)
            stop = states.index(target_precision)
            if stop < start:
                raise ValueError(
                    f"ga_precision_candidate_above_baseline:{locus}:{target_precision}"
                )
            for position in range(start + 1, stop + 1):
                precision = dict(current.precision_genes)
                precision[locus] = states[position]
                successor = CandidateGenotype(
                    pruning_width_genes=dict(genotype.pruning_width_genes),
                    precision_genes=precision,
                    meta={"created_by": "stage1_adjacent_precision_accumulation"},
                )
                successor_phenotype = canonicalize_candidate(successor, self.space)
                j_wq += float(
                    self.weight_proxy.weight_quantization_action_breakdown(
                        current_phenotype, successor_phenotype
                    )["delta_J_WQ"]
                )
                j_aq += float(
                    self.activation_cache.action_breakdown(
                        current_phenotype, successor_phenotype
                    )["delta_J_AQ"]
                )
                current = successor
                current_phenotype = successor_phenotype
        j_aq_fitness = self.activation_taylor_fitness_weight * j_aq
        j_total = j_struct + j_wq + j_aq_fitness
        if not math.isfinite(j_total) or min(j_struct, j_wq, j_aq) < 0.0:
            raise RuntimeError("ga_stage1_taylor_invalid")
        size = dict(self.size_evaluator(candidate_phenotype))
        result = {
            **identity,
            "J_struct_gate": j_struct,
            "J_WQ": j_wq,
            "J_AQ": j_aq,
            "J_AQ_fitness_contribution": j_aq_fitness,
            "J_total": j_total,
            "F1": j_total,
            "R_bops_vs_fp32": retention,
            "bops_deviation": deviation,
            "bops_feasible": feasible,
            "bops_hard_gate_passed": feasible,
            "R_parameter_retention": float(size["R_parameter_retention"]),
            "mixed_weight_retention": float(size["R_size_vs_fp32"]),
            "structural_repair_count": 0,
            "precision_repair_count": 0,
            "budget_repair_count": 0,
            "joint_taylor_used_for_fitness": False,
            "cross_residual_used_for_fitness": False,
            "legacy_weight_taylor_used_for_fitness": False,
            "activation_taylor_fitness_weight": (
                self.activation_taylor_fitness_weight
            ),
            "activation_taylor_used_for_fitness": bool(
                self.activation_taylor_fitness_weight != 0.0
            ),
            "genotype": genotype,
            "stage1_cache_hit": False,
            "taylor_evaluated_after_bops_hard_gate": True,
        }
        self._cache[cache_key] = {key: value for key, value in result.items() if key != "genotype"}
        return result


def _taylor_equal(left: float, right: float) -> bool:
    return abs(left - right) <= max(1.0e-12, 1.0e-8 * max(abs(left), abs(right)))


def rank_stage1(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Taylor-primary ordering; compression is never rewarded."""

    feasible = [dict(row) for row in rows if bool(row.get("bops_feasible", False))]

    def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
        jl, jr = float(left["J_total"]), float(right["J_total"])
        if not _taylor_equal(jl, jr):
            return -1 if jl < jr else 1
        left_key = (
            float(left["bops_deviation"]),
            -float(left["R_parameter_retention"]),
            -float(left["mixed_weight_retention"]),
            str(left["complete_phenotype_hash"]),
        )
        right_key = (
            float(right["bops_deviation"]),
            -float(right["R_parameter_retention"]),
            -float(right["mixed_weight_retention"]),
            str(right["complete_phenotype_hash"]),
        )
        return (left_key > right_key) - (left_key < right_key)

    return sorted(feasible, key=functools.cmp_to_key(compare))


@dataclass(frozen=True)
class Stage2Result:
    complete_phenotype_hash: str
    genotype: CandidateGenotype
    status: str
    map: float | None = None
    p50_ms: float | None = None
    requested_realized_exact: bool = False
    evaluated: int = 0
    skipped: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def deployable(self) -> bool:
        return (
            self.status == "ok"
            and self.map is not None
            and self.p50_ms is not None
            and math.isfinite(float(self.map))
            and math.isfinite(float(self.p50_ms))
            and self.requested_realized_exact
            and self.skipped == 0
        )


def score_stage2(
    result: Stage2Result,
    *,
    greedy_map: float,
    greedy_p50_ms: float,
    accuracy_tolerance: float = 0.005,
) -> dict[str, Any]:
    eligible = bool(
        result.deployable
        and float(result.map) >= float(greedy_map) - float(accuracy_tolerance)
    )
    if not eligible:
        return {"eligible": False, "F_S2": float("inf")}
    accuracy = max(
        -1.0,
        min(1.0, (float(greedy_map) - float(result.map)) / accuracy_tolerance),
    )
    latency = float(result.p50_ms) / float(greedy_p50_ms)
    return {
        "eligible": True,
        "A_0.005": accuracy,
        "latency_ratio": latency,
        "F_S2": 0.2 * accuracy + 0.8 * latency,
    }


def select_generation_winner(
    results: Sequence[Stage2Result],
    *,
    greedy_map: float,
    greedy_p50_ms: float,
    stage1_by_hash: Mapping[str, Mapping[str, Any]],
) -> tuple[Stage2Result | None, list[dict[str, Any]]]:
    scored = []
    for result in results:
        score = score_stage2(
            result, greedy_map=greedy_map, greedy_p50_ms=greedy_p50_ms
        )
        scored.append({"result": result, **score})
    eligible = [row for row in scored if row["eligible"]]
    if not eligible:
        return None, scored
    highest_map = max(float(row["result"].map) for row in eligible)
    lowest_p50 = min(float(row["result"].p50_ms) for row in eligible)
    dominant = [
        row
        for row in eligible
        if float(row["result"].map) == highest_map
        and float(row["result"].p50_ms) == lowest_p50
    ]
    if dominant:
        winner = min(
            dominant, key=lambda row: row["result"].complete_phenotype_hash
        )
        return winner["result"], scored

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        result = row["result"]
        stage1 = stage1_by_hash[result.complete_phenotype_hash]
        return (
            float(row["F_S2"]),
            -float(result.map),
            float(result.p50_ms),
            -float(stage1["R_parameter_retention"]),
            -float(stage1["mixed_weight_retention"]),
            float(stage1["bops_deviation"]),
            result.complete_phenotype_hash,
        )

    return min(eligible, key=key)["result"], scored


@dataclass
class GlobalRealAnchors:
    greedy: Stage2Result
    lowest_f: Stage2Result | None = None
    highest_map: Stage2Result | None = None
    lowest_p50: Stage2Result | None = None

    def update(
        self,
        results: Sequence[Stage2Result],
        *,
        greedy_map: float,
        greedy_p50_ms: float,
    ) -> None:
        eligible = [
            row
            for row in results
            if score_stage2(
                row, greedy_map=greedy_map, greedy_p50_ms=greedy_p50_ms
            )["eligible"]
        ]
        if not eligible:
            return
        pool = {
            row.complete_phenotype_hash: row
            for row in (
                [value for value in (self.lowest_f, self.highest_map, self.lowest_p50) if value]
                + eligible
            )
        }
        values = list(pool.values())
        self.lowest_f = min(
            values,
            key=lambda row: (
                float(
                    score_stage2(
                        row, greedy_map=greedy_map, greedy_p50_ms=greedy_p50_ms
                    )["F_S2"]
                ),
                row.complete_phenotype_hash,
            ),
        )
        self.highest_map = min(
            values,
            key=lambda row: (-float(row.map), row.complete_phenotype_hash),
        )
        self.lowest_p50 = min(
            values,
            key=lambda row: (float(row.p50_ms), row.complete_phenotype_hash),
        )

    def unique(self) -> list[Stage2Result]:
        rows = [
            self.greedy,
            *[
                row
                for row in (self.lowest_f, self.highest_map, self.lowest_p50)
                if row is not None
            ],
        ]
        return list({row.complete_phenotype_hash: row for row in rows}.values())


def select_new_stage2_candidates(
    ranked_stage1: Sequence[Mapping[str, Any]],
    *,
    evaluated_hashes: set[str],
    quota: int = 5,
) -> list[dict[str, Any]]:
    if quota > 5:
        raise ValueError("ga_stage2_new_candidate_quota_exceeds_five")
    selected = []
    seen_physical: set[tuple[str, str]] = set()
    for row in ranked_stage1:
        identity = str(row["complete_phenotype_hash"])
        if identity in evaluated_hashes:
            continue
        physical = (
            str(row["physical_structure_hash"]), str(row["precision_map_hash"])
        )
        if physical in seen_physical:
            continue
        seen_physical.add(physical)
        selected.append(dict(row))
        if len(selected) >= quota:
            break
    return selected


class StrictStage12V3Runner:
    """Run generation 0 plus the explicitly contracted evolution generations."""

    def __init__(
        self,
        space: SearchSpaceSpec,
        config: StrictGAConfig,
        *,
        stage1_evaluator: Callable[[CandidateGenotype], Mapping[str, Any]],
        stage2_evaluator: Callable[[CandidateGenotype, int], Stage2Result],
    ) -> None:
        self.space = space
        self.config = config
        self.stage1_evaluator = stage1_evaluator
        self.stage2_evaluator = stage2_evaluator
        self.rng = random.Random(config.random_seed)

    def run(
        self,
        initial_population: Sequence[CandidateGenotype],
        *,
        greedy_anchor: Stage2Result,
        generation_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if len(initial_population) != self.config.population_size:
            raise ValueError("ga_initial_population_size_mismatch")
        initial_hashes: set[str] = set()
        for candidate in initial_population:
            validate_genotype_schema(candidate, self.space)
            identity = phenotype_identity(candidate, self.space)["complete_phenotype_hash"]
            if identity in initial_hashes:
                raise ValueError("ga_initial_population_duplicate_phenotype")
            initial_hashes.add(identity)
            if not bool(self.stage1_evaluator(candidate)["bops_feasible"]):
                raise ValueError("ga_initial_population_outside_bops_band")
        greedy_stage1 = dict(self.stage1_evaluator(greedy_anchor.genotype))
        if not greedy_stage1["bops_feasible"]:
            raise ValueError("ga_greedy_anchor_outside_budget_band")
        population = list(initial_population)
        if not any(
            phenotype_identity(row, self.space)["complete_phenotype_hash"]
            == greedy_anchor.complete_phenotype_hash
            for row in population
        ):
            population[-1] = greedy_anchor.genotype
        anchors = GlobalRealAnchors(greedy=greedy_anchor)
        evaluated: dict[str, Stage2Result] = {
            greedy_anchor.complete_phenotype_hash: greedy_anchor
        }
        history: list[dict[str, Any]] = [
            {
                "generation": 0,
                "initialization_only": True,
                "counted_as_evolution_generation": False,
                "population_size": len(population),
                "population_unique_count": len(initial_hashes),
                "population_legal_count": len(population),
                "population_in_band_count": len(population),
                "stage2_new_candidate_count": 0,
            }
        ]
        if generation_callback:
            generation_callback(history[-1])

        termination = f"completed_generation_{self.config.generations}"
        for generation in range(1, self.config.generations + 1):
            scored_population = [dict(self.stage1_evaluator(row)) for row in population]
            ranked_population = rank_stage1(scored_population)
            offspring: list[CandidateGenotype] = []
            offspring_hashes: set[str] = set()
            attempts = 0
            maximum_attempts = self.config.offspring_size * 500
            while len(offspring) < self.config.offspring_size and attempts < maximum_attempts:
                attempts += 1
                left = self.rng.choice(ranked_population)["genotype"]
                right = self.rng.choice(ranked_population)["genotype"]
                child = same_locus_crossover(left, right, self.space, self.rng)
                child = adjacent_mutation(child, self.space, self.rng)
                metrics = dict(self.stage1_evaluator(child))
                if not metrics["bops_feasible"]:
                    continue
                identity = str(metrics["complete_phenotype_hash"])
                if identity in offspring_hashes:
                    continue
                offspring_hashes.add(identity)
                offspring.append(child)
            if len(offspring) != self.config.offspring_size:
                raise RuntimeError(
                    "ga_offspring_population_not_exactly_64:"
                    f"generated={len(offspring)}:attempts={attempts}"
                )

            combined_by_hash: dict[str, dict[str, Any]] = {}
            for candidate in [*population, *offspring]:
                metrics = dict(self.stage1_evaluator(candidate))
                if metrics["bops_feasible"]:
                    combined_by_hash[str(metrics["complete_phenotype_hash"])] = metrics
            combined_by_hash[greedy_anchor.complete_phenotype_hash] = greedy_stage1
            ranked = rank_stage1(combined_by_hash.values())
            new_rows = select_new_stage2_candidates(
                ranked,
                evaluated_hashes=set(evaluated),
                quota=self.config.stage2_new_candidate_quota,
            )
            # Stage-2 candidates are independent once Stage-1 has fixed their
            # deterministic order.  A deployment evaluator may therefore
            # execute the batch on isolated GPUs.  Results MUST be returned in
            # ``new_rows`` order so scheduling latency cannot perturb GA state.
            evaluate_many = getattr(self.stage2_evaluator, "evaluate_many", None)
            if callable(evaluate_many):
                new_results = list(
                    evaluate_many(
                        [row["genotype"] for row in new_rows], generation
                    )
                )
                if len(new_results) != len(new_rows):
                    raise RuntimeError("ga_stage2_batch_result_count_mismatch")
                expected_hashes = [
                    str(row["complete_phenotype_hash"]) for row in new_rows
                ]
                realized_hashes = [
                    result.complete_phenotype_hash for result in new_results
                ]
                if realized_hashes != expected_hashes:
                    raise RuntimeError("ga_stage2_batch_result_order_mismatch")
            else:
                new_results = [
                    self.stage2_evaluator(row["genotype"], generation)
                    for row in new_rows
                ]
            for result in new_results:
                evaluated[result.complete_phenotype_hash] = result
            stage1_by_hash = {
                str(row["complete_phenotype_hash"]): row for row in ranked
            }
            generation_winner, stage2_rows = select_generation_winner(
                new_results,
                greedy_map=float(greedy_anchor.map),
                greedy_p50_ms=float(greedy_anchor.p50_ms),
                stage1_by_hash=stage1_by_hash,
            )
            anchors.update(
                new_results,
                greedy_map=float(greedy_anchor.map),
                greedy_p50_ms=float(greedy_anchor.p50_ms),
            )
            feedback_rows = []
            if generation_winner is not None:
                feedback_rows.append(generation_winner)
            feedback_rows.extend(
                row for row in anchors.unique()
                if row.complete_phenotype_hash != greedy_anchor.complete_phenotype_hash
            )
            feedback_by_hash = {
                row.complete_phenotype_hash: row for row in feedback_rows
            }
            injected_feedback_hashes = sorted(feedback_by_hash)
            forced_results = [greedy_anchor, *feedback_by_hash.values()]
            forced = [row.genotype for row in forced_results]
            forced_hashes = {
                phenotype_identity(row, self.space)["complete_phenotype_hash"]
                for row in forced
            }
            next_population = list(forced)
            for row in ranked:
                identity = str(row["complete_phenotype_hash"])
                if identity in forced_hashes:
                    continue
                next_population.append(row["genotype"])
                if len(next_population) == self.config.population_size:
                    break
            population = next_population[: self.config.population_size]
            survivor_hashes = {
                phenotype_identity(row, self.space)["complete_phenotype_hash"]
                for row in population
            }
            if (
                len(population) != self.config.population_size
                or len(survivor_hashes) != self.config.population_size
            ):
                raise RuntimeError(
                    "ga_survivor_population_not_exactly_64_unique:"
                    f"size={len(population)}:unique={len(survivor_hashes)}"
                )
            survivor_metrics = [self.stage1_evaluator(row) for row in population]
            if not all(bool(row["bops_feasible"]) for row in survivor_metrics):
                raise RuntimeError("ga_survivor_outside_bops_band")
            record = {
                "generation": generation,
                "initialization_only": False,
                "counted_as_evolution_generation": True,
                "offspring_requested": self.config.offspring_size,
                "offspring_generated": len(offspring),
                "offspring_unique_count": len(offspring_hashes),
                "offspring_attempts": attempts,
                "survivor_size": len(population),
                "survivor_unique_count": len(survivor_hashes),
                "survivor_legal_count": len(population),
                "survivor_in_band_count": len(population),
                "stage2_new_candidate_count": len(new_results),
                "stage2_new_candidate_hashes": [
                    row.complete_phenotype_hash for row in new_results
                ],
                "generation_winner_hash": (
                    generation_winner.complete_phenotype_hash
                    if generation_winner is not None
                    else None
                ),
                "real_feedback_injected_next_generation": bool(
                    injected_feedback_hashes
                ),
                "real_feedback_injected_hashes": injected_feedback_hashes,
                "greedy_anchor_retained": any(
                    phenotype_identity(row, self.space)["complete_phenotype_hash"]
                    == greedy_anchor.complete_phenotype_hash
                    for row in population
                ),
                "global_anchor_hashes": [
                    row.complete_phenotype_hash for row in anchors.unique()
                ],
                "stage2_rows": stage2_rows,
            }
            history.append(record)
            if generation_callback:
                generation_callback(record)

        return {
            "generation_zero_counted": False,
            "completed_evolution_generations": sum(
                bool(row["counted_as_evolution_generation"]) for row in history
            ),
            "termination_reason": termination,
            "history": history,
            "anchors": anchors,
            "evaluated": evaluated,
            "final_population": population,
            "formal_ga_repair_counts": {
                "structural": 0,
                "precision": 0,
                "budget": 0,
            },
        }


__all__ = [
    "GlobalRealAnchors",
    "Stage2Result",
    "StrictGAConfig",
    "StrictStage12V3Runner",
    "UnifiedTaylorStage1Evaluator",
    "adjacent_mutation",
    "phenotype_identity",
    "rank_stage1",
    "same_locus_crossover",
    "score_stage2",
    "select_generation_winner",
    "select_new_stage2_candidates",
    "validate_genotype_schema",
]
