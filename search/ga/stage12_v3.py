"""Deployment-closed GA Stage-1/Stage-2 and V1-V3 real-anchor contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cmp_to_key
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping, Sequence

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate, repair_genotype
from ..hashing import candidate_hash


def _mapping_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(sorted(value.items())),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Stage1Policy:
    target_bops_retention: float
    absolute_bops_tolerance: float = 0.005
    numerical_tie_absolute: float = 1.0e-12
    numerical_tie_relative: float = 1.0e-8
    taylor_equivalence_band: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_bops_retention <= 1.0:
            raise ValueError("stage1_target_bops_retention_invalid")
        if self.absolute_bops_tolerance < 0.0:
            raise ValueError("stage1_absolute_bops_tolerance_invalid")
        if self.taylor_equivalence_band < 0.0:
            raise ValueError("stage1_taylor_equivalence_band_invalid")


def strict_taylor_tie(left: float, right: float, policy: Stage1Policy) -> bool:
    threshold = max(
        float(policy.numerical_tie_absolute),
        float(policy.numerical_tie_relative) * max(abs(left), abs(right)),
    )
    if abs(left - right) <= threshold:
        return True
    return bool(
        policy.taylor_equivalence_band > 0.0
        and abs(left - right)
        <= policy.taylor_equivalence_band * max(abs(left), abs(right), 1.0e-30)
    )


def _validate_complete_chromosome(
    genotype: CandidateGenotype, space: SearchSpaceSpec
) -> CandidateGenotype:
    expected_widths = {str(domain.domain_id) for domain in space.pruning_domains}
    supplied_widths = set(genotype.pruning_width_genes)
    if supplied_widths != expected_widths:
        raise ValueError(
            "ga_structure_chromosome_incomplete:"
            f"missing={sorted(expected_widths-supplied_widths)}:"
            f"extra={sorted(supplied_widths-expected_widths)}"
        )
    expected_precision = set(space.precision_gene_ids)
    supplied_precision = set(genotype.precision_genes)
    if supplied_precision != expected_precision:
        raise ValueError(
            "ga_precision_chromosome_incomplete_or_fixed_locus_present:"
            f"missing={sorted(expected_precision-supplied_precision)}:"
            f"extra={sorted(supplied_precision-expected_precision)}"
        )
    validated = repair_genotype(genotype, space)
    if (
        validated.pruning_width_genes != genotype.pruning_width_genes
        or validated.precision_genes != genotype.precision_genes
    ):
        raise RuntimeError("ga_candidate_would_require_repair")
    return validated


def adjacent_legal_mutation(
    candidate: CandidateGenotype,
    space: SearchSpaceSpec,
    rng: random.Random,
    *,
    locus_kind: str | None = None,
) -> CandidateGenotype:
    """Mutate exactly one locus to an adjacent legal state without repair."""

    source = _validate_complete_chromosome(candidate, space)
    actions: list[tuple[str, str, tuple[Any, ...]]] = []
    for domain in space.pruning_domains:
        states = tuple(int(value) for value in domain.legal_widths)
        if len(states) > 1 and locus_kind in {None, "structure"}:
            actions.append(("structure", str(domain.domain_id), states))
    groups = {group.group_id: group for group in space.quantization_groups}
    precision_order = ("FP32", "FP16", "INT8")
    for gene_id in space.precision_gene_ids:
        states = tuple(
            value
            for value in precision_order
            if value in groups[gene_id].allowed_precisions
        )
        if len(states) > 1 and locus_kind in {None, "precision"}:
            actions.append(("precision", gene_id, states))
    if not actions:
        raise RuntimeError("ga_adjacent_mutation_no_legal_locus")
    kind, gene_id, states = rng.choice(actions)
    widths = dict(source.pruning_width_genes)
    precision = dict(source.precision_genes)
    current = widths[gene_id] if kind == "structure" else precision[gene_id]
    index = states.index(current)
    neighbors = [position for position in (index - 1, index + 1) if 0 <= position < len(states)]
    if not neighbors:
        raise RuntimeError("ga_adjacent_mutation_no_neighbor")
    successor = states[rng.choice(neighbors)]
    if kind == "structure":
        widths[gene_id] = int(successor)
    else:
        precision[gene_id] = str(successor)
    return _validate_complete_chromosome(
        CandidateGenotype(
            pruning_genes={},
            precision_genes=precision,
            pruning_width_genes=widths,
            meta={"created_by": "stage12_v3_adjacent_mutation", "repair_count": 0},
        ),
        space,
    )


def same_locus_crossover(
    left: CandidateGenotype,
    right: CandidateGenotype,
    space: SearchSpaceSpec,
    rng: random.Random,
) -> CandidateGenotype:
    """Exchange only homologous legal loci from two complete chromosomes."""

    lhs = _validate_complete_chromosome(left, space)
    rhs = _validate_complete_chromosome(right, space)
    widths = {
        gene_id: (lhs.pruning_width_genes[gene_id] if rng.random() < 0.5 else rhs.pruning_width_genes[gene_id])
        for gene_id in lhs.pruning_width_genes
    }
    precision = {
        gene_id: (lhs.precision_genes[gene_id] if rng.random() < 0.5 else rhs.precision_genes[gene_id])
        for gene_id in lhs.precision_genes
    }
    return _validate_complete_chromosome(
        CandidateGenotype(
            pruning_genes={},
            precision_genes=precision,
            pruning_width_genes=widths,
            meta={"created_by": "stage12_v3_same_locus_crossover", "repair_count": 0},
        ),
        space,
    )


def _stage1_compare(left: Mapping[str, Any], right: Mapping[str, Any], policy: Stage1Policy) -> int:
    lscore, rscore = float(left["J_total"]), float(right["J_total"])
    if not strict_taylor_tie(lscore, rscore, policy):
        return -1 if lscore < rscore else 1
    lkey = (
        float(left["BOPS_deviation"]),
        -float(left["parameter_retention"]),
        -float(left["mixed_weight_retention"]),
        str(left["canonical_hash"]),
    )
    rkey = (
        float(right["BOPS_deviation"]),
        -float(right["parameter_retention"]),
        -float(right["mixed_weight_retention"]),
        str(right["canonical_hash"]),
    )
    return -1 if lkey < rkey else (1 if lkey > rkey else 0)


def evaluate_stage1_population(
    population: Sequence[CandidateGenotype],
    space: SearchSpaceSpec,
    *,
    policy: Stage1Policy,
    resource_evaluator: Callable[[Any], Mapping[str, Any]],
    proxy_evaluator: Callable[[Any], Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate, deduplicate, hard-gate and only then evaluate Taylor risk."""

    records: list[dict[str, Any]] = []
    seen_physical: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for index, raw in enumerate(population):
        row: dict[str, Any] = {
            "population_index": index,
            "legality_status": "invalid",
            "dedup_status": "not_checked",
            "stage2_eligibility": False,
            "structural_repair_count": 0,
            "precision_repair_count": 0,
            "budget_repair_count": 0,
        }
        try:
            genotype = _validate_complete_chromosome(raw, space)
            phenotype = canonicalize_candidate(genotype, space)
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}:{exc}"
            records.append(row)
            continue
        physical_hash = _mapping_hash(genotype.pruning_width_genes)
        precision_hash = _mapping_hash(genotype.precision_genes)
        canonical_hash = candidate_hash(phenotype, space)
        row.update(
            {
                "legality_status": "legal",
                "structure_hash": physical_hash,
                "precision_hash": precision_hash,
                "physical_hash": canonical_hash,
                "canonical_hash": canonical_hash,
            }
        )
        if canonical_hash in seen_physical:
            row["dedup_status"] = "duplicate_physical_hash"
            records.append(row)
            continue
        seen_physical.add(canonical_hash)
        row["dedup_status"] = "unique"
        resources = dict(resource_evaluator(phenotype))
        retention = float(resources["R_BOPS"])
        deviation = abs(retention - policy.target_bops_retention)
        row.update(
            {
                "BOPS": float(resources["BOPS"]),
                "R_BOPS": retention,
                "BOPS_deviation": deviation,
                "params": float(resources["params"]),
                "parameter_retention": float(resources["parameter_retention"]),
                "mixed_weight_size": float(resources["mixed_weight_size"]),
                "mixed_weight_retention": float(resources["mixed_weight_retention"]),
            }
        )
        if deviation > policy.absolute_bops_tolerance:
            row["hard_gate_status"] = "rejected_outside_bops_band"
            records.append(row)
            continue
        row["hard_gate_status"] = "passed"
        proxy = dict(proxy_evaluator(phenotype))
        j_struct = float(proxy["J_struct"])
        j_wq = float(proxy["J_WQ"])
        j_aq = float(proxy["J_AQ"])
        values = (j_struct, j_wq, j_aq, j_struct + j_wq + j_aq)
        if min(values) < 0.0 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("stage1_proxy_nonfinite_or_negative")
        row.update(
            {
                "J_struct": j_struct,
                "J_WQ": j_wq,
                "J_AQ": j_aq,
                "J_total": values[-1],
                "stage2_eligibility": True,
                "genotype": genotype,
                "phenotype": phenotype,
            }
        )
        records.append(row)
        eligible.append(row)
    eligible.sort(key=cmp_to_key(lambda left, right: _stage1_compare(left, right, policy)))
    for rank, row in enumerate(eligible, start=1):
        row["stage1_rank"] = rank
    return {
        "policy": asdict(policy),
        "records": records,
        "eligible": eligible,
        "eligible_count": len(eligible),
        "repair_count": 0,
        "order": [
            "static_contract_validation",
            "physical_hash_dedup",
            "BOPS_hard_gate",
            "Taylor_proxy",
            "Taylor_sort",
            "strict_tie_resource_selection",
            "Stage2_selection",
        ],
    }


@dataclass(frozen=True)
class Stage2Policy:
    accuracy_tolerance: float = 0.005
    accuracy_weight: float = 0.2
    latency_weight: float = 0.8
    new_engine_quota: int = 5


STAGE2_STEPS = (
    "physical_materialization",
    "s32_model",
    "s32_fixed50",
    "fresh_train200_calibration",
    "jmix_strongly_typed_onnx",
    "tensorrt_engine_build",
    "requested_realized_precision",
    "jmix_fixed50",
    "screening_latency",
)


def run_stage2_pipeline(
    candidate: Mapping[str, Any],
    *,
    step_runner: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    state: dict[str, Any] = dict(candidate)
    completed: list[str] = []
    for step in STAGE2_STEPS:
        result = dict(step_runner(step, candidate, state))
        state.update(result)
        completed.append(step)
        if str(result.get("status", "ok")) != "ok":
            return {
                **state,
                "status": "stage2_invalid",
                "failed_step": step,
                "completed_steps": completed,
                "precision_fallback_allowed": False,
            }
    return {**state, "status": "ok", "completed_steps": completed}


def score_stage2_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    greedy_anchor: Mapping[str, Any],
    policy: Stage2Policy | None = None,
) -> dict[str, Any]:
    config = policy or Stage2Policy()
    greedy_map = float(greedy_anchor["mAP"])
    greedy_latency = float(greedy_anchor["p50_ms"])
    if greedy_latency <= 0.0:
        raise ValueError("stage2_greedy_latency_nonpositive")
    scored: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        candidate_map = float(row.get("mAP", float("-inf")))
        candidate_latency = float(row.get("p50_ms", float("inf")))
        exact = bool(row.get("requested_realized_exact", False))
        status_ok = str(row.get("status", "")) == "ok"
        accuracy_ok = candidate_map >= greedy_map - config.accuracy_tolerance
        row["accuracy_gate_passed"] = accuracy_ok
        row["stage2_eligible"] = status_ok and exact and accuracy_ok and math.isfinite(candidate_latency)
        if row["stage2_eligible"]:
            normalized_accuracy = max(
                -1.0,
                min(1.0, (greedy_map - candidate_map) / config.accuracy_tolerance),
            )
            row["A_0.005"] = normalized_accuracy
            row["F_S2"] = (
                config.accuracy_weight * normalized_accuracy
                + config.latency_weight * candidate_latency / greedy_latency
            )
            eligible.append(row)
        else:
            row["F_S2"] = float("inf")
        scored.append(row)
    winner: dict[str, Any] | None = None
    reason = "no_eligible_candidate"
    if eligible:
        highest_map = max(eligible, key=lambda row: float(row["mAP"]))
        lowest_latency = min(eligible, key=lambda row: float(row["p50_ms"]))
        if str(highest_map["physical_hash"]) == str(lowest_latency["physical_hash"]):
            winner = highest_map
            reason = "dominant_highest_map_and_lowest_latency"
        else:
            winner = min(
                eligible,
                key=lambda row: (
                    float(row["F_S2"]),
                    -float(row["mAP"]),
                    float(row["p50_ms"]),
                    -float(row["parameter_retention"]),
                    -float(row["mixed_weight_retention"]),
                    float(row["BOPS_deviation"]),
                    str(row["physical_hash"]),
                ),
            )
            reason = "minimum_F_S2"
    return {
        "policy": asdict(config),
        "rows": scored,
        "eligible_count": len(eligible),
        "winner": winner,
        "winner_reason": reason,
        "new_generation_winner_created": winner is not None,
    }


def select_stage2_new_candidates(
    stage1_rows: Sequence[Mapping[str, Any]],
    *,
    evaluated_hashes: set[str],
    historical_real_elites: Sequence[Mapping[str, Any]] = (),
    quota: int = 5,
) -> dict[str, Any]:
    historical = []
    seen: set[str] = set()
    for raw in historical_real_elites:
        physical_hash = str(raw["physical_hash"])
        if physical_hash not in seen:
            historical.append(dict(raw))
            seen.add(physical_hash)
    fresh = []
    for raw in stage1_rows:
        physical_hash = str(raw["physical_hash"])
        if physical_hash in evaluated_hashes or physical_hash in seen:
            continue
        fresh.append(dict(raw))
        seen.add(physical_hash)
        if len(fresh) >= int(quota):
            break
    return {
        "historical_real_elites": historical,
        "new_candidates": fresh,
        "new_engine_quota": int(quota),
        "new_engine_count": len(fresh),
        "historical_elites_consume_quota": False,
    }


@dataclass
class RealAnchorArchive:
    """V1 Greedy plus V2/V3 globally deduplicated real Stage-2 anchors."""

    greedy_anchor: dict[str, Any]
    evaluated: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = str(self.greedy_anchor["physical_hash"])
        self.greedy_anchor = dict(self.greedy_anchor)
        self.evaluated[key] = dict(self.greedy_anchor)

    def update(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for raw in rows:
            row = dict(raw)
            if not bool(row.get("stage2_eligible", False)):
                continue
            self.evaluated[str(row["physical_hash"])] = row

    def anchors(self) -> dict[str, Any]:
        eligible = [row for row in self.evaluated.values() if bool(row.get("stage2_eligible", True))]
        non_greedy = [
            row for row in eligible
            if str(row["physical_hash"]) != str(self.greedy_anchor["physical_hash"])
        ]
        selected: list[dict[str, Any]] = [dict(self.greedy_anchor)]
        roles: dict[str, list[str]] = {str(self.greedy_anchor["physical_hash"]): ["greedy_anchor"]}
        choices = []
        if non_greedy:
            choices = [
                ("lowest_global_F_S2", min(non_greedy, key=lambda row: float(row["F_S2"]))),
                ("highest_global_mAP", max(non_greedy, key=lambda row: float(row["mAP"]))),
                ("lowest_global_p50", min(non_greedy, key=lambda row: float(row["p50_ms"]))),
            ]
        by_hash = {str(self.greedy_anchor["physical_hash"]): selected[0]}
        for role, row in choices:
            key = str(row["physical_hash"])
            if key not in by_hash:
                by_hash[key] = dict(row)
                selected.append(by_hash[key])
            roles.setdefault(key, []).append(role)
        return {
            "anchors": selected,
            "roles_by_physical_hash": roles,
            "greedy_anchor_preserved": True,
            "black_box_surrogate": False,
            "anchor_count_after_dedup": len(selected),
        }


__all__ = [
    "RealAnchorArchive",
    "STAGE2_STEPS",
    "Stage1Policy",
    "Stage2Policy",
    "adjacent_legal_mutation",
    "evaluate_stage1_population",
    "run_stage2_pipeline",
    "same_locus_crossover",
    "score_stage2_candidates",
    "select_stage2_new_candidates",
    "strict_taylor_tie",
]
