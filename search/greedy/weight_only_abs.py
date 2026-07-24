"""Repair-free Greedy loop using conservative weight-only Taylor action costs."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..hashing import candidate_hash
from ..proxy.joint_weight_taylor import JointWeightTaylorProxy
from .engine import GreedyBudgetSearch, GreedySearchConfig


BreakdownEvaluator = Callable[[Any], dict[str, Any]]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _state(candidate: CandidateGenotype) -> dict[str, Any]:
    return {
        "widths": dict(sorted(candidate.pruning_width_genes.items())),
        "precision": dict(sorted(candidate.precision_genes.items())),
    }


def run_weight_only_abs_greedy(
    space: SearchSpaceSpec,
    *,
    weight_proxy: JointWeightTaylorProxy,
    bops_evaluator: BreakdownEvaluator,
    size_evaluator: BreakdownEvaluator,
    target: float = 0.05,
    tolerance_abs: float = 0.005,
    epsilon: float = 1.0e-12,
    maximum_steps: int = 10000,
) -> dict[str, Any]:
    """Run one deterministic path without model execution or artifact export."""
    neighbor_engine = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(float(target),),
            bops_tolerance_abs=float(tolerance_abs),
            maximum_steps=int(maximum_steps),
            run_to_exhaustion=True,
            enable_budget_recovery=False,
        ),
    )
    current = neighbor_engine._initial_candidate()
    current_phenotype = canonicalize_candidate(current, space)
    baseline_bops = bops_evaluator(current_phenotype)
    bops_before = float(baseline_bops["bops_total"])
    if bops_before <= 0.0 or not math.isfinite(bops_before):
        raise RuntimeError("weight_only_greedy_baseline_bops_invalid")
    cumulative_prune = 0.0
    cumulative_wq = 0.0
    selected_hashes = [candidate_hash(current_phenotype, space)]
    trace: list[dict[str, Any]] = []
    band: dict[str, dict[str, Any]] = {}
    selected_steps: list[dict[str, Any]] = []
    termination = ""

    for step in range(1, int(maximum_steps) + 1):
        neighbors = neighbor_engine._neighbors(current)
        if not neighbors:
            termination = "no_remaining_legal_action"
            break
        action_rows: list[dict[str, Any]] = []
        for successor, action in neighbors:
            successor_phenotype = canonicalize_candidate(successor, space)
            bops = bops_evaluator(successor_phenotype)
            bops_after = float(bops["bops_total"])
            delta_bops = bops_before - bops_after
            if not math.isfinite(delta_bops) or delta_bops <= float(epsilon):
                continue
            if action["kind"] == "domain_width":
                risk = weight_proxy.pruning_action_breakdown(
                    current_phenotype, successor_phenotype
                )
            elif action["kind"] == "precision":
                risk = weight_proxy.weight_quantization_action_breakdown(
                    current_phenotype, successor_phenotype
                )
            else:
                raise RuntimeError(f"unsupported_weight_only_greedy_action:{action}")
            delta_prune = float(risk["delta_J_prune"])
            delta_wq = float(risk["delta_J_WQ"])
            delta_action = delta_prune + delta_wq
            if delta_action < 0.0 or not math.isfinite(delta_action):
                raise RuntimeError("weight_only_greedy_action_risk_invalid")
            utility = delta_action / max(delta_bops, float(epsilon))
            if not math.isfinite(utility):
                raise RuntimeError("weight_only_greedy_utility_nonfinite")
            retention = bops_after / float(baseline_bops["bops_total"])
            phenotype_hash = candidate_hash(successor_phenotype, space)
            sizes = size_evaluator(successor_phenotype)
            cumulative = cumulative_prune + cumulative_wq + delta_action
            row = {
                "step": step,
                "action_type": action["kind"],
                "domain_layer": action["gene_id"],
                "domain_type": action.get("domain_type", ""),
                "module_path": action.get("module_path", ""),
                "state_before": json.dumps(_state(current), sort_keys=True),
                "state_after": json.dumps(_state(successor), sort_keys=True),
                "newly_pruned_parameter_count": int(
                    risk.get("newly_pruned_parameter_count", 0)
                ),
                "delta_J_prune": delta_prune,
                "delta_J_WQ": delta_wq,
                "activation_taylor_diagnostic": 0.0,
                "activation_taylor_used_for_fitness": False,
                "joint_taylor_diagnostic": 0.0,
                "joint_taylor_used_for_fitness": False,
                "cross_residual_diagnostic": 0.0,
                "cross_residual_used_for_fitness": False,
                "first_order_abs_sum": float(risk["first_order_abs_sum"]),
                "second_order_abs_sum": float(risk["second_order_abs_sum"]),
                "delta_BOPS": delta_bops,
                "utility": utility,
                "BOPS_before": bops_before,
                "BOPS_after": bops_after,
                "current_retention": retention,
                "global_rank": 0,
                "selected": False,
                "cumulative_proxy": cumulative,
                "cumulative_pruning_taylor": cumulative_prune + delta_prune,
                "cumulative_weight_quantization_taylor": cumulative_wq + delta_wq,
                "structure_hash": _stable_hash(successor.pruning_width_genes),
                "precision_hash": _stable_hash(successor.precision_genes),
                "candidate_hash": phenotype_hash,
                "mixed_weight_size_bytes": float(sizes["size_bits_total"]) / 8.0,
                "R_parameter_retention": float(sizes["R_parameter_retention"]),
                "structural_repair_count": 0,
                "precision_repair_count": 0,
                "budget_projection_count": 0,
                "candidate": successor,
                "phenotype": successor_phenotype,
                "risk": risk,
                "bops_breakdown": bops,
                "size_breakdown": sizes,
            }
            action_rows.append(row)
        if not action_rows:
            termination = "no_positive_bops_reduction_action"
            break
        action_rows.sort(
            key=lambda row: (
                row["utility"],
                row["domain_layer"],
                row["candidate_hash"],
            )
        )
        for rank, row in enumerate(action_rows, start=1):
            row["global_rank"] = rank
            public = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "candidate",
                    "phenotype",
                    "risk",
                    "bops_breakdown",
                    "size_breakdown",
                }
            }
            if abs(float(row["current_retention"]) - float(target)) <= float(
                tolerance_abs
            ):
                incumbent = band.get(row["candidate_hash"])
                candidate_entry = {**row, "trace_row": public}
                if incumbent is None or (
                    row["cumulative_proxy"],
                    abs(row["current_retention"] - target),
                    row["mixed_weight_size_bytes"],
                    row["candidate_hash"],
                ) < (
                    incumbent["cumulative_proxy"],
                    abs(incumbent["current_retention"] - target),
                    incumbent["mixed_weight_size_bytes"],
                    incumbent["candidate_hash"],
                ):
                    band[row["candidate_hash"]] = candidate_entry
            trace.append(public)
        selected = action_rows[0]
        trace[-len(action_rows)]["selected"] = True
        selected["selected"] = True
        selected_steps.append(selected)
        current = selected["candidate"]
        current_phenotype = selected["phenotype"]
        cumulative_prune += float(selected["delta_J_prune"])
        cumulative_wq += float(selected["delta_J_WQ"])
        bops_before = float(selected["BOPS_after"])
        selected_hashes.append(selected["candidate_hash"])
    else:
        termination = "maximum_steps_reached"

    ordered_band = sorted(
        band.values(),
        key=lambda row: (
            row["cumulative_proxy"],
            abs(row["current_retention"] - target),
            row["mixed_weight_size_bytes"],
            row["candidate_hash"],
        ),
    )
    if not ordered_band and not selected_steps:
        raise RuntimeError("weight_only_greedy_no_legal_candidate")
    winner = ordered_band[0] if ordered_band else min(
        selected_steps,
        key=lambda row: (
            abs(row["current_retention"] - target),
            row["cumulative_proxy"],
            row["candidate_hash"],
        ),
    )
    return {
        "winner_candidate": winner["candidate"],
        "winner_phenotype": winner["phenotype"],
        "winner_metrics": {
            key: value
            for key, value in winner.items()
            if key
            not in {
                "candidate",
                "phenotype",
                "risk",
                "bops_breakdown",
                "size_breakdown",
                "trace_row",
            }
        },
        "winner_bops_breakdown": winner["bops_breakdown"],
        "winner_size_breakdown": winner["size_breakdown"],
        "trace": trace,
        "selected_candidate_hashes": selected_hashes,
        "selected_step_count": len(selected_steps),
        "visited_action_count": len(trace),
        "budget_band_candidate_count": len(ordered_band),
        "budget_reached": bool(ordered_band),
        "budget_unreachable": not bool(ordered_band),
        "termination_reason": termination,
        "search_loop_forward_calls": 0,
        "search_loop_backward_calls": 0,
        "search_loop_physical_exports": 0,
        "search_loop_onnx_exports": 0,
        "search_loop_trt_builds": 0,
        "activation_taylor_used_for_fitness": False,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
        "elementwise_abs_before_reduction": True,
    }
