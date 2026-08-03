"""Repair-free Greedy helpers for conservative structure/precision Taylor."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..hashing import candidate_hash
from .engine import GreedyBudgetSearch, GreedySearchConfig


def empty_search_loop_runtime_audit() -> dict[str, int]:
    return {
        "search_loop_forward_calls": 0,
        "search_loop_backward_calls": 0,
        "search_loop_physical_exports": 0,
        "search_loop_onnx_exports": 0,
        "search_loop_trt_builds": 0,
    }


def combine_precision_action_risk(
    weight: Mapping[str, Any], activation: Mapping[str, Any]
) -> dict[str, Any]:
    delta_wq = float(weight["delta_J_WQ"])
    delta_aq = float(activation["delta_J_AQ"])
    total = delta_wq + delta_aq
    if min(delta_wq, delta_aq, total) < 0.0 or not math.isfinite(total):
        raise RuntimeError("conservative_precision_action_risk_invalid")
    return {
        "delta_J_WQ": delta_wq,
        "delta_J_AQ": delta_aq,
        "delta_J_precision": total,
        "first_order_abs_sum": float(weight.get("first_order_abs_sum", 0.0))
        + float(activation.get("first_order_abs_sum", 0.0)),
        "second_order_abs_sum": float(weight.get("second_order_abs_sum", 0.0))
        + float(activation.get("second_order_abs_sum", 0.0)),
        "joint_taylor_diagnostic": 0.0,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_diagnostic": 0.0,
        "cross_residual_used_for_fitness": False,
        "risk_refund": 0.0,
        "elementwise_abs_before_reduction": True,
    }


def _relative_equal(left: float, right: float, tolerance: float) -> bool:
    threshold = max(1.0e-12, float(tolerance) * max(abs(left), abs(right)))
    return abs(left - right) <= threshold


def absolute_fp32_retention(bops: Mapping[str, Any]) -> float:
    """Return the formal B0-relative retention; never renormalize legal start."""

    value = float(bops.get("R_bops_vs_fp32", bops.get("R_bops", float("nan"))))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("conservative_greedy_absolute_fp32_retention_invalid")
    return value


def select_budget_winner(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: float,
    relative_taylor_equality: float = 1.0e-8,
) -> dict[str, Any]:
    """Apply the formal selector without rewarding additional compression."""

    if not rows:
        raise RuntimeError("conservative_winner_selector_empty")
    minimum = min(float(row["cumulative_total_taylor"]) for row in rows)
    equivalent = [
        dict(row)
        for row in rows
        if _relative_equal(
            float(row["cumulative_total_taylor"]),
            minimum,
            relative_taylor_equality,
        )
    ]
    equivalent.sort(
        key=lambda row: (
            abs(float(row["current_retention"]) - float(target)),
            -float(row["R_parameter_retention"]),
            -float(row.get("mixed_weight_retention", 0.0)),
            str(row["candidate_hash"]),
        )
    )
    return equivalent[0]


def select_stage2_budget_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: float,
    maximum: int = 5,
) -> list[dict[str, Any]]:
    """Select up to five physically different budget-band candidates."""

    # Stage-2 first screens physical S32 structures. Precision-only variants of
    # the same structure would become byte-identical after the S32 override and
    # must not consume another engine-build slot.
    deduplicated: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = str(row["physical_hash"])
        incumbent = deduplicated.get(key)
        if incumbent is None or (
            float(row["cumulative_total_taylor"]),
            str(row["candidate_hash"]),
        ) < (
            float(incumbent["cumulative_total_taylor"]),
            str(incumbent["candidate_hash"]),
        ):
            deduplicated[key] = row
    pool = list(deduplicated.values())
    if not pool:
        return []
    criteria = [
        (
            "lowest_total_taylor",
            lambda row: (
                float(row["cumulative_total_taylor"]),
                abs(float(row["current_retention"]) - float(target)),
                str(row["candidate_hash"]),
            ),
        ),
        (
            "highest_parameter_retention",
            lambda row: (
                -float(row["R_parameter_retention"]),
                float(row["cumulative_total_taylor"]),
                str(row["candidate_hash"]),
            ),
        ),
        (
            "least_structure_most_precision",
            lambda row: (
                int(row["pruned_unit_count"]),
                -int(row["int8_count"]),
                float(row["cumulative_total_taylor"]),
                str(row["candidate_hash"]),
            ),
        ),
        (
            "widest_attention_ffn",
            lambda row: (
                -int(row["attention_ffn_width_sum"]),
                float(row["cumulative_total_taylor"]),
                str(row["candidate_hash"]),
            ),
        ),
        (
            "nearest_taylor_distinct_physical",
            lambda row: (
                float(row["cumulative_total_taylor"]),
                abs(float(row["current_retention"]) - float(target)),
                str(row["candidate_hash"]),
            ),
        ),
    ]
    selected: dict[str, dict[str, Any]] = {}
    for reason, key_fn in criteria:
        eligible = [
            row
            for row in pool
            if str(row["physical_hash"]) not in selected
        ]
        if not eligible:
            break
        row = min(eligible, key=key_fn)
        key = str(row["physical_hash"])
        selected[key] = {**row, "selection_reasons": [reason]}
        if len(selected) >= int(maximum):
            break
    # Preserve the semantic selection order above, then append a reason when
    # another criterion resolves to an already-selected candidate.
    for reason, key_fn in criteria:
        row = min(pool, key=key_fn)
        key = str(row["physical_hash"])
        if key in selected and reason not in selected[key]["selection_reasons"]:
            selected[key]["selection_reasons"].append(reason)
    return list(selected.values())[: int(maximum)]


def run_conservative_joint_greedy(
    space: SearchSpaceSpec,
    *,
    structural_proxy: Any,
    weight_proxy: Any,
    activation_proxy: Any,
    bops_evaluator: Any,
    size_evaluator: Any,
    target: float = 0.30,
    tolerance_abs: float = 0.005,
    epsilon: float = 1.0e-12,
    maximum_steps: int = 10000,
) -> dict[str, Any]:
    """Run a monotonic path using only pre-collected action statistics."""

    engine = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(float(target),),
            bops_tolerance_abs=float(tolerance_abs),
            maximum_steps=int(maximum_steps),
            run_to_exhaustion=True,
            enable_budget_recovery=False,
        ),
    )
    current = engine._initial_candidate()
    current_phenotype = canonicalize_candidate(current, space)
    baseline_bops = bops_evaluator(current_phenotype)
    baseline_total = float(baseline_bops["bops_total"])
    if baseline_total <= 0.0 or not math.isfinite(baseline_total):
        raise RuntimeError("conservative_greedy_baseline_bops_invalid")
    current_bops = baseline_total
    cumulative_struct = 0.0
    cumulative_wq = 0.0
    cumulative_aq = 0.0
    trace: list[dict[str, Any]] = []
    band: dict[str, dict[str, Any]] = {}
    selected_steps: list[dict[str, Any]] = []
    selected_hashes = [candidate_hash(current_phenotype, space)]
    termination = ""

    for step in range(1, int(maximum_steps) + 1):
        neighbors = engine._neighbors(current)
        if not neighbors:
            termination = "no_remaining_legal_action"
            break
        action_rows: list[dict[str, Any]] = []
        for successor, action in neighbors:
            phenotype = canonicalize_candidate(successor, space)
            bops = bops_evaluator(phenotype)
            after = float(bops["bops_total"])
            delta_bops = current_bops - after
            if delta_bops <= float(epsilon) or not math.isfinite(delta_bops):
                continue
            if action["kind"] == "domain_width":
                risk = structural_proxy.pruning_action_breakdown(
                    current_phenotype, phenotype
                )
                delta_struct = float(risk["delta_J_struct"])
                delta_wq = 0.0
                delta_aq = 0.0
            elif action["kind"] == "precision":
                weight = weight_proxy.weight_quantization_action_breakdown(
                    current_phenotype, phenotype
                )
                activation = activation_proxy.quantization_action_breakdown(
                    current_phenotype,
                    phenotype,
                    changed_gene_id=str(action["gene_id"]),
                )
                risk = combine_precision_action_risk(weight, activation)
                delta_struct = 0.0
                delta_wq = float(risk["delta_J_WQ"])
                delta_aq = float(risk["delta_J_AQ"])
            else:
                raise RuntimeError(f"conservative_greedy_action_unsupported:{action}")
            delta_action = delta_struct + delta_wq + delta_aq
            if delta_action < 0.0 or not math.isfinite(delta_action):
                raise RuntimeError("conservative_greedy_action_risk_invalid")
            utility = delta_action / max(delta_bops, float(epsilon))
            if not math.isfinite(utility):
                raise RuntimeError("conservative_greedy_utility_nonfinite")
            retention = absolute_fp32_retention(bops)
            phenotype_hash = candidate_hash(phenotype, space)
            action_rows.append(
                {
                    "step": step,
                    "action_type": action["kind"],
                    "domain_layer": str(action["gene_id"]),
                    "domain_type": str(action.get("domain_type", "")),
                    "module_path": str(action.get("module_path", "")),
                    "delta_J_struct": delta_struct,
                    "delta_J_WQ": delta_wq,
                    "delta_J_AQ": delta_aq,
                    "delta_J_action": delta_action,
                    "delta_BOPS": delta_bops,
                    "utility": utility,
                    "BOPS_before": current_bops,
                    "BOPS_after": after,
                    "current_retention": retention,
                    "R_bops_vs_fp32": retention,
                    "R_bops_reference": "original_fp32",
                    "cumulative_total_taylor": cumulative_struct
                    + cumulative_wq
                    + cumulative_aq
                    + delta_action,
                    "cumulative_structural_taylor": cumulative_struct + delta_struct,
                    "cumulative_weight_quant_taylor": cumulative_wq + delta_wq,
                    "cumulative_activation_quant_taylor": cumulative_aq + delta_aq,
                    "candidate_hash": phenotype_hash,
                    "physical_hash": _stable_mapping_hash(successor.pruning_width_genes),
                    "precision_hash": _stable_mapping_hash(successor.precision_genes),
                    "phenotype_hash": phenotype_hash,
                    "selected": False,
                    "global_rank": 0,
                    "structural_repair_count": 0,
                    "precision_repair_count": 0,
                    "budget_projection_count": 0,
                    "candidate": successor,
                    "phenotype": phenotype,
                    "bops_breakdown": bops,
                    "risk": risk,
                }
            )
        if not action_rows:
            termination = "no_positive_bops_reduction_action"
            break
        action_rows.sort(
            key=lambda row: (
                float(row["utility"]),
                str(row["domain_layer"]),
                str(row["candidate_hash"]),
            )
        )
        selected = action_rows[0]
        selected["selected"] = True
        for rank, row in enumerate(action_rows, start=1):
            row["global_rank"] = rank
            in_band = abs(float(row["current_retention"]) - float(target)) <= float(
                tolerance_abs
            )
            if in_band or row is selected:
                sizes = size_evaluator(row["phenotype"])
                row["R_parameter_retention"] = float(
                    sizes["R_parameter_retention"]
                )
                row["mixed_weight_size_bytes"] = float(
                    sizes["size_bits_total"]
                ) / 8.0
            else:
                row["R_parameter_retention"] = float("nan")
                row["mixed_weight_size_bytes"] = float("nan")
            public = {
                key: value
                for key, value in row.items()
                if key not in {"candidate", "phenotype", "bops_breakdown", "risk"}
            }
            trace.append(public)
            if in_band:
                value = {
                    **row,
                    "pruned_unit_count": len(row["phenotype"].pruned_unit_ids),
                    "int8_count": sum(
                        precision == "INT8"
                        for precision in row[
                            "phenotype"
                        ].realized_precision_profile.values()
                    ),
                    "attention_ffn_width_sum": sum(
                        int(width)
                        for domain_id, width in row[
                            "candidate"
                        ].pruning_width_genes.items()
                        if domain_id.startswith("attention_dh::")
                        or domain_id.startswith("ffn_hidden::")
                    ),
                }
                incumbent = band.get(row["candidate_hash"])
                if incumbent is None or float(value["cumulative_total_taylor"]) < float(
                    incumbent["cumulative_total_taylor"]
                ):
                    band[row["candidate_hash"]] = value
        selected_steps.append(selected)
        current = selected["candidate"]
        current_phenotype = selected["phenotype"]
        current_bops = float(selected["BOPS_after"])
        cumulative_struct += float(selected["delta_J_struct"])
        cumulative_wq += float(selected["delta_J_WQ"])
        cumulative_aq += float(selected["delta_J_AQ"])
        selected_hashes.append(str(selected["candidate_hash"]))
    else:
        termination = "maximum_steps_reached"

    band_rows = list(band.values())
    if not band_rows:
        raise RuntimeError("conservative_greedy_budget_band_empty")
    winner = select_budget_winner(band_rows, target=target)
    return {
        "winner_candidate": winner["candidate"],
        "winner_phenotype": winner["phenotype"],
        "winner_metrics": _public_candidate_row(winner),
        "winner_bops_breakdown": winner["bops_breakdown"],
        "winner_size_breakdown": size_evaluator(winner["phenotype"]),
        "budget_candidates": band_rows,
        "trace": trace,
        "selected_candidate_hashes": selected_hashes,
        "selected_step_count": len(selected_steps),
        "visited_action_count": len(trace),
        "budget_band_candidate_count": len(band_rows),
        "budget_reached": True,
        "budget_unreachable": False,
        "termination_reason": termination,
        **empty_search_loop_runtime_audit(),
        "activation_taylor_used_for_fitness": True,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
        "elementwise_abs_before_reduction": True,
    }


def run_conservative_joint_greedy_multi_budget(
    space: SearchSpaceSpec,
    *,
    structural_proxy: Any,
    weight_proxy: Any,
    activation_proxy: Any,
    bops_evaluator: Any,
    size_evaluator: Any,
    targets: Sequence[float] = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05),
    tolerance_abs: float = 0.005,
    epsilon: float = 1.0e-12,
    maximum_steps: int = 10000,
) -> dict[str, Any]:
    """Run one monotonic trajectory and capture every requested budget band."""

    normalized_targets = tuple(sorted({float(value) for value in targets}, reverse=True))
    if not normalized_targets or any(not 0.0 < value < 1.0 for value in normalized_targets):
        raise ValueError("conservative_multi_budget_targets_invalid")
    engine = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=normalized_targets,
            bops_tolerance_abs=float(tolerance_abs),
            maximum_steps=int(maximum_steps),
            run_to_exhaustion=True,
            enable_budget_recovery=False,
        ),
    )
    current = engine._initial_candidate()
    current_phenotype = canonicalize_candidate(current, space)
    baseline_bops = bops_evaluator(current_phenotype)
    current_bops = float(baseline_bops["bops_total"])
    if current_bops <= 0.0 or not math.isfinite(current_bops):
        raise RuntimeError("conservative_multi_budget_baseline_bops_invalid")
    cumulative_struct = 0.0
    cumulative_wq = 0.0
    cumulative_aq = 0.0
    trace: list[dict[str, Any]] = []
    bands: dict[float, dict[str, dict[str, Any]]] = {
        target: {} for target in normalized_targets
    }
    selected_path: list[dict[str, Any]] = []
    selected_hashes = [candidate_hash(current_phenotype, space)]
    termination = ""

    for step in range(1, int(maximum_steps) + 1):
        neighbors = engine._neighbors(current)
        if not neighbors:
            termination = "no_remaining_legal_action"
            break
        action_rows: list[dict[str, Any]] = []
        for successor, action in neighbors:
            phenotype = canonicalize_candidate(successor, space)
            bops = bops_evaluator(phenotype)
            after = float(bops["bops_total"])
            delta_bops = current_bops - after
            if delta_bops <= float(epsilon) or not math.isfinite(delta_bops):
                continue
            if action["kind"] == "domain_width":
                risk = structural_proxy.pruning_action_breakdown(
                    current_phenotype, phenotype
                )
                delta_struct = float(risk["delta_J_struct"])
                delta_wq = 0.0
                delta_aq = 0.0
            elif action["kind"] == "precision":
                weight = weight_proxy.weight_quantization_action_breakdown(
                    current_phenotype, phenotype
                )
                activation = activation_proxy.quantization_action_breakdown(
                    current_phenotype,
                    phenotype,
                    changed_gene_id=str(action["gene_id"]),
                )
                risk = combine_precision_action_risk(weight, activation)
                delta_struct = 0.0
                delta_wq = float(risk["delta_J_WQ"])
                delta_aq = float(risk["delta_J_AQ"])
            else:
                raise RuntimeError(f"conservative_multi_budget_action_unsupported:{action}")
            delta_action = delta_struct + delta_wq + delta_aq
            if delta_action < 0.0 or not math.isfinite(delta_action):
                raise RuntimeError("conservative_multi_budget_action_risk_invalid")
            utility = delta_action / max(delta_bops, float(epsilon))
            if not math.isfinite(utility):
                raise RuntimeError("conservative_multi_budget_utility_nonfinite")
            retention = absolute_fp32_retention(bops)
            phenotype_hash = candidate_hash(phenotype, space)
            matched_targets = tuple(
                target
                for target in normalized_targets
                if abs(retention - target) <= float(tolerance_abs)
            )
            action_rows.append(
                {
                    "step": step,
                    "action_type": action["kind"],
                    "domain_layer": str(action["gene_id"]),
                    "domain_type": str(action.get("domain_type", "")),
                    "module_path": str(action.get("module_path", "")),
                    "delta_J_struct": delta_struct,
                    "delta_J_WQ": delta_wq,
                    "delta_J_AQ": delta_aq,
                    "delta_J_action": delta_action,
                    "delta_BOPS": delta_bops,
                    "utility": utility,
                    "BOPS_before": current_bops,
                    "BOPS_after": after,
                    "current_retention": retention,
                    "R_bops_vs_fp32": retention,
                    "R_bops_reference": "original_fp32",
                    "cumulative_total_taylor": cumulative_struct
                    + cumulative_wq
                    + cumulative_aq
                    + delta_action,
                    "cumulative_structural_taylor": cumulative_struct + delta_struct,
                    "cumulative_weight_quant_taylor": cumulative_wq + delta_wq,
                    "cumulative_activation_quant_taylor": cumulative_aq + delta_aq,
                    "candidate_hash": phenotype_hash,
                    "physical_hash": _stable_mapping_hash(successor.pruning_width_genes),
                    "precision_hash": _stable_mapping_hash(successor.precision_genes),
                    "phenotype_hash": phenotype_hash,
                    "matched_budget_targets": matched_targets,
                    "selected": False,
                    "global_rank": 0,
                    "structural_repair_count": 0,
                    "precision_repair_count": 0,
                    "budget_projection_count": 0,
                    "candidate": successor,
                    "phenotype": phenotype,
                    "bops_breakdown": bops,
                    "risk": risk,
                }
            )
        if not action_rows:
            termination = "no_positive_bops_reduction_action"
            break
        action_rows.sort(
            key=lambda row: (
                float(row["utility"]),
                str(row["domain_layer"]),
                str(row["candidate_hash"]),
            )
        )
        selected = action_rows[0]
        selected["selected"] = True
        for rank, row in enumerate(action_rows, start=1):
            row["global_rank"] = rank
            if row["matched_budget_targets"] or row is selected:
                sizes = size_evaluator(row["phenotype"])
                row["R_parameter_retention"] = float(sizes["R_parameter_retention"])
                row["parameter_count"] = float(sizes["parameter_count_after"])
                row["mixed_weight_size_bytes"] = float(sizes["size_bits_total"]) / 8.0
                row["mixed_weight_retention"] = float(sizes["R_size_vs_fp32"])
            else:
                row["R_parameter_retention"] = float("nan")
                row["parameter_count"] = float("nan")
                row["mixed_weight_size_bytes"] = float("nan")
                row["mixed_weight_retention"] = float("nan")
            public = {
                key: value
                for key, value in row.items()
                if key not in {"candidate", "phenotype", "bops_breakdown", "risk"}
            }
            trace.append(public)
            for target in row["matched_budget_targets"]:
                value = {
                    **row,
                    "budget_target": target,
                    "BOPS_deviation": abs(float(row["current_retention"]) - target),
                    "pruned_unit_count": len(row["phenotype"].pruned_unit_ids),
                    "int8_count": sum(
                        precision == "INT8"
                        for precision in row["phenotype"].realized_precision_profile.values()
                    ),
                    "attention_ffn_width_sum": sum(
                        int(width)
                        for domain_id, width in row["candidate"].pruning_width_genes.items()
                        if domain_id.startswith("attention_dh::")
                        or domain_id.startswith("ffn_hidden::")
                    ),
                }
                incumbent = bands[target].get(str(row["candidate_hash"]))
                if incumbent is None or float(value["cumulative_total_taylor"]) < float(
                    incumbent["cumulative_total_taylor"]
                ):
                    bands[target][str(row["candidate_hash"])] = value
        selected_path.append(
            {
                key: value
                for key, value in selected.items()
                if key not in {"candidate", "phenotype", "bops_breakdown", "risk"}
            }
        )
        current = selected["candidate"]
        current_phenotype = selected["phenotype"]
        current_bops = float(selected["BOPS_after"])
        cumulative_struct += float(selected["delta_J_struct"])
        cumulative_wq += float(selected["delta_J_WQ"])
        cumulative_aq += float(selected["delta_J_AQ"])
        selected_hashes.append(str(selected["candidate_hash"]))
        if (
            all(bands[target] for target in normalized_targets)
            and float(selected["current_retention"])
            < min(normalized_targets) - float(tolerance_abs)
        ):
            termination = "all_budgets_captured_and_path_below_minimum_band"
            break
    else:
        termination = "maximum_steps_reached"

    missing = [target for target in normalized_targets if not bands[target]]
    if missing:
        raise RuntimeError(f"conservative_multi_budget_band_empty:{missing}")
    winners = {
        target: select_budget_winner(list(bands[target].values()), target=target)
        for target in normalized_targets
    }
    return {
        "targets": normalized_targets,
        "winners": winners,
        "budget_candidates": {
            target: list(bands[target].values()) for target in normalized_targets
        },
        "trace": trace,
        "selected_path": selected_path,
        "selected_candidate_hashes": selected_hashes,
        "selected_step_count": len(selected_path),
        "visited_action_count": len(trace),
        "budget_band_candidate_counts": {
            target: len(bands[target]) for target in normalized_targets
        },
        "budget_reached": {target: True for target in normalized_targets},
        "termination_reason": termination,
        **empty_search_loop_runtime_audit(),
        "activation_taylor_used_for_fitness": True,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
        "elementwise_abs_before_reduction": True,
        "repair_count": 0,
    }


def _stable_mapping_hash(value: Mapping[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            dict(sorted(value.items())),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _public_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"candidate", "phenotype", "bops_breakdown", "risk"}
    }


__all__ = [
    "combine_precision_action_risk",
    "empty_search_loop_runtime_audit",
    "run_conservative_joint_greedy",
    "run_conservative_joint_greedy_multi_budget",
    "select_budget_winner",
    "select_stage2_budget_pool",
]
