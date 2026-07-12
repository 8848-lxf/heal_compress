"""Structure pairing and descriptive analysis helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


STRUCTURE_FIELDS = (
    "canonical_module_name",
    "module_type",
    "weight_shape",
    "bias_shape",
    "in_channels",
    "out_channels",
    "in_features",
    "out_features",
    "num_features",
    "groups",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "output_padding",
)


def _shape_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    modules = snapshot.get("modules", [])
    if isinstance(modules, Mapping):
        rows = [
            {"canonical_module_name": str(name), **dict(value)}
            for name, value in modules.items()
        ]
    else:
        rows = [dict(value) for value in modules]
    normalized = [
        {field: row.get(field) for field in STRUCTURE_FIELDS}
        for row in rows
    ]
    normalized.sort(key=lambda row: str(row.get("canonical_module_name", "")))
    return {
        "parameter_count": int(snapshot.get("parameter_count", 0)),
        "modules": normalized,
    }


def physical_shape_hash(snapshot: Mapping[str, Any]) -> str:
    raw = json.dumps(_shape_payload(snapshot), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_pairwise_structure(
    left_snapshot: Mapping[str, Any],
    right_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    left = _shape_payload(left_snapshot)
    right = _shape_payload(right_snapshot)
    left_modules = {row["canonical_module_name"]: row for row in left["modules"]}
    right_modules = {row["canonical_module_name"]: row for row in right["modules"]}
    names = sorted(set(left_modules) | set(right_modules))
    mismatches = [
        {
            "module_path": name,
            "left": left_modules.get(name),
            "right": right_modules.get(name),
        }
        for name in names
        if left_modules.get(name) != right_modules.get(name)
    ]
    parameter_count_match = left["parameter_count"] == right["parameter_count"]
    left_hash = physical_shape_hash(left_snapshot)
    right_hash = physical_shape_hash(right_snapshot)
    valid = not mismatches and parameter_count_match and left_hash == right_hash
    return {
        "comparison_valid": valid,
        "failure_reason": "" if valid else "structure_mismatch",
        "parameter_count_match": parameter_count_match,
        "left_parameter_count": left["parameter_count"],
        "right_parameter_count": right["parameter_count"],
        "physical_shape_hash_match": left_hash == right_hash,
        "left_physical_shape_hash": left_hash,
        "right_physical_shape_hash": right_hash,
        "mismatches": mismatches,
    }


def cumulative_interaction(
    *,
    all_stage_drop: float,
    single_stage_drops: Mapping[str, float],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    summed = sum(float(value) for value in single_stage_drops.values())
    interaction = float(all_stage_drop) - summed
    if interaction > tolerance:
        classification = "superadditive"
    elif interaction < -tolerance:
        classification = "subadditive_or_saturation"
    else:
        classification = "approximately_additive"
    return {
        "all_stage_delta_map": float(all_stage_drop),
        "single_stage_delta_map": {key: float(value) for key, value in single_stage_drops.items()},
        "sum_single_stage_delta_map": summed,
        "cumulative_interaction": interaction,
        "interaction_class": classification,
        "interpretation_scope": "descriptive_fixed_500_frame_evaluation",
    }


def build_strategy_comparisons(
    models: list[Mapping[str, Any]],
    pair_rows: list[Mapping[str, Any]],
    *,
    material_difference: float = 0.005,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("stage_scope")), str(row.get("strength")), str(row.get("strategy"))): row
        for row in models
    }
    output: list[dict[str, Any]] = []
    metric_keys = ("ap_0.03", "ap_0.30", "ap_0.50", "ap_0.70", "mAP")
    for pair in pair_rows:
        scope = str(pair["stage_scope"])
        strength = str(pair["strength"])
        independent = by_key.get((scope, strength, "taylor_independent_group_ranking"))
        shared = by_key.get((scope, strength, "torch_pruning_l2_shared_position"))
        valid = bool(pair.get("comparison_valid")) and independent is not None and shared is not None
        row: dict[str, Any] = {
            "stage_scope": scope,
            "strength": strength,
            "comparison_valid": valid,
            "failure_reason": "" if valid else str(pair.get("failure_reason") or "missing_evaluation_row"),
            "independent_model_id": independent.get("model_id") if independent else None,
            "tp_shared_model_id": shared.get("model_id") if shared else None,
        }
        if not valid:
            for key in metric_keys:
                row[f"{key}_difference_independent_minus_tp"] = None
            row["retention_difference_independent_minus_tp"] = None
            row["strategy_verdict"] = "inconclusive"
            output.append(row)
            continue
        for key in metric_keys:
            row[f"{key}_difference_independent_minus_tp"] = float(independent[key]) - float(shared[key])
        difference = row["mAP_difference_independent_minus_tp"]
        row["retention_difference_independent_minus_tp"] = float(
            independent["mAP_retention"]
        ) - float(shared["mAP_retention"])
        row["independent_mAP"] = float(independent["mAP"])
        row["tp_shared_mAP"] = float(shared["mAP"])
        row["actual_parameter_reduction_match"] = int(
            independent["actual_parameter_reduction"]
        ) == int(shared["actual_parameter_reduction"])
        if difference > material_difference:
            verdict = "independent_better"
        elif difference < -material_difference:
            verdict = "tp_shared_better"
        else:
            verdict = "no_material_difference"
        row["strategy_verdict"] = verdict
        row["material_difference_threshold"] = float(material_difference)
        output.append(row)
    return output


def build_sensitivity_rankings(models: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in models
        if str(row.get("stage_scope")) in {"stage0_only", "stage1_only", "stage2_only"}
    ]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in candidates:
        groups.setdefault((str(row["strategy"]), str(row["strength"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (strategy, strength), rows in sorted(groups.items()):
        absolute_order = sorted(
            rows,
            key=lambda row: (-float(row["mAP_absolute_drop"]), str(row["stage_scope"])),
        )
        normalized_order = sorted(
            rows,
            key=lambda row: (
                -float(row["delta_map_per_million_pruned_parameters"]),
                str(row["stage_scope"]),
            ),
        )
        absolute_rank = {str(row["model_id"]): rank for rank, row in enumerate(absolute_order, 1)}
        normalized_rank = {
            str(row["model_id"]): rank for rank, row in enumerate(normalized_order, 1)
        }
        for row in sorted(rows, key=lambda item: str(item["stage_scope"])):
            scope = str(row["stage_scope"])
            output.append(
                {
                    "strategy": strategy,
                    "strength": strength,
                    "stage": scope.removesuffix("_only"),
                    "stage_scope": scope,
                    "model_id": row["model_id"],
                    "delta_mAP": float(row["mAP_absolute_drop"]),
                    "mAP_retention": float(row["mAP_retention"]),
                    "delta_map_per_million_pruned_parameters": float(
                        row["delta_map_per_million_pruned_parameters"]
                    ),
                    "actual_parameter_reduction": int(row["actual_parameter_reduction"]),
                    "grouped_layer_parameter_reduction": int(
                        row.get("grouped_layer_parameter_reduction", 0)
                    ),
                    "closure_module_count": int(row.get("closure_module_count", 0)),
                    "active_root_parameter_reduction": int(
                        row.get("active_root_parameter_reduction", 0)
                    ),
                    "dependency_driven_parameter_reduction": int(
                        row.get("dependency_driven_parameter_reduction", 0)
                    ),
                    "absolute_sensitivity_rank": absolute_rank[str(row["model_id"])],
                    "parameter_normalized_sensitivity_rank": normalized_rank[
                        str(row["model_id"])
                    ],
                }
            )
    return output


def stage1_attribution(
    models: list[Mapping[str, Any]],
    *,
    material_difference: float = 0.005,
) -> dict[str, Any]:
    by_key = {
        (str(row.get("stage_scope")), str(row.get("strength")), str(row.get("strategy"))): row
        for row in models
    }
    strategies = (
        "taylor_independent_group_ranking",
        "torch_pruning_l2_shared_position",
    )
    stage1 = {
        strategy: by_key.get(("stage1_only", "mild", strategy)) for strategy in strategies
    }
    stage2 = {
        strategy: by_key.get(("stage2_only", "mild", strategy)) for strategy in strategies
    }
    comparable = all(stage1.values()) and all(stage2.values())
    intrinsic = bool(comparable) and all(
        float(stage1[strategy]["mAP_absolute_drop"])
        > float(stage2[strategy]["mAP_absolute_drop"]) + material_difference
        for strategy in strategies
    )
    stage1_strategy_gap = (
        float(stage1[strategies[0]]["mAP"]) - float(stage1[strategies[1]]["mAP"])
        if comparable
        else None
    )
    shared_exacerbation = bool(comparable) and float(stage1_strategy_gap) > material_difference
    all_stage_rows = [
        row for row in models if str(row.get("stage_scope")) == "all_grouped_stages"
    ]
    return {
        "stage1_intrinsic_sensitivity": "supported" if intrinsic else ("not supported" if comparable else "inconclusive"),
        "shared_position_exacerbation": "supported" if shared_exacerbation else ("not supported" if comparable else "inconclusive"),
        "combined_attribution": "supported" if intrinsic and shared_exacerbation else ("not supported" if comparable else "inconclusive"),
        "stage1_independent_mAP": float(stage1[strategies[0]]["mAP"]) if comparable else None,
        "stage1_tp_shared_mAP": float(stage1[strategies[1]]["mAP"]) if comparable else None,
        "stage2_independent_mAP": float(stage2[strategies[0]]["mAP"]) if comparable else None,
        "stage2_tp_shared_mAP": float(stage2[strategies[1]]["mAP"]) if comparable else None,
        "stage1_strategy_gap_independent_minus_tp": stage1_strategy_gap,
        "all_stage_cumulative_effect": "observed" if all_stage_rows else "inconclusive",
        "all_stage_cumulative_effect_reason": "" if all_stage_rows else "all_grouped_stages candidates were infeasible under the declared output-width contract",
        "material_difference_threshold": float(material_difference),
    }


__all__ = [
    "build_sensitivity_rankings",
    "build_strategy_comparisons",
    "cumulative_interaction",
    "physical_shape_hash",
    "stage1_attribution",
    "validate_pairwise_structure",
]
