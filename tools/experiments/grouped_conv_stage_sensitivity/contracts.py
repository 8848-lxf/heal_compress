"""Pure experiment contracts for grouped-convolution stage sensitivity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pruning.config import GroupedConvConfig, GroupedConvSelectionPolicy
from pruning.selection.grouped_conv import select_grouped_conv_channels


STAGE_PREFIXES = {
    "stage0": "pyramid_backbone.resnet.layer0.",
    "stage1": "pyramid_backbone.resnet.layer1.",
    "stage2": "pyramid_backbone.resnet.layer2.",
}
EXPECTED_STAGE_COUNTS = {"stage0": 3, "stage1": 5, "stage2": 8}
EXPECTED_MODULE_PATHS = {
    stage: [f"{STAGE_PREFIXES[stage]}{index}.conv2" for index in range(count)]
    for stage, count in EXPECTED_STAGE_COUNTS.items()
}
LEGAL_WIDTHS = (4, 8, 16, 32, 64, 128, 256, 512)
STRATEGIES = (
    "taylor_independent_group_ranking",
    "torch_pruning_l2_shared_position",
)
STAGE_SCOPES = ("stage0_only", "stage1_only", "stage2_only", "all_grouped_stages")
STRENGTHS = ("mild", "aggressive")


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def classify_stage(module_path: str, prefixes: Mapping[str, str] = STAGE_PREFIXES) -> str | None:
    """Classify only through explicit module prefixes."""

    matches = [stage for stage, prefix in prefixes.items() if str(module_path).startswith(str(prefix))]
    if len(matches) > 1:
        raise ValueError(f"ambiguous stage prefixes for {module_path}: {matches}")
    return matches[0] if matches else None


def build_stage_inventory(
    source_rows: Sequence[Any],
    *,
    prefixes: Mapping[str, str] = STAGE_PREFIXES,
) -> dict[str, Any]:
    """Normalize grouped Conv rows and report the explicit 3/5/8 contract."""

    rows: list[dict[str, Any]] = []
    unclassified: list[str] = []
    for source in source_rows:
        module_type = str(_value(source, "module_type", ""))
        groups = int(_value(source, "groups", 1) or 1)
        if module_type not in {"Conv2d", "ConvTranspose2d"} or groups <= 1:
            continue
        path = str(_value(source, "module_path", ""))
        stage = classify_stage(path, prefixes)
        if stage is None:
            unclassified.append(path)
            continue
        in_channels = int(_value(source, "in_channels", 0) or 0)
        out_channels = int(_value(source, "out_channels", 0) or 0)
        in_divisible = in_channels > 0 and in_channels % groups == 0
        out_divisible = out_channels > 0 and out_channels % groups == 0
        scope_ids = _value(source, "scope_ids", None)
        if scope_ids is None:
            scope_id = _value(source, "scope_id", "")
            scope_ids = [scope_id] if scope_id else []
        normalized = {
            "stage": stage,
            "module_path": path,
            "module_type": module_type,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "groups": groups,
            "input_channels_per_group": in_channels // groups if in_divisible else None,
            "output_channels_per_group": out_channels // groups if out_divisible else None,
            "in_channels_divisible_by_groups": in_divisible,
            "out_channels_divisible_by_groups": out_divisible,
            "scope_id": str(scope_ids[0]) if len(scope_ids) == 1 else "",
            "scope_ids": sorted(str(value) for value in scope_ids),
            "root_pruning_allowed": bool(_value(source, "root_pruning_allowed", False)),
            "dependency_input_pruning_allowed": bool(
                _value(
                    source,
                    "dependency_input_pruning_allowed",
                    _value(source, "input_dependency_pruning_allowed", False),
                )
            ),
            "root_axis": str(_value(source, "root_axis", "out")),
            "protected_reason": str(_value(source, "protected_reason", "")),
            "trace_source": str(_value(source, "trace_source", "formal_trace")),
        }
        rows.append(normalized)
    rows.sort(key=lambda row: row["module_path"])
    observed = {
        stage: sum(row["stage"] == stage for row in rows)
        for stage in prefixes
    }
    observed_paths = {stage: {row["module_path"] for row in rows if row["stage"] == stage} for stage in prefixes}
    expected_paths = {stage: set(EXPECTED_MODULE_PATHS.get(stage, ())) for stage in prefixes}
    missing = sorted(path for stage in prefixes for path in expected_paths[stage] - observed_paths[stage])
    additional = sorted(path for stage in prefixes for path in observed_paths[stage] - expected_paths[stage])
    legality_errors = [
        row["module_path"]
        for row in rows
        if not row["in_channels_divisible_by_groups"]
        or not row["out_channels_divisible_by_groups"]
        or not row["scope_ids"]
    ]
    mismatch = (
        observed != dict(EXPECTED_STAGE_COUNTS)
        or bool(unclassified)
        or bool(missing)
        or bool(additional)
        or bool(legality_errors)
    )
    return {
        "stage_prefixes": dict(prefixes),
        "expected_stage_counts": dict(EXPECTED_STAGE_COUNTS),
        "observed_stage_counts": observed,
        "expected_total_count": sum(EXPECTED_STAGE_COUNTS.values()),
        "total_count": len(rows),
        "inventory_count_mismatch": mismatch,
        "missing_module_paths": missing,
        "additional_module_paths": additional,
        "unclassified_grouped_modules": sorted(unclassified),
        "legality_error_module_paths": legality_errors,
        "rows": rows,
        "schema_version": "grouped-conv-stage-inventory-v2",
    }


def compute_pruning_strength_plan(
    inventory_rows: Sequence[Mapping[str, Any]],
    *,
    legal_widths: Sequence[int] = LEGAL_WIDTHS,
) -> dict[str, Any]:
    """Apply the common-width mild/aggressive algorithm exactly per stage."""

    stage_plans: dict[str, dict[str, Any]] = {}
    for stage in STAGE_PREFIXES:
        rows = [dict(row) for row in inventory_rows if str(row.get("stage")) == stage]
        widths = [int(row["output_channels_per_group"]) for row in rows]
        common = [
            int(width)
            for width in legal_widths
            if widths and all(int(width) < original for original in widths)
        ]
        if not common:
            stage_plans[stage] = {
                "stage": stage,
                "module_count": len(rows),
                "module_paths": [row["module_path"] for row in rows],
                "original_output_channels_per_group": widths,
                "common_legal_widths": [],
                "status": "stage_output_pruning_infeasible",
                "mild": None,
                "mild_status": "stage_output_pruning_infeasible",
                "aggressive": None,
                "aggressive_status": "stage_output_pruning_infeasible",
            }
            continue
        mild = min(
            common,
            key=lambda target: (
                sum(abs(target / original - 0.5) for original in widths) / len(widths),
                -target,
            ),
        )
        aggressive_options = [width for width in common if width < mild]
        aggressive = max(aggressive_options) if aggressive_options else None
        stage_plans[stage] = {
            "stage": stage,
            "module_count": len(rows),
            "module_paths": [row["module_path"] for row in rows],
            "original_output_channels_per_group": widths,
            "common_legal_widths": common,
            "status": "feasible",
            "mild": mild,
            "mild_status": "planned",
            "aggressive": aggressive,
            "aggressive_status": "planned" if aggressive is not None else "aggressive_unavailable",
        }
    return {
        "legal_widths": [int(value) for value in legal_widths],
        "stages": stage_plans,
        "schema_version": "grouped-conv-pruning-strength-plan-v2",
    }


def build_candidate_matrix(strength_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build all 16 declared rows while preserving explicit infeasibility."""

    stages = dict(strength_plan["stages"])
    rows: list[dict[str, Any]] = []
    for stage_scope in STAGE_SCOPES:
        scope_stages = list(STAGE_PREFIXES) if stage_scope == "all_grouped_stages" else [stage_scope.removesuffix("_only")]
        for strength in STRENGTHS:
            targets = {stage: stages[stage].get(strength) for stage in scope_stages}
            unavailable = [stage for stage, target in targets.items() if target is None]
            status = "infeasible" if unavailable else "planned"
            if unavailable:
                reasons = [
                    f"{stage}:{stages[stage].get(f'{strength}_status', stages[stage].get('status', 'unavailable'))}"
                    for stage in unavailable
                ]
                reason = ";".join(reasons)
            else:
                reason = ""
            for strategy in STRATEGIES:
                candidate_id = f"{stage_scope}__{strength}__{strategy}"
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "stage_scope": stage_scope,
                        "strength": strength,
                        "strategy": strategy,
                        "active_stages": scope_stages,
                        "target_widths_by_stage": {
                            stage: int(target) for stage, target in targets.items() if target is not None
                        },
                        "candidate_status": status,
                        "infeasible_reason": reason,
                        "source_model": "original_checkpoint_one_shot",
                    }
                )
    return rows


def select_independent_group_positions(
    per_group_scores: Mapping[int, Sequence[float]],
    *,
    keep_width: int,
) -> dict[str, Any]:
    decision = select_grouped_conv_channels(
        per_group_scores,
        final_channels_per_group=int(keep_width),
        config=GroupedConvConfig(
            allowed_channels_per_group=LEGAL_WIDTHS,
            selection_policy=GroupedConvSelectionPolicy.INDEPENDENT_GROUP_TOPK,
        ),
    )
    return {
        **decision.to_dict(),
        "group_keep_map": dict(decision.group_keep_map),
        "group_prune_map": dict(decision.group_prune_map),
        "selection": "independent_group_topk",
    }


def select_tp_shared_positions(
    per_group_scores: Mapping[int, Sequence[float]],
    *,
    keep_width: int,
) -> dict[str, Any]:
    scores = {int(group): [float(value) for value in values] for group, values in per_group_scores.items()}
    decision = select_grouped_conv_channels(
        scores,
        final_channels_per_group=int(keep_width),
        config=GroupedConvConfig(
            allowed_channels_per_group=LEGAL_WIDTHS,
            selection_policy=GroupedConvSelectionPolicy.SHARED_LOCAL_MEAN,
        ),
    )
    width = len(next(iter(scores.values())))
    shared = [sum(values[index] for values in scores.values()) / len(scores) for index in range(width)]
    return {
        **decision.to_dict(),
        "group_keep_map": dict(decision.group_keep_map),
        "group_prune_map": dict(decision.group_prune_map),
        "selection": "shared_position_topk",
        "shared_local_mean_scores": shared,
        "torch_pruning_semantics": "base_pruner.py:imp.view(ch_groups,-1).mean(dim=0)",
    }


def filter_active_root_units(units: Sequence[Any], allowed_roots: set[str]) -> list[Any]:
    """Filter candidates by their real root path without touching closure members."""

    return [unit for unit in units if str(_value(unit, "root_module_path", "")) in allowed_roots]


__all__ = [
    "EXPECTED_MODULE_PATHS",
    "EXPECTED_STAGE_COUNTS",
    "LEGAL_WIDTHS",
    "STAGE_PREFIXES",
    "STAGE_SCOPES",
    "STRATEGIES",
    "STRENGTHS",
    "build_candidate_matrix",
    "build_stage_inventory",
    "classify_stage",
    "compute_pruning_strength_plan",
    "filter_active_root_units",
    "select_independent_group_positions",
    "select_tp_shared_positions",
]
