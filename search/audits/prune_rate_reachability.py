from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import floor, prod
from typing import Any


def _rate(removed: int, original: int) -> float:
    if original <= 0:
        raise ValueError("the original count must be positive")
    return float(removed) / float(original)


def prune_rate_metrics(
    *,
    requested_prune_rate: float,
    predicted_original_params: int,
    predicted_candidate_params: int,
    physical_original_params: int,
    physical_candidate_params: int,
    prunable_original_params: int,
    atomic_unit_count: int,
    pruned_atomic_unit_count: int,
    channel_count: int,
    pruned_channel_count: int,
) -> dict[str, float]:
    """Return explicitly named pruning metrics without aliasing their denominators."""

    return {
        "requested_prune_rate": float(requested_prune_rate),
        "predicted_full_param_prune_rate": _rate(
            predicted_original_params - predicted_candidate_params,
            predicted_original_params,
        ),
        "physical_full_param_prune_rate": _rate(
            physical_original_params - physical_candidate_params,
            physical_original_params,
        ),
        "physical_prunable_param_prune_rate": _rate(
            physical_original_params - physical_candidate_params,
            prunable_original_params,
        ),
        "atomic_unit_prune_ratio": _rate(
            pruned_atomic_unit_count, atomic_unit_count
        ),
        "channel_prune_ratio": _rate(pruned_channel_count, channel_count),
    }


def _slice_linear_indices(shape: Sequence[int], axis: int, indices: Iterable[int]) -> set[int]:
    if not shape:
        raise ValueError("parameter shape must not be empty")
    axis = int(axis)
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f"axis {axis} is invalid for shape {tuple(shape)}")

    before = prod(int(value) for value in shape[:axis])
    selected = sorted({int(value) for value in indices})
    axis_width = int(shape[axis])
    after = prod(int(value) for value in shape[axis + 1 :])
    if any(value < 0 or value >= axis_width for value in selected):
        raise ValueError(
            f"slice indices {selected} are invalid for axis width {axis_width}"
        )

    result: set[int] = set()
    block_width = axis_width * after
    for prefix in range(before):
        base = prefix * block_width
        for value in selected:
            start = base + value * after
            result.update(range(start, start + after))
    return result


def parameter_slice_union(
    parameter_shapes: Mapping[str, Sequence[int]],
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count dependency slices by element-wise union, retaining overlap evidence."""

    union_by_parameter: dict[str, set[int]] = {}
    raw_element_sum = 0
    row_results: list[dict[str, Any]] = []

    for row in rows:
        name = str(row["parameter_name"])
        if name not in parameter_shapes:
            raise KeyError(f"parameter slice refers to missing parameter: {name}")
        linear = _slice_linear_indices(
            parameter_shapes[name], int(row["axis"]), row["indices"]
        )
        seen = union_by_parameter.setdefault(name, set())
        overlap_count = len(seen.intersection(linear))
        seen.update(linear)
        raw_element_sum += len(linear)
        row_results.append(
            {
                **dict(row),
                "element_count": len(linear),
                "overlap_count": overlap_count,
            }
        )

    counts = {name: len(values) for name, values in sorted(union_by_parameter.items())}
    union_count = sum(counts.values())
    return {
        "raw_element_sum": raw_element_sum,
        "global_union_element_count": union_count,
        "duplicate_or_overlap_element_count": raw_element_sum - union_count,
        "global_union_by_parameter": counts,
        "row_results": row_results,
    }


def _dense_max_pruned(domain: Mapping[str, Any], cap: float) -> tuple[list[str], float]:
    unit_ids = list(domain["unit_ids"])
    width = len(unit_ids)
    alignment = max(1, int(domain.get("alignment", 1)))
    minimum_width = max(0, int(domain.get("minimum_width", 0)))
    maximum_by_width = max(0, width - minimum_width)
    maximum_by_cap = floor(cap * width + 1e-12)
    maximum = min(maximum_by_width, maximum_by_cap)
    pruned_count = (maximum // alignment) * alignment
    return unit_ids[:pruned_count], _rate(pruned_count, width)


def _grouped_max_pruned(domain: Mapping[str, Any], cap: float) -> tuple[list[str], float]:
    physical_groups = {
        int(group_id): list(unit_ids)
        for group_id, unit_ids in domain["physical_groups"].items()
    }
    if not physical_groups:
        return [], 0.0
    widths = {len(unit_ids) for unit_ids in physical_groups.values()}
    if len(widths) != 1:
        raise ValueError("grouped domains require equal physical-group widths")
    group_width = widths.pop()
    allowed_keep = sorted(
        {
            int(value)
            for value in domain.get("allowed_channels_per_group", [group_width])
            if 0 < int(value) <= group_width
        }
    )
    if group_width not in allowed_keep:
        allowed_keep.append(group_width)
        allowed_keep.sort()

    legal_keep = [
        keep
        for keep in allowed_keep
        if (group_width - keep) / group_width <= cap + 1e-12
    ]
    selected_keep = min(legal_keep) if legal_keep else group_width
    per_group_pruned = group_width - selected_keep
    pruned: list[str] = []
    for group_id in sorted(physical_groups):
        pruned.extend(physical_groups[group_id][:per_group_pruned])
    total_width = group_width * len(physical_groups)
    return pruned, _rate(len(pruned), total_width)


def solve_independent_domain_max(
    domains: Mapping[str, Mapping[str, Any]],
    *,
    per_domain_cap: float,
) -> dict[str, Any]:
    """Maximize legal removals by enumerating each domain's width constraints.

    The solver deliberately ignores unit importance order. It is an independent
    reachability bound, not a candidate-ranking implementation.
    """

    if not 0.0 <= per_domain_cap <= 1.0:
        raise ValueError("per_domain_cap must be within [0, 1]")

    all_pruned: list[str] = []
    counts: dict[str, int] = {}
    rates: dict[str, float] = {}
    for domain_id in sorted(domains):
        domain = domains[domain_id]
        kind = str(domain.get("kind", "dense"))
        if kind == "grouped":
            pruned, rate = _grouped_max_pruned(domain, per_domain_cap)
        elif kind == "dense":
            pruned, rate = _dense_max_pruned(domain, per_domain_cap)
        else:
            raise ValueError(f"unsupported domain kind: {kind}")
        all_pruned.extend(pruned)
        counts[domain_id] = len(pruned)
        rates[domain_id] = rate

    return {
        "pruned_unit_ids": all_pruned,
        "domain_pruned_counts": counts,
        "maximum_domain_prune_rates": rates,
    }


def compare_reachability_masks(
    *,
    planner_pruned_ids: Iterable[str],
    independent_pruned_ids: Iterable[str],
    planner_predicted_rate: float,
    independent_predicted_rate: float,
) -> dict[str, Any]:
    planner = set(planner_pruned_ids)
    independent = set(independent_pruned_ids)
    mask_equal = planner == independent
    difference = round(float(independent_predicted_rate) - float(planner_predicted_rate), 12)
    if mask_equal:
        reason = "masks_equal"
    elif difference > 0:
        reason = "planner_selection_or_projection_under_reaches_legal_space"
    else:
        reason = "different_masks_without_independent_rate_gain"
    return {
        "mask_equal": mask_equal,
        "planner_only_unit_ids": sorted(planner - independent),
        "independent_only_unit_ids": sorted(independent - planner),
        "independent_minus_planner_rate": difference,
        "difference_reason": reason,
    }


def diagnose_historical_mask(
    input_unit_ids: Sequence[str],
    *,
    current_unit_ids: set[str],
    protected_unit_ids: set[str],
    rejection_reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Classify an old mask under current rules without changing the mask."""

    explicit = dict(rejection_reasons or {})
    rejected: dict[str, str] = {}
    for unit_id in input_unit_ids:
        if unit_id in protected_unit_ids:
            rejected[unit_id] = "protected"
        elif unit_id not in current_unit_ids:
            rejected[unit_id] = explicit.get(unit_id, "missing_unit_mapping")
        elif unit_id in explicit:
            rejected[unit_id] = explicit[unit_id]
    return {
        "input_unit_ids": list(input_unit_ids),
        "output_unit_ids": list(input_unit_ids),
        "repair_applied": False,
        "passed": not rejected,
        "rejections": rejected,
    }


def classify_replay_failure(
    *,
    physical_export_success: bool,
    forward_success: bool,
    evaluation_success: bool,
    map_value: float | None,
    map_reference: float,
    collapse_absolute_drop: float = 0.1,
) -> str:
    if not physical_export_success or not forward_success:
        return "STRUCTURAL_FAILURE"
    if not evaluation_success or map_value is None:
        return "EVALUATION_FAILURE"
    if float(map_reference) - float(map_value) > float(collapse_absolute_drop):
        return "ACCURACY_COLLAPSE"
    return "PASS"
