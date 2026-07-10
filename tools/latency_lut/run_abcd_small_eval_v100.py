#!/usr/bin/env python3
"""Small full-model A/B/C/D pruning evaluation for lidar_pyramid v10.0.

This runner treats A/B/C/D as local ordinary grouped-Conv2d resolver policies
inside the latest supported full-model pruning surface.  The primary pruning
metric is always the full physical model parameter ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface  # noqa: E402
from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator  # noqa: E402
from heal_compress.pruning.grouped_conv import (  # noqa: E402
    resolve_grouped_conv_d_compact_frontfill_reblock,
    resolve_grouped_conv_input_keep,
)
from heal_compress.pruning.grouped_conv_policy_registry import (  # noqa: E402
    GroupedConvPolicyRegistry,
    parse_group_conv_policy_choice,
)
from heal_compress.pruning.physical_prune_plan import (  # noqa: E402
    GlobalPhysicalPrunePlan,
    ModuleAxisPruneRequest,
)
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan  # noqa: E402
from heal_compress.pruning.units import (  # noqa: E402
    atomic_prune_unit_rows,
    concrete_pruning_group_rows,
    coupled_channel_unit_rows,
    dataclass_to_json_dict,
)
from heal_compress.search.importance import (  # noqa: E402
    compute_group_importance,
    compute_scope_channel_importance_map,
)
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import build_op_graph  # noqa: E402
from heal_compress.utils.model_utils import resolve_device  # noqa: E402
from heal_compress.pruning.model_io import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    build_importance_calibration_data,
    build_protected_layers,
    configure_grouped_conv_pruning_fns,
    count_params,
    load_heal_model,
    move_batch_to_device,
    setup_logger as setup_prune_logger,
)


OUT_DEFAULT = Path("outputs/latency_lut/abcd_small_eval_v100")
POLICY_TO_MODE = {
    "A": "flat_output_groups_fixed",
    "B": "group_balanced_output_groups_fixed",
    "C": "true_group_block_pruning",
    "D": "group_coarsening_zero_padded_reblock",
}
STRATEGY_NAMES = {
    "A": "flat_output_groups_fixed",
    "B": "group_balanced_output_groups_fixed",
    "C": "true_group_block_pruning",
    "D": "compact_frontfill_zero_padded_reblock",
}
FAILURE_ENUM = {
    "shape_simulator_illegal",
    "physical_prune_failed",
    "synthetic_forward_failed",
    "eval_forward_failed",
    "AP_eval_failed",
    "latency_eval_failed",
    "missing_upstream_closure",
    "missing_downstream_closure",
    "missing_bn_closure",
    "residual_closure_incomplete",
    "concat_closure_incomplete",
    "grouped_conv_strategy_bug",
    "ordered_keep_indices_mismatch",
    "frontfill_source_kernel_too_wide",
    "target_budget_unreachable",
    "domain_unit_selection_bug",
    "coupled_unit_mapping_missing",
    "unsupported_pruning_domain",
    "cuda_oom",
    "data_loader_error",
    "metric_evaluator_error",
    "unknown_exception",
}
WHY_NOT_REACHED_VALUES = (
    "protected surface too large",
    "domain unit ratio does not match parameter ratio",
    "grouped conv policy constraints",
    "residual/concat closure constraints",
    "min channel constraints",
    "shape alignment constraints",
    "strategy-specific legality constraints",
    "forward failure",
    "eval failure",
    "unknown",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _module_param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters(recurse=False)))


def _is_grouped_conv2d(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and int(getattr(module, "groups", 1)) > 1


def _is_ordinary_grouped_conv2d(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and int(module.groups) > 1
        and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
    )


def compute_param_inventory(model: nn.Module) -> dict[str, int]:
    total_params = int(sum(p.numel() for p in model.parameters()))
    grouped_conv_params = 0
    non_grouped_conv_params = 0
    total_conv_params = 0
    ordinary_grouped_conv_params = 0
    depthwise_conv_params = 0
    bn_params = 0
    linear_params = 0
    for module in model.modules():
        params = _module_param_count(module)
        if isinstance(module, nn.Conv2d):
            total_conv_params += params
            if int(module.groups) > 1:
                grouped_conv_params += params
                if _is_ordinary_grouped_conv2d(module):
                    ordinary_grouped_conv_params += params
                else:
                    depthwise_conv_params += params
            else:
                non_grouped_conv_params += params
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            bn_params += params
        elif isinstance(module, nn.Linear):
            linear_params += params
    return {
        "total_params": total_params,
        "grouped_conv_params": int(grouped_conv_params),
        "ordinary_grouped_conv_params": int(ordinary_grouped_conv_params),
        "depthwise_conv_params": int(depthwise_conv_params),
        "non_grouped_conv_params": int(non_grouped_conv_params),
        "total_conv_params": int(total_conv_params),
        "bn_params": int(bn_params),
        "linear_params": int(linear_params),
    }


def _ratio(before: int | float, after: int | float) -> float:
    before_f = float(before)
    if before_f <= 0:
        return 0.0
    return 1.0 - float(after) / before_f


def _budget_reasons(
    *,
    actual_full: float,
    target_full: float,
    actual_domain: float,
    target_domain: float,
    forward_ok: bool = True,
    eval_ok: bool = True,
) -> str:
    reasons: list[str] = []
    if not forward_ok:
        reasons.append("forward failure")
    if not eval_ok:
        reasons.append("eval failure")
    if actual_full + 1e-9 < target_full:
        reasons.append("domain unit ratio does not match parameter ratio")
    if actual_full + 1e-9 < target_full and abs(actual_domain - target_domain) > 0.025:
        reasons.append("min channel constraints")
    return ";".join(dict.fromkeys(reasons))


def build_target_budget_report(
    *,
    strategy: str,
    target_param_prune_ratio_full_model: float,
    target_domain_unit_prune_ratio: float,
    before: dict[str, int],
    after: dict[str, int],
    supported_surface_params_before: int,
    supported_surface_params_after: int,
    total_units: int,
    pruned_units: int,
    forward_ok: bool = True,
    eval_ok: bool = True,
) -> dict[str, Any]:
    actual_full = _ratio(before.get("total_params", 0), after.get("total_params", 0))
    actual_domain = float(pruned_units) / float(total_units) if total_units else 0.0
    why = _budget_reasons(
        actual_full=actual_full,
        target_full=target_param_prune_ratio_full_model,
        actual_domain=actual_domain,
        target_domain=target_domain_unit_prune_ratio,
        forward_ok=forward_ok,
        eval_ok=eval_ok,
    )
    return {
        "strategy": strategy,
        "target_param_prune_ratio_full_model": float(target_param_prune_ratio_full_model),
        "actual_param_prune_ratio_full_model": actual_full,
        "budget_error_full_model": float(target_param_prune_ratio_full_model) - actual_full,
        "target_domain_unit_prune_ratio": float(target_domain_unit_prune_ratio),
        "actual_domain_unit_prune_ratio": actual_domain,
        "domain_unit_budget_error": float(target_domain_unit_prune_ratio) - actual_domain,
        "baseline_total_params": int(before.get("total_params", 0)),
        "pruned_total_params": int(after.get("total_params", 0)),
        "full_model_param_saving": int(before.get("total_params", 0) - after.get("total_params", 0)),
        "supported_surface_param_before": int(supported_surface_params_before),
        "supported_surface_param_after": int(supported_surface_params_after),
        "actual_param_prune_ratio_supported_surface": _ratio(supported_surface_params_before, supported_surface_params_after),
        "grouped_conv_param_before": int(before.get("grouped_conv_params", 0)),
        "grouped_conv_param_after": int(after.get("grouped_conv_params", 0)),
        "actual_grouped_conv_param_prune_ratio": _ratio(before.get("grouped_conv_params", 0), after.get("grouped_conv_params", 0)),
        "non_grouped_conv_param_before": int(before.get("non_grouped_conv_params", 0)),
        "non_grouped_conv_param_after": int(after.get("non_grouped_conv_params", 0)),
        "actual_non_grouped_conv_param_prune_ratio": _ratio(before.get("non_grouped_conv_params", 0), after.get("non_grouped_conv_params", 0)),
        "total_conv_param_before": int(before.get("total_conv_params", 0)),
        "total_conv_param_after": int(after.get("total_conv_params", 0)),
        "actual_total_conv_param_prune_ratio": _ratio(before.get("total_conv_params", 0), after.get("total_conv_params", 0)),
        "why_not_reached": why,
    }


def _mod(value: int, divisor: int) -> int:
    return int(value) % int(divisor)


def _friendly_channel_count(channels: int) -> bool:
    return int(channels) > 0 and int(channels) % 8 == 0


def compute_shape_alignment_report(model: nn.Module, *, policy: str = "", old_group_map: dict[str, Any] | None = None) -> dict[str, Any]:
    old_group_map = old_group_map or {}
    conv_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    unaligned: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        c_in = int(module.in_channels)
        c_out = int(module.out_channels)
        groups = int(module.groups)
        row = {
            "module_name": name,
            "module_type": "Conv2d",
            "C_in": c_in,
            "C_out": c_out,
            "groups": groups,
            "C_in_mod4": _mod(c_in, 4),
            "C_in_mod8": _mod(c_in, 8),
            "C_in_mod16": _mod(c_in, 16),
            "C_in_mod32": _mod(c_in, 32),
            "C_out_mod4": _mod(c_out, 4),
            "C_out_mod8": _mod(c_out, 8),
            "C_out_mod16": _mod(c_out, 16),
            "C_out_mod32": _mod(c_out, 32),
            "is_tensorcore_friendly_like": bool(_friendly_channel_count(c_in) and _friendly_channel_count(c_out)),
            "is_likely_hardware_friendly": bool(_friendly_channel_count(c_in) and _friendly_channel_count(c_out)),
        }
        conv_rows.append(row)
        if groups <= 1:
            continue
        in_per = c_in // groups if groups else 0
        out_per = c_out // groups if groups else 0
        reasons: list[str] = []
        if c_in % 8:
            reasons.append("C_in_not_multiple_of_8")
        if c_out % 8:
            reasons.append("C_out_not_multiple_of_8")
        if in_per % 4:
            reasons.append("in_per_group_not_multiple_of_4")
        if out_per % 4:
            reasons.append("out_per_group_not_multiple_of_4")
        if groups % 4:
            reasons.append("groups_not_multiple_of_4")
        grouped = {
            **row,
            "in_per_group": in_per,
            "out_per_group": out_per,
            "in_per_group_mod4": _mod(in_per, 4),
            "in_per_group_mod8": _mod(in_per, 8),
            "in_per_group_mod16": _mod(in_per, 16),
            "out_per_group_mod4": _mod(out_per, 4),
            "out_per_group_mod8": _mod(out_per, 8),
            "out_per_group_mod16": _mod(out_per, 16),
            "groups_mod2": _mod(groups, 2),
            "groups_mod4": _mod(groups, 4),
            "groups_mod8": _mod(groups, 8),
            "groups_old": old_group_map.get(name, {}).get("groups_old", ""),
            "groups_new": old_group_map.get(name, {}).get("groups_new", groups),
            "policy": policy,
            "is_unaligned_reason": ";".join(reasons),
            "is_likely_hardware_friendly": not reasons,
        }
        grouped_rows.append(grouped)
        if reasons:
            unaligned.append(grouped)
    return {
        "policy": policy,
        "num_conv2d": len(conv_rows),
        "num_grouped_conv2d": len(grouped_rows),
        "num_unaligned_grouped_conv_shapes": len(unaligned),
        "conv2d": conv_rows,
        "grouped_conv2d": grouped_rows,
        "unaligned_grouped_conv_shapes": unaligned,
    }


def _percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * float(pct)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - rank) + vals[hi] * (rank - lo)


def summarize_latency(values: Sequence[float], *, baseline: dict[str, Any] | None = None) -> dict[str, float]:
    vals = [float(v) for v in values if float(v) > 0.0]
    mean_ms = statistics.mean(vals) if vals else 0.0
    p50_ms = statistics.median(vals) if vals else 0.0
    row: dict[str, float] = {
        "mean_ms": round(mean_ms, 6),
        "median_ms": round(p50_ms, 6),
        "p50_ms": round(p50_ms, 6),
        "p90_ms": round(_percentile(vals, 0.90), 6),
        "p95_ms": round(_percentile(vals, 0.95), 6),
        "min_ms": round(min(vals), 6) if vals else 0.0,
        "max_ms": round(max(vals), 6) if vals else 0.0,
        "std_ms": round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0,
        "FPS_p50": round(1000.0 / p50_ms, 6) if p50_ms > 0 else 0.0,
        "FPS_mean": round(1000.0 / mean_ms, 6) if mean_ms > 0 else 0.0,
        "num_timed_frames": float(len(vals)),
    }
    if baseline:
        base_p50 = float(baseline.get("p50_ms", 0.0) or 0.0)
        base_mean = float(baseline.get("mean_ms", 0.0) or 0.0)
        row["speedup_p50"] = round(base_p50 / p50_ms, 6) if p50_ms > 0 and base_p50 > 0 else 0.0
        row["speedup_mean"] = round(base_mean / mean_ms, 6) if mean_ms > 0 and base_mean > 0 else 0.0
    else:
        row["speedup_p50"] = 1.0
        row["speedup_mean"] = 1.0
    return row


def _ordered_unique_valid(indices: Iterable[int], total: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        idx = int(idx)
        if idx < 0 or idx >= int(total) or idx in seen:
            continue
        out.append(idx)
        seen.add(idx)
    return out


def _divisors(value: int) -> list[int]:
    return [v for v in range(1, int(value) + 1) if int(value) % v == 0]


def _expand_or_trim_keep(preferred: list[int], *, total: int, target_count: int) -> list[int]:
    keep = list(preferred)
    if len(keep) > target_count:
        return keep[:target_count]
    if len(keep) < target_count:
        seen = set(keep)
        for idx in range(int(total)):
            if idx not in seen:
                keep.append(idx)
                seen.add(idx)
                if len(keep) >= target_count:
                    break
    return keep


def choose_d_compact_frontfill_keep(
    module: nn.Conv2d,
    preferred_keep: Sequence[int],
    *,
    groups_new: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Choose a legal D compact-frontfill keep list and metadata.

    The preferred order is preserved whenever it is already legal.  If the keep
    count is not divisible by any legal ``groups_new``, the count is repaired to
    the closest legal compact count and the selected old filters remain ordered.
    """
    if not _is_ordinary_grouped_conv2d(module):
        raise ValueError("d_strategy_requires_ordinary_grouped_conv")
    groups_old = int(module.groups)
    c_in = int(module.in_channels)
    c_out = int(module.out_channels)
    old_slice_width = int(module.weight.shape[1])
    preferred = _ordered_unique_valid(preferred_keep, c_out)
    if not preferred:
        raise ValueError("d_strategy_requires_nonempty_preferred_keep")
    group_candidates = [int(groups_new)] if groups_new else [g for g in _divisors(c_in) if 0 < g < groups_old]
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for g_new in sorted(set(group_candidates), reverse=True):
        if g_new <= 0 or g_new >= groups_old:
            continue
        if c_in % g_new != 0:
            continue
        in_per_new = c_in // g_new
        if in_per_new < old_slice_width:
            continue
        legal_counts = [count for count in range(g_new, c_out) if count % g_new == 0]
        for count in legal_counts:
            repaired_keep = _expand_or_trim_keep(preferred, total=c_out, target_count=count)
            resolved = resolve_grouped_conv_d_compact_frontfill_reblock(
                module,
                old_output_keep_indices=repaired_keep,
                old_input_keep_indices=range(c_in),
                groups_new=g_new,
            )
            if not resolved.get("legal", False):
                continue
            distance = abs(count - len(preferred))
            prune_bias = 0 if count <= len(preferred) else 1
            candidates.append((distance, prune_bias, -g_new, resolved))
    if not candidates:
        raise ValueError("d_compact_frontfill_no_legal_groups_new")
    candidates.sort(key=lambda item: item[:3])
    resolved = dict(candidates[0][3])
    keep = [int(v) for v in resolved["old_output_keep_indices"]]
    resolved.update(
        {
            "axis": "grouped_d_compact_frontfill_reblock",
            "replay_axis": "grouped_d_compact_frontfill_reblock",
            "ordered_keep_indices": keep,
            "ordered_keep_indices_used": True,
        }
    )
    return keep, resolved


def _channels_for_item(item: Any) -> int:
    module = item.module
    if item.direction == "out":
        return int(getattr(module, "out_channels", getattr(module, "num_features", getattr(module, "out_features", 0))))
    return int(getattr(module, "in_channels", getattr(module, "in_features", 0)))


def _regular_grouped_conv_output_item(scope: Any) -> Any | None:
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if _is_ordinary_grouped_conv2d(module) and getattr(item, "direction", "") == "out":
            return item
    return None


def _regular_grouped_conv_input_item(scope: Any) -> Any | None:
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if _is_ordinary_grouped_conv2d(module) and getattr(item, "direction", "") == "in":
            return item
    return None


def _replay_axis_for_item(item: Any) -> str:
    fn_name = getattr(item.pruning_fn, "__name__", "")
    module = getattr(item, "module", None)
    if isinstance(module, nn.Conv2d) and int(module.groups) > 1 and getattr(item, "direction", "") == "in":
        return "grouped_input_balanced"
    if fn_name == "prune_grouped_conv_input_balanced":
        return "grouped_input_balanced"
    if fn_name == "prune_grouped_flat_output_groups_fixed":
        return "grouped_flat_output"
    if fn_name == "prune_grouped_group_balanced_output_groups_fixed":
        return "grouped_group_balanced_output"
    if fn_name == "prune_grouped_remove_groups":
        return "grouped_remove"
    if fn_name == "prune_grouped_conv_true_group_block":
        return "grouped_true_group_block"
    if fn_name == "prune_grouped_conv_d_compact_frontfill_reblock":
        return "grouped_d_compact_frontfill_reblock"
    return item.direction


def build_global_plan_from_concrete_v100(
    groups: list[Any],
    concrete_groups: list[Any],
    policy_key: str,
    *,
    grouped_input_reports: list[dict[str, Any]] | None = None,
    grouped_d_reports: list[dict[str, Any]] | None = None,
    grouped_c_reports: list[dict[str, Any]] | None = None,
) -> GlobalPhysicalPrunePlan:
    scope_by_id = {group.group_id: group for group in groups}
    global_plan = GlobalPhysicalPrunePlan()
    for concrete in concrete_groups:
        scope = scope_by_id.get(concrete.scope_id)
        if scope is None:
            continue
        concrete_keep = list(concrete.keep_indices)

        grouped_input_item = _regular_grouped_conv_input_item(scope)
        if grouped_input_item is not None and policy_key != "C":
            local_keep = sorted(int(v) for v in grouped_input_item.local_keep(concrete_keep))
            if local_keep != sorted(concrete_keep):
                if grouped_input_reports is not None:
                    grouped_input_reports.append(
                        {
                            "scope_id": concrete.scope_id,
                            "module_name": grouped_input_item.name,
                            "status": "skipped",
                            "reason": "grouped_input_non_identity_index_transform",
                            "preferred_keep_count": len(local_keep),
                        }
                    )
                continue
            resolved = resolve_grouped_conv_input_keep(
                grouped_input_item.module,
                local_keep,
                allow_repair=True,
                min_in_per_group=1,
            )
            report = {
                "scope_id": concrete.scope_id,
                "module_name": grouped_input_item.name,
                "status": "repaired" if resolved.get("repaired") else ("accepted" if resolved.get("legal") else "skipped"),
                "reason": resolved.get("reason", ""),
                "preferred_keep_count": len(local_keep),
                "actual_keep_count": len(resolved.get("keep_indices", [])),
                "groups": resolved.get("groups", getattr(grouped_input_item.module, "groups", 0)),
                "in_per_group_before": resolved.get("in_per_group_before"),
                "in_per_group_after": resolved.get("in_per_group_after"),
                "per_group_kept_count": resolved.get("per_group_kept_count", {}),
                "group_keep_map": resolved.get("group_keep_map", {}),
            }
            if grouped_input_reports is not None:
                grouped_input_reports.append(report)
            if not resolved.get("legal", False):
                continue
            concrete_keep = sorted(int(v) for v in resolved.get("keep_indices", []))

        c_root_item = _regular_grouped_conv_output_item(scope) if policy_key == "C" else None
        c_pruned_groups: list[int] | None = None
        c_kept_groups: list[int] | None = None
        if c_root_item is not None:
            module = c_root_item.module
            groups_old = int(module.groups)
            out_per = int(module.out_channels) // groups_old
            root_keep = sorted(int(v) for v in c_root_item.local_keep(concrete_keep))
            c_kept_groups = sorted({idx // out_per for idx in root_keep})
            c_pruned_groups = [idx for idx in range(groups_old) if idx not in set(c_kept_groups)]
            if grouped_c_reports is not None:
                grouped_c_reports.append(
                    {
                        "scope_id": concrete.scope_id,
                        "module_name": c_root_item.name,
                        "groups_old": groups_old,
                        "groups_after": len(c_kept_groups),
                        "kept_old_groups": c_kept_groups,
                        "pruned_old_groups": c_pruned_groups,
                        "in_per_group_before": int(module.in_channels) // groups_old,
                        "in_per_group_after": int(module.in_channels) // groups_old,
                        "out_per_group_before": out_per,
                        "out_per_group_after": out_per,
                        "true_group_block": True,
                    }
                )

        d_root_item = _regular_grouped_conv_output_item(scope) if policy_key == "D" else None
        d_keep: list[int] | None = None
        d_metadata: dict[str, Any] = {}
        if d_root_item is not None:
            preferred_keep = [int(v) for v in d_root_item.local_keep(concrete_keep)]
            try:
                d_keep, d_metadata = choose_d_compact_frontfill_keep(d_root_item.module, preferred_keep)
                if grouped_d_reports is not None:
                    grouped_d_reports.append(
                        {
                            "scope_id": concrete.scope_id,
                            "module_name": d_root_item.name,
                            "status": "accepted",
                            **{k: v for k, v in d_metadata.items() if k != "copy_plan"},
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                if grouped_d_reports is not None:
                    grouped_d_reports.append(
                        {
                            "scope_id": concrete.scope_id,
                            "module_name": d_root_item.name,
                            "status": "skipped",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue

        for item in scope.items:
            if c_root_item is not None and item is c_root_item:
                axis = "grouped_true_group_block"
                prune = list(c_pruned_groups or [])
                if not prune:
                    continue
                metadata = {
                    "reason": item.reason,
                    "replay_axis": "grouped_true_group_block",
                    "policy": policy_key,
                    "true_group_block": True,
                    "groups_old": int(item.module.groups),
                    "kept_old_groups": list(c_kept_groups or []),
                    "pruned_old_groups": prune,
                }
                global_plan.add_request(
                    ModuleAxisPruneRequest(
                        module_name=item.name,
                        axis=axis,
                        prune_indices=prune,
                        source_recipe_id=concrete.concrete_group_id,
                        metadata=metadata,
                    )
                )
                continue
            if c_root_item is not None and item.module is c_root_item.module and getattr(item, "direction", "") == "in":
                continue

            if d_keep is not None:
                local_keep = [int(v) for v in item.local_keep(d_keep)]
            else:
                local_keep = [int(v) for v in item.local_keep(concrete_keep)]
            total = _channels_for_item(item)
            keep_set = set(local_keep)
            prune = [idx for idx in range(total) if idx not in keep_set]
            if not prune:
                continue
            axis = item.direction
            metadata = {
                "reason": item.reason,
                "replay_axis": _replay_axis_for_item(item),
                "policy": policy_key,
            }
            if d_root_item is not None:
                metadata["ordered_keep_indices"] = local_keep
                metadata["ordered_keep_indices_used"] = True
                if item is d_root_item:
                    axis = "grouped_d_compact_frontfill_reblock"
                    metadata.update(d_metadata)
            if (
                getattr(item.pruning_fn, "__name__", "") == "prune_grouped_conv_input_balanced"
                or (
                    isinstance(getattr(item, "module", None), nn.Conv2d)
                    and int(getattr(item.module, "groups", 1)) > 1
                    and getattr(item, "direction", "") == "in"
                )
            ):
                axis = "grouped_input_balanced"
                metadata["replay_axis"] = "grouped_input_balanced"
            global_plan.add_request(
                ModuleAxisPruneRequest(
                    module_name=item.name,
                    axis=axis,
                    prune_indices=prune,
                    source_recipe_id=concrete.concrete_group_id,
                    metadata=metadata,
                )
            )
    return global_plan


def _module_params_by_name(model: nn.Module) -> dict[str, int]:
    return {name: _module_param_count(module) for name, module in model.named_modules() if name}


def _supported_surface_module_names(groups: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for group in groups:
        if bool(getattr(group, "protected", False)):
            continue
        for item in getattr(group, "items", []):
            module = getattr(item, "module", None)
            if module is not None and _module_param_count(module) > 0:
                names.add(str(getattr(item, "name", "")))
    return names


def _sum_module_params(model: nn.Module, names: Iterable[str]) -> int:
    modules = dict(model.named_modules())
    return int(sum(_module_param_count(modules[name]) for name in set(names) if name in modules))


def _strategy_metadata(policy_key: str, d_reports: Sequence[dict[str, Any]], c_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if policy_key == "A":
        return {
            "groups_fixed": True,
            "flat_output_groups_fixed": True,
            "group_balanced": False,
        }
    if policy_key == "B":
        return {
            "groups_fixed": True,
            "group_balanced": True,
            "flat_output_groups_fixed": False,
        }
    if policy_key == "C":
        return {
            "true_group_block": True,
            "reports": list(c_reports),
        }
    if policy_key == "D":
        return {
            "semantic_preserved": False,
            "compact_first": True,
            "frontfill_weight_transplant": True,
            "weight_values_retained": True,
            "new_connections_zero_initialized": True,
            "requires_recovery_finetune": True,
            "ordered_keep_indices_used": True,
            "reports": list(d_reports),
        }
    return {}


def build_pruning_domain_report(
    *,
    groups: Sequence[Any],
    plan: Any,
    policy_key: str,
    target_domain_unit_prune_ratio: float,
    module_params_before: dict[str, int],
    module_params_after: dict[str, int] | None = None,
) -> dict[str, Any]:
    module_params_after = module_params_after or module_params_before
    concrete_by_scope = {group.scope_id: group for group in getattr(plan, "concrete_groups", [])}
    rows: list[dict[str, Any]] = []
    for group in groups:
        concrete = concrete_by_scope.get(group.group_id)
        item_names = [str(getattr(item, "name", "")) for item in getattr(group, "items", [])]
        unique_names = sorted(set(item_names))
        before = int(sum(module_params_before.get(name, 0) for name in unique_names))
        after = int(sum(module_params_after.get(name, module_params_before.get(name, 0)) for name in unique_names))
        grouped_convs = [
            name
            for name in unique_names
            if isinstance(getattr(dict((m.name, m.module) for m in getattr(group, "items", [])), "get", lambda _n: None)(name), nn.Conv2d)
        ]
        constraints = {
            "min_channel_constraint": True,
            "residual_closure_constraint": (getattr(group, "meta", {}) or {}).get("group_type", "") == "add",
            "concat_offset_constraint": (getattr(group, "meta", {}) or {}).get("group_type", "") == "cat",
            "grouped_conv_policy_constraint": policy_key if any(_is_ordinary_grouped_conv2d(getattr(item, "module", None)) for item in getattr(group, "items", [])) else "",
            "shape_alignment_constraint": False,
            "fixed_shape_contract": bool(getattr(group, "protected", False)),
        }
        pruned = len(getattr(concrete, "prune_indices", []) or []) if concrete is not None else 0
        total = int(getattr(group, "num_channels", 0) or 0)
        rows.append(
            {
                "domain_id": group.group_id,
                "domain_type": (getattr(group, "meta", {}) or {}).get("group_type", "plain"),
                "total_coupled_units": total,
                "pruned_coupled_units": pruned,
                "kept_coupled_units": max(total - pruned, 0),
                "target_domain_unit_prune_ratio": float(target_domain_unit_prune_ratio),
                "actual_domain_unit_prune_ratio": pruned / total if total else 0.0,
                "param_before_in_domain": before,
                "param_after_in_domain": after,
                "param_saving_in_domain": before - after,
                "involved_modules": unique_names,
                "involved_grouped_convs": grouped_convs,
                "grouped_conv_policy_used": policy_key if grouped_convs else "",
                "contains_grouped_conv": bool(grouped_convs),
                "contains_residual": (getattr(group, "meta", {}) or {}).get("group_type", "") == "add",
                "contains_concat": (getattr(group, "meta", {}) or {}).get("group_type", "") == "cat",
                "contains_convtranspose": any(isinstance(getattr(item, "module", None), nn.ConvTranspose2d) for item in getattr(group, "items", [])),
                "contains_protected_boundary": bool(getattr(group, "protected", False)),
                "selected_unit_indices": list(getattr(concrete, "prune_indices", []) or []) if concrete is not None else [],
                "protected_unit_indices": [] if not getattr(group, "protected", False) else list(range(total)),
                "protected_reason_if_any": getattr(group, "protected_reason", ""),
                "domain_constraints": constraints,
                "not_pruned_reason": "" if concrete is not None else (getattr(group, "protected_reason", "") or "not_selected_by_domain_unit_ratio"),
            }
        )
    return {
        "strategy": policy_key,
        "domains": rows,
        "num_domains_total": len(rows),
        "num_domains_pruned": sum(1 for row in rows if row["pruned_coupled_units"] > 0),
        "num_coupled_units_total": sum(int(row["total_coupled_units"]) for row in rows),
        "num_coupled_units_pruned": sum(int(row["pruned_coupled_units"]) for row in rows),
        "num_coupled_units_kept": sum(int(row["kept_coupled_units"]) for row in rows),
    }


def classify_exception(exc: BaseException | str, stage: str) -> str:
    text = str(exc).lower()
    if "cuda" in text and "out of memory" in text:
        return "cuda_oom"
    if "ordered_keep_indices_mismatch" in text:
        return "ordered_keep_indices_mismatch"
    if "frontfill_source_kernel_too_wide" in text:
        return "frontfill_source_kernel_too_wide"
    if "shape" in text or "size mismatch" in text or "mat1" in text:
        if stage == "simulator":
            return "shape_simulator_illegal"
        if stage == "physical":
            return "physical_prune_failed"
        if stage == "forward":
            return "synthetic_forward_failed"
        return "eval_forward_failed"
    if stage == "simulator":
        return "shape_simulator_illegal"
    if stage == "physical":
        return "physical_prune_failed"
    if stage == "forward":
        return "synthetic_forward_failed"
    if stage == "eval":
        return "eval_forward_failed"
    return "unknown_exception"


def write_failure_report(path: Path, *, strategy: str, stage: str, exc: BaseException | None = None, traceback_text: str = "", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    if exc is None:
        report = {
            "strategy": strategy,
            "status": "success",
            "failure_reason": "",
            "failure_category": "",
            "traceback": "",
        }
    else:
        category = classify_exception(exc, stage)
        if category not in FAILURE_ENUM:
            category = "unknown_exception"
        report = {
            "strategy": strategy,
            "status": "failed",
            "stage": stage,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_category": category,
            "traceback": traceback_text,
        }
    report.update(extra or {})
    write_json(path, report)
    return report


def setup_v100_logger(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("abcd_small_eval_v100")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(out / "v100_run.log", mode="w", encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def build_model_args_for_strategy(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        heal_root=args.heal_root,
        device=getattr(args, "device", ""),
        extra_protected_prefix=list(getattr(args, "extra_protected_prefix", []) or []),
        num_calib_batches=int(getattr(args, "num_calib_batches", 0) or 0),
        importance_mode=str(getattr(args, "importance_mode", "l1_norm")),
    )


def _load_eval_helpers():
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from heal_compress.pruning.eval.prune_and_eval import build_dataset, evaluate_one_model

    return HEALLiDARAdapter, build_dataset, evaluate_one_model


def build_eval_report(summary: dict[str, Any], rows: Sequence[dict[str, Any]], *, baseline_ap: float | None = None) -> dict[str, Any]:
    ap = float(summary.get("AP_0_30", 0.0) or 0.0)
    evaluated = int(summary.get("actual_frames", 0) or 0)
    skipped = int(summary.get("missing_frames", 0) or 0)
    report = {
        "eval_status": "success" if evaluated > 0 and not summary.get("first_failure") else ("partial" if evaluated > 0 else "failed"),
        "AP_0_03": summary.get("AP_0_03", 0.0),
        "AP_0_30": summary.get("AP_0_30", 0.0),
        "AP_0_50": summary.get("AP_0_50", 0.0),
        "AP_0_70": summary.get("AP_0_70", 0.0),
        "AP": ap,
        "mAP": ap,
        "evaluated_frames": evaluated,
        "skipped_frames": skipped,
        "skip_ratio": skipped / max(evaluated + skipped, 1),
        "first_failure": summary.get("first_failure", ""),
        "per_frame_rows": len(rows),
    }
    if baseline_ap is not None:
        drop = float(baseline_ap) - ap
        report["AP_baseline"] = float(baseline_ap)
        report["AP_pruned"] = ap
        report["AP_drop_abs"] = drop
        report["AP_drop_rel"] = drop / float(baseline_ap) if float(baseline_ap) else 0.0
    return report


def build_latency_report(summary: dict[str, Any], rows: Sequence[dict[str, Any]], *, baseline_latency: dict[str, Any] | None = None) -> dict[str, Any]:
    forward_values = [float(row.get("forward_time_ms", 0.0) or 0.0) for row in rows if row.get("success")]
    report = summarize_latency(forward_values, baseline=baseline_latency)
    report.update(
        {
            "latency_status": "success" if forward_values else "failed",
            "timing_source": "PyTorch forward_time_ms",
            "warmup_excluded": True,
            "summary_forward_time_mean_ms": summary.get("forward_time_mean_ms", 0.0),
            "summary_forward_time_p50_ms": summary.get("forward_time_p50_ms", 0.0),
        }
    )
    return report


def write_shape_alignment_artifacts(out: Path, model: nn.Module, *, policy: str, old_group_map: dict[str, Any] | None = None, file_stem: str = "shape_alignment_report") -> dict[str, Any]:
    report = compute_shape_alignment_report(model, policy=policy, old_group_map=old_group_map)
    write_json(out / f"{file_stem}.json", report)
    write_csv(out / "unaligned_grouped_conv_shapes.csv", report.get("unaligned_grouped_conv_shapes", []))
    return report


def run_baseline(args: argparse.Namespace, out: Path, dataset: Any, loader: Any, device: torch.device, logger: logging.Logger) -> dict[str, Any]:
    baseline_dir = out / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    prune_logger = setup_prune_logger(baseline_dir)
    model_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        heal_root=args.heal_root,
    )
    model, adapter = load_heal_model(model_args, device, prune_logger)
    inventory = compute_param_inventory(model)
    baseline_supported_surface_params = 0
    baseline_surface_report: dict[str, Any] = {}
    try:
        sample = adapter.build_synthetic_batch(model)
        trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
        protected_layers = build_protected_layers(
            model,
            adapter_protected=adapter.get_protected_layers(model),
            extra_prefixes=args.extra_protected_prefix or [],
        )
        op_graph = build_op_graph(trace, model, protected_layers=protected_layers)
        groups = GroupBuilder(
            op_graph,
            align=args.align,
            grouped_conv_mode="flat_output_groups_fixed",
            protect_residual_add=False,
        ).build()
        baseline_surface_report = apply_full_model_prunable_surface(
            groups,
            group_conv_policy="A",
            total_model_params=inventory["total_params"],
        )
        baseline_supported_surface_params = int(baseline_surface_report.get("total_prunable_params", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        baseline_surface_report = {
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    write_json(
        baseline_dir / "baseline_model_report.json",
        {
            "baseline_total_params": inventory["total_params"],
            "baseline_supported_surface_params": baseline_supported_surface_params,
            "baseline_grouped_conv_params": inventory["grouped_conv_params"],
            "baseline_non_grouped_conv_params": inventory["non_grouped_conv_params"],
            "baseline_total_conv_params": inventory["total_conv_params"],
            "inventory": inventory,
            "supported_surface": baseline_surface_report,
        },
    )
    shape_report = write_shape_alignment_artifacts(baseline_dir, model, policy="baseline", file_stem="baseline_shape_alignment_report")
    rows, summary = evaluate_model_for_v100(
        model=model,
        checkpoint=args.checkpoint,
        metadata={"target_prune_ratio": 0.0, "actual_param_prune_ratio": 0.0},
        model_type="baseline",
        dataset=dataset,
        loader=loader,
        device=device,
        round_id=0,
        max_frames=args.eval_frames,
        warmup_frames=args.latency_warmup,
        logger=logger,
    )
    eval_report = build_eval_report(summary, rows)
    latency_report = build_latency_report(summary, rows)
    write_json(baseline_dir / "baseline_eval_report.json", eval_report)
    write_json(baseline_dir / "baseline_latency_report.json", latency_report)
    write_csv(baseline_dir / "baseline_per_frame_latency.csv", rows)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "inventory": inventory,
        "eval_report": eval_report,
        "latency_report": latency_report,
        "shape_alignment_report": shape_report,
        "baseline_supported_surface_params": baseline_supported_surface_params,
    }


def evaluate_model_for_v100(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _HEALLiDARAdapter, _build_dataset, evaluate_one_model = _load_eval_helpers()
    return evaluate_one_model(**kwargs)


def run_strategy(
    *,
    args: argparse.Namespace,
    out: Path,
    strategy: str,
    baseline: dict[str, Any],
    dataset: Any,
    loader: Any,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    strategy_dir = out / f"{strategy}_020"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    policy = parse_group_conv_policy_choice(strategy)
    mode = POLICY_TO_MODE[policy.key]
    model_args = build_model_args_for_strategy(args)
    prune_logger = setup_prune_logger(strategy_dir)
    failure_extra: dict[str, Any] = {
        "strategy": strategy,
        "target_param_prune_ratio_full_model": args.target_param_prune_ratio_full_model,
        "target_domain_unit_prune_ratio": args.target_domain_unit_prune_ratio,
        "selected_pruning_domains": [],
        "selected_coupled_units": [],
        "selected_grouped_convs": [],
    }
    model: nn.Module | None = None
    groups: list[Any] = []
    plan: Any | None = None
    global_plan: GlobalPhysicalPrunePlan | None = None
    supported_names: set[str] = set()
    module_params_before: dict[str, int] = {}
    before_inventory = dict(baseline["inventory"])
    after_inventory = dict(before_inventory)
    supported_before = 0
    supported_after = 0
    surgery: dict[str, Any] = {"operations": [], "num_operations": 0}
    sim_report: dict[str, Any] = {"legal": False, "issues": []}
    forward_report: dict[str, Any] = {"forward_smoke_status": "not_run"}
    eval_report: dict[str, Any] = {"eval_status": "not_run"}
    latency_report: dict[str, Any] = {"latency_status": "not_run"}
    domain_report: dict[str, Any] = {}
    d_reports: list[dict[str, Any]] = []
    c_reports: list[dict[str, Any]] = []
    grouped_input_reports: list[dict[str, Any]] = []
    try:
        model, adapter = load_heal_model(model_args, device, prune_logger)
        module_params_before = _module_params_by_name(model)
        sample = adapter.build_synthetic_batch(model)
        trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
        protected_layers = build_protected_layers(
            model,
            adapter_protected=adapter.get_protected_layers(model),
            extra_prefixes=args.extra_protected_prefix or [],
        )
        op_graph = build_op_graph(trace, model, protected_layers=protected_layers)
        effective_grouped_mode = mode if policy.key in {"A", "B", "D"} else "remove_groups"
        groups = GroupBuilder(
            op_graph,
            align=args.align,
            grouped_conv_mode=effective_grouped_mode,
            protect_residual_add=False,
        ).build()
        if args.surface != "full_model_all_safe_coupled_units":
            raise ValueError(f"unsupported_surface_for_v100:{args.surface}")
        surface_report = apply_full_model_prunable_surface(
            groups,
            group_conv_policy=policy.key,
            total_model_params=before_inventory["total_params"],
        )
        configure_grouped_conv_pruning_fns(
            groups,
            argparse.Namespace(group_conv_selection_mode=mode, allow_remove_groups=(policy.key == "C")),
        )
        supported_names = _supported_surface_module_names(groups)
        supported_before = _sum_module_params(model, supported_names)
        for param in model.parameters():
            param.requires_grad_(True)
        calibration_data = build_importance_calibration_data(adapter, model_args, prune_logger)
        if calibration_data is not None:
            calibration_data = [move_batch_to_device(batch, device) for batch in calibration_data]
        _importance, importance_records = compute_group_importance(
            model,
            groups,
            method=args.importance_mode,
            forward_fn=adapter.forward_for_task if calibration_data is not None else None,
            calibration_data=calibration_data,
            loss_fn=adapter.compute_task_loss if calibration_data is not None else None,
            num_calib_batches=int(args.num_calib_batches or 0),
            strict_grad=args.importance_mode in {"first_order_taylor", "second_order_fisher"},
        )
        scope_importance, scope_records = compute_scope_channel_importance_map(groups, method=args.importance_mode)
        cfg = SelectionConfig(
            prune_ratio=float(args.target_domain_unit_prune_ratio),
            selection_mode=args.selection_mode,
            group_conv_selection_mode=mode,
            align=args.align,
            group_conv_align=args.group_conv_align,
            allow_remove_groups=(policy.key == "C"),
            min_channels=max(1, min(args.align, 8)),
            importance_mode=args.importance_mode,
            min_groups_after_prune=1,
            groups_align=1,
        )
        plan = build_pruning_plan(groups, scope_importance, cfg)
        global_plan = build_global_plan_from_concrete_v100(
            groups,
            plan.concrete_groups,
            policy.key,
            grouped_input_reports=grouped_input_reports,
            grouped_d_reports=d_reports,
            grouped_c_reports=c_reports,
        )
        failure_extra.update(
            {
                "selected_pruning_domains": sorted({cg.scope_id for cg in plan.concrete_groups}),
                "selected_coupled_units": sorted(plan.selected_coupled_unit_ids),
                "selected_grouped_convs": sorted(
                    {
                        row.get("module_name", "")
                        for row in list(getattr(plan, "grouped_conv_reports", [])) + d_reports + c_reports
                        if row.get("module_name")
                    }
                ),
            }
        )
        write_json(strategy_dir / "prunable_surface_used.json", surface_report)
        write_json(
            strategy_dir / "selected_coupled_units_report.json",
            {
                "strategy": strategy,
                "num_coupled_units_total": len(plan.coupled_units),
                "num_atomic_units_total": len(plan.atomic_units),
                "num_selected_atomic_units": len(plan.selected_atomic_units),
                "num_selected_coupled_units": len(plan.selected_coupled_unit_ids),
                "selected_coupled_unit_ids": sorted(plan.selected_coupled_unit_ids),
                "coupled_units_sample": coupled_channel_unit_rows(plan.coupled_units[:100]),
                "selected_atomic_units": atomic_prune_unit_rows(plan.selected_atomic_units),
                "concrete_pruning_groups": concrete_pruning_group_rows(plan.concrete_groups),
                "grouped_conv_selection_reports": getattr(plan, "grouped_conv_reports", []),
                "importance_records_count": len(importance_records),
                "scope_importance_records_count": len(scope_records),
            },
        )
        write_json(
            strategy_dir / "prune_plan.json",
            {
                "strategy": strategy,
                "policy": policy.__dict__,
                "strategy_metadata": _strategy_metadata(policy.key, d_reports, c_reports),
                "global_plan": global_plan.audit(),
                "grouped_input_reports": grouped_input_reports,
                "grouped_d_reports": d_reports,
                "grouped_c_reports": c_reports,
            },
        )
        sim = GlobalPlanShapeSimulator(
            model,
            global_plan,
            op_graph=op_graph,
            group_conv_align=args.group_conv_align,
            allow_convtranspose=False,
            allow_fixed_shape_pruning=False,
        )
        sim_report = sim.simulate()
        write_json(strategy_dir / "shape_simulator_report.json", sim_report)
        if not sim_report.get("legal", False):
            exc = RuntimeError("shape_simulator_illegal")
            raise exc
        surgery = global_plan.apply_one_shot(model)
        write_json(strategy_dir / "physical_prune_report.json", surgery)
        after_inventory = compute_param_inventory(model)
        supported_after = _sum_module_params(model, supported_names)
        module_params_after = _module_params_by_name(model)
        domain_report = build_pruning_domain_report(
            groups=groups,
            plan=plan,
            policy_key=policy.key,
            target_domain_unit_prune_ratio=args.target_domain_unit_prune_ratio,
            module_params_before=module_params_before,
            module_params_after=module_params_after,
        )
        write_json(strategy_dir / "pruning_domain_report.json", domain_report)
        old_group_map = {
            str(row.get("module_name")): row
            for row in d_reports + c_reports
            if row.get("module_name")
        }
        shape_report = write_shape_alignment_artifacts(strategy_dir, model, policy=policy.key, old_group_map=old_group_map)
        try:
            model.eval()
            with torch.no_grad():
                adapter.forward_for_task(model, sample)
            forward_report = {
                "forward_smoke_status": "forward_passed",
                "forward_smoke_passed": True,
                "failure_reason": "",
            }
        except Exception as exc:  # noqa: BLE001
            forward_report = {
                "forward_smoke_status": "forward_failed",
                "forward_smoke_passed": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            write_json(strategy_dir / "forward_smoke_report.json", forward_report)
            raise RuntimeError(f"synthetic_forward_failed:{exc}") from exc
        write_json(strategy_dir / "forward_smoke_report.json", forward_report)
        budget_report = build_target_budget_report(
            strategy=strategy,
            target_param_prune_ratio_full_model=args.target_param_prune_ratio_full_model,
            target_domain_unit_prune_ratio=args.target_domain_unit_prune_ratio,
            before=before_inventory,
            after=after_inventory,
            supported_surface_params_before=supported_before,
            supported_surface_params_after=supported_after,
            total_units=len(plan.coupled_units),
            pruned_units=len(plan.selected_coupled_unit_ids),
            forward_ok=True,
            eval_ok=True,
        )
        rows, eval_summary = evaluate_model_for_v100(
            model=model,
            checkpoint=f"in_memory_{strategy}",
            metadata={
                "target_prune_ratio": args.target_param_prune_ratio_full_model,
                "actual_param_prune_ratio": budget_report["actual_param_prune_ratio_full_model"],
            },
            model_type=f"{strategy}_pruned",
            dataset=dataset,
            loader=loader,
            device=device,
            round_id=1,
            max_frames=args.eval_frames,
            warmup_frames=args.latency_warmup,
            logger=logger,
        )
        eval_report = build_eval_report(
            eval_summary,
            rows,
            baseline_ap=float(baseline["eval_report"].get("AP", baseline["eval_report"].get("AP_0_30", 0.0)) or 0.0),
        )
        latency_report = build_latency_report(eval_summary, rows, baseline_latency=baseline["latency_report"])
        write_json(strategy_dir / "eval_report_50.json", eval_report)
        write_json(strategy_dir / "latency_report_50.json", latency_report)
        write_csv(strategy_dir / "per_frame_latency_50.csv", rows)
        budget_report = build_target_budget_report(
            strategy=strategy,
            target_param_prune_ratio_full_model=args.target_param_prune_ratio_full_model,
            target_domain_unit_prune_ratio=args.target_domain_unit_prune_ratio,
            before=before_inventory,
            after=after_inventory,
            supported_surface_params_before=supported_before,
            supported_surface_params_after=supported_after,
            total_units=len(plan.coupled_units),
            pruned_units=len(plan.selected_coupled_unit_ids),
            forward_ok=bool(forward_report.get("forward_smoke_passed", False)),
            eval_ok=eval_report.get("eval_status") in {"success", "partial"},
        )
        write_json(strategy_dir / "target_budget_achievement_report.json", budget_report)
        failure = write_failure_report(
            strategy_dir / "failure_report.json",
            strategy=strategy,
            stage="",
            exc=None,
            extra=failure_extra,
        )
        result = {
            **budget_report,
            "strategy": strategy,
            "strategy_name": STRATEGY_NAMES[strategy],
            "forward_smoke_passed": bool(forward_report.get("forward_smoke_passed", False)),
            "eval_passed": eval_report.get("eval_status") in {"success", "partial"},
            "latency_passed": latency_report.get("latency_status") == "success",
            "AP_baseline": eval_report.get("AP_baseline", 0.0),
            "AP_pruned": eval_report.get("AP_pruned", 0.0),
            "AP_drop_abs": eval_report.get("AP_drop_abs", 0.0),
            "AP_drop_rel": eval_report.get("AP_drop_rel", 0.0),
            "latency_baseline_p50_ms": baseline["latency_report"].get("p50_ms", 0.0),
            "latency_pruned_p50_ms": latency_report.get("p50_ms", 0.0),
            "speedup_p50": latency_report.get("speedup_p50", 0.0),
            "latency_baseline_mean_ms": baseline["latency_report"].get("mean_ms", 0.0),
            "latency_pruned_mean_ms": latency_report.get("mean_ms", 0.0),
            "speedup_mean": latency_report.get("speedup_mean", 0.0),
            "latency_pruned_p90_ms": latency_report.get("p90_ms", 0.0),
            "latency_pruned_p95_ms": latency_report.get("p95_ms", 0.0),
            "selected_pruning_domains": failure_extra["selected_pruning_domains"],
            "selected_coupled_units": failure_extra["selected_coupled_units"],
            "selected_grouped_convs": failure_extra["selected_grouped_convs"],
            "num_domains_total": domain_report.get("num_domains_total", 0),
            "num_domains_pruned": domain_report.get("num_domains_pruned", 0),
            "num_coupled_units_total": domain_report.get("num_coupled_units_total", 0),
            "num_coupled_units_pruned": domain_report.get("num_coupled_units_pruned", 0),
            "num_coupled_units_kept": domain_report.get("num_coupled_units_kept", 0),
            "num_grouped_conv_ops_pruned": sum(1 for op in surgery.get("operations", []) if "grouped" in str(op.get("physical_axis", "")) or "grouped" in str(op.get("axis", ""))),
            "num_non_grouped_conv_ops_pruned": sum(
                1
                for op in surgery.get("operations", [])
                if op.get("module_name") and isinstance(dict(model.named_modules()).get(op["module_name"]), nn.Conv2d) and "grouped" not in str(op.get("physical_axis", ""))
            ),
            "num_bn_ops_pruned": sum(
                1
                for op in surgery.get("operations", [])
                if op.get("module_name") and isinstance(dict(model.named_modules()).get(op["module_name"]), nn.modules.batchnorm._BatchNorm)
            ),
            "num_downstream_input_ops_pruned": sum(1 for op in surgery.get("operations", []) if op.get("physical_axis") in {"in", "grouped_input_balanced"}),
            "failure_status": failure.get("status", ""),
            "shape_alignment_unfriendly_grouped": shape_report.get("num_unaligned_grouped_conv_shapes", 0),
            "strategy_specific_metadata": _strategy_metadata(policy.key, d_reports, c_reports),
        }
        return result
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        stage = "runner"
        if str(exc) == "shape_simulator_illegal":
            stage = "simulator"
        elif "synthetic_forward_failed" in str(exc):
            stage = "forward"
        failure = write_failure_report(
            strategy_dir / "failure_report.json",
            strategy=strategy,
            stage=stage,
            exc=exc,
            traceback_text=tb,
            extra=failure_extra,
        )
        if not (strategy_dir / "shape_simulator_report.json").exists():
            write_json(strategy_dir / "shape_simulator_report.json", sim_report)
        if not (strategy_dir / "physical_prune_report.json").exists():
            write_json(strategy_dir / "physical_prune_report.json", surgery)
        if not (strategy_dir / "forward_smoke_report.json").exists():
            write_json(strategy_dir / "forward_smoke_report.json", forward_report)
        if not (strategy_dir / "eval_report_50.json").exists():
            write_json(strategy_dir / "eval_report_50.json", eval_report)
        if not (strategy_dir / "latency_report_50.json").exists():
            write_json(strategy_dir / "latency_report_50.json", latency_report)
        if not (strategy_dir / "pruning_domain_report.json").exists():
            if plan is not None and groups:
                domain_report = build_pruning_domain_report(
                    groups=groups,
                    plan=plan,
                    policy_key=strategy,
                    target_domain_unit_prune_ratio=args.target_domain_unit_prune_ratio,
                    module_params_before=module_params_before,
                )
            else:
                domain_report = {"strategy": strategy, "domains": [], "num_domains_total": 0, "num_domains_pruned": 0}
            write_json(strategy_dir / "pruning_domain_report.json", domain_report)
        if not (strategy_dir / "selected_coupled_units_report.json").exists():
            write_json(
                strategy_dir / "selected_coupled_units_report.json",
                {
                    "strategy": strategy,
                    "num_coupled_units_total": len(getattr(plan, "coupled_units", []) or []) if plan is not None else 0,
                    "num_selected_coupled_units": len(getattr(plan, "selected_coupled_unit_ids", []) or []) if plan is not None else 0,
                },
            )
        if not (strategy_dir / "prune_plan.json").exists():
            write_json(strategy_dir / "prune_plan.json", {"strategy": strategy, "global_plan": global_plan.audit() if global_plan else {}})
        if not (strategy_dir / "shape_alignment_report.json").exists() and model is not None:
            write_shape_alignment_artifacts(strategy_dir, model, policy=strategy)
        budget_report = build_target_budget_report(
            strategy=strategy,
            target_param_prune_ratio_full_model=args.target_param_prune_ratio_full_model,
            target_domain_unit_prune_ratio=args.target_domain_unit_prune_ratio,
            before=before_inventory,
            after=after_inventory,
            supported_surface_params_before=supported_before,
            supported_surface_params_after=supported_after,
            total_units=len(getattr(plan, "coupled_units", []) or []) if plan is not None else 0,
            pruned_units=len(getattr(plan, "selected_coupled_unit_ids", []) or []) if plan is not None else 0,
            forward_ok=False,
            eval_ok=False,
        )
        write_json(strategy_dir / "target_budget_achievement_report.json", budget_report)
        return {
            **budget_report,
            "strategy": strategy,
            "strategy_name": STRATEGY_NAMES.get(strategy, strategy),
            "forward_smoke_passed": False,
            "eval_passed": False,
            "latency_passed": False,
            "AP_baseline": baseline["eval_report"].get("AP", 0.0),
            "AP_pruned": 0.0,
            "AP_drop_abs": baseline["eval_report"].get("AP", 0.0),
            "AP_drop_rel": 1.0 if baseline["eval_report"].get("AP", 0.0) else 0.0,
            "latency_baseline_p50_ms": baseline["latency_report"].get("p50_ms", 0.0),
            "latency_pruned_p50_ms": 0.0,
            "speedup_p50": 0.0,
            "latency_baseline_mean_ms": baseline["latency_report"].get("mean_ms", 0.0),
            "latency_pruned_mean_ms": 0.0,
            "speedup_mean": 0.0,
            "latency_pruned_p90_ms": 0.0,
            "latency_pruned_p95_ms": 0.0,
            "selected_pruning_domains": failure_extra["selected_pruning_domains"],
            "selected_coupled_units": failure_extra["selected_coupled_units"],
            "selected_grouped_convs": failure_extra["selected_grouped_convs"],
            "num_domains_total": domain_report.get("num_domains_total", 0),
            "num_domains_pruned": domain_report.get("num_domains_pruned", 0),
            "num_coupled_units_total": domain_report.get("num_coupled_units_total", 0),
            "num_coupled_units_pruned": domain_report.get("num_coupled_units_pruned", 0),
            "num_coupled_units_kept": domain_report.get("num_coupled_units_kept", 0),
            "num_grouped_conv_ops_pruned": 0,
            "num_non_grouped_conv_ops_pruned": 0,
            "num_bn_ops_pruned": 0,
            "num_downstream_input_ops_pruned": 0,
            "failure_status": failure.get("status", ""),
            "failure_reason": failure.get("failure_reason", ""),
            "failure_category": failure.get("failure_category", ""),
            "strategy_specific_metadata": _strategy_metadata(strategy, d_reports, c_reports),
        }
    finally:
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_eval_sample_manifest(args: argparse.Namespace, strategy_results: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    first_eval_frame = int(args.latency_warmup)
    last_eval_frame = first_eval_frame + int(args.eval_frames)
    return {
        "sample_ids": list(range(first_eval_frame, last_eval_frame)),
        "total_requested_frames": int(args.eval_frames),
        "latency_warmup_frames": int(args.latency_warmup),
        "actually_evaluated_frames": int(args.eval_frames),
        "skipped_frames": 0,
        "skip_reasons": [],
        "same_manifest_reused_by_all_strategies": True,
        "manifest_basis": "DataLoader shuffle=False frame order; sample_ids are post-warmup evaluation frame ordinals.",
        "strategy_status": [
            {
                "strategy": row.get("strategy", ""),
                "eval_passed": row.get("eval_passed", False),
                "latency_passed": row.get("latency_passed", False),
            }
            for row in (strategy_results or [])
        ],
    }


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "strategy",
        "strategy_name",
        "target_param_prune_ratio_full_model",
        "actual_param_prune_ratio_full_model",
        "budget_error_full_model",
        "target_domain_unit_prune_ratio",
        "actual_domain_unit_prune_ratio",
        "domain_unit_budget_error",
        "actual_grouped_conv_param_prune_ratio",
        "actual_non_grouped_conv_param_prune_ratio",
        "actual_total_conv_param_prune_ratio",
        "actual_param_prune_ratio_supported_surface",
        "forward_smoke_passed",
        "eval_passed",
        "latency_passed",
        "AP_baseline",
        "AP_pruned",
        "AP_drop_abs",
        "AP_drop_rel",
        "latency_baseline_p50_ms",
        "latency_pruned_p50_ms",
        "speedup_p50",
        "latency_baseline_mean_ms",
        "latency_pruned_mean_ms",
        "speedup_mean",
        "latency_pruned_p90_ms",
        "latency_pruned_p95_ms",
        "num_domains_total",
        "num_domains_pruned",
        "num_coupled_units_total",
        "num_coupled_units_pruned",
        "num_coupled_units_kept",
        "num_grouped_conv_ops_pruned",
        "num_non_grouped_conv_ops_pruned",
        "num_bn_ops_pruned",
        "num_downstream_input_ops_pruned",
        "failure_status",
        "failure_reason",
        "failure_category",
        "why_not_reached",
    ]
    return {key: result.get(key, "") for key in keys}


def _best_strategy(rows: Sequence[dict[str, Any]], key: str, *, reverse: bool = True) -> str:
    valid = [row for row in rows if isinstance(row.get(key), (int, float))]
    if not valid:
        return ""
    valid.sort(key=lambda row: float(row.get(key, 0.0) or 0.0), reverse=reverse)
    return str(valid[0].get("strategy", ""))


def write_summary(out: Path, baseline: dict[str, Any], results: Sequence[dict[str, Any]]) -> None:
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(row) for row in results]
    write_csv(summary_dir / "abcd_strategy_comparison.csv", rows)
    write_csv(
        summary_dir / "abcd_failure_matrix.csv",
        [
            {
                "strategy": row.get("strategy", ""),
                "forward_smoke_passed": row.get("forward_smoke_passed", False),
                "eval_passed": row.get("eval_passed", False),
                "latency_passed": row.get("latency_passed", False),
                "failure_status": row.get("failure_status", ""),
                "failure_reason": row.get("failure_reason", ""),
                "failure_category": row.get("failure_category", ""),
            }
            for row in results
        ],
    )
    write_csv(
        summary_dir / "abcd_latency_speedup_report.csv",
        [
            {
                "strategy": row.get("strategy", ""),
                "latency_pruned_p50_ms": row.get("latency_pruned_p50_ms", 0.0),
                "latency_pruned_mean_ms": row.get("latency_pruned_mean_ms", 0.0),
                "latency_pruned_p90_ms": row.get("latency_pruned_p90_ms", 0.0),
                "latency_pruned_p95_ms": row.get("latency_pruned_p95_ms", 0.0),
                "speedup_p50": row.get("speedup_p50", 0.0),
                "speedup_mean": row.get("speedup_mean", 0.0),
            }
            for row in results
        ],
    )
    write_csv(
        summary_dir / "abcd_ap_drop_report.csv",
        [
            {
                "strategy": row.get("strategy", ""),
                "AP_baseline": row.get("AP_baseline", 0.0),
                "AP_pruned": row.get("AP_pruned", 0.0),
                "AP_drop_abs": row.get("AP_drop_abs", 0.0),
                "AP_drop_rel": row.get("AP_drop_rel", 0.0),
            }
            for row in results
        ],
    )
    write_csv(
        summary_dir / "abcd_prune_ratio_report.csv",
        [
            {
                "strategy": row.get("strategy", ""),
                "target_param_prune_ratio_full_model": row.get("target_param_prune_ratio_full_model", 0.0),
                "actual_param_prune_ratio_full_model": row.get("actual_param_prune_ratio_full_model", 0.0),
                "actual_grouped_conv_param_prune_ratio": row.get("actual_grouped_conv_param_prune_ratio", 0.0),
                "actual_non_grouped_conv_param_prune_ratio": row.get("actual_non_grouped_conv_param_prune_ratio", 0.0),
                "actual_param_prune_ratio_supported_surface": row.get("actual_param_prune_ratio_supported_surface", 0.0),
            }
            for row in results
        ],
    )
    write_csv(
        summary_dir / "abcd_domain_unit_prune_report.csv",
        [
            {
                "strategy": row.get("strategy", ""),
                "target_domain_unit_prune_ratio": row.get("target_domain_unit_prune_ratio", 0.0),
                "actual_domain_unit_prune_ratio": row.get("actual_domain_unit_prune_ratio", 0.0),
                "num_coupled_units_total": row.get("num_coupled_units_total", 0),
                "num_coupled_units_pruned": row.get("num_coupled_units_pruned", 0),
            }
            for row in results
        ],
    )
    write_csv(
        summary_dir / "abcd_shape_alignment_comparison.csv",
        [
            {
                "strategy": row.get("strategy", ""),
                "shape_alignment_unfriendly_grouped": row.get("shape_alignment_unfriendly_grouped", ""),
            }
            for row in results
        ],
    )
    write_csv(
        summary_dir / "abcd_strategy_specific_metadata.csv",
        [
            {
                "strategy": row.get("strategy", ""),
                "metadata": json.dumps(row.get("strategy_specific_metadata", {}), ensure_ascii=False, default=str),
            }
            for row in results
        ],
    )
    lines = [
        "# v10.0 ABCD Small Eval Verdict",
        "",
        f"Baseline AP@0.30: {baseline['eval_report'].get('AP', baseline['eval_report'].get('AP_0_30', 0.0))}",
        f"Baseline PyTorch forward latency p50/mean: {baseline['latency_report'].get('p50_ms', 0.0)} / {baseline['latency_report'].get('mean_ms', 0.0)} ms",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Full Param Prune | Domain Unit Prune | Grouped Conv Prune | AP Drop | p50 ms | mean ms | p90 ms | p95 ms | speedup p50 | speedup mean | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            "| {strategy} | {full:.6f} | {domain:.6f} | {grouped:.6f} | {apdrop:.6f} | {p50:.3f} | {mean:.3f} | {p90:.3f} | {p95:.3f} | {sp50:.4f} | {smean:.4f} | {status} |".format(
                strategy=row.get("strategy", ""),
                full=float(row.get("actual_param_prune_ratio_full_model", 0.0) or 0.0),
                domain=float(row.get("actual_domain_unit_prune_ratio", 0.0) or 0.0),
                grouped=float(row.get("actual_grouped_conv_param_prune_ratio", 0.0) or 0.0),
                apdrop=float(row.get("AP_drop_abs", 0.0) or 0.0),
                p50=float(row.get("latency_pruned_p50_ms", 0.0) or 0.0),
                mean=float(row.get("latency_pruned_mean_ms", 0.0) or 0.0),
                p90=float(row.get("latency_pruned_p90_ms", 0.0) or 0.0),
                p95=float(row.get("latency_pruned_p95_ms", 0.0) or 0.0),
                sp50=float(row.get("speedup_p50", 0.0) or 0.0),
                smean=float(row.get("speedup_mean", 0.0) or 0.0),
                status="ok" if row.get("eval_passed") and row.get("latency_passed") else row.get("failure_category", "failed"),
            )
        )
    best_prune = _best_strategy(results, "actual_param_prune_ratio_full_model")
    best_ap = _best_strategy(results, "AP_drop_abs", reverse=False)
    best_speed = _best_strategy(results, "speedup_mean")
    stable_speed = [
        row.get("strategy", "")
        for row in results
        if float(row.get("speedup_mean", 0.0) or 0.0) > 1.02
        and float(row.get("speedup_p50", 0.0) or 0.0) > 1.02
        and row.get("latency_passed")
    ]
    any_unstable = any(float(row.get("speedup_mean", 0.0) or 0.0) <= 1.02 for row in results if row.get("latency_passed"))
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Baseline AP / latency: AP@0.30={baseline['eval_report'].get('AP', baseline['eval_report'].get('AP_0_30', 0.0))}, p50={baseline['latency_report'].get('p50_ms', 0.0)} ms, mean={baseline['latency_report'].get('mean_ms', 0.0)} ms.",
            "2. A/B/C/D full-model actual prune ratios: "
            + ", ".join(f"{row.get('strategy')}={float(row.get('actual_param_prune_ratio_full_model', 0.0) or 0.0):.6f}" for row in results),
            "3. Target=0.20 reached: "
            + ", ".join(f"{row.get('strategy')}={float(row.get('actual_param_prune_ratio_full_model', 0.0) or 0.0) >= 0.20}" for row in results),
            "4. Domain unit target/actual: "
            + ", ".join(f"{row.get('strategy')}={row.get('target_domain_unit_prune_ratio', 0.0)}/{float(row.get('actual_domain_unit_prune_ratio', 0.0) or 0.0):.6f}" for row in results),
            "5. Grouped-conv param prune ratios: "
            + ", ".join(f"{row.get('strategy')}={float(row.get('actual_grouped_conv_param_prune_ratio', 0.0) or 0.0):.6f}" for row in results),
            "6. AP drops: "
            + ", ".join(f"{row.get('strategy')}={float(row.get('AP_drop_abs', 0.0) or 0.0):.6f}" for row in results),
            "7. Latency p50/mean/p90/p95: "
            + ", ".join(
                f"{row.get('strategy')}={float(row.get('latency_pruned_p50_ms', 0.0) or 0.0):.3f}/"
                f"{float(row.get('latency_pruned_mean_ms', 0.0) or 0.0):.3f}/"
                f"{float(row.get('latency_pruned_p90_ms', 0.0) or 0.0):.3f}/"
                f"{float(row.get('latency_pruned_p95_ms', 0.0) or 0.0):.3f} ms"
                for row in results
            ),
            "8. Speedups p50/mean: "
            + ", ".join(f"{row.get('strategy')}={float(row.get('speedup_p50', 0.0) or 0.0):.4f}/{float(row.get('speedup_mean', 0.0) or 0.0):.4f}" for row in results),
            "9. Forward/eval/latency bugs: "
            + ", ".join(f"{row.get('strategy')}={row.get('failure_category', '') or 'none'}" for row in results),
            f"10. Highest full-model actual prune ratio: {best_prune}.",
            f"11. Smallest AP drop: {best_ap}.",
            f"12. Most stable PyTorch latency speedup by mean/p50 threshold: {','.join(stable_speed) if stable_speed else 'none'}. Best mean speedup: {best_speed}.",
            "13. Strategies without actual acceleration: "
            + ",".join(row.get("strategy", "") for row in results if row.get("latency_passed") and float(row.get("speedup_mean", 0.0) or 0.0) <= 1.0),
            f"14. If speedup is unstable, shape alignment is a plausible contributor: {bool(any_unstable)}. See per-strategy shape_alignment_report.json and unaligned_grouped_conv_shapes.csv.",
            "15. A/B alignment-aware next step: yes, if A/B pass but speedup is unstable, per-group kept count and C_out alignment need explicit control.",
            "16. C alignment-aware next step: yes, groups_after and per-group width should be constrained before larger searches.",
            "17. D alignment-aware next step: yes, groups_new/in_per_group_new/out_per_group_new should be selected with hardware-friendly constraints after recovery-finetune evidence.",
            "18. D should stay in the search space only if its forward/eval/latency row is successful and AP drop is acceptable; D is not semantic-preserving and requires recovery finetune.",
            "19. GA / latency proxy readiness: stay cautious. Proceed only if at least one strategy has AP+latency success and stable speedup; otherwise prioritize alignment-aware A/B/C/D.",
            "",
            "These are 50-frame quick diagnostics, not final validation accuracy. PyTorch latency is deployment guidance only and not TensorRT evidence.",
        ]
    )
    (summary_dir / "abcd_strategy_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (summary_dir / "v100_abcd_small_eval_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v10.0 full-model ABCD small eval")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--surface", default="full_model_all_safe_coupled_units")
    parser.add_argument("--strategies", default="A,B,C,D")
    parser.add_argument("--target-param-prune-ratio-full-model", type=float, default=0.20)
    parser.add_argument("--target-domain-unit-prune-ratio", type=float, default=0.20)
    parser.add_argument("--eval-frames", type=int, default=50)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--selection-mode", default="root_node_local_unit_ratio")
    parser.add_argument("--importance-mode", default="l1_norm", choices=["l1_norm", "l2_norm", "first_order_taylor", "second_order_fisher"])
    parser.add_argument("--num-calib-batches", type=int, default=1)
    parser.add_argument("--align", type=int, default=1)
    parser.add_argument("--group-conv-align", type=int, default=1)
    parser.add_argument("--extra-protected-prefix", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_v100_logger(out)
    logger.info("v10.0 ABCD small eval args: %s", json.dumps(vars(args), ensure_ascii=False, default=str))
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    write_json(out / "v100_config.json", vars(args))
    write_json(out / "grouped_conv_policy_registry_report.json", GroupedConvPolicyRegistry.default().report())

    HEALLiDARAdapter, build_dataset, _evaluate_one_model = _load_eval_helpers()
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    dataset, loader = build_dataset(adapter, args.model_config)
    baseline = run_baseline(args, out, dataset, loader, device, logger)

    strategies = [item.strip().upper() for item in str(args.strategies).split(",") if item.strip()]
    results: list[dict[str, Any]] = []
    for strategy in strategies:
        if strategy not in {"A", "B", "C", "D"}:
            logger.warning("Skipping unsupported strategy: %s", strategy)
            continue
        logger.info("Running strategy %s@%.2f", strategy, args.target_param_prune_ratio_full_model)
        result = run_strategy(
            args=args,
            out=out,
            strategy=strategy,
            baseline=baseline,
            dataset=dataset,
            loader=loader,
            device=device,
            logger=logger,
        )
        results.append(result)
        write_json(out / f"{strategy}_020" / "strategy_result_summary.json", result)

    manifest = build_eval_sample_manifest(args, results)
    write_json(out / "eval_sample_manifest.json", manifest)
    write_json(out / "summary" / "eval_sample_manifest.json", manifest)
    write_summary(out, baseline, results)
    print(json.dumps({"success": True, "output_dir": str(out), "num_strategies": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
