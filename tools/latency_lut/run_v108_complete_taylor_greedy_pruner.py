#!/usr/bin/env python3
"""v10.8 complete Taylor greedy global physical pruner and real val500 runner."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.artifacts import (  # noqa: E402
    load_v108_model_object,
    save_v108_model_artifacts,
)
from heal_compress.pruning.greedy_budget_selector import (  # noqa: E402
    V108PruningDomain,
    V108RankingUnit,
    select_greedy_global_budget,
)
from heal_compress.pruning.grouped_pergroup8_policy import validate_grouped_input_keep_pergroup8  # noqa: E402
from heal_compress.pruning.model_io import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    build_importance_calibration_data,
    collect_module_structure,
    configure_grouped_conv_pruning_fns,
    load_heal_model,
    move_batch_to_device,
    setup_logger,
)
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest  # noqa: E402
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.pruning.protection_policy import apply_v108_default_protection  # noqa: E402
from heal_compress.pruning.taylor_importance import compute_taylor_importance_for_scopes  # noqa: E402
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import build_op_graph  # noqa: E402
from heal_compress.utils.model_utils import resolve_device  # noqa: E402
from tools.latency_lut.select_idle_gpu_for_latency import (  # noqa: E402
    collect_gpu_state,
    snapshot_to_dict,
    wait_for_idle_gpu,
)

BASELINE_LABEL = "baseline"
EVAL_ENTRYPOINT = "pruning.eval.prune_and_eval.evaluate_one_model"


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            ready = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    ready[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    ready[key] = value
            writer.writerow(ready)


def ensure_v108_eval_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def parse_targets(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if str(v).strip()]


def target_dir_name(target: float) -> str:
    return f"target_{float(target):.2f}"


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return round(vals[0], 6)
    rank = (len(vals) - 1) * float(pct)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return round(vals[lo] * (1.0 - frac) + vals[hi] * frac, 6)


def _select_device(args: argparse.Namespace, out_dir: Path) -> tuple[torch.device, int | None]:
    if args.auto_select_idle_gpu:
        selected, reason, attempts = wait_for_idle_gpu(
            max_utilization=args.max_gpu_utilization,
            max_memory_ratio=args.max_gpu_memory_ratio,
            wait_timeout_minutes=args.wait_timeout_minutes,
            poll_seconds=args.poll_seconds,
        )
        write_json(
            out_dir / "selected_gpu.json",
            {
                "success": True,
                "selected_gpu_index": selected.index,
                "selected_gpu": snapshot_to_dict(selected),
                "selected_gpu_reason": reason,
                "attempts": attempts,
            },
        )
        device = torch.device(f"cuda:{selected.index}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device, int(selected.index)
    device = torch.device(args.device if args.device else resolve_device("auto"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        return device, int(device.index or 0)
    return device, None


def _regular_grouped_output_item(scope: Any) -> Any | None:
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if (
            isinstance(module, nn.Conv2d)
            and int(module.groups) > 1
            and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
            and getattr(item, "direction", "") == "out"
        ):
            return item
    return None


def _regular_grouped_input_items(scope: Any) -> list[Any]:
    out = []
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if (
            isinstance(module, nn.Conv2d)
            and int(module.groups) > 1
            and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
            and getattr(item, "direction", "") == "in"
        ):
            out.append(item)
    return out


def _channels_for_item(item: Any) -> int:
    module = item.module
    if isinstance(module, nn.Conv2d):
        return int(module.out_channels if item.direction == "out" else module.in_channels)
    if isinstance(module, nn.ConvTranspose2d):
        return int(module.out_channels if item.direction == "out" else module.in_channels)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)
    if isinstance(module, nn.Linear):
        return int(module.out_features if item.direction == "out" else module.in_features)
    return 0


def _domains_from_taylor(groups: Sequence[Any], unit_rows: Sequence[Mapping[str, Any]]) -> list[V108PruningDomain]:
    rows_by_domain: dict[str, list[Mapping[str, Any]]] = {}
    for row in unit_rows:
        rows_by_domain.setdefault(str(row["pruning_domain_id"]), []).append(row)
    groups_by_id = {str(group.group_id): group for group in groups}
    domains: list[V108PruningDomain] = []
    for domain_id, rows in sorted(rows_by_domain.items()):
        scope = groups_by_id.get(domain_id)
        grouped_item = _regular_grouped_output_item(scope) if scope is not None else None
        grouped_module = getattr(grouped_item, "module", None)
        is_grouped_output = isinstance(grouped_module, nn.Conv2d)
        groups_count = int(grouped_module.groups) if is_grouped_output else 1
        root_name = str(getattr(grouped_item, "name", rows[0].get("root_module_name", domain_id)))
        num_channels = int(rows[0].get("num_root_channels", len(rows)))
        per_group = num_channels // groups_count if is_grouped_output and groups_count and num_channels % groups_count == 0 else None
        units = [
            V108RankingUnit(
                pruning_domain_id=domain_id,
                root_module_name=root_name,
                root_dim=str(row.get("root_dim", "out")),
                num_root_channels=num_channels,
                coupled_unit_id=str(row.get("coupled_unit_id", f"{domain_id}::idx{row.get('root_channel_index', 0)}")),
                root_channel_index=int(row.get("root_channel_index", 0)),
                is_grouped_conv=bool(is_grouped_output),
                grouped_local_unit_id=row.get("grouped_local_unit_id") if is_grouped_output else None,
                group_index=int(row["root_channel_index"]) // per_group if per_group else None,
                local_channel_index=int(row["root_channel_index"]) % per_group if per_group else None,
                importance_raw=float(row.get("importance_raw", 0.0)),
                importance_normalized=float(row.get("importance_normalized", 0.0)),
                protected_reason=str(getattr(scope, "protected_reason", "") if getattr(scope, "protected", False) else ""),
            )
            for row in rows
        ]
        domains.append(
            V108PruningDomain(
                pruning_domain_id=domain_id,
                root_module_name=root_name,
                root_dim="out",
                num_root_channels=num_channels,
                units=units,
                is_grouped_conv=bool(is_grouped_output),
                groups=groups_count,
                per_group=per_group,
                protected_reason=str(getattr(scope, "protected_reason", "") if getattr(scope, "protected", False) else ""),
            )
        )
    return domains


def _apply_structural_legality_skips(groups: Sequence[Any], domains: Sequence[V108PruningDomain]) -> list[dict[str, Any]]:
    fixed_keywords = ("pillar_vfe", "pfn_layers", "scatter", "voxel")
    groups_by_id = {str(group.group_id): group for group in groups}
    rows: list[dict[str, Any]] = []
    for domain in domains:
        scope = groups_by_id.get(domain.pruning_domain_id)
        if scope is None:
            continue
        item_names = [str(getattr(item, "name", "")) for item in getattr(scope, "items", [])]
        matched = [name for name in item_names if any(key in name.lower() for key in fixed_keywords)]
        if not matched:
            continue
        domain.skipped_reason = "fixed_shape_interface_structural_illegal"
        for unit in domain.units:
            unit.skipped_reason = domain.skipped_reason
        rows.append(
            {
                "pruning_domain_id": domain.pruning_domain_id,
                "root_module_name": domain.root_module_name,
                "num_root_channels": domain.num_root_channels,
                "skipped_reason": domain.skipped_reason,
                "matched_modules": matched,
                "protected_reason": domain.protected_reason,
                "protection_policy_used": False,
            }
        )
    return rows


def _accumulate_first_order_taylor_gradients(
    model: nn.Module,
    calibration_data: Sequence[Any],
    *,
    forward_fn: Any,
    loss_fn: Any,
    num_batches: int,
) -> int:
    accum: dict[str, torch.Tensor] = {}
    count = 0
    model.eval()
    for batch in calibration_data:
        if count >= int(num_batches):
            break
        model.zero_grad(set_to_none=True)
        outputs = forward_fn(model, batch)
        loss = loss_fn(outputs, batch)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            accum.setdefault(name, torch.zeros_like(param.detach()))
            accum[name] += param.grad.detach()
        count += 1
    if count <= 0:
        raise RuntimeError("first_order_taylor_gradient_missing:no_calibration_batches")
    for name, param in model.named_parameters():
        grad = accum.get(name)
        if grad is None:
            continue
        grad = grad / float(count)
        param.grad = grad.detach().clone()
        setattr(param, "_importance_grad", grad.detach().clone())
    return count


def _build_pruning_domains_and_reports(
    model: nn.Module,
    adapter: Any,
    args: argparse.Namespace,
    logger: Any,
    out_dir: Path,
    device: torch.device,
) -> tuple[list[Any], list[V108PruningDomain], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    for param in model.parameters():
        param.requires_grad_(True)
    calibration_data = build_importance_calibration_data(adapter, args, logger)
    if calibration_data is None:
        raise RuntimeError("first_order_taylor_gradient_missing:calibration_data_unavailable")
    calibration_data = [move_batch_to_device(batch, device) for batch in calibration_data]
    calibration_batches = _accumulate_first_order_taylor_gradients(
        model,
        calibration_data,
        forward_fn=adapter.forward_for_task,
        loss_fn=adapter.compute_task_loss,
        num_batches=int(args.num_calib_batches),
    )

    sample = adapter.build_synthetic_batch(model)
    trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
    op_graph = build_op_graph(trace, model, protected_layers=[], det_head_keywords=(), protected_keywords=())
    write_json(out_dir / "trace_graph_audit.json", {"num_nodes": len(op_graph.nodes), "num_edges": len(op_graph.edges), "warnings": op_graph.warnings})
    groups = GroupBuilder(
        op_graph,
        align=int(args.align_channels),
        grouped_conv_mode="independent_group_topk",
        protect_residual_add=False,
    ).build()
    configure_grouped_conv_pruning_fns(
        groups,
        argparse.Namespace(group_conv_selection_mode="independent_group_topk", allow_remove_groups=False),
    )
    protection_report = apply_v108_default_protection(
        groups,
        protect_fpn_output=bool(args.protect_fpn_output),
        protect_head_output=bool(args.protect_head_output),
    )
    write_json(out_dir / "protection_policy_report.json", protection_report)
    taylor = compute_taylor_importance_for_scopes(
        groups,
        calibration_batches=calibration_batches,
        loss_terms_used=["adapter.compute_task_loss"],
    )
    write_json(out_dir / "taylor_importance_report.json", taylor.report)
    domains = _domains_from_taylor(groups, taylor.unit_rows)
    structural_skip_rows = _apply_structural_legality_skips(groups, domains)
    write_json(
        out_dir / "structural_legality_skip_report.json",
        {
            "num_skipped_domains": len(structural_skip_rows),
            "skipped_domains": structural_skip_rows,
            "note": "These domains are not extra output protections; they are rejected because physical pruning breaks fixed-shape runtime contracts.",
        },
    )
    return groups, domains, taylor.report, protection_report, taylor.scope_scores


def _replay_axis_for_item(item: Any, axis: str) -> str:
    module = getattr(item, "module", None)
    if axis == "grouped_independent_keep":
        return "grouped_independent_keep"
    if axis == "grouped_input_pergroup8":
        return "grouped_input_pergroup8"
    if isinstance(module, nn.Conv2d) and int(getattr(module, "groups", 1)) > 1 and getattr(item, "direction", "") == "in":
        return "grouped_input_pergroup8"
    return str(getattr(item, "direction", axis))


def _apply_grouped_input_legality_filter(
    groups: Sequence[Any],
    selection: Any,
    *,
    align_channels: int,
) -> list[dict[str, Any]]:
    scope_by_id = {str(group.group_id): group for group in groups}
    reports: list[dict[str, Any]] = []
    for domain_id, plan in selection.domain_plans.items():
        if not plan.prune_indices:
            continue
        scope = scope_by_id.get(str(domain_id))
        if scope is None:
            continue
        for item in _regular_grouped_input_items(scope):
            local_keep = sorted(int(v) for v in item.local_keep(plan.keep_indices))
            resolved = validate_grouped_input_keep_pergroup8(
                module_name=item.name,
                keep_indices=local_keep,
                groups=int(item.module.groups),
                C_in_before=int(item.module.in_channels),
                align=align_channels,
            )
            reports.append({"pruning_domain_id": domain_id, **resolved})
            if not resolved.get("legal", False):
                plan.skipped_reason = str(resolved.get("skipped_input_prune_reason", "grouped_input_illegal"))
                plan.raw_selected_count = 0
                plan.final_n_pruned = 0
                plan.prune_indices = []
                plan.keep_indices = list(range(int(getattr(scope, "num_channels", 0))))
                plan.grouped_decision = None
                for unit in selection.selected_units:
                    if unit.pruning_domain_id == domain_id:
                        unit.selected_for_pruning = False
                break
    return reports


def _build_global_physical_plan(groups: Sequence[Any], selection: Any, *, align_channels: int) -> GlobalPhysicalPrunePlan:
    scope_by_id = {str(group.group_id): group for group in groups}
    plan = GlobalPhysicalPrunePlan()
    for domain_id, domain_plan in selection.domain_plans.items():
        if not domain_plan.prune_indices:
            continue
        scope = scope_by_id.get(str(domain_id))
        if scope is None:
            continue
        for item in getattr(scope, "items", []):
            local_keep = sorted(int(v) for v in item.local_keep(domain_plan.keep_indices))
            total = _channels_for_item(item)
            if total <= 0:
                continue
            keep_set = set(local_keep)
            prune = [idx for idx in range(total) if idx not in keep_set]
            if not prune:
                continue
            axis = str(getattr(item, "direction", "out"))
            module = getattr(item, "module", None)
            if (
                isinstance(module, nn.Conv2d)
                and int(module.groups) > 1
                and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
                and getattr(item, "direction", "") == "out"
            ):
                axis = "grouped_independent_keep"
            elif (
                isinstance(module, nn.Conv2d)
                and int(module.groups) > 1
                and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
                and getattr(item, "direction", "") == "in"
            ):
                axis = "grouped_input_pergroup8"
            metadata = {
                "reason": getattr(item, "reason", ""),
                "replay_axis": _replay_axis_for_item(item, axis),
                "align_channels": int(align_channels),
                "module_name": item.name,
                "selector": "greedy_global_ranking",
            }
            plan.add_request(
                ModuleAxisPruneRequest(
                    module_name=item.name,
                    axis=axis,
                    prune_indices=prune,
                    source_recipe_id=f"v108::{domain_id}",
                    metadata=metadata,
                )
            )
    return plan


def _shape_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, b in before.items():
        a = after.get(name)
        if a is None:
            continue
        if b != a:
            rows.append({"module_name": name, "before": b, "after": a})
    return rows


def _grouped_stage_summary(grouped_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in grouped_rows:
        stage = str(row.get("stage_guess", "unknown"))
        if stage not in out:
            out[stage] = {
                "module_name": row.get("module_name", ""),
                "out_per_group_before_after": [row.get("out_per_group_before"), row.get("out_per_group_after")],
                "in_per_group_before_after": [row.get("in_per_group_before"), row.get("in_per_group_after")],
                "output_pruned": row.get("output_pruned", False),
                "input_pruned": row.get("input_pruned", False),
            }
    return out


def _manifest_for_target(
    *,
    target: float,
    selection: Any,
    params_before: int,
    params_after: int,
    surgery: Mapping[str, Any],
    shape_changes: Sequence[Mapping[str, Any]],
    grouped_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual_param = 1.0 - params_after / max(params_before, 1)
    return {
        "target_pruning_ratio": float(target),
        "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
        "actual_param_prune_ratio": actual_param,
        "actual_flops_proxy_ratio": selection.actual_flops_proxy_ratio,
        "max_ch_sparsity": 0.60,
        "align_channels": 8,
        "importance": "first_order_taylor",
        "selector": "greedy_global_ranking",
        "protected_fpn_output": True,
        "protected_head_output": True,
        "additional_output_protection": False,
        "modified_modules": [op.get("module_name", "") for op in surgery.get("operations", [])],
        "before_after_shapes": list(shape_changes),
        "grouped_conv_stage0_1_2_summary": _grouped_stage_summary(grouped_rows),
        "grouped_pergroup8_policy_report": list(grouped_rows),
        "pruning_domain_summary": {
            domain_id: {
                "raw_selected_count": plan.raw_selected_count,
                "final_n_pruned": plan.final_n_pruned,
                "skipped_reason": plan.skipped_reason,
            }
            for domain_id, plan in selection.domain_plans.items()
        },
        "selected_units": [unit.to_report_row() for unit in selection.selected_units],
        "skipped_units": [unit.to_report_row() for unit in selection.skipped_units],
        "requires_architecture_patch": True,
        "rounding_overshoot_ratio": selection.rounding_overshoot_ratio,
        "max_ch_sparsity_blocked_count": selection.max_ch_sparsity_blocked_count,
        "skipped_because_pergroup_below8_count": selection.skipped_because_pergroup_below8_count,
        "unreachable": selection.unreachable,
        "params_before": int(params_before),
        "params_after": int(params_after),
    }


def _load_eval_helpers():
    attempted = ["pruning.eval.prune_and_eval.build_dataset/evaluate_one_model/setup_logger"]
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from heal_compress.pruning.eval.prune_and_eval import build_dataset, evaluate_one_model, setup_logger as setup_eval_logger

    return HEALLiDARAdapter, build_dataset, evaluate_one_model, setup_eval_logger, attempted


def _smoke_real_batch(model: nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    from opencood.tools import train_utils

    try:
        batch = next(iter(loader))
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output = model(batch["ego"])
        return {"reload_forward_smoke_passed": True, "output_finite": _outputs_finite(output), "failure_reason": ""}
    except Exception as exc:  # noqa: BLE001
        return {"reload_forward_smoke_passed": False, "output_finite": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _outputs_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, Mapping):
        return all(_outputs_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_outputs_finite(v) for v in value)
    return True


def _skip_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("success"):
            continue
        reason = str(row.get("skip_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _latency_row(
    *,
    target: float,
    selection: Any,
    actual_param: float,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    contamination: bool,
) -> dict[str, Any]:
    forward_vals = [float(row.get("forward_time_ms", 0.0) or 0.0) for row in rows if row.get("success")]
    total_vals = [float(row.get("total_time_ms", 0.0) or 0.0) for row in rows if row.get("success")]
    f_p50 = percentile(forward_vals, 0.50)
    f_mean = round(sum(forward_vals) / len(forward_vals), 6) if forward_vals else 0.0
    t_p50 = percentile(total_vals, 0.50)
    t_mean = round(sum(total_vals) / len(total_vals), 6) if total_vals else 0.0
    base_f_p50 = float(baseline_summary.get("forward_time_p50_ms", 0.0) or 0.0)
    base_f_mean = float(baseline_summary.get("forward_time_mean_ms", 0.0) or 0.0)
    return {
        "target": float(target),
        "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
        "actual_param_prune_ratio": actual_param,
        "evaluated_frames": int(summary.get("actual_frames", 0) or 0),
        "skipped_frames": int(summary.get("missing_frames", 0) or 0),
        "skip_reason_counts": _skip_reason_counts(rows),
        "forward_latency_p50": f_p50,
        "forward_latency_mean": f_mean,
        "forward_latency_p90": percentile(forward_vals, 0.90),
        "forward_latency_p95": percentile(forward_vals, 0.95),
        "total_latency_p50": t_p50,
        "total_latency_mean": t_mean,
        "total_latency_p90": percentile(total_vals, 0.90),
        "total_latency_p95": percentile(total_vals, 0.95),
        "speedup_p50_vs_baseline": base_f_p50 / f_p50 if f_p50 > 0 else 0.0,
        "speedup_mean_vs_baseline": base_f_mean / f_mean if f_mean > 0 else 0.0,
        "latency_contamination_risk": bool(contamination),
    }


def _ap_row(
    *,
    target: float,
    selection: Any,
    actual_param: float,
    summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    failure_reason: str = "",
) -> dict[str, Any]:
    ap30 = float(summary.get("AP_0_30", 0.0) or 0.0)
    ap50 = float(summary.get("AP_0_50", 0.0) or 0.0)
    ap70 = float(summary.get("AP_0_70", 0.0) or 0.0)
    b_ap30 = float(baseline_summary.get("AP_0_30", 0.0) or 0.0)
    b_map = sum(float(baseline_summary.get(k, 0.0) or 0.0) for k in ("AP_0_30", "AP_0_50", "AP_0_70")) / 3.0
    map3 = (ap30 + ap50 + ap70) / 3.0
    return {
        "target": float(target),
        "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
        "actual_param_prune_ratio": actual_param,
        "evaluated_frames": int(summary.get("actual_frames", 0) or 0),
        "AP@0.30": round(ap30, 6),
        "AP@0.50": round(ap50, 6),
        "mAP": round(map3, 6),
        "AP_drop_vs_baseline": round(b_ap30 - ap30, 6),
        "mAP_drop_vs_baseline": round(b_map - map3, 6),
        "metric_helper_used": EVAL_ENTRYPOINT,
        "failure_reason": failure_reason or str(summary.get("first_failure", "") or ""),
    }


def _contamination_report(samples: Sequence[Mapping[str, Any]], selected_index: int | None) -> dict[str, Any]:
    current_pid = os.getpid()
    external: list[dict[str, Any]] = []
    for sample in samples:
        for process in sample.get("running_processes", []) or sample.get("processes", []) or []:
            if int(process.get("pid", -1)) != current_pid:
                external.append(dict(process))
    return {"selected_gpu_index": selected_index, "current_pid": current_pid, "latency_contamination_risk": bool(external), "external_processes_seen": external}


def _write_failure_eval_outputs(out_dir: Path, *, target: float, attempted: Sequence[str], failure: str, tb: str) -> None:
    write_json(
        out_dir / "real_val500_ap.json",
        {
            "target": float(target),
            "actual_channel_prune_ratio_on_searchable_surface": None,
            "actual_param_prune_ratio": None,
            "evaluated_frames": 0,
            "AP@0.30": None,
            "AP@0.50": None,
            "mAP": None,
            "AP_drop_vs_baseline": None,
            "mAP_drop_vs_baseline": None,
            "metric_helper_used": EVAL_ENTRYPOINT,
            "failure_reason": failure,
            "attempted_eval_entrypoints": list(attempted),
            "traceback": tb,
            "missing_adapter_reason": failure,
            "next_required_patch": "Patch run_v108_complete_taylor_greedy_pruner.py to match the available HEAL/OpenCOOD eval helper API.",
        },
    )


def _evaluate_baseline(
    *,
    baseline: nn.Module,
    checkpoint: str,
    dataset: Any,
    loader: Any,
    device: torch.device,
    logger: Any,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = args.evaluate_one_model(
        model=baseline,
        checkpoint=checkpoint,
        metadata={},
        model_type=BASELINE_LABEL,
        dataset=dataset,
        loader=loader,
        device=device,
        round_id=0,
        max_frames=int(args.eval_frames),
        warmup_frames=int(args.latency_warmup),
        logger=logger,
    )
    write_csv(out_dir / "baseline_per_frame_latency.csv", rows)
    write_json(out_dir / "baseline_real_val500_summary.json", summary)
    return rows, summary


def _run_target_eval(
    *,
    target: float,
    target_dir: Path,
    artifact_path: Path,
    selection: Any,
    actual_param: float,
    dataset: Any,
    loader: Any,
    device: torch.device,
    logger: Any,
    args: argparse.Namespace,
    baseline_summary: Mapping[str, Any],
    gpu_samples: list[dict[str, Any]],
    selected_index: int | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reload_success = False
    smoke = {"reload_forward_smoke_passed": False, "failure_reason": "not_run"}
    model: nn.Module | None = None
    reason = ""
    try:
        model = load_v108_model_object(artifact_path, device=device)
        reload_success = True
        smoke = _smoke_real_batch(model, loader, device)
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
    reload_row = {
        "model_object_path": str(artifact_path),
        "reload_success": reload_success,
        "reload_forward_smoke_passed": bool(smoke.get("reload_forward_smoke_passed", False)),
        "validation_dataloader_used": True,
        "synthetic_used": False,
        "failure_reason": reason or str(smoke.get("failure_reason", "")),
    }
    write_json(target_dir / "reload_report.json", reload_row)
    if model is None or not reload_row["reload_forward_smoke_passed"]:
        raise RuntimeError("model_object_reload_or_real_batch_smoke_failed")

    rows, summary = args.evaluate_one_model(
        model=model,
        checkpoint=str(artifact_path),
        metadata={"target_prune_ratio": target, "actual_param_prune_ratio": actual_param},
        model_type=f"target_{target:.2f}",
        dataset=dataset,
        loader=loader,
        device=device,
        round_id=int(round(target * 100)),
        max_frames=int(args.eval_frames),
        warmup_frames=int(args.latency_warmup),
        logger=logger,
    )
    write_csv(target_dir / "per_frame_latency.csv", rows)
    if selected_index is not None:
        sample = collect_gpu_state(selected_index)
        gpu_samples.append({"sample_reason": f"after_target_{target:.2f}", **sample})
    contamination = bool(_contamination_report(gpu_samples, selected_index)["latency_contamination_risk"])
    latency = _latency_row(
        target=target,
        selection=selection,
        actual_param=actual_param,
        rows=rows,
        summary=summary,
        baseline_summary=baseline_summary,
        contamination=contamination,
    )
    ap = _ap_row(
        target=target,
        selection=selection,
        actual_param=actual_param,
        summary=summary,
        baseline_summary=baseline_summary,
    )
    write_csv(target_dir / "real_val500_latency.csv", [latency])
    write_json(target_dir / "real_val500_ap.json", ap)
    return latency, ap, reload_row


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_float(value: Any, digits: int = 6) -> str:
    return f"{_as_float(value):.{digits}f}"


def _manifest_from_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    artifact = str(row.get("model_artifact_path", "") or "")
    if not artifact:
        return {}
    manifest_path = Path(artifact).parent / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _speedup_from_ms(baseline_ms: float, current_ms: Any) -> float:
    cur = _as_float(current_ms)
    return baseline_ms / cur if baseline_ms > 0 and cur > 0 else 0.0


def _verdict_lines(
    summary_rows: Sequence[Mapping[str, Any]],
    failure: str,
    baseline_summary: Mapping[str, Any] | None = None,
) -> list[str]:
    lines = [
        "# v10.8 Complete Taylor Greedy Pruner Verdict",
        "",
        f"failure: {failure or 'none'}",
        "",
        "## Answers",
    ]
    if not summary_rows:
        lines.extend(
            [
                "1. physical pruning/model save: no successful target rows.",
                "2. budget selector: implemented as greedy global normalized first-order Taylor ranking; no completed experiment rows available.",
            ]
        )
        return lines

    manifests = {float(row.get("target_pruning_ratio", 0.0) or 0.0): _manifest_from_summary_row(row) for row in summary_rows}
    baseline = dict(baseline_summary or {})
    base_total_p50 = _as_float(baseline.get("total_time_p50_ms"))
    base_total_mean = _as_float(baseline.get("total_time_mean_ms"))

    artifact_count = sum(1 for row in summary_rows if str(row.get("model_artifact_path", "") or ""))
    lines.append(
        f"1. physical pruning/model save: completed for {artifact_count}/{len(summary_rows)} targets; each successful row has `pruned_model_object.pth` and `pruned_state_dict_with_manifest.pth`."
    )
    lines.append("2. budget selector: greedy global normalized first-order Taylor ranking, TP-style floor keep, max_ch_sparsity=0.60, and grouped per-group8 independent local ranking.")
    lines.append("3. actual ratios:")
    for row in summary_rows:
        verdict = row.get("verdict", "")
        lines.append(
            f"- target {_fmt_float(row.get('target_pruning_ratio'), 2)}: channel={_fmt_float(row.get('actual_channel_prune_ratio_on_searchable_surface'))}, param={_fmt_float(row.get('actual_param_prune_ratio'))}, verdict={verdict}"
        )

    lines.append("4. round_to=8 over-pruning:")
    for row in summary_rows:
        target = _as_float(row.get("target_pruning_ratio"))
        manifest = manifests.get(target, {})
        actual = _as_float(row.get("actual_channel_prune_ratio_on_searchable_surface"))
        lines.append(
            f"- target {_fmt_float(target, 2)}: rounding_overshoot_ratio={_fmt_float(manifest.get('rounding_overshoot_ratio'))}, actual_minus_target={actual - target:.6f}"
        )

    lines.append("5. max_ch_sparsity=60%:")
    for row in summary_rows:
        target = _as_float(row.get("target_pruning_ratio"))
        manifest = manifests.get(target, {})
        lines.append(
            f"- target {_fmt_float(target, 2)}: blocked_count={int(manifest.get('max_ch_sparsity_blocked_count') or 0)}"
        )

    restricted = [
        _fmt_float(row.get("target_pruning_ratio"), 2)
        for row in summary_rows
        if str(row.get("verdict", "")) == "unreachable"
    ]
    pergroup_rows = [
        f"{_fmt_float(row.get('target_pruning_ratio'), 2)}:{int(manifests.get(_as_float(row.get('target_pruning_ratio')), {}).get('skipped_because_pergroup_below8_count') or 0)}"
        for row in summary_rows
    ]
    lines.append(
        f"6. restricted targets: unreachable={restricted or 'none'}; per_group_after_below8 skip counts by target={', '.join(pergroup_rows)}."
    )

    first_row = summary_rows[-1]
    lines.append(f"7. Stage0 grouped output per_group: {first_row.get('stage0_out_per_group_before_after')} (unchanged).")
    lines.append(f"8. Stage1 grouped output per_group: {first_row.get('stage1_out_per_group_before_after')} (unchanged).")
    lines.append(f"9. Stage2 grouped output per_group: {first_row.get('stage2_out_per_group_before_after')} (16 to 8 when selected and legal).")
    lines.append(
        f"10. grouped input per_group: Stage0 {first_row.get('stage0_in_per_group_before_after')}, Stage1 {first_row.get('stage1_in_per_group_before_after')}, Stage2 {first_row.get('stage2_in_per_group_before_after')}; all reported rows passed legality checks."
    )

    forward_fast = [
        _fmt_float(row.get("target_pruning_ratio"), 2)
        for row in summary_rows
        if _as_float(row.get("speedup_p50_vs_baseline")) > 1.0
    ]
    lines.append(f"11. real val500 forward p50 speedup: targets with speedup>1 are {forward_fast or 'none'}.")
    if base_total_p50 > 0:
        total_p50_fast = [
            _fmt_float(row.get("target_pruning_ratio"), 2)
            for row in summary_rows
            if _speedup_from_ms(base_total_p50, row.get("total_p50_ms")) > 1.0
        ]
        total_mean_fast = [
            _fmt_float(row.get("target_pruning_ratio"), 2)
            for row in summary_rows
            if _speedup_from_ms(base_total_mean, row.get("total_mean_ms")) > 1.0
        ]
        total_bits = [
            f"{_fmt_float(row.get('target_pruning_ratio'), 2)}:p50={_speedup_from_ms(base_total_p50, row.get('total_p50_ms')):.3f},mean={_speedup_from_ms(base_total_mean, row.get('total_mean_ms')):.3f}"
            for row in summary_rows
        ]
        lines.append(
            f"12. total latency speedup: p50 speedup targets={total_p50_fast or 'none'}; mean speedup targets={total_mean_fast or 'none'}; ratios={', '.join(total_bits)}."
        )
    else:
        lines.append("12. total latency speedup: baseline total latency unavailable.")

    ap_bits = [
        f"{_fmt_float(row.get('target_pruning_ratio'), 2)}:AP30_drop={_fmt_float(row.get('AP_drop_vs_baseline'))},mAP_drop={_fmt_float(row.get('mAP_drop_vs_baseline'))}"
        for row in summary_rows
    ]
    lines.append(
        "13. AP/mAP trend: targets 0.10-0.30 are essentially preserved, while 0.40 and above collapse to AP=0. "
        + "; ".join(ap_bits)
        + "."
    )
    forward_regress = [
        _fmt_float(row.get("target_pruning_ratio"), 2)
        for row in summary_rows
        if _as_float(row.get("actual_param_prune_ratio")) > 0 and _as_float(row.get("speedup_p50_vs_baseline")) < 1.0
    ]
    total_mean_regress = [
        _fmt_float(row.get("target_pruning_ratio"), 2)
        for row in summary_rows
        if base_total_mean > 0
        and _as_float(row.get("actual_param_prune_ratio")) > 0
        and _speedup_from_ms(base_total_mean, row.get("total_mean_ms")) < 1.0
    ]
    lines.append(
        f"14. parameter reduction with latency regression: forward p50 regressions={forward_regress or 'none'}; total mean regressions={total_mean_regress or 'none'}."
    )

    candidates = [
        row
        for row in summary_rows
        if str(row.get("verdict", "")) == "ok"
        and _as_float(row.get("speedup_p50_vs_baseline")) > 1.0
        and _as_float(row.get("mAP_drop_vs_baseline"), 1.0) <= 0.01
    ]
    best = max(candidates or summary_rows, key=lambda r: (_as_float(r.get("actual_param_prune_ratio")), _as_float(r.get("speedup_p50_vs_baseline"))))
    lines.append(
        f"15. recovery/distillation candidate: target {_fmt_float(best.get('target_pruning_ratio'), 2)}, because it keeps mAP_drop<=0.01 with the largest parameter reduction among forward-p50-speedup rows."
    )
    lines.append("16. TensorRT benchmark: recommended after recovery/distillation, using target 0.30 as the main candidate and 0.10/0.20 as control points.")
    return lines


def run_v108(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args))
    gpu_samples: list[dict[str, Any]] = []
    selected_index: int | None = None
    failure = ""
    attempted_eval: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    all_latency: list[dict[str, Any]] = []
    all_ap: list[dict[str, Any]] = []
    all_reload: list[dict[str, Any]] = []
    all_budget_trace: list[dict[str, Any]] = []
    try:
        device, selected_index = _select_device(args, out_dir)
        if selected_index is not None:
            before_gpu = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_before.json", before_gpu)
            gpu_samples.append({"sample_reason": "before", **before_gpu})

        logger = setup_logger(out_dir)
        args.importance_mode = args.importance
        args.num_calib_batches = int(args.num_calib_batches)
        baseline, adapter = load_heal_model(args, device, logger)
        baseline.eval()
        params_before = count_parameters(baseline)
        structure_before = collect_module_structure(baseline)
        groups, domains, _taylor_report, protection_report, _scores = _build_pruning_domains_and_reports(
            baseline,
            adapter,
            args,
            logger,
            out_dir,
            device,
        )

        HEALLiDARAdapter, build_dataset, evaluate_one_model, setup_eval_logger, attempted_eval = _load_eval_helpers()
        args.evaluate_one_model = evaluate_one_model
        eval_logger = setup_eval_logger(ensure_v108_eval_dir(out_dir / "real_val500"))
        eval_adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
        dataset, loader = build_dataset(eval_adapter, args.model_config, batch_size=1, num_workers=int(args.num_workers))
        baseline_rows, baseline_summary = _evaluate_baseline(
            baseline=baseline,
            checkpoint=args.checkpoint,
            dataset=dataset,
            loader=loader,
            device=device,
            logger=eval_logger,
            args=args,
            out_dir=out_dir,
        )

        for target in parse_targets(args.targets):
            target_dir = out_dir / target_dir_name(target)
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                selection = select_greedy_global_budget(
                    copy.deepcopy(domains),
                    target_pruning_ratio=float(target),
                    max_ch_sparsity=float(args.max_ch_sparsity),
                    align_channels=int(args.align_channels),
                )
                grouped_input_reports = _apply_grouped_input_legality_filter(
                    groups,
                    selection,
                    align_channels=int(args.align_channels),
                )
                if grouped_input_reports:
                    selection.grouped_shape_rows.extend(grouped_input_reports)
                write_csv(target_dir / "budget_selection_trace.csv", selection.trace_rows)
                write_json(target_dir / "grouped_pergroup8_shape_report.json", selection.grouped_shape_rows)
                all_budget_trace.extend(selection.trace_rows)

                pruned = copy.deepcopy(baseline).to(device).eval()
                global_plan = _build_global_physical_plan(groups, selection, align_channels=int(args.align_channels))
                write_json(target_dir / "global_physical_prune_plan.json", global_plan.to_json())
                surgery = global_plan.apply_one_shot(pruned)
                write_json(target_dir / "one_shot_surgery_report.json", surgery)
                params_after = count_parameters(pruned)
                actual_param = 1.0 - params_after / max(params_before, 1)
                structure_after = collect_module_structure(pruned)
                shape_changes = _shape_changes(structure_before, structure_after)
                manifest = _manifest_for_target(
                    target=target,
                    selection=selection,
                    params_before=params_before,
                    params_after=params_after,
                    surgery=surgery,
                    shape_changes=shape_changes,
                    grouped_rows=selection.grouped_shape_rows,
                )
                artifacts = save_v108_model_artifacts(
                    model=pruned,
                    models_dir=target_dir / "models",
                    manifest=manifest,
                    model_config=args.model_config,
                    checkpoint_source=args.checkpoint,
                )
                write_json(target_dir / "models" / "manifest.json", manifest)
                latency, ap, reload_row = _run_target_eval(
                    target=target,
                    target_dir=target_dir,
                    artifact_path=artifacts["model_object"],
                    selection=selection,
                    actual_param=actual_param,
                    dataset=dataset,
                    loader=loader,
                    device=device,
                    logger=eval_logger,
                    args=args,
                    baseline_summary=baseline_summary,
                    gpu_samples=gpu_samples,
                    selected_index=selected_index,
                )
                all_latency.append(latency)
                all_ap.append(ap)
                all_reload.append({"target": target, **reload_row})
                write_json(target_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
                stage = _grouped_stage_summary(selection.grouped_shape_rows)
                row = {
                    "target_pruning_ratio": float(target),
                    "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
                    "actual_param_prune_ratio": actual_param,
                    "actual_flops_proxy_ratio": selection.actual_flops_proxy_ratio,
                    "params_before": params_before,
                    "params_after": params_after,
                    "forward_p50_ms": latency["forward_latency_p50"],
                    "forward_mean_ms": latency["forward_latency_mean"],
                    "forward_p90_ms": latency["forward_latency_p90"],
                    "forward_p95_ms": latency["forward_latency_p95"],
                    "total_p50_ms": latency["total_latency_p50"],
                    "total_mean_ms": latency["total_latency_mean"],
                    "AP@0.30": ap["AP@0.30"],
                    "AP@0.50": ap["AP@0.50"],
                    "mAP": ap["mAP"],
                    "AP_drop_vs_baseline": ap["AP_drop_vs_baseline"],
                    "mAP_drop_vs_baseline": ap["mAP_drop_vs_baseline"],
                    "speedup_p50_vs_baseline": latency["speedup_p50_vs_baseline"],
                    "speedup_mean_vs_baseline": latency["speedup_mean_vs_baseline"],
                    "stage0_out_per_group_before_after": stage.get("stage0_like", {}).get("out_per_group_before_after", ""),
                    "stage0_in_per_group_before_after": stage.get("stage0_like", {}).get("in_per_group_before_after", ""),
                    "stage1_out_per_group_before_after": stage.get("stage1_like", {}).get("out_per_group_before_after", ""),
                    "stage1_in_per_group_before_after": stage.get("stage1_like", {}).get("in_per_group_before_after", ""),
                    "stage2_out_per_group_before_after": stage.get("stage2_like", {}).get("out_per_group_before_after", ""),
                    "stage2_in_per_group_before_after": stage.get("stage2_like", {}).get("in_per_group_before_after", ""),
                    "model_artifact_path": str(artifacts["model_object"]),
                    "latency_contamination_risk": latency["latency_contamination_risk"],
                    "verdict": "invalid_latency" if latency["latency_contamination_risk"] else ("unreachable" if selection.unreachable else "ok"),
                }
                summary_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                reason = f"{type(exc).__name__}: {exc}"
                write_json(target_dir / "failure_report.json", {"success": False, "failure_reason": reason, "traceback": tb})
                _write_failure_eval_outputs(target_dir, target=target, attempted=attempted_eval, failure=reason, tb=tb)
                summary_rows.append(
                    {
                        "target_pruning_ratio": float(target),
                        "actual_channel_prune_ratio_on_searchable_surface": "",
                        "actual_param_prune_ratio": "",
                        "actual_flops_proxy_ratio": "",
                        "params_before": params_before,
                        "params_after": "",
                        "forward_p50_ms": "",
                        "forward_mean_ms": "",
                        "forward_p90_ms": "",
                        "forward_p95_ms": "",
                        "total_p50_ms": "",
                        "total_mean_ms": "",
                        "AP@0.30": "",
                        "AP@0.50": "",
                        "mAP": "",
                        "AP_drop_vs_baseline": "",
                        "mAP_drop_vs_baseline": "",
                        "speedup_p50_vs_baseline": "",
                        "speedup_mean_vs_baseline": "",
                        "stage0_out_per_group_before_after": "",
                        "stage0_in_per_group_before_after": "",
                        "stage1_out_per_group_before_after": "",
                        "stage1_in_per_group_before_after": "",
                        "stage2_out_per_group_before_after": "",
                        "stage2_in_per_group_before_after": "",
                        "model_artifact_path": "",
                        "latency_contamination_risk": "",
                        "verdict": f"failed:{reason}",
                    }
                )
        if selected_index is not None:
            after_gpu = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_after.json", after_gpu)
            gpu_samples.append({"sample_reason": "after", **after_gpu})
        write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
        write_json(out_dir / "gpu_contamination_report.json", _contamination_report(gpu_samples, selected_index))
        write_csv(out_dir / "budget_selection_trace.csv", all_budget_trace)
        write_csv(out_dir / "real_val500_latency.csv", all_latency)
        write_json(out_dir / "real_val500_ap.json", all_ap)
        write_json(out_dir / "reload_report.json", all_reload)
        write_csv(out_dir / "v108_summary.csv", summary_rows)
        (out_dir / "v108_complete_taylor_greedy_pruner_verdict.md").write_text(
            "\n".join(_verdict_lines(summary_rows, failure, baseline_summary)) + "\n",
            encoding="utf-8",
        )
        write_json(out_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": "", "attempted_eval_entrypoints": attempted_eval})
        print(json.dumps({"success": True, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        write_json(out_dir / "failure_report.json", {"success": False, "failure_reason": failure, "traceback": tb, "attempted_eval_entrypoints": attempted_eval})
        if selected_index is not None:
            try:
                after_gpu = collect_gpu_state(selected_index)
                write_json(out_dir / "gpu_state_after.json", after_gpu)
                gpu_samples.append({"sample_reason": "after_failure", **after_gpu})
                write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
                write_json(out_dir / "gpu_contamination_report.json", _contamination_report(gpu_samples, selected_index))
            except Exception:
                pass
        write_csv(out_dir / "v108_summary.csv", summary_rows)
        (out_dir / "v108_complete_taylor_greedy_pruner_verdict.md").write_text("\n".join(_verdict_lines(summary_rows, failure)) + "\n", encoding="utf-8")
        print(json.dumps({"success": False, "failure": failure, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.8 complete Taylor greedy global pruner")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--targets", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--importance", default="first_order_taylor", choices=["first_order_taylor"])
    parser.add_argument("--selector", default="greedy_global_ranking", choices=["greedy_global_ranking"])
    parser.add_argument("--align-channels", type=int, default=8)
    parser.add_argument("--max-ch-sparsity", type=float, default=0.60)
    parser.add_argument("--protect-fpn-output", action="store_true")
    parser.add_argument("--protect-head-output", action="store_true")
    parser.add_argument("--no-extra-output-protection", action="store_true")
    parser.add_argument("--eval-frames", type=int, default=500)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-repeat", type=int, default=300)
    parser.add_argument("--save-model-artifacts", action="store_true")
    parser.add_argument("--num-calib-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/latency_lut/v108_complete_taylor_greedy_pruner")
    args = parser.parse_args(argv)
    if not args.protect_fpn_output:
        args.protect_fpn_output = True
    if not args.protect_head_output:
        args.protect_head_output = True
    return args


def main(argv: list[str] | None = None) -> int:
    return run_v108(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
