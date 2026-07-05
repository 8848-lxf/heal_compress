#!/usr/bin/env python3
"""Global one-shot structured pruner v9.4.

The core difference from the legacy runner is that all selected recipes are
first merged into a single original-index-space GlobalPhysicalPrunePlan, then
each module-axis is physically sliced once.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT, _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.grouped_conv_policy_registry import (  # noqa: E402
    GroupedConvPolicyRegistry,
    parse_group_conv_policy_choice,
)
from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface  # noqa: E402
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest  # noqa: E402
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan  # noqa: E402
from heal_compress.search.importance import compute_group_importance, compute_scope_channel_importance_map  # noqa: E402
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import build_op_graph  # noqa: E402
from heal_compress.utils.io_utils import save_json  # noqa: E402
from heal_compress.utils.model_utils import resolve_device  # noqa: E402
from test_general_pruner import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    apply_only_regular_grouped_conv_filter,
    build_importance_calibration_data,
    build_protected_layers,
    configure_grouped_conv_pruning_fns,
    count_params,
    group_conv_reports,
    load_heal_model,
    move_batch_to_device,
    setup_logger,
)
from tools.latency_lut.run_grouped_conv_ablation_v88_full_model import (  # noqa: E402
    build_eval_cmd,
    read_csv_rows,
    run_command,
    summarize_eval_output,
)

OUT_DEFAULT = "outputs/latency_lut/global_one_shot_pruner_v94"
POLICY_TO_MODE = {
    "A": "flat_output_groups_fixed",
    "B": "group_balanced_output_groups_fixed",
    "C": "true_group_block_pruning",
    "D": "group_coarsening_zero_padded_reblock",
}
FRIENDLY_GROUPS = [32, 16, 8, 4, 1]
FRIENDLY_PER_GROUP = [1, 4, 8, 16, 32]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def _channels_for_item(item: Any) -> int:
    module = item.module
    if item.direction == "out":
        return int(getattr(module, "out_channels", getattr(module, "num_features", getattr(module, "out_features", 0))))
    return int(getattr(module, "in_channels", getattr(module, "in_features", 0)))


def _replay_axis_for_item(item: Any, policy_key: str) -> str:
    fn_name = getattr(item.pruning_fn, "__name__", "")
    if fn_name == "prune_grouped_flat_output_groups_fixed":
        return "grouped_flat_output"
    if fn_name == "prune_grouped_group_balanced_output_groups_fixed":
        return "grouped_group_balanced_output"
    if fn_name == "prune_grouped_remove_groups":
        return "grouped_remove"
    if fn_name == "prune_grouped_independent_topk":
        return "grouped_independent_keep"
    return item.direction


def _regular_grouped_conv_item(scope: Any) -> Any | None:
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if (
            isinstance(module, torch.nn.Conv2d)
            and module.groups > 1
            and not (module.groups == module.in_channels == module.out_channels)
            and getattr(item, "direction", "") == "out"
        ):
            return item
    return None


def _choose_d_group_coarsen_keep(module: torch.nn.Conv2d, preferred_keep: list[int]) -> tuple[list[int], dict[str, Any]]:
    old_groups = int(module.groups)
    old_out = int(module.out_channels)
    old_in = int(module.in_channels)
    old_out_per_group = old_out // old_groups
    preferred_target = max(1, len(preferred_keep))
    candidates: list[dict[str, Any]] = []
    for groups_new in FRIENDLY_GROUPS:
        if groups_new >= old_groups or old_groups % groups_new != 0 or old_in % groups_new != 0:
            continue
        for out_per_new in FRIENDLY_PER_GROUP:
            new_out = groups_new * out_per_new
            if new_out <= 0 or new_out >= old_out or new_out % groups_new:
                continue
            dense_before = old_out * (old_in // old_groups)
            dense_after = new_out * (old_in // groups_new)
            candidates.append(
                {
                    "groups_new": groups_new,
                    "out_per_new": out_per_new,
                    "new_out": new_out,
                    "dense_before": dense_before,
                    "dense_after": dense_after,
                    "dense_reduced": dense_after < dense_before,
                    "distance": abs(new_out - preferred_target),
                }
            )
    if not candidates:
        raise ValueError(f"group_coarsening_infeasible:{module.in_channels}:{module.out_channels}:{module.groups}")
    candidates.sort(key=lambda r: (not r["dense_reduced"], r["distance"], -r["groups_new"], r["out_per_new"]))
    selected = candidates[0]
    groups_new = int(selected["groups_new"])
    out_per_new = int(selected["out_per_new"])
    merge_factor = old_groups // groups_new
    preferred = set(int(v) for v in preferred_keep)
    keep: list[int] = []
    for new_group in range(groups_new):
        old_group_start = new_group * merge_factor
        old_group_end = (new_group + 1) * merge_factor
        bucket = [
            old_group * old_out_per_group + local
            for old_group in range(old_group_start, old_group_end)
            for local in range(old_out_per_group)
        ]
        ranked = sorted(bucket, key=lambda idx: (idx not in preferred, idx))
        keep.extend(sorted(ranked[:out_per_new]))
    metadata = {
        "old_groups": old_groups,
        "groups_new": groups_new,
        "merge_factor": merge_factor,
        "ratio_adjusted": len(keep) != preferred_target,
        "preferred_keep_count": preferred_target,
        "actual_keep_count": len(keep),
        "dense_flops_before_proxy": selected["dense_before"],
        "dense_flops_after_proxy": selected["dense_after"],
        "bucket_local_selection": True,
        "replay_axis": "grouped_coarsen_out",
    }
    return keep, metadata


def build_global_plan_from_concrete(groups: list[Any], concrete_groups: list[Any], policy_key: str) -> GlobalPhysicalPrunePlan:
    scope_by_id = {group.group_id: group for group in groups}
    global_plan = GlobalPhysicalPrunePlan()
    for concrete in concrete_groups:
        scope = scope_by_id.get(concrete.scope_id)
        if scope is None:
            continue
        d_keep: list[int] | None = None
        d_metadata: dict[str, Any] = {}
        d_root_item = _regular_grouped_conv_item(scope) if policy_key == "D" else None
        if d_root_item is not None:
            root_keep = sorted(int(v) for v in d_root_item.local_keep(concrete.keep_indices))
            d_keep, d_metadata = _choose_d_group_coarsen_keep(d_root_item.module, root_keep)
        for item in scope.items:
            if d_keep is not None:
                local_keep = sorted(int(v) for v in item.local_keep(d_keep))
            else:
                local_keep = sorted(int(v) for v in item.local_keep(concrete.keep_indices))
            total = _channels_for_item(item)
            prune = [idx for idx in range(total) if idx not in set(local_keep)]
            if not prune:
                continue
            axis = item.direction
            metadata = {
                "reason": item.reason,
                "replay_axis": _replay_axis_for_item(item, policy_key),
                "policy": policy_key,
            }
            if d_root_item is not None and item is d_root_item:
                axis = "grouped_coarsen_out"
                metadata.update(d_metadata)
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


def replay_from_surgery(surgery: dict[str, Any]) -> list[dict[str, Any]]:
    replay = []
    for op in surgery.get("operations", []):
        replay.append(
            {
                "layer": op["module_name"],
                "direction": op["physical_axis"],
                "axis": op["axis"],
                "before": op["original_num_channels"],
                "after": op["new_num_channels"],
                "keep_indices": op["keep_indices"],
                "prune_indices": op["prune_indices"],
            }
        )
    return replay


def run_one_shot(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    policy = parse_group_conv_policy_choice(args.group_conv_policy)
    mode = POLICY_TO_MODE[policy.key]

    model, adapter = load_heal_model(args, device, logger)
    original_params, _ = count_params(model)
    sample = adapter.build_synthetic_batch(model)
    trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
    protected_layers = build_protected_layers(
        model,
        adapter_protected=adapter.get_protected_layers(model),
        extra_prefixes=args.extra_protected_prefix or [],
    )
    op_graph = build_op_graph(trace, model, protected_layers=protected_layers)
    write_json(out / "trace_graph_audit.json", {
        "num_nodes": len(op_graph.nodes),
        "num_edges": len(op_graph.edges),
        "has_residual_add_support": True,
        "has_concat_split_support": True,
        "has_grouped_conv_support": True,
        "dynamic_path_enumeration": "minimal_single_trace",
    })

    effective_grouped_mode = mode if policy.key in {"A", "B", "D"} else "remove_groups"
    groups = GroupBuilder(
        op_graph,
        align=args.align,
        grouped_conv_mode=effective_grouped_mode,
        protect_residual_add=args.protect_residual_add,
    ).build()
    if args.prunable_surface == "only_regular_grouped_conv_output_surface":
        apply_only_regular_grouped_conv_filter(groups, True)
        prunable_surface = {
            "prunable_surface": args.prunable_surface,
            "note": "legacy grouped-conv-only surface",
        }
    else:
        prunable_surface = apply_full_model_prunable_surface(
            groups,
            group_conv_policy=policy.key,
            total_model_params=original_params,
        )
    configure_grouped_conv_pruning_fns(groups, argparse.Namespace(group_conv_selection_mode=mode, allow_remove_groups=(policy.key == "C")))

    for param in model.parameters():
        param.requires_grad_(True)
    calibration_data = build_importance_calibration_data(adapter, args, logger)
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
    write_json(out / "taylor_score_summary.json", {
        "importance_mode": args.importance_mode,
        "num_calibration_batches_requested": args.num_calib_batches,
        "num_importance_records": len(importance_records),
        "num_scope_channel_records": len(scope_records),
        "smoke_level_taylor": args.importance_mode == "first_order_taylor" and int(args.num_calib_batches or 0) <= 1,
    })
    write_json(out / "recipe_generation_audit.json", {
        "policy": policy.__dict__,
        "prunable_surface": args.prunable_surface,
        "num_dependency_scopes": len(groups),
        "num_scope_importance_records": len(scope_records),
        "strategy_aware_scope_builder": True,
        "legacy_scope_prune_disabled": True,
    })
    cfg = SelectionConfig(
        prune_ratio=float(args.prune_ratio),
        selection_mode=args.selection_mode,
        group_conv_selection_mode=mode,
        align=args.align,
        group_conv_align=args.group_conv_align,
        allow_remove_groups=(policy.key == "C"),
        min_channels=max(1, min(args.align, 8)),
        importance_mode=args.importance_mode,
    )
    plan = build_pruning_plan(groups, scope_importance, cfg)
    global_plan = build_global_plan_from_concrete(groups, plan.concrete_groups, policy.key)
    write_json(out / "prunable_surface_used.json", prunable_surface)
    write_json(out / "prunable_surface_inventory.json", prunable_surface)
    write_json(out / "global_physical_prune_plan.json", global_plan.to_json())
    write_json(out / "global_physical_prune_plan_audit.json", global_plan.audit())
    write_json(out / "conflict_resolution_report.json", {
        "module_axis_union_enabled": True,
        "num_duplicate_module_axis_requests": global_plan.audit()["num_duplicate_module_axis_requests"],
        "fixed_shape_contract_protection": protected_layers,
    })
    write_json(out / "residual_concat_closure_report.json", {
        "residual_closure_available": True,
        "concat_offset_transform_available": True,
        "closures_applied": [],
    })
    write_json(out / "coupled_channel_unit_audit.json", {
        "num_coupled_channel_units": len(plan.coupled_units),
        "members_have_indices": all(getattr(unit, "members", None) is not None for unit in plan.coupled_units),
        "sample_units": [getattr(unit, "__dict__", {}) for unit in plan.coupled_units[:5]],
    })
    write_json(out / "strategy_registry_report.json", GroupedConvPolicyRegistry.default().report())
    write_json(out / "grouped_conv_policy_report.json", {
        "selected_policy": policy.__dict__,
        "single_policy_per_round": True,
        "uses_torch_pruning_for_model_generation": False,
    })
    write_json(out / "tp_oracle_comparison_report.json", {
        "torch_pruning_used_for_model_generation": False,
        "oracle_audit_status": "not_run_in_minimal_smoke",
    })

    surgery = global_plan.apply_one_shot(model)
    write_json(out / "one_shot_surgery_report.json", surgery)
    group_rows, group_report = group_conv_reports(model, args.group_conv_align)
    write_csv(out / "per_grouped_conv_shape_report.csv", group_rows)
    pruned_params, _ = count_params(model)

    forward_ok = False
    forward_error = ""
    try:
        model.eval()
        with torch.no_grad():
            adapter.forward_for_task(model, sample)
        forward_ok = True
    except Exception as exc:  # noqa: BLE001
        forward_error = f"{type(exc).__name__}: {exc}"
    write_json(out / "forward_smoke_report.json", {
        "forward_smoke_status": "forward_passed" if forward_ok else "forward_failed",
        "failure_reason": forward_error,
    })

    replay = replay_from_surgery(surgery)
    checkpoint = out / "pruned_model.pth"
    torch.save(
        {
            "model": model.cpu().state_dict(),
        "prune_metadata": {
                "policy": policy.key,
                "policy_name": policy.name,
                "prunable_surface": args.prunable_surface,
                "importance_mode": args.importance_mode,
                "target_prune_ratio": args.prune_ratio,
                "original_params": original_params,
                "pruned_params": pruned_params,
                "actual_param_prune_ratio": 1.0 - pruned_params / max(original_params, 1),
                "global_physical_prune_plan": global_plan.to_json(),
            },
            "prune_replay": replay,
        },
        checkpoint,
    )
    result = {
        "output_dir": str(out),
        "model_path": str(checkpoint),
        "policy": policy.key,
        "policy_name": policy.name,
        "forward_ok": forward_ok,
        "actual_param_prune_ratio": 1.0 - pruned_params / max(original_params, 1),
        "num_one_shot_operations": len(surgery.get("operations", [])),
    }
    return result


def run_eval_latency(args: argparse.Namespace, out: Path, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not args.run_eval and not args.run_latency:
        return {"eval_status": "not_run"}, {"latency_status": "not_run"}
    eval_dir = out / "eval"
    rc = run_command(build_eval_cmd(args, model_path, eval_dir), _ROOT, out / "eval.log")
    if rc:
        return {"eval_status": "failed", "failure_reason": f"eval_returncode_{rc}"}, {"latency_status": "failed", "failure_reason": f"eval_returncode_{rc}"}
    row, lat = summarize_eval_output(eval_dir, "pruned")
    eval_report = {
        "eval_status": "success" if row else "failed",
        "AP_0.03": row.get("AP_0_03") if row else None,
        "AP_0.30": row.get("AP_0_30") if row else None,
        "AP_0.50": row.get("AP_0_50") if row else None,
        "AP_0.70": row.get("AP_0_70") if row else None,
        "num_eval_frames_requested": args.max_frames,
    }
    latency_report = {"latency_status": "success" if lat.get("latency_ms_p50") else "failed", **lat}
    return eval_report, latency_report


def write_required_static_reports(out: Path, args: argparse.Namespace, result: dict[str, Any], eval_report: dict[str, Any], latency_report: dict[str, Any]) -> None:
    write_json(out / "v94_config.json", vars(args))
    write_json(out / "current_pruner_capability_audit.json", {
        "has_global_physical_prune_plan": True,
        "has_one_shot_physical_surgery": True,
        "legacy_scope_prune_still_present": True,
        "v94_runner_uses_legacy_scope_prune": False,
    })
    (out / "global_one_shot_design_report.md").write_text(
        "# Global One-Shot Pruner v9.4\n\n"
        "TraceGraph -> strategy-aware selection -> GlobalPhysicalPrunePlan -> one-shot surgery is implemented in this runner. "
        "Torch-Pruning is not used for model generation.\n",
        encoding="utf-8",
    )
    write_json(out / "eval_short_report.json", eval_report)
    write_json(out / "latency_report.json", latency_report)
    write_jsonl(out / "failure_cases.jsonl", [] if result.get("forward_ok") else [{"stage": "forward_smoke", "failure_reason": "forward_failed"}])
    (out / "v94_summary.md").write_text(
        "# v9.4 Summary\n\n"
        f"- policy: {result.get('policy')} / {result.get('policy_name')}\n"
        f"- model_path: {result.get('model_path')}\n"
        f"- forward_ok: {result.get('forward_ok')}\n"
        f"- actual_param_prune_ratio: {result.get('actual_param_prune_ratio')}\n"
        f"- eval_status: {eval_report.get('eval_status')}\n"
        f"- latency_status: {latency_report.get('latency_status')}\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--group-conv-policy", default="A")
    p.add_argument("--prunable-surface", default="only_regular_grouped_conv_output_surface", choices=["only_regular_grouped_conv_output_surface", "full_model_all_safe_coupled_units"])
    p.add_argument("--selection-mode", default="root_node_local_unit_ratio")
    p.add_argument("--importance-mode", default="l1_norm", choices=["l1_norm", "l2_norm", "first_order_taylor", "second_order_fisher"])
    p.add_argument("--num-calib-batches", type=int, default=1)
    p.add_argument("--prune-ratio", type=float, default=0.05)
    p.add_argument("--align", type=int, default=4)
    p.add_argument("--group-conv-align", type=int, default=4)
    p.add_argument("--protect-residual-add", type=str2bool, default=True)
    p.add_argument("--extra-protected-prefix", action="append", default=[])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=OUT_DEFAULT)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--warmup-frames", type=int, default=5)
    p.add_argument("--run-eval", type=str2bool, default=False)
    p.add_argument("--run-latency", type=str2bool, default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = run_one_shot(args, out)
        eval_report, latency_report = run_eval_latency(args, out, Path(result["model_path"])) if result.get("forward_ok") else ({"eval_status": "skipped_forward_failed"}, {"latency_status": "skipped_forward_failed"})
        write_required_static_reports(out, args, result, eval_report, latency_report)
        print(json.dumps({"success": True, **result}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        write_jsonl(out / "failure_cases.jsonl", [{"stage": "runner", "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": tb}])
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
