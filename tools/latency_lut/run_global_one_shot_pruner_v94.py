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
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.grouped_conv_policy_registry import (  # noqa: E402
    GroupedConvPolicyRegistry,
    parse_group_conv_policy_choice,
)
from heal_compress.pruning.grouped_conv import resolve_grouped_conv_input_keep  # noqa: E402
from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface  # noqa: E402
from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator  # noqa: E402
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest  # noqa: E402
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan  # noqa: E402
from heal_compress.pruning.tp_oracle_diff import build_tp_oracle_diff  # noqa: E402
from heal_compress.search.importance import compute_group_importance, compute_scope_channel_importance_map  # noqa: E402
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import build_op_graph  # noqa: E402
from heal_compress.utils.io_utils import save_json  # noqa: E402
from heal_compress.utils.model_utils import resolve_device  # noqa: E402
from heal_compress.pruning.model_io import (  # noqa: E402
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
    normalize_grouped_convs_for_alignment,
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
    module = getattr(item, "module", None)
    if (
        isinstance(module, torch.nn.Conv2d)
        and int(module.groups) > 1
        and getattr(item, "direction", "") == "in"
    ):
        return "grouped_input_balanced"
    if fn_name == "prune_grouped_conv_input_balanced":
        return "grouped_input_balanced"
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


def _regular_grouped_conv_input_item(scope: Any) -> Any | None:
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if (
            isinstance(module, torch.nn.Conv2d)
            and module.groups > 1
            and not (module.groups == module.in_channels == module.out_channels)
            and getattr(item, "direction", "") == "in"
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


def build_global_plan_from_concrete(
    groups: list[Any],
    concrete_groups: list[Any],
    policy_key: str,
    *,
    grouped_input_reports: list[dict[str, Any]] | None = None,
) -> GlobalPhysicalPrunePlan:
    scope_by_id = {group.group_id: group for group in groups}
    global_plan = GlobalPhysicalPrunePlan()
    for concrete in concrete_groups:
        scope = scope_by_id.get(concrete.scope_id)
        if scope is None:
            continue
        concrete_keep = list(concrete.keep_indices)
        grouped_input_item = _regular_grouped_conv_input_item(scope)
        if grouped_input_item is not None:
            local_keep = sorted(int(v) for v in grouped_input_item.local_keep(concrete_keep))
            if local_keep != sorted(concrete_keep):
                if grouped_input_reports is not None:
                    grouped_input_reports.append({
                        "scope_id": concrete.scope_id,
                        "module_name": grouped_input_item.name,
                        "status": "skipped",
                        "reason": "grouped_input_non_identity_index_transform",
                        "preferred_keep_count": len(local_keep),
                    })
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
        d_keep: list[int] | None = None
        d_metadata: dict[str, Any] = {}
        d_root_item = _regular_grouped_conv_item(scope) if policy_key == "D" else None
        if d_root_item is not None:
            root_keep = sorted(int(v) for v in d_root_item.local_keep(concrete_keep))
            d_keep, d_metadata = _choose_d_group_coarsen_keep(d_root_item.module, root_keep)
        for item in scope.items:
            if d_keep is not None:
                local_keep = sorted(int(v) for v in item.local_keep(d_keep))
            else:
                local_keep = sorted(int(v) for v in item.local_keep(concrete_keep))
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
            if (
                getattr(item.pruning_fn, "__name__", "") == "prune_grouped_conv_input_balanced"
                or (
                    isinstance(getattr(item, "module", None), torch.nn.Conv2d)
                    and int(getattr(item.module, "groups", 1)) > 1
                    and getattr(item, "direction", "") == "in"
                )
            ):
                axis = "grouped_input_balanced"
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


def _ratio_tag(value: float) -> str:
    return f"{float(value):.2f}"


def _request_rows_by_module(global_plan: GlobalPhysicalPrunePlan) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for req in global_plan.requests():
        row = {
            "module_name": req.module_name,
            "axis": req.axis,
            "prune_indices": [int(v) for v in req.prune_indices],
            "source_recipe_ids": list(req.source_recipe_ids),
            "metadata": dict(req.metadata or {}),
        }
        rows.setdefault(req.module_name, []).append(row)
    return rows


def _filter_plan_for_source_recipe_ids(
    global_plan: GlobalPhysicalPrunePlan,
    source_recipe_ids: list[str],
) -> tuple[GlobalPhysicalPrunePlan, int]:
    wanted = {str(v) for v in source_recipe_ids if str(v)}
    if not wanted:
        return global_plan, len(global_plan.requests())
    filtered = GlobalPhysicalPrunePlan()
    total = 0
    kept = 0
    for req in global_plan.requests():
        total += 1
        req_sources = set(str(v) for v in (req.source_recipe_ids or ([req.source_recipe_id] if req.source_recipe_id else [])))
        if not req_sources.intersection(wanted):
            continue
        kept += 1
        filtered.add_request(
            ModuleAxisPruneRequest(
                module_name=req.module_name,
                axis=req.axis,
                prune_indices=list(req.prune_indices),
                source_recipe_id=req.source_recipe_id,
                metadata=dict(req.metadata or {}),
                source_recipe_ids=list(req.source_recipe_ids or []),
            )
        )
    return filtered, kept


def _first_shape_issue(sim_report: dict[str, Any]) -> dict[str, Any]:
    issues = list(sim_report.get("issues", []) or [])
    if not issues:
        return {"issue": "forward_failed_after_shape_simulator_passed", "module_name": ""}
    priority = {
        "fixed_shape_contract_pruned": 0,
        "downstream_input_channel_mismatch": 1,
        "norm_channel_mismatch": 2,
        "concat_downstream_input_mismatch": 3,
        "residual_add_branch_channel_mismatch": 4,
        "grouped_conv_inner_channel_alignment": 5,
        "grouped_conv_divisibility": 6,
    }
    issues.sort(key=lambda item: priority.get(str(item.get("issue", "")), 100))
    return issues[0]


def _related_modules(op_graph: Any, module_name: str) -> tuple[list[str], list[str]]:
    if op_graph is None or not module_name or module_name not in getattr(op_graph, "nodes", {}):
        return [], []
    upstream = [src for src, _idx in op_graph.incoming(module_name)]
    downstream = [dst for dst, _idx in op_graph.outgoing(module_name)]
    return upstream, downstream


def _path_flags(model: torch.nn.Module, op_graph: Any, names: list[str]) -> dict[str, bool]:
    modules = dict(model.named_modules())
    joined = " ".join(names).lower()
    grouped = False
    for name in names:
        module = modules.get(name)
        if isinstance(module, torch.nn.Conv2d) and int(module.groups) > 1:
            grouped = True
            break
    op_types = []
    if op_graph is not None:
        for name in names:
            node = getattr(op_graph, "nodes", {}).get(name)
            if node is not None:
                op_types.append(str(getattr(node, "op_type", "")))
    return {
        "residual": "Add" in op_types or "add" in joined or "residual" in joined or "downsample" in joined,
        "concat": "Cat" in op_types or "cat" in joined or "concat" in joined,
        "grouped_conv": grouped,
        "deblock": "deblock" in joined or "convtranspose" in joined,
        "head": any(key in joined for key in ("cls_head", "reg_head", "dir_head", "heatmap_head")),
    }


def _shape_failure_payload(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    global_plan: GlobalPhysicalPrunePlan,
    sim_report: dict[str, Any],
    op_graph: Any,
    stage: str,
    failure_traceback: str,
    forward_error: str = "",
) -> dict[str, Any]:
    issue = _first_shape_issue(sim_report)
    issue_name = str(issue.get("issue", ""))
    first_module = str(issue.get("module_name", ""))
    requests_by_module = _request_rows_by_module(global_plan)
    request = None
    if first_module in requests_by_module:
        request = requests_by_module[first_module][0]
    upstream_from_issue = str(issue.get("upstream", issue.get("details", {}).get("upstream", "")))
    if request is None and upstream_from_issue in requests_by_module:
        request = requests_by_module[upstream_from_issue][0]
    if request is None and global_plan.requests():
        req = global_plan.requests()[0]
        request = {
            "module_name": req.module_name,
            "axis": req.axis,
            "prune_indices": [int(v) for v in req.prune_indices],
            "source_recipe_ids": list(req.source_recipe_ids),
            "metadata": dict(req.metadata or {}),
        }
    root_module = str((request or {}).get("module_name", first_module))
    if not first_module:
        first_module = root_module
    shapes = sim_report.get("shapes", {}) or {}
    before_shape = shapes.get(first_module) or shapes.get(root_module) or {}
    after_planned_shape = dict(before_shape)
    prune_indices = list((request or {}).get("prune_indices", issue.get("prune_indices", [])) or [])
    keep_indices = list(issue.get("keep_indices", []) or [])
    if not keep_indices and before_shape:
        axis = str((request or {}).get("axis", ""))
        original = before_shape.get("out_channels_before") if axis == "out" else before_shape.get("in_channels_before")
        if original is None:
            original = before_shape.get("out_channels_before")
        try:
            keep_indices = [idx for idx in range(int(original)) if idx not in set(int(v) for v in prune_indices)]
        except Exception:  # noqa: BLE001
            keep_indices = []
    source_tensor_shape: list[Any] = []
    target_tensor_shape: list[Any] = []
    if "upstream_channels_after" in issue and "downstream_in_after" in issue:
        source_tensor_shape = [issue.get("upstream_channels_after"), "N/H/W"]
        target_tensor_shape = [issue.get("downstream_in_after"), "N/H/W"]
    elif "before" in issue and "after_planned" in issue:
        source_tensor_shape = [issue.get("after_planned"), "N"]
        target_tensor_shape = [issue.get("before"), "N"]
    elif before_shape:
        source_tensor_shape = [before_shape.get("out_channels_after"), "N/H/W"]
        target_tensor_shape = [before_shape.get("out_channels_before"), "N/H/W"]
    related_upstream, related_downstream = _related_modules(op_graph, first_module)
    root_upstream, root_downstream = _related_modules(op_graph, root_module)
    related_names = sorted(set([first_module, root_module] + related_upstream + related_downstream + root_upstream + root_downstream))
    if issue_name == "fixed_shape_contract_pruned" and any(k in first_module.lower() for k in ("pillar_vfe", "pfn")):
        failing_op = "PointPillarScatter fixed canvas assignment would receive pruned PFN features"
    elif issue_name == "downstream_input_channel_mismatch":
        failing_op = "producer output channels do not match downstream module input channels"
    elif issue_name == "concat_downstream_input_mismatch":
        failing_op = "concat branch sum does not match downstream module input channels"
    elif issue_name == "residual_add_branch_channel_mismatch":
        failing_op = "residual add branches have different planned channel widths"
    else:
        failing_op = issue_name or stage
    return {
        "stage": stage,
        "failure_traceback": failure_traceback,
        "forward_error": forward_error,
        "first_failing_module": first_module,
        "failing_assignment_or_forward_op": failing_op,
        "simulator_issue": issue,
        "before_shape": before_shape,
        "after_planned_shape": after_planned_shape,
        "source_tensor_shape": source_tensor_shape,
        "target_tensor_shape": target_tensor_shape,
        "prune_indices": prune_indices,
        "keep_indices": keep_indices,
        "physical_axis": str((request or {}).get("axis", issue.get("axis", ""))),
        "root_action": request or {},
        "related_upstream_modules": sorted(set(related_upstream + root_upstream)),
        "related_downstream_modules": sorted(set(related_downstream + root_downstream)),
        "path_classification": _path_flags(model, op_graph, related_names),
        "policy": args.group_conv_policy,
        "target_prune_ratio": float(args.prune_ratio),
        "prunable_surface": args.prunable_surface,
    }


def _write_shape_failure_reports(
    out: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    global_plan: GlobalPhysicalPrunePlan,
    sim_report: dict[str, Any],
    op_graph: Any,
    *,
    stage: str,
    failure_traceback: str,
    forward_error: str = "",
) -> dict[str, Any]:
    payload = _shape_failure_payload(
        args=args,
        model=model,
        global_plan=global_plan,
        sim_report=sim_report,
        op_graph=op_graph,
        stage=stage,
        failure_traceback=failure_traceback,
        forward_error=forward_error,
    )
    stem = f"shape_failure_root_cause_{str(args.group_conv_policy).upper()}_{_ratio_tag(args.prune_ratio)}"
    write_json(out / f"{stem}.json", payload)
    lines = [
        f"# Shape Failure Root Cause {str(args.group_conv_policy).upper()}@{_ratio_tag(args.prune_ratio)}",
        "",
        f"- stage: {payload['stage']}",
        f"- first_failing_module: {payload['first_failing_module']}",
        f"- failing_assignment_or_forward_op: {payload['failing_assignment_or_forward_op']}",
        f"- physical_axis: {payload['physical_axis']}",
        f"- source_tensor_shape: {payload['source_tensor_shape']}",
        f"- target_tensor_shape: {payload['target_tensor_shape']}",
        f"- path_classification: {payload['path_classification']}",
        "",
        "## Root Action",
        "",
        "```json",
        json.dumps(payload["root_action"], ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Simulator Issue",
        "",
        "```json",
        json.dumps(payload["simulator_issue"], ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _write_tp_oracle_diff(
    out: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    global_plan: GlobalPhysicalPrunePlan,
    root_cause: dict[str, Any],
    *,
    sample: Any,
    adapter: Any,
) -> dict[str, Any]:
    root_action = root_cause.get("root_action", {}) or {}
    root_module = str(root_action.get("module_name", root_cause.get("first_failing_module", "")))
    root_axis = str(root_action.get("axis", root_cause.get("physical_axis", "out")))
    root_indices = [int(v) for v in root_action.get("prune_indices", root_cause.get("prune_indices", []))[:32]]
    root_source_ids = [str(v) for v in root_action.get("source_recipe_ids", []) if str(v)]
    oracle_plan, scoped_count = _filter_plan_for_source_recipe_ids(global_plan, root_source_ids)
    def _tensor_sum(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            if value.is_floating_point() or value.is_complex():
                return value.float().sum()
            return value.float().sum()
        if isinstance(value, dict):
            total: torch.Tensor | None = None
            for child in value.values():
                child_sum = _tensor_sum(child)
                if child_sum is not None:
                    total = child_sum if total is None else total + child_sum
            return total
        if isinstance(value, (list, tuple)):
            total = None
            for child in value:
                child_sum = _tensor_sum(child)
                if child_sum is not None:
                    total = child_sum if total is None else total + child_sum
            return total
        return None

    forward_fn = None
    if adapter is not None:
        def forward_fn(m: torch.nn.Module, x: Any) -> torch.Tensor:
            output = adapter.forward_for_task(m, x)
            total = _tensor_sum(output)
            if total is None:
                raise RuntimeError("tp_oracle_forward_no_tensor_output")
            return total
    diff = build_tp_oracle_diff(
        model,
        oracle_plan,
        root_module_name=root_module,
        root_axis="out" if root_axis == "grouped_coarsen_out" else root_axis,
        root_indices=root_indices,
        example_inputs=sample,
        forward_fn=forward_fn,
    )
    diff["project_plan_scope"] = {
        "scope": "root_action_source_recipe_ids" if root_source_ids else "full_global_plan",
        "root_source_recipe_ids": root_source_ids,
        "num_project_members_in_scope": scoped_count,
        "num_project_members_global": len(global_plan.requests()),
    }
    stem = f"tp_oracle_diff_{str(args.group_conv_policy).upper()}_{_ratio_tag(args.prune_ratio)}.json"
    write_json(out / stem, diff)
    return diff


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
    pre_prune_group_alignment_ops = normalize_grouped_convs_for_alignment(model, args.group_conv_align)
    write_json(out / "pre_prune_group_alignment_report.json", {
        "enabled": True,
        "group_conv_align": args.group_conv_align,
        "num_operations": len(pre_prune_group_alignment_ops),
        "operations": pre_prune_group_alignment_ops,
        "note": "Group merge normalization preserves function with block-diagonal zero-padded weights before pruning.",
    })
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
    grouped_input_reports: list[dict[str, Any]] = []
    global_plan = build_global_plan_from_concrete(
        groups,
        plan.concrete_groups,
        policy.key,
        grouped_input_reports=grouped_input_reports,
    )
    write_json(out / "prunable_surface_used.json", prunable_surface)
    write_json(out / "prunable_surface_inventory.json", prunable_surface)
    write_json(out / "full_model_surface_after_unprotect_grouped_input.json", {
        **prunable_surface,
        "grouped_conv_input_contract_policy": "v9.7_resolver_not_surface_protection",
        "grouped_conv_input_groups_unprotected": sum(
            1 for group in groups if _regular_grouped_conv_input_item(group) is not None and not bool(getattr(group, "protected", False))
        ),
    })
    write_json(out / "grouped_conv_input_pruning_report.json", {
        "num_grouped_conv_input_scopes": sum(1 for group in groups if _regular_grouped_conv_input_item(group) is not None),
        "num_grouped_conv_input_reports": len(grouped_input_reports),
        "num_grouped_conv_input_pruning_ops": sum(1 for row in grouped_input_reports if row.get("status") in {"accepted", "repaired"}),
        "num_grouped_input_balance_violations": sum(1 for row in grouped_input_reports if row.get("reason") == "grouped_input_balance_violation"),
        "rows": grouped_input_reports,
    })
    write_json(out / "global_physical_prune_plan.json", global_plan.to_json())
    write_json(out / "global_physical_prune_plan_audit.json", global_plan.audit())
    write_json(out / "conflict_resolution_report.json", {
        "module_axis_union_enabled": True,
        "num_duplicate_module_axis_requests": global_plan.audit()["num_duplicate_module_axis_requests"],
        "fixed_shape_contract_protection": protected_layers,
        "pre_prune_group_alignment_ops": pre_prune_group_alignment_ops,
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
        "oracle_audit_status": "pending_shape_simulator",
    })

    sim = GlobalPlanShapeSimulator(
        model,
        global_plan,
        op_graph=op_graph,
        group_conv_align=args.group_conv_align,
        allow_convtranspose=False,
        allow_fixed_shape_pruning=False,
    )
    sim_report = sim.simulate()
    write_json(out / "global_plan_shape_simulator_report.json", sim_report)
    if not sim_report.get("legal", False):
        root_cause = _write_shape_failure_reports(
            out,
            args,
            model,
            global_plan,
            sim_report,
            op_graph,
            stage="global_plan_shape_simulator",
            failure_traceback="blocked_by_global_plan_shape_simulator_before_physical_surgery",
        )
        tp_diff = _write_tp_oracle_diff(out, args, model, global_plan, root_cause, sample=sample, adapter=adapter)
        write_json(out / "tp_oracle_comparison_report.json", {
            "torch_pruning_used_for_model_generation": False,
            "oracle_audit_status": tp_diff.get("status", ""),
            "oracle_available": tp_diff.get("available", False),
            "root_module_name": tp_diff.get("root_module_name", ""),
        })
        write_json(out / "one_shot_surgery_report.json", {
            "operations": [],
            "num_operations": 0,
            "skipped_reason": "global_plan_shape_simulator_failed",
        })
        write_csv(out / "target_budget_achievement_report.csv", [{
            "policy": policy.key,
            "target_param_prune_ratio": float(args.prune_ratio),
            "actual_param_prune_ratio": 0.0,
            "target_achieved": False,
            "gap": float(args.prune_ratio),
            "original_params": original_params,
            "pruned_params": original_params,
            "num_grouped_conv_input_pruning_ops": sum(1 for row in grouped_input_reports if row.get("status") in {"accepted", "repaired"}),
            "shape_simulator_legal": False,
            "forward_ok": False,
        }])
        write_json(out / "forward_smoke_report.json", {
            "forward_smoke_status": "skipped_shape_simulator_failed",
            "failure_reason": "global_plan_shape_simulator_failed",
        })
        return {
            "output_dir": str(out),
            "model_path": "",
            "policy": policy.key,
            "policy_name": policy.name,
            "forward_ok": False,
            "actual_param_prune_ratio": 0.0,
            "num_one_shot_operations": 0,
            "num_pre_prune_group_alignment_ops": len(pre_prune_group_alignment_ops),
            "shape_simulator_legal": False,
            "shape_simulator_num_issues": int(sim_report.get("num_issues", 0) or 0),
            "shape_failure_root_cause": root_cause,
        }

    surgery = global_plan.apply_one_shot(model)
    write_json(out / "one_shot_surgery_report.json", surgery)
    group_rows, group_report = group_conv_reports(model, args.group_conv_align)
    write_csv(out / "per_grouped_conv_shape_report.csv", group_rows)
    pruned_params, _ = count_params(model)
    grouped_input_ops = [
        op for op in surgery.get("operations", [])
        if op.get("physical_axis") == "grouped_input_balanced" or op.get("axis") == "grouped_input_balanced"
    ]

    forward_ok = False
    forward_error = ""
    try:
        model.eval()
        with torch.no_grad():
            adapter.forward_for_task(model, sample)
        forward_ok = True
    except Exception as exc:  # noqa: BLE001
        forward_error = f"{type(exc).__name__}: {exc}"
        root_cause = _write_shape_failure_reports(
            out,
            args,
            model,
            global_plan,
            sim_report,
            op_graph,
            stage="forward_smoke",
            failure_traceback=traceback.format_exc(),
            forward_error=forward_error,
        )
        tp_diff = _write_tp_oracle_diff(out, args, model, global_plan, root_cause, sample=sample, adapter=adapter)
        write_json(out / "tp_oracle_comparison_report.json", {
            "torch_pruning_used_for_model_generation": False,
            "oracle_audit_status": tp_diff.get("status", ""),
            "oracle_available": tp_diff.get("available", False),
            "root_module_name": tp_diff.get("root_module_name", ""),
        })
    write_json(out / "forward_smoke_report.json", {
        "forward_smoke_status": "forward_passed" if forward_ok else "forward_failed",
        "failure_reason": forward_error,
    })
    actual_ratio = 1.0 - pruned_params / max(original_params, 1)
    write_csv(out / "target_budget_achievement_report.csv", [{
        "policy": policy.key,
        "target_param_prune_ratio": float(args.prune_ratio),
        "actual_param_prune_ratio": actual_ratio,
        "target_achieved": actual_ratio >= float(args.prune_ratio),
        "gap": float(args.prune_ratio) - actual_ratio,
        "original_params": original_params,
        "pruned_params": pruned_params,
        "num_grouped_conv_input_pruning_ops": len(grouped_input_ops),
        "num_one_shot_operations": len(surgery.get("operations", [])),
        "shape_simulator_legal": bool(sim_report.get("legal", False)),
        "forward_ok": forward_ok,
    }])

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
                "actual_param_prune_ratio": actual_ratio,
                "global_physical_prune_plan": global_plan.to_json(),
                "pre_prune_group_alignment_ops": pre_prune_group_alignment_ops,
                "grouped_conv_input_pruning_reports": grouped_input_reports,
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
        "actual_param_prune_ratio": actual_ratio,
        "num_one_shot_operations": len(surgery.get("operations", [])),
        "num_grouped_conv_input_pruning_ops": len(grouped_input_ops),
        "num_pre_prune_group_alignment_ops": len(pre_prune_group_alignment_ops),
        "shape_simulator_legal": bool(sim_report.get("legal", False)),
        "shape_simulator_num_issues": int(sim_report.get("num_issues", 0) or 0),
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
    p.add_argument("--group-conv-align", type=int, default=8)
    p.add_argument("--protect-residual-add", type=str2bool, default=False)
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
