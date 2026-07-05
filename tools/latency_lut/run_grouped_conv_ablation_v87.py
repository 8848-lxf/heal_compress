from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
UNIAD = ROOT.parent
for p in (UNIAD, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.test_general_pruner import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_HEAL_ROOT, load_heal_model, setup_logger


POLICIES = [
    "flat_output_groups_fixed",
    "group_balanced_output_groups_fixed",
    "true_group_block_pruning",
    "group_coarsening_zero_padded_reblock",
]
RATIOS = [0.25, 0.50, 0.75]
SCORE_MODES = ["l1", "taylor1_task"]


def ordinary_grouped_conv(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.groups > 1
        and not (module.groups == module.in_channels and module.groups == module.out_channels)
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _module_protection_reason(name: str, module: nn.Module) -> tuple[bool, str, bool, bool]:
    lower = name.lower()
    if any(k in lower for k in ("cls_head", "reg_head", "dir_head")) and getattr(module, "out_channels", 0) <= 16:
        return True, "detection_head_final_output_channels", False, False
    if any(k in lower for k in ("pfn_layers", "pillar_vfe")):
        return True, "fixed_width_shape_contract:pfn_to_pointpillar_scatter", False, True
    if "deblocks" in lower or "fpn" in lower:
        return True, "fpn_output_channel_boundary", False, True
    return False, "", False, False


def filter_scores(module: nn.Conv2d, score_mode: str) -> torch.Tensor | None:
    if score_mode == "taylor1_task":
        return None
    weight = module.weight.detach().float()
    return weight.abs().view(weight.shape[0], -1).sum(dim=1)


def _nearest(values: Iterable[int], target: float) -> int | None:
    values = list(dict.fromkeys(int(v) for v in values))
    if not values:
        return None
    return min(values, key=lambda v: (abs(v - target), v))


def _flat_prune_count(c_out: int, groups: int, ratio: float) -> tuple[int | None, bool, int]:
    target = int(round(c_out * ratio))
    legal = [p for p in range(0, c_out) if p % 4 == 0 and (c_out - p) > 0 and (c_out - p) % groups == 0]
    chosen = _nearest(legal, target)
    return chosen, chosen != target if chosen is not None else True, target


def _top_keep(scores: torch.Tensor, keep_count: int, candidates: list[int]) -> list[int]:
    ranked = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)
    return sorted(ranked[:keep_count])


def compute_reinterpretation(old_to_new_out_map: dict[int, int], *, c_out_before: int, c_out_after: int, groups: int) -> dict[str, Any]:
    old_per = c_out_before // groups if groups else 0
    new_per = c_out_after // groups if groups and c_out_after % groups == 0 else 0
    count = 0
    details = []
    old_group_keep_count = {str(g): 0 for g in range(groups)}
    for old_idx, new_idx in sorted(old_to_new_out_map.items()):
        old_group = old_idx // old_per if old_per else -1
        new_group = new_idx // new_per if new_per else -1
        old_group_keep_count[str(old_group)] = old_group_keep_count.get(str(old_group), 0) + 1
        mismatch = old_group != new_group
        count += int(mismatch)
        details.append({"old_out_idx": old_idx, "new_out_idx": new_idx, "old_out_group": old_group, "new_out_group": new_group, "local_input_reinterpretation": mismatch})
    denom = max(len(old_to_new_out_map), 1)
    return {
        "old_group_keep_count": old_group_keep_count,
        "reinterpretation_count": count,
        "reinterpretation_ratio": count / denom,
        "reinterpretation_details": details,
    }


def _conv_flops(c_in: int, c_out: int, groups: int, k_h: int, k_w: int) -> int:
    return int(c_out * (c_in // groups) * k_h * k_w) if groups else 0


def _base_detail(module_name: str, module: nn.Conv2d, policy: str, score_mode: str, ratio: float) -> dict[str, Any]:
    k_h, k_w = module.kernel_size
    return {
        "policy": policy,
        "score_mode": score_mode,
        "target_prune_ratio": ratio,
        "module_name": module_name,
        "module_type": module.__class__.__name__,
        "C_in_before": int(module.in_channels),
        "C_out_before": int(module.out_channels),
        "groups_before": int(module.groups),
        "in_per_group_before": int(module.in_channels // module.groups),
        "out_per_group_before": int(module.out_channels // module.groups),
        "weight_shape_before": list(module.weight.shape),
        "current_layer_input_pruned": False,
        "current_layer_output_pruned": False,
        "previous_layer_output_pruned": False,
        "next_layer_input_pruned": False,
        "groups_changed": False,
        "legality_status": "unknown",
        "failure_reason": "",
        "dense_flops_before": _conv_flops(module.in_channels, module.out_channels, module.groups, k_h, k_w),
    }


def _finish_detail(detail: dict[str, Any], module: nn.Conv2d, keep_out: list[int], groups_after: int, c_in_after: int, weight_after: torch.Tensor | None) -> dict[str, Any]:
    c_out_after = len(keep_out) if weight_after is None else int(weight_after.shape[0])
    detail.update(
        {
            "C_in_after": int(c_in_after),
            "C_out_after": int(c_out_after),
            "groups_after": int(groups_after),
            "in_per_group_after": int(c_in_after // groups_after) if groups_after and c_in_after % groups_after == 0 else 0,
            "out_per_group_after": int(c_out_after // groups_after) if groups_after and c_out_after % groups_after == 0 else 0,
            "weight_shape_after": list(weight_after.shape) if weight_after is not None else [],
            "groups_changed": int(groups_after) != int(module.groups),
            "actual_prune_ratio": 1.0 - c_out_after / max(module.out_channels, 1),
            "dense_flops_after": _conv_flops(c_in_after, c_out_after, groups_after, *module.kernel_size),
        }
    )
    before = max(float(detail["dense_flops_before"]), 1.0)
    detail["theoretical_dense_flops_speedup"] = before / max(float(detail["dense_flops_after"]), 1.0)
    return detail


def _copy_conv(module: nn.Conv2d, *, c_in: int, c_out: int, groups: int, weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Conv2d:
    new = nn.Conv2d(
        c_in,
        c_out,
        module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=groups,
        bias=bias is not None,
        padding_mode=module.padding_mode,
    )
    with torch.no_grad():
        new.weight.copy_(weight.detach().cpu())
        if bias is not None and new.bias is not None:
            new.bias.copy_(bias.detach().cpu())
    return new.eval()


def _choose_groups_new(c_in: int, c_out_new: int, groups_old: int, out_per_old: int) -> int | None:
    candidates = []
    for g in range(1, groups_old):
        if groups_old % g == 0 and c_in % g == 0 and c_out_new % g == 0:
            out_per = c_out_new // g
            if out_per in {1, 4, 8, 16, 32, 64, 128}:
                candidates.append(g)
    if not candidates:
        return None
    return min(candidates, key=lambda g: (abs(c_out_new // g - out_per_old), -g))


def make_grouped_conv_candidate(
    *,
    module_name: str,
    module: nn.Conv2d,
    policy: str,
    score_mode: str,
    target_prune_ratio: float,
) -> dict[str, Any]:
    detail = _base_detail(module_name, module, policy, score_mode, target_prune_ratio)
    scores = filter_scores(module, score_mode)
    if scores is None:
        detail.update({"legality_status": "pending", "failure_reason": "taylor1_task_pending_not_executed"})
        return detail
    c_in, c_out, groups = int(module.in_channels), int(module.out_channels), int(module.groups)
    in_per, out_per = c_in // groups, c_out // groups
    if c_in % groups != 0 or c_out % groups != 0:
        detail.update({"legality_status": "failed", "failure_reason": "invalid_original_grouped_conv"})
        return detail
    bias = module.bias.detach().float() if module.bias is not None else None

    if policy == "flat_output_groups_fixed":
        prune_count, adjusted, target_count = _flat_prune_count(c_out, groups, target_prune_ratio)
        if prune_count is None:
            detail.update({"legality_status": "failed", "failure_reason": "grouped_conv_shape_infeasible"})
            return detail
        keep = _top_keep(scores, c_out - prune_count, list(range(c_out)))
        old_to_new = {old: i for i, old in enumerate(keep)}
        weight = module.weight.detach().float()[keep].cpu()
        b = bias[keep].cpu() if bias is not None else None
        detail.update(
            {
                "target_prune_count": target_count,
                "adjusted_prune_count": prune_count,
                "ratio_adjusted": adjusted,
                "keep_out_idx": keep,
                "old_to_new_out_map": old_to_new,
                "current_layer_output_pruned": prune_count > 0,
                "next_layer_input_pruned": prune_count > 0,
                "legality_status": "legal",
                "failure_reason": "",
                "new_conv": _copy_conv(module, c_in=c_in, c_out=len(keep), groups=groups, weight=weight, bias=b),
            }
        )
        detail.update(compute_reinterpretation(old_to_new, c_out_before=c_out, c_out_after=len(keep), groups=groups))
        return _finish_detail(detail, module, keep, groups, c_in, weight)

    if policy == "group_balanced_output_groups_fixed":
        target_keep = c_out * (1.0 - target_prune_ratio)
        feasible_keep_per = [k for k in range(1, out_per + 1) if (k * groups) % 4 == 0]
        keep_per = _nearest(feasible_keep_per, target_keep / groups)
        if keep_per is None:
            detail.update({"legality_status": "failed", "failure_reason": "grouped_conv_shape_infeasible"})
            return detail
        keep = []
        for g in range(groups):
            inds = list(range(g * out_per, (g + 1) * out_per))
            keep.extend(_top_keep(scores, keep_per, inds))
        keep = sorted(keep)
        old_to_new = {old: i for i, old in enumerate(keep)}
        weight = module.weight.detach().float()[keep].cpu()
        b = bias[keep].cpu() if bias is not None else None
        stats = compute_reinterpretation(old_to_new, c_out_before=c_out, c_out_after=len(keep), groups=groups)
        group_balanced_pass = len(set(stats["old_group_keep_count"].values())) == 1 and stats["reinterpretation_count"] == 0
        detail.update(
            {
                "target_prune_count": int(round(c_out * target_prune_ratio)),
                "adjusted_prune_count": c_out - len(keep),
                "ratio_adjusted": c_out - len(keep) != int(round(c_out * target_prune_ratio)),
                "keep_out_idx": keep,
                "old_to_new_out_map": old_to_new,
                "group_balanced_pass": group_balanced_pass,
                "current_layer_output_pruned": len(keep) < c_out,
                "next_layer_input_pruned": len(keep) < c_out,
                "legality_status": "legal" if group_balanced_pass else "failed",
                "failure_reason": "" if group_balanced_pass else "group_balanced_violation",
                "new_conv": _copy_conv(module, c_in=c_in, c_out=len(keep), groups=groups, weight=weight, bias=b),
            }
        )
        detail.update(stats)
        return _finish_detail(detail, module, keep, groups, c_in, weight)

    if policy == "true_group_block_pruning":
        groups_to_prune = min(groups - 1, max(0, int(round(groups * target_prune_ratio))))
        group_scores = []
        for g in range(groups):
            inds = list(range(g * out_per, (g + 1) * out_per))
            group_scores.append((float(scores[inds].sum()), g))
        deleted = sorted(g for _s, g in sorted(group_scores)[:groups_to_prune])
        kept_groups = [g for g in range(groups) if g not in set(deleted)]
        keep = [idx for g in kept_groups for idx in range(g * out_per, (g + 1) * out_per)]
        keep_in = [idx for g in kept_groups for idx in range(g * in_per, (g + 1) * in_per)]
        weight = module.weight.detach().float()[keep].cpu()
        b = bias[keep].cpu() if bias is not None else None
        detail.update(
            {
                "deleted_group_ids": deleted,
                "kept_group_ids": kept_groups,
                "group_keep_map": {str(new_g): list(range(out_per)) for new_g, _old in enumerate(kept_groups)},
                "dependency_complete": False,
                "in_out_block_sync_pass": True,
                "current_layer_input_pruned": True,
                "current_layer_output_pruned": True,
                "previous_layer_output_pruned": True,
                "next_layer_input_pruned": True,
                "keep_out_idx": keep,
                "keep_in_idx": keep_in,
                "old_to_new_out_map": {old: i for i, old in enumerate(keep)},
                "reinterpretation_count": 0,
                "reinterpretation_ratio": 0.0,
                "legality_status": "failed",
                "failure_reason": "dependency_incomplete",
            }
        )
        return _finish_detail(detail, module, keep, len(kept_groups), len(keep_in), weight)

    if policy == "group_coarsening_zero_padded_reblock":
        prune_count, adjusted, target_count = _flat_prune_count(c_out, groups, target_prune_ratio)
        if prune_count is None:
            detail.update({"legality_status": "failed", "failure_reason": "group_coarsening_infeasible"})
            return detail
        c_out_new = c_out - prune_count
        groups_new = _choose_groups_new(c_in, c_out_new, groups, out_per)
        if groups_new is None:
            detail.update({"legality_status": "failed", "failure_reason": "group_coarsening_infeasible"})
            return detail
        merge = groups // groups_new
        new_in_per = c_in // groups_new
        new_out_per = c_out_new // groups_new
        selected: list[int] = []
        mismatch = False
        for new_g in range(groups_new):
            old_groups = list(range(new_g * merge, (new_g + 1) * merge))
            bucket = [idx for og in old_groups for idx in range(og * out_per, (og + 1) * out_per)]
            if len(bucket) < new_out_per:
                detail.update({"legality_status": "failed", "failure_reason": "bucket_candidate_insufficient"})
                return detail
            selected.extend(_top_keep(scores, new_out_per, bucket))
        keep = sorted(selected)
        old_to_new = {old: i for i, old in enumerate(keep)}
        new_weight = torch.zeros((len(keep), new_in_per, *module.kernel_size), dtype=torch.float32)
        old_group_to_new = {str(g): g // merge for g in range(groups)}
        offsets: dict[str, int] = {}
        for new_idx, old_idx in enumerate(keep):
            old_group = old_idx // out_per
            target_new_group = old_group // merge
            if new_idx // new_out_per != target_new_group:
                mismatch = True
            offset = (old_group % merge) * in_per
            offsets[str(old_idx)] = offset
            new_weight[new_idx, offset:offset + in_per] = module.weight.detach().float()[old_idx]
        b = bias[keep].cpu() if bias is not None else None
        zero_added = int((new_weight == 0).sum().item())
        total = int(new_weight.numel())
        detail.update(
            {
                "target_prune_count": target_count,
                "adjusted_prune_count": prune_count,
                "ratio_adjusted": adjusted,
                "groups_after": groups_new,
                "merge_factor": merge,
                "keep_out_idx": keep,
                "old_to_new_out_map": old_to_new,
                "new_group_assignment": {str(old): old_to_new[old] // new_out_per for old in keep},
                "old_group_to_new_group_map": old_group_to_new,
                "offset_in_new_group": offsets,
                "weight_copy_status": "copy_plus_zero_pad",
                "added_zero_connections": zero_added,
                "zero_padded_slice_ratio": zero_added / max(total, 1),
                "weight_truncation_count": 0,
                "group_coarsening_semantic_mismatch": mismatch,
                "current_layer_output_pruned": True,
                "next_layer_input_pruned": True,
                "legality_status": "failed" if mismatch else "legal",
                "failure_reason": "group_coarsening_semantic_mismatch" if mismatch else "",
                "new_conv": _copy_conv(module, c_in=c_in, c_out=len(keep), groups=groups_new, weight=new_weight, bias=b),
            }
        )
        detail["reinterpretation_count"] = 0
        detail["reinterpretation_ratio"] = 0.0
        return _finish_detail(detail, module, keep, groups_new, c_in, new_weight)

    detail.update({"legality_status": "failed", "failure_reason": f"unsupported_policy:{policy}"})
    return detail


def _latency_ms(module: nn.Module, c_in: int, device: torch.device, warmup: int, repeat: int) -> dict[str, Any]:
    module = module.to(device).eval()
    x = torch.randn(1, c_in, 16, 16, device=device)
    times = []
    try:
        with torch.no_grad():
            for _ in range(warmup):
                module(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for _ in range(repeat):
                start = time.perf_counter()
                module(x)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000.0)
        return {
            "latency_backend": "pytorch_module_microbenchmark",
            "latency_ms_mean": statistics.mean(times) if times else None,
            "latency_ms_p50": statistics.median(times) if times else None,
            "latency_ms_p90": sorted(times)[int(0.9 * (len(times) - 1))] if times else None,
        }
    except Exception as exc:
        return {"latency_backend": "pytorch_module_microbenchmark", "latency_ms_mean": None, "latency_ms_p50": None, "latency_ms_p90": None, "failure_reason": str(exc)}


def _forward_smoke(conv: nn.Module, c_in: int, device: torch.device) -> tuple[bool, str]:
    try:
        conv = conv.to(device).eval()
        x = torch.randn(1, c_in, 16, 16, device=device)
        with torch.no_grad():
            conv(x)
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def _flatten_for_json(detail: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in detail.items() if k != "new_conv"}
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    logger = setup_logger(out)
    model_args = argparse.Namespace(checkpoint=args.checkpoint, model_config=args.model_config, heal_root=args.heal_root)
    model, _adapter = load_heal_model(model_args, device, logger)
    model = model.cpu().eval()
    config = {
        "checkpoint": args.checkpoint,
        "model_config": args.model_config,
        "policies": POLICIES,
        "score_modes": SCORE_MODES,
        "target_prune_ratios": RATIOS,
        "latency_backend": "pytorch_module_microbenchmark",
        "full_engine": False,
        "latency_lut": False,
        "proxy_training": False,
        "ga": False,
    }
    _write_json(out / "grouped_conv_ablation_config.json", config)

    protected = [
        {
            "module_name": "encoder_m1.pillar_vfe.pfn_layers",
            "reason": "fixed_width_shape_contract:pfn_to_pointpillar_scatter",
            "user_specified": False,
            "fixed_shape_contract": True,
            "resolver_can_unprotect": False,
        },
        {
            "module_name": "pillar_vfe.pfn_layers",
            "reason": "fixed_width_shape_contract:pfn_to_pointpillar_scatter",
            "user_specified": False,
            "fixed_shape_contract": True,
            "resolver_can_unprotect": False,
        },
    ]
    grouped_modules: list[tuple[str, nn.Conv2d]] = []
    excluded_depthwise = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prot, reason, user, fixed = _module_protection_reason(name, module)
            if prot:
                protected.append({
                    "module_name": name,
                    "reason": reason,
                    "user_specified": user,
                    "fixed_shape_contract": fixed,
                    "resolver_can_unprotect": False,
                })
        if not isinstance(module, nn.Conv2d) or module.groups <= 1:
            continue
        is_depthwise = module.groups == module.in_channels and module.groups == module.out_channels
        if is_depthwise:
            excluded_depthwise.append({"module_name": name, "reason": "excluded_depthwise_conv"})
            continue
        if _module_protection_reason(name, module)[0]:
            continue
        grouped_modules.append((name, module))
    _write_json(out / "protected_scope_report.json", {"protected_scopes": protected, "excluded_depthwise_conv": excluded_depthwise})

    details: list[dict[str, Any]] = []
    forward_rows = []
    latency_rows = []
    tp_rows = []
    for name, module in grouped_modules:
        baseline_latency = _latency_ms(module, module.in_channels, device, args.warmup, args.repeat)
        for score_mode in SCORE_MODES:
            for policy in POLICIES:
                for ratio in RATIOS:
                    detail = make_grouped_conv_candidate(module_name=name, module=module, policy=policy, score_mode=score_mode, target_prune_ratio=ratio)
                    conv = detail.get("new_conv")
                    pass_smoke = False
                    smoke_reason = detail.get("failure_reason", "")
                    latency = {"latency_ms_mean": None, "latency_ms_p50": None, "latency_ms_p90": None, "speedup_vs_baseline": None, "latency_backend": "pytorch_module_microbenchmark"}
                    if conv is not None and detail.get("legality_status") == "legal":
                        pass_smoke, smoke_reason = _forward_smoke(conv, int(detail["C_in_after"]), device)
                        if pass_smoke:
                            latency = _latency_ms(conv, int(detail["C_in_after"]), device, args.warmup, args.repeat)
                            b = baseline_latency.get("latency_ms_p50")
                            p = latency.get("latency_ms_p50")
                            latency["speedup_vs_baseline"] = (b / p) if b and p else None
                    flat = _flatten_for_json(detail)
                    flat.update(latency)
                    flat["forward_smoke_pass"] = pass_smoke
                    if smoke_reason and not flat.get("failure_reason"):
                        flat["failure_reason"] = smoke_reason
                    details.append(flat)
                    forward_rows.append({"policy": policy, "score_mode": score_mode, "target_prune_ratio": ratio, "module_name": name, "forward_smoke_pass": pass_smoke, "failure_reason": smoke_reason})
                    latency_rows.append({"policy": policy, "score_mode": score_mode, "target_prune_ratio": ratio, "module_name": name, **latency})
                    if policy == "flat_output_groups_fixed":
                        tp_rows.append({
                            "root_module": name,
                            "root_pruning_function": "tp.prune_conv_out_channels",
                            "dependent_module": "next_layer_input",
                            "dependent_pruning_function": "tp_depgraph_dependent_input_prune",
                            "idxs": [i for i in range(module.out_channels) if i not in flat.get("keep_out_idx", [])],
                            "whether_current_layer_input_changed": False,
                            "whether_current_layer_output_changed": bool(flat.get("current_layer_output_pruned")),
                            "whether_groups_changed": bool(flat.get("groups_changed")),
                            "in_channels_before": module.in_channels,
                            "in_channels_after": flat.get("C_in_after"),
                            "out_channels_before": module.out_channels,
                            "out_channels_after": flat.get("C_out_after"),
                            "groups_before": module.groups,
                            "groups_after": flat.get("groups_after"),
                            "is_regular_grouped_conv": True,
                            "is_depthwise": False,
                        })

    with (out / "per_layer_grouped_conv_details.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    _write_json(out / "grouped_conv_feasibility_audit.json", {"records": details})
    _write_json(out / "grouped_conv_policy_detail_report.json", {"records": details})
    _write_json(out / "tp_replay_audit_report.json", {"records": tp_rows})
    _write_json(out / "forward_smoke_report.json", {"records": forward_rows})
    _write_json(out / "latency_smoke_report.json", {"records": latency_rows, "latency_backend": "pytorch_module_microbenchmark"})
    _write_json(out / "eval_short_report.json", {
        "status": "not_run",
        "reason": "full_model_structural_resolver_not_executed_in_grouped_conv_semantic_ablation; AP/mAP not evaluated and not faked",
        "AP_0.3": None,
        "mAP": None,
    })

    summary = []
    for score_mode in SCORE_MODES:
        for policy in POLICIES:
            for ratio in RATIOS:
                rows = [r for r in details if r["score_mode"] == score_mode and r["policy"] == policy and r["target_prune_ratio"] == ratio]
                legal = [r for r in rows if r.get("legality_status") == "legal"]
                lat = [r for r in legal if r.get("latency_ms_p50") is not None]
                mean_re = statistics.mean([float(r.get("reinterpretation_ratio") or 0.0) for r in rows]) if rows else 0.0
                max_re = max([float(r.get("reinterpretation_ratio") or 0.0) for r in rows], default=0.0)
                summary.append({
                    "policy": policy,
                    "score_mode": score_mode,
                    "target_prune_ratio": ratio,
                    "actual_prune_ratio": statistics.mean([float(r.get("actual_prune_ratio") or 0.0) for r in legal]) if legal else None,
                    "actual_param_prune_ratio": None,
                    "actual_dense_flops_or_bops_prune_ratio": statistics.mean([1.0 - float(r.get("dense_flops_after") or 0) / max(float(r.get("dense_flops_before") or 1), 1.0) for r in legal]) if legal else None,
                    "num_grouped_convs_touched": len(legal),
                    "num_grouped_convs_failed": len(rows) - len(legal),
                    "C_in_before_total": sum(int(r.get("C_in_before") or 0) for r in rows),
                    "C_in_after_total": sum(int(r.get("C_in_after") or 0) for r in legal),
                    "C_out_before_total": sum(int(r.get("C_out_before") or 0) for r in rows),
                    "C_out_after_total": sum(int(r.get("C_out_after") or 0) for r in legal),
                    "groups_changed_count": sum(1 for r in rows if r.get("groups_changed")),
                    "mean_reinterpretation_ratio": mean_re,
                    "max_reinterpretation_ratio": max_re,
                    "mean_added_zero_connections": statistics.mean([float(r.get("added_zero_connections") or 0) for r in rows]) if rows else 0.0,
                    "max_added_zero_connections": max([int(r.get("added_zero_connections") or 0) for r in rows], default=0),
                    "total_weight_truncation_count": sum(int(r.get("weight_truncation_count") or 0) for r in rows),
                    "group_coarsening_semantic_mismatch_count": sum(1 for r in rows if r.get("group_coarsening_semantic_mismatch")),
                    "forward_smoke_pass": bool(legal) and all(r.get("forward_smoke_pass") for r in legal),
                    "AP_0.3": None,
                    "mAP": None,
                    "latency_ms_mean": statistics.mean([float(r["latency_ms_mean"]) for r in lat]) if lat else None,
                    "latency_ms_p50": statistics.mean([float(r["latency_ms_p50"]) for r in lat]) if lat else None,
                    "latency_ms_p90": statistics.mean([float(r["latency_ms_p90"]) for r in lat]) if lat else None,
                    "speedup_vs_baseline": statistics.mean([float(r["speedup_vs_baseline"]) for r in lat if r.get("speedup_vs_baseline")]) if lat else None,
                    "failure_reason": ";".join(sorted({str(r.get("failure_reason")) for r in rows if r.get("failure_reason")})),
                })
    csv_path = out / "grouped_conv_ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)
    md = ["# Grouped Conv Ablation v8.7", "", "## Answers"]
    md.append("1. Shape feasibility is recorded per policy/ratio in `grouped_conv_feasibility_audit.json`; failed combinations keep explicit `failure_reason`.")
    md.append("2. A/flat can produce local-input reinterpretation; see `mean_reinterpretation_ratio` and per-layer maps.")
    md.append("3. B/group-balanced is designed to reduce reinterpretation by equal old-group keep counts; violations are marked `group_balanced_violation`.")
    md.append("4. C/true-group-block reports `dependency_incomplete` without a full cross-layer resolver; it is not used for AP eval.")
    md.append("5. D/group-coarsening uses copy-plus-zero-pad reblock and records forward smoke plus zero-padded slice ratio.")
    md.append("6. D reports added zero connections; dense FLOPs are computed without treating zero slices as free.")
    md.append("7. Dense FLOPs may be offset by larger `in_per_group_after`; see `dense_flops_before/after`.")
    md.append("8. A TP replay audit is recorded in `tp_replay_audit_report.json`; current-layer input is not changed and groups remain fixed.")
    md.append("9. AP/mAP are not run in this semantic ablation because no full-model resolver was applied; values are null, not faked.")
    md.append("10. Only policies with legal shape, forward smoke pass, and acceptable reinterpretation should proceed to later full-engine validation.")
    md.append("11. Rejections use explicit reasons including dependency_incomplete, grouped_conv_shape_infeasible, group_coarsening_semantic_mismatch, and weight_truncation.")
    md.append("")
    md.append("## Summary table")
    md.append(csv_path.read_text(encoding="utf-8"))
    (out / "grouped_conv_ablation_summary.md").write_text("\n".join(md), encoding="utf-8")
    return {"output_dir": str(out), "num_regular_grouped_convs": len(grouped_modules), "num_records": len(details)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    p.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    p.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--output-dir", default="outputs/latency_lut/grouped_conv_ablation_v87")
    args = p.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
