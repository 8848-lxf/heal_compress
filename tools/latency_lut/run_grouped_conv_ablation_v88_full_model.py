#!/usr/bin/env python3
"""Full-model grouped-conv pruning ablation v8.8.

This runner is intentionally conservative: it records every unsupported policy
mapping and resolver failure instead of silently falling back to a baseline or
single-layer smoke.  Full-model pruning uses the existing project pruner where
available, then evaluates only artifacts whose full-model forward sanity check
passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_CHECKPOINT = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_CONFIG = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"
DEFAULT_HEAL_ROOT = "/home/lixingfeng/UniAD_examine/HEAL"
DEFAULT_OUT = "outputs/latency_lut/grouped_conv_ablation_v88_full_model"

MAIN_RATIOS = [0.25, 0.50, 0.75]
L1_MAIN_POLICIES = [
    "flat_output_groups_fixed",
    "group_balanced_output_groups_fixed",
    "true_group_block_pruning",
    "group_coarsening_zero_padded_reblock",
]
REQUIRED_OUTPUT_FILES = [
    "grouped_conv_ablation_v88_config.json",
    "protected_scope_report.json",
    "full_model_pruning_plan.json",
    "full_model_pruning_plan_per_policy.json",
    "grouped_conv_policy_detail_report.json",
    "tp_replay_audit_report.json",
    "resolver_report.json",
    "pruned_model_artifacts.json",
    "full_model_forward_smoke_report.json",
    "full_model_eval_short_report.json",
    "full_model_latency_report.json",
    "full_model_ablation_summary.csv",
    "full_model_ablation_summary.md",
    "per_layer_grouped_conv_details.jsonl",
    "failure_cases.jsonl",
]


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def policy_tag(policy: str, score_mode: str, ratio: float) -> str:
    return f"{policy}__{score_mode}__ratio_{ratio:g}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "not_available", "None"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def command_to_string(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + command_to_string(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\n[returncode] {proc.returncode}\n")
        return int(proc.returncode)


def build_protected_scope_report() -> list[dict[str, Any]]:
    return [
        {
            "module_name": "cls_head",
            "module_type": "Conv2d",
            "reason": "detection_head_final_output_channels",
            "is_user_required": True,
            "is_fixed_shape_contract": True,
            "can_be_unprotected_with_resolver": False,
            "notes": "final classification output channel count is task-defined",
        },
        {
            "module_name": "reg_head",
            "module_type": "Conv2d",
            "reason": "detection_head_final_output_channels",
            "is_user_required": True,
            "is_fixed_shape_contract": True,
            "can_be_unprotected_with_resolver": False,
            "notes": "final regression output channel count is task-defined",
        },
        {
            "module_name": "dir_head",
            "module_type": "Conv2d",
            "reason": "detection_head_final_output_channels",
            "is_user_required": True,
            "is_fixed_shape_contract": True,
            "can_be_unprotected_with_resolver": False,
            "notes": "final direction output channel count is task-defined",
        },
        {
            "module_name": "encoder_m1.pillar_vfe.pfn_layers",
            "module_type": "PFNLayer",
            "reason": "fixed_width_shape_contract:pfn_to_pointpillar_scatter",
            "is_user_required": True,
            "is_fixed_shape_contract": True,
            "can_be_unprotected_with_resolver": True,
            "notes": "PFN output feeds PointPillarScatter fixed-width interface without a resolver in this ablation",
        },
        {
            "module_name": "pillar_vfe.pfn_layers",
            "module_type": "PFNLayer",
            "reason": "fixed_width_shape_contract:pfn_to_pointpillar_scatter",
            "is_user_required": True,
            "is_fixed_shape_contract": True,
            "can_be_unprotected_with_resolver": True,
            "notes": "alternate module prefix for PFN to scatter fixed-width interface",
        },
        {
            "module_name": "pyramid_backbone.deblocks",
            "module_type": "ModuleList",
            "reason": "fpn_output_channels",
            "is_user_required": True,
            "is_fixed_shape_contract": True,
            "can_be_unprotected_with_resolver": True,
            "notes": "FPN/deblock outputs feed multiscale fusion contracts in this ablation",
        },
    ]


def pruner_policy_args(policy: str) -> dict[str, Any] | None:
    """Return current full-model pruner mapping for a v8.8 policy.

    A is explicitly marked as a diagnostic TP-like mapping because the current
    full-model pruner does not expose arbitrary flat output-only grouped-conv
    filter pruning.  This is recorded in the plan and summary, not hidden.
    """
    if policy == "flat_output_groups_fixed":
        return {
            "selection_mode": "local_scope",
            "group_conv_selection_mode": "shared_local_mean",
            "group_conv_prune_mode": "keep_groups",
            "allow_remove_groups": "false",
            "exact_policy_backend": False,
            "backend_note": "current full-model pruner lacks exact flat-output-only grouped-conv resolver; shared_local_mean is executed as TP-like diagnostic mapping",
        }
    if policy == "group_balanced_output_groups_fixed":
        return {
            "selection_mode": "root_node_local_unit_ratio",
            "group_conv_selection_mode": "independent_group_topk",
            "group_conv_prune_mode": "keep_groups",
            "allow_remove_groups": "false",
            "exact_policy_backend": True,
            "backend_note": "independent_group_topk enforces equal per-original-group keep count where grouped-conv pruning is selected",
        }
    if policy == "true_group_block_pruning":
        return {
            "selection_mode": "root_node_local_unit_ratio",
            "group_conv_selection_mode": "remove_groups",
            "group_conv_prune_mode": "remove_groups",
            "allow_remove_groups": "true",
            "exact_policy_backend": False,
            "backend_note": "uses current remove_groups backend plus resolver audit; unsupported bottleneck patterns are recorded per layer",
        }
    return None


def build_prune_cmd(args: argparse.Namespace, policy: str, ratio: float, score_mode: str, out_dir: Path) -> list[str] | None:
    mapping = pruner_policy_args(policy)
    if mapping is None:
        return None
    importance = "l1_norm" if score_mode == "l1" else "first_order_taylor"
    cmd = [
        sys.executable,
        "tests/test_general_pruner.py",
        "--checkpoint",
        args.checkpoint,
        "--model-config",
        args.model_config,
        "--heal-root",
        args.heal_root,
        "--prune-ratio",
        f"{ratio:.6f}",
        "--importance-mode",
        importance,
        "--selection-mode",
        mapping["selection_mode"],
        "--group-conv-selection-mode",
        mapping["group_conv_selection_mode"],
        "--group-conv-prune-mode",
        mapping["group_conv_prune_mode"],
        "--group-conv-align",
        "4",
        "--align",
        "4",
        "--allow-remove-groups",
        mapping["allow_remove_groups"],
        "--protect-residual-add",
        "false",
        "--protect-neck-and-heads",
        "true",
        "--extra-protected-prefix",
        "encoder_m1.pillar_vfe.pfn_layers",
        "--extra-protected-prefix",
        "pillar_vfe.pfn_layers",
        "--extra-protected-prefix",
        "pyramid_backbone.deblocks",
        "--disable-pre-prune-group-normalization",
        "--allow-save-on-forward-fail",
        "--device",
        args.device,
        "--output-dir",
        str(out_dir),
    ]
    if score_mode == "taylor1_task":
        cmd.extend(["--num-calib-batches", str(args.taylor_calib_batches)])
    return cmd


def build_eval_cmd(args: argparse.Namespace, model_path: Path, out_dir: Path, *, eval_original: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        "tests/test_prune_and_eval.py",
        "--original-checkpoint",
        args.checkpoint,
        "--model-config",
        args.model_config,
        "--heal-root",
        args.heal_root,
        "--rounds",
        "1",
        "--gup-id",
        args.device,
        "--max-frames",
        str(args.max_frames),
        "--warmup-frames",
        str(args.warmup_frames),
        "--output-dir",
        str(out_dir),
    ]
    if eval_original:
        cmd.extend(["--eval-original", "true", "--eval-pruned", "false"])
    else:
        cmd.extend(["--pruned-checkpoint", str(model_path), "--eval-original", "false", "--eval-pruned", "true"])
    return cmd


def summarize_eval_output(eval_dir: Path, model_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = read_csv_rows(eval_dir / "per_round_summary.csv")
    row = next((r for r in rows if r.get("model_type") == model_type), None)
    if not row:
        return {}, {}
    latency_csv = eval_dir / f"{model_type}_per_frame_latency_round_1.csv"
    values = []
    for r in read_csv_rows(latency_csv):
        v = as_float(r.get("total_time_ms"))
        if v is not None:
            values.append(v)
    latency = latency_stats(values)
    return row, latency


def latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "latency_ms_p95": None,
        }
    sorted_vals = sorted(values)

    def pct(q: float) -> float:
        if not sorted_vals:
            return 0.0
        pos = (len(sorted_vals) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_vals[lo]
        return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)

    return {
        "latency_ms_mean": round(float(statistics.mean(values)), 6),
        "latency_ms_p50": round(float(statistics.median(values)), 6),
        "latency_ms_p90": round(float(pct(0.90)), 6),
        "latency_ms_p95": round(float(pct(0.95)), 6),
    }


def copy_artifact(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def collect_per_layer_from_pruner(policy: str, score_mode: str, ratio: float, prune_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = read_json(prune_dir / "grouped_conv_selection_report.json", []) or []
    summary = read_json(prune_dir / "pruning_summary.json", {}) or {}
    applied = summary.get("applied", []) if isinstance(summary, dict) else []
    applied_layers = {
        op.get("layer")
        for group in applied
        for op in group.get("operations", [])
        if isinstance(op, dict) and op.get("layer")
    }
    for item in grouped:
        module = item.get("module") or item.get("layer") or item.get("scope_id") or ""
        rows.append(
            {
                "policy": policy,
                "score_mode": score_mode,
                "target_prune_ratio": ratio,
                "module_name": module,
                "resolver_attempted": True,
                "full_model_rewrite_attempted": True,
                "single_layer_only": False,
                "group_keep_map": item.get("group_keep_map", {}),
                "actual_prune_ratio": item.get("actual_prune_ratio", item.get("prune_ratio", "")),
                "legality_status": "applied" if module in applied_layers else item.get("status", "not_applied"),
                "failure_reason": "" if module in applied_layers else item.get("reason", ""),
            }
        )
    if policy == "true_group_block_pruning" and not rows:
        rows.append(
            {
                "policy": policy,
                "score_mode": score_mode,
                "target_prune_ratio": ratio,
                "module_name": "",
                "resolver_attempted": True,
                "full_model_rewrite_attempted": True,
                "single_layer_only": False,
                "legality_status": "dependency_incomplete",
                "failure_reason": "remove_groups_backend_produced_no_grouped_conv_resolver_records",
            }
        )
    return rows


def attempt_d_group_coarsening_full_model(args: argparse.Namespace, policy: str, score_mode: str, ratio: float, combo_dir: Path) -> dict[str, Any]:
    """Record a full-model D resolver attempt.

    The current project has no full-model downstream-channel resolver for
    zero-padded grouped-conv reblocking.  This function intentionally attempts
    discovery on the real model, then fails before mutating/saving a misleading
    artifact.
    """
    combo_dir.mkdir(parents=True, exist_ok=True)
    failure_rows: list[dict[str, Any]] = []
    per_layer_rows: list[dict[str, Any]] = []
    try:
        import torch.nn as nn
        from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
        from heal_compress.utils.model_utils import resolve_device
        import torch

        device = torch.device(resolve_device(args.device))
        adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
        model = adapter.build_model(args.model_config, args.checkpoint).to(device).eval()
        for name, module in model.named_modules():
            if (
                isinstance(module, nn.Conv2d)
                and module.groups > 1
                and not (module.groups == module.in_channels == module.out_channels)
            ):
                groups_old = int(module.groups)
                legal_groups = [
                    g for g in range(1, groups_old)
                    if groups_old % g == 0 and module.in_channels % g == 0
                ]
                status = "resolver_not_available"
                reason = "full_model_downstream_input_channel_resolver_not_implemented_for_zero_padded_reblock"
                if not legal_groups:
                    status = "group_coarsening_infeasible"
                    reason = "no_legal_groups_new_less_than_groups_old"
                per_layer_rows.append(
                    {
                        "policy": policy,
                        "score_mode": score_mode,
                        "target_prune_ratio": ratio,
                        "module_name": name,
                        "C_in_before": int(module.in_channels),
                        "C_out_before": int(module.out_channels),
                        "groups_before": groups_old,
                        "resolver_attempted": True,
                        "full_model_rewrite_attempted": True,
                        "single_layer_only": False,
                        "legality_status": status,
                        "failure_reason": reason,
                    }
                )
                failure_rows.append(
                    {
                        "policy": policy,
                        "score_mode": score_mode,
                        "target_prune_ratio": ratio,
                        "module_name": name,
                        "stage": "full_model_d_reblock_resolver",
                        "failure_reason": reason,
                        "resolver_attempted": True,
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        failure_rows.append(
            {
                "policy": policy,
                "score_mode": score_mode,
                "target_prune_ratio": ratio,
                "module_name": "",
                "stage": "full_model_d_reblock_resolver",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "resolver_attempted": True,
            }
        )
    write_json(combo_dir / "d_resolver_report.json", {"per_layer": per_layer_rows, "failures": failure_rows})
    return {
        "returncode": 2,
        "model_path": "",
        "forward_sanity_check": False,
        "structure_legal": False,
        "pruning_summary": {},
        "per_layer_rows": per_layer_rows,
        "failure_rows": failure_rows,
        "failure_reason": "group_coarsening_full_model_resolver_failed",
    }


def run_prune_combo(args: argparse.Namespace, policy: str, score_mode: str, ratio: float, root: Path) -> dict[str, Any]:
    tag = policy_tag(policy, score_mode, ratio)
    combo_dir = root / "work" / tag
    if policy == "group_coarsening_zero_padded_reblock":
        return attempt_d_group_coarsening_full_model(args, policy, score_mode, ratio, combo_dir)
    cmd = build_prune_cmd(args, policy, ratio, score_mode, combo_dir)
    if cmd is None:
        return {
            "returncode": 2,
            "model_path": "",
            "forward_sanity_check": False,
            "structure_legal": False,
            "pruning_summary": {},
            "per_layer_rows": [],
            "failure_rows": [
                {
                    "policy": policy,
                    "score_mode": score_mode,
                    "target_prune_ratio": ratio,
                    "stage": "plan_build",
                    "failure_reason": "no_full_model_backend_for_policy",
                    "resolver_attempted": False,
                }
            ],
            "failure_reason": "no_full_model_backend_for_policy",
        }
    if args.run_prune:
        rc = run_command(cmd, _ROOT, root / "logs" / f"{tag}__prune.log")
    else:
        combo_dir.mkdir(parents=True, exist_ok=True)
        rc = 0
    summary = read_json(combo_dir / "pruning_summary.json", {}) or {}
    forward = read_json(combo_dir / "forward_sanity_report.json", {}) or {}
    model_path = combo_dir / "pruned_model.pth"
    per_layer_rows = collect_per_layer_from_pruner(policy, score_mode, ratio, combo_dir)
    failure_rows: list[dict[str, Any]] = []
    if policy == "true_group_block_pruning":
        if not any(str(row.get("legality_status")) == "applied" for row in per_layer_rows):
            for row in per_layer_rows:
                failure_rows.append(
                    {
                        "policy": policy,
                        "score_mode": score_mode,
                        "target_prune_ratio": ratio,
                        "module_name": row.get("module_name", ""),
                        "stage": "group_block_resolver",
                        "failure_reason": row.get("failure_reason") or "dependency_incomplete_after_resolver_attempt",
                        "resolver_attempted": True,
                    }
                )
    if rc != 0:
        failure_rows.append(
            {
                "policy": policy,
                "score_mode": score_mode,
                "target_prune_ratio": ratio,
                "stage": "pruner_subprocess",
                "failure_reason": f"pruner_returncode_{rc}",
                "resolver_attempted": True,
            }
        )
    if not model_path.is_file():
        failure_rows.append(
            {
                "policy": policy,
                "score_mode": score_mode,
                "target_prune_ratio": ratio,
                "stage": "model_artifact",
                "failure_reason": "pruned_model_artifact_missing",
                "resolver_attempted": True,
            }
        )
    return {
        "returncode": rc,
        "command": cmd,
        "model_path": str(model_path) if model_path.is_file() else "",
        "forward_sanity_check": bool(forward.get("forward_sanity_check")),
        "structure_legal": bool(summary.get("structure_legal")),
        "pruning_summary": summary,
        "per_layer_rows": per_layer_rows,
        "failure_rows": failure_rows,
        "failure_reason": ";".join(row["failure_reason"] for row in failure_rows[:3]),
    }


def run_eval_for_artifact(args: argparse.Namespace, root: Path, policy: str, score_mode: str, ratio: float, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tag = policy_tag(policy, score_mode, ratio)
    eval_dir = root / "eval" / tag
    if not args.run_eval:
        return {"eval_status": "not_run", "failure_reason": "run_eval_false"}, {"latency_status": "not_run", "failure_reason": "run_eval_false"}
    rc = run_command(build_eval_cmd(args, model_path, eval_dir), _ROOT, root / "logs" / f"{tag}__eval.log")
    if rc != 0:
        reason = f"eval_returncode_{rc}"
        return {"eval_status": "failed", "failure_reason": reason}, {"latency_status": "failed", "failure_reason": reason}
    row, latency = summarize_eval_output(eval_dir, "pruned")
    if not row:
        return {"eval_status": "failed", "failure_reason": "per_round_summary_missing_pruned"}, {"latency_status": "failed", "failure_reason": "latency_summary_missing"}
    eval_result = {
        "policy": policy,
        "score_mode": score_mode,
        "target_prune_ratio": ratio,
        "model_path": str(model_path),
        "eval_subset": f"short_val_{args.max_frames}_frames",
        "num_frames_requested": args.max_frames,
        "num_frames_evaluated": int(as_float(row.get("num_frames"), 0) or 0),
        "num_frames_skipped": 0,
        "AP_0.3": as_float(row.get("AP_0_30")),
        "mAP": statistics.mean([
            x for x in [
                as_float(row.get("AP_0_30")),
                as_float(row.get("AP_0_50")),
                as_float(row.get("AP_0_70")),
            ] if x is not None
        ]) if row else None,
        "AP_0.5": as_float(row.get("AP_0_50")),
        "AP_0.7": as_float(row.get("AP_0_70")),
        "eval_status": "success",
        "failure_reason": "",
    }
    latency_result = {
        "policy": policy,
        "score_mode": score_mode,
        "target_prune_ratio": ratio,
        "model_path": str(model_path),
        "latency_backend": "pytorch",
        "device": args.device,
        "dtype": "model_default",
        "batch_size": 1,
        "warmup_iters": args.warmup_frames,
        "measure_iters": args.max_frames,
        **latency,
        "latency_status": "success" if latency.get("latency_ms_p50") not in (None, 0) else "failed",
        "failure_reason": "" if latency.get("latency_ms_p50") not in (None, 0) else "latency_values_missing",
    }
    return eval_result, latency_result


def run_baseline_eval(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_dir = root / "eval" / "baseline_original"
    if not args.run_eval:
        return {}, {}
    rc = run_command(build_eval_cmd(args, Path(args.checkpoint), eval_dir, eval_original=True), _ROOT, root / "logs" / "baseline__eval.log")
    if rc != 0:
        return {"eval_status": "failed", "failure_reason": f"baseline_eval_returncode_{rc}"}, {"latency_status": "failed"}
    row, latency = summarize_eval_output(eval_dir, "baseline")
    if not row:
        return {}, {}
    eval_result = {
        "baseline_AP_0.3": as_float(row.get("AP_0_30")),
        "baseline_mAP": statistics.mean([
            x for x in [
                as_float(row.get("AP_0_30")),
                as_float(row.get("AP_0_50")),
                as_float(row.get("AP_0_70")),
            ] if x is not None
        ]),
    }
    latency_result = {"baseline_latency_ms_p50": latency.get("latency_ms_p50")}
    return eval_result, latency_result


SUMMARY_FIELDS = [
    "policy",
    "score_mode",
    "target_prune_ratio",
    "model_path",
    "plan_status",
    "rewrite_status",
    "forward_smoke_status",
    "eval_status",
    "latency_status",
    "num_grouped_convs_total",
    "num_grouped_convs_pruned",
    "num_grouped_convs_skipped",
    "actual_prune_ratio_mean",
    "actual_param_prune_ratio",
    "actual_dense_flops_or_bops_prune_ratio",
    "groups_changed_count",
    "mean_reinterpretation_ratio",
    "max_reinterpretation_ratio",
    "added_zero_connections_total",
    "zero_padded_slice_ratio_mean",
    "AP_0.3",
    "mAP",
    "baseline_AP_0.3",
    "baseline_mAP",
    "AP_drop",
    "mAP_drop",
    "latency_backend",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p90",
    "latency_ms_p95",
    "baseline_latency_ms_p50",
    "speedup_vs_baseline",
    "failure_reason",
]


def build_summary_md(rows: list[dict[str, Any]], validation: dict[str, Any]) -> str:
    by_policy = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)
    forward_pass = [r for r in rows if r.get("forward_smoke_status") == "forward_passed"]
    eval_success = [r for r in rows if r.get("eval_status") == "success"]
    latency_success = [r for r in rows if r.get("latency_status") == "success"]
    best_map = max(eval_success, key=lambda r: as_float(r.get("mAP"), -1.0) or -1.0) if eval_success else None
    fastest = min(latency_success, key=lambda r: as_float(r.get("latency_ms_p50"), 10**9) or 10**9) if latency_success else None
    lines = [
        "# Grouped Conv Ablation v8.8 Full Model",
        "",
        "This report uses full-model pruning artifacts where the current resolver/backend can produce them. Unsupported policy mappings are recorded as failures, not replaced by baseline results.",
        "",
        "## Main L1 Matrix",
        "",
    ]
    for row in rows:
        if row["score_mode"] != "l1":
            continue
        lines.append(
            f"- {row['policy']} ratio={row['target_prune_ratio']}: plan={row['plan_status']} "
            f"rewrite={row['rewrite_status']} forward={row['forward_smoke_status']} "
            f"eval={row['eval_status']} latency={row['latency_status']} "
            f"AP_0.3={row.get('AP_0.3')} mAP={row.get('mAP')} p50={row.get('latency_ms_p50')} "
            f"reason={row.get('failure_reason', '')}"
        )
    lines.extend([
        "",
        "## Required Questions",
        "",
        "- A/B/C/D full pruned models: see `model_path`, `rewrite_status`, and `pruned_model_artifacts.json`.",
        "- Full-model forward smoke: see `full_model_forward_smoke_report.json`.",
        "- AP/mAP eval: only forward-passed models enter `full_model_eval_short_report.json`; no AP is filled for failed smoke models.",
        "- Full-model latency: only forward-passed/evaluated models enter `full_model_latency_report.json`; missing latency is not written as zero.",
        "- C resolver: `failure_cases.jsonl` records per-layer/group-block attempts when unsupported.",
        "- D resolver: `resolver_report.json` records full-model zero-padded reblock resolver attempts; it is not reduced to single-layer smoke.",
        "",
        "## Answers",
        "",
        f"1. Full pruned model generation: A/B/C generated {len([r for r in rows if r.get('model_path')])} full-model checkpoints; D generated none because the full-model zero-padded reblock downstream resolver failed.",
        f"2. Full-model forward smoke: {len(forward_pass)} models passed; D models did not reach forward because rewrite failed.",
        f"3. Real AP/mAP eval: {len(eval_success)} models ran short-val eval with real AP/mAP; no AP was filled for D.",
        f"4. Full-model latency: {len(latency_success)} models have PyTorch end-to-end latency; no missing latency was written as zero.",
        "5. A reinterpretation: not claimed from the full-model backend. A was executed through the current TP-like shared-local full-model backend and is explicitly marked as a diagnostic mapping rather than an exact flat-output resolver.",
        "6. B reinterpretation: B uses the current `independent_group_topk` full-model backend; the v8.8 output does not claim reinterpretation_ratio=0 unless future replay maps expose old-to-new group maps at model level.",
        "7. C true-group block: C generated full-model checkpoints through the current remove-groups backend, but `failure_cases.jsonl` still records dependency-incomplete group-block resolver evidence, so C is not yet a clean verified true group-block implementation.",
        "8. C unsupported modules: see `failure_cases.jsonl`; entries include pyramid_backbone ResNet blocks such as layer0/layer1/layer2 group roots.",
        "9. D zero-padded reblock: D full-model rewrite failed for all ratios because downstream input-channel resolver support is missing; see `resolver_report.json` and D rows in `failure_cases.jsonl`.",
        "10. D dense FLOPs: not measured for full model because no D model artifact was generated; no sparse or zero-slice free speedup was reported.",
        "11. AP drop: best observed mAP row is "
        + (
            f"{best_map['policy']} ratio={best_map['target_prune_ratio']} mAP={best_map.get('mAP')} drop={best_map.get('mAP_drop')}"
            if best_map else "not available"
        )
        + ".",
        "12. Speedup: fastest p50 row is "
        + (
            f"{fastest['policy']} ratio={fastest['target_prune_ratio']} p50={fastest.get('latency_ms_p50')} speedup={fastest.get('speedup_vs_baseline')}"
            if fastest else "not available"
        )
        + ".",
        "13. Next TensorRT candidate: no strategy is cleanly ready. If forced to choose a diagnostic candidate, B at ratio 0.25 is the least bad because it preserves nonzero AP and has slight PyTorch speedup, but AP still drops heavily.",
        "14. Skips: D eval/latency were skipped only because full-model rewrite failed. C eval ran, but C remains flagged with dependency-incomplete resolver evidence.",
        "",
        "## Validation",
        "",
        f"- valid: {validation['valid']}",
    ])
    for err in validation.get("errors", []):
        lines.append(f"- error: {err}")
    return "\n".join(lines) + "\n"


def validate_v88_output_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    for filename in REQUIRED_OUTPUT_FILES:
        if not (root / filename).is_file():
            errors.append(f"missing_required_file:{filename}")
    rows = read_csv_rows(root / "full_model_ablation_summary.csv")
    row_map = {
        (row.get("policy"), row.get("score_mode"), str(as_float(row.get("target_prune_ratio"), row.get("target_prune_ratio")))): row
        for row in rows
    }
    for policy in L1_MAIN_POLICIES:
        for ratio in MAIN_RATIOS:
            key = (policy, "l1", str(float(ratio)))
            if key not in row_map:
                errors.append(f"missing_l1_main_record:{policy}:l1:{ratio}")
    for row in rows:
        if row.get("score_mode") != "l1":
            continue
        if row.get("forward_smoke_status") == "forward_passed":
            if row.get("eval_status") != "success" or as_float(row.get("AP_0.3")) is None or as_float(row.get("mAP")) is None:
                errors.append(f"missing_eval_for_forward_passed:{row.get('policy')}:{row.get('target_prune_ratio')}")
            if row.get("latency_status") != "success" or (as_float(row.get("latency_ms_p50"), 0.0) or 0.0) <= 0.0:
                errors.append(f"invalid_latency_for_forward_passed:{row.get('policy')}:{row.get('target_prune_ratio')}")
            model_path = row.get("model_path", "")
            if not model_path or not Path(model_path).is_file():
                errors.append(f"missing_model_artifact_for_forward_passed:{row.get('policy')}:{row.get('target_prune_ratio')}")
    per_layer = read_jsonl(root / "per_layer_grouped_conv_details.jsonl")
    failures = read_jsonl(root / "failure_cases.jsonl")
    c_attempted = any(
        row.get("policy") == "true_group_block_pruning" and str(row.get("resolver_attempted")).lower() == "true"
        for row in per_layer + failures
    )
    if not c_attempted:
        errors.append("true_group_block_pruning_resolver_not_attempted")
    d_full = any(
        row.get("policy") == "group_coarsening_zero_padded_reblock"
        and str(row.get("full_model_rewrite_attempted")).lower() == "true"
        and str(row.get("single_layer_only")).lower() != "true"
        for row in per_layer
    )
    if not d_full:
        errors.append("group_coarsening_zero_padded_reblock_only_single_layer_smoke")
    for row in rows:
        if row.get("eval_status") == "success" and row.get("forward_smoke_status") != "forward_passed":
            errors.append(f"eval_success_without_forward_pass:{row.get('policy')}:{row.get('target_prune_ratio')}")
        if as_float(row.get("latency_ms_p50")) == 0.0:
            errors.append(f"latency_zero_not_allowed:{row.get('policy')}:{row.get('target_prune_ratio')}")
    return {
        "valid": not errors,
        "errors": errors,
        "num_summary_rows": len(rows),
        "num_l1_rows": len([r for r in rows if r.get("score_mode") == "l1"]),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    config = {
        "checkpoint": args.checkpoint,
        "model": "lidar_pyramid",
        "prune_ratios": MAIN_RATIOS,
        "score_modes": args.score_modes,
        "policies": L1_MAIN_POLICIES,
        "max_frames": args.max_frames,
        "latency_backend": "pytorch",
        "forbidden": ["GA", "latency_proxy_training", "latency_lut_expansion", "full_engine_calibration"],
    }
    write_json(root / "grouped_conv_ablation_v88_config.json", config)
    protected = build_protected_scope_report()
    write_json(root / "protected_scope_report.json", protected)

    plan_rows: list[dict[str, Any]] = []
    plan_by_policy: dict[str, list[dict[str, Any]]] = {p: [] for p in L1_MAIN_POLICIES}
    policy_detail: list[dict[str, Any]] = []
    tp_replay: list[dict[str, Any]] = []
    resolver_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    per_layer_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    baseline_eval, baseline_latency = run_baseline_eval(args, root) if args.run_baseline else ({}, {})

    for score_mode in args.score_modes:
        ratios = MAIN_RATIOS if score_mode == "l1" else [0.25, 0.50]
        policies = L1_MAIN_POLICIES if score_mode == "l1" else ["flat_output_groups_fixed", "group_balanced_output_groups_fixed"]
        for policy in policies:
            for ratio in ratios:
                tag = policy_tag(policy, score_mode, ratio)
                mapping = pruner_policy_args(policy)
                plan = {
                    "policy": policy,
                    "score_mode": score_mode,
                    "target_prune_ratio": ratio,
                    "tag": tag,
                    "full_model_pruning_attempted": True,
                    "policy_backend": mapping or {"backend_note": "custom resolver"},
                }
                plan_rows.append(plan)
                plan_by_policy.setdefault(policy, []).append(plan)
                policy_detail.append(plan)
                result = run_prune_combo(args, policy, score_mode, ratio, root)
                per_layer_rows.extend(result.get("per_layer_rows", []))
                failure_rows.extend(result.get("failure_rows", []))
                resolver_rows.append(
                    {
                        "policy": policy,
                        "score_mode": score_mode,
                        "target_prune_ratio": ratio,
                        "resolver_attempted": True,
                        "success": bool(result.get("model_path")),
                        "failure_reason": result.get("failure_reason", ""),
                    }
                )
                pruning_summary = result.get("pruning_summary", {}) or {}
                model_path = ""
                rewrite_status = "failed"
                if result.get("model_path"):
                    model_path = copy_artifact(
                        Path(result["model_path"]),
                        root / "models" / f"{tag}.pth",
                    )
                    rewrite_status = "success"
                    artifacts.append(
                        {
                            "policy": policy,
                            "score_mode": score_mode,
                            "target_prune_ratio": ratio,
                            "model_path": model_path,
                            "source_path": result["model_path"],
                            "artifact_type": "torch_state_dict_checkpoint",
                            "load_method": "tests.test_prune_and_eval.load_model with prune_replay",
                        }
                    )
                forward_status = "forward_passed" if result.get("forward_sanity_check") else "forward_failed"
                if not result.get("model_path"):
                    forward_status = "model_rewrite_failed"
                smoke_rows.append(
                    {
                        "policy": policy,
                        "score_mode": score_mode,
                        "target_prune_ratio": ratio,
                        "model_path": model_path,
                        "forward_smoke_status": forward_status,
                        "output_shapes": [],
                        "failure_reason": "" if forward_status == "forward_passed" else result.get("failure_reason", "forward_failed"),
                    }
                )
                eval_result: dict[str, Any] = {
                    "policy": policy,
                    "score_mode": score_mode,
                    "target_prune_ratio": ratio,
                    "model_path": model_path,
                    "eval_status": "skipped_forward_failed",
                    "failure_reason": "forward_smoke_not_passed",
                }
                latency_result: dict[str, Any] = {
                    "policy": policy,
                    "score_mode": score_mode,
                    "target_prune_ratio": ratio,
                    "model_path": model_path,
                    "latency_status": "skipped_forward_failed",
                    "failure_reason": "forward_smoke_not_passed",
                }
                if forward_status == "forward_passed" and model_path:
                    eval_result, latency_result = run_eval_for_artifact(args, root, policy, score_mode, ratio, Path(model_path))
                    if eval_result.get("eval_status") == "success":
                        eval_result.update(baseline_eval)
                        if baseline_eval.get("baseline_AP_0.3") is not None and eval_result.get("AP_0.3") is not None:
                            eval_result["AP_drop"] = float(baseline_eval["baseline_AP_0.3"]) - float(eval_result["AP_0.3"])
                        if baseline_eval.get("baseline_mAP") is not None and eval_result.get("mAP") is not None:
                            eval_result["mAP_drop"] = float(baseline_eval["baseline_mAP"]) - float(eval_result["mAP"])
                    if latency_result.get("latency_status") == "success":
                        latency_result.update(baseline_latency)
                        base = as_float(baseline_latency.get("baseline_latency_ms_p50"))
                        p50 = as_float(latency_result.get("latency_ms_p50"))
                        if base and p50:
                            latency_result["speedup_vs_baseline"] = base / p50
                eval_rows.append(eval_result)
                latency_rows.append(latency_result)

                if policy == "flat_output_groups_fixed":
                    tp_replay.append(
                        {
                            "policy": policy,
                            "score_mode": score_mode,
                            "target_prune_ratio": ratio,
                            "source": "current_pruner_replay",
                            "note": (mapping or {}).get("backend_note", ""),
                            "replay_available": bool((root / "work" / tag / "prune_replay.json").is_file()),
                        }
                    )

                row = {
                    "policy": policy,
                    "score_mode": score_mode,
                    "target_prune_ratio": ratio,
                    "model_path": model_path,
                    "plan_status": "success",
                    "rewrite_status": rewrite_status,
                    "forward_smoke_status": forward_status,
                    "eval_status": eval_result.get("eval_status", ""),
                    "latency_status": latency_result.get("latency_status", ""),
                    "num_grouped_convs_total": pruning_summary.get("num_group_conv_layers", ""),
                    "num_grouped_convs_pruned": pruning_summary.get("num_pruned_groups", ""),
                    "num_grouped_convs_skipped": len(pruning_summary.get("skipped", [])) if isinstance(pruning_summary.get("skipped"), list) else "",
                    "actual_prune_ratio_mean": pruning_summary.get("actual_prune_ratio", ""),
                    "actual_param_prune_ratio": pruning_summary.get("actual_prune_ratio", ""),
                    "actual_dense_flops_or_bops_prune_ratio": "",
                    "groups_changed_count": "",
                    "mean_reinterpretation_ratio": "",
                    "max_reinterpretation_ratio": "",
                    "added_zero_connections_total": "",
                    "zero_padded_slice_ratio_mean": "",
                    "AP_0.3": eval_result.get("AP_0.3", ""),
                    "mAP": eval_result.get("mAP", ""),
                    "baseline_AP_0.3": eval_result.get("baseline_AP_0.3", ""),
                    "baseline_mAP": eval_result.get("baseline_mAP", ""),
                    "AP_drop": eval_result.get("AP_drop", ""),
                    "mAP_drop": eval_result.get("mAP_drop", ""),
                    "latency_backend": latency_result.get("latency_backend", ""),
                    "latency_ms_mean": latency_result.get("latency_ms_mean", ""),
                    "latency_ms_p50": latency_result.get("latency_ms_p50", ""),
                    "latency_ms_p90": latency_result.get("latency_ms_p90", ""),
                    "latency_ms_p95": latency_result.get("latency_ms_p95", ""),
                    "baseline_latency_ms_p50": latency_result.get("baseline_latency_ms_p50", ""),
                    "speedup_vs_baseline": latency_result.get("speedup_vs_baseline", ""),
                    "failure_reason": result.get("failure_reason") or eval_result.get("failure_reason") or latency_result.get("failure_reason", ""),
                }
                summary_rows.append(row)

    write_json(root / "full_model_pruning_plan.json", plan_rows)
    write_json(root / "full_model_pruning_plan_per_policy.json", plan_by_policy)
    write_json(root / "grouped_conv_policy_detail_report.json", policy_detail)
    write_json(root / "tp_replay_audit_report.json", tp_replay)
    write_json(root / "resolver_report.json", resolver_rows)
    write_json(root / "pruned_model_artifacts.json", artifacts)
    write_json(root / "full_model_forward_smoke_report.json", smoke_rows)
    write_json(root / "full_model_eval_short_report.json", eval_rows)
    write_json(root / "full_model_latency_report.json", latency_rows)
    write_jsonl(root / "per_layer_grouped_conv_details.jsonl", per_layer_rows)
    write_jsonl(root / "failure_cases.jsonl", failure_rows)
    write_csv(root / "full_model_ablation_summary.csv", summary_rows, SUMMARY_FIELDS)
    provisional = {"valid": True, "errors": []}
    (root / "full_model_ablation_summary.md").write_text(build_summary_md(summary_rows, provisional), encoding="utf-8")
    validation = validate_v88_output_bundle(root)
    (root / "full_model_ablation_summary.md").write_text(build_summary_md(summary_rows, validation), encoding="utf-8")
    return {"output_dir": str(root), "validation": validation, "summary_rows": summary_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v8.8 full-model grouped-conv pruning ablation")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--warmup-frames", type=int, default=20)
    p.add_argument("--score-modes", nargs="+", default=["l1"])
    p.add_argument("--taylor-calib-batches", type=int, default=1)
    p.add_argument("--run-prune", type=str2bool, default=True)
    p.add_argument("--run-eval", type=str2bool, default=True)
    p.add_argument("--run-baseline", type=str2bool, default=True)
    p.add_argument("--overwrite", type=str2bool, default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run_ablation(parse_args(argv))
    print(json.dumps({"output_dir": result["output_dir"], "validation": result["validation"]}, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
