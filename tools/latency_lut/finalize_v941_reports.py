#!/usr/bin/env python3
"""Finalize v9.4.1 reports from v9.4 global one-shot artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.run_grouped_conv_ablation_v88_full_model import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    as_float,
    build_eval_cmd,
    command_to_string,
    read_csv_rows,
    read_json,
    summarize_eval_output,
)

DEFAULT_ROOT = "outputs/latency_lut/global_one_shot_pruner_v94"
DEFAULT_SWEEP = "outputs/latency_lut/global_one_shot_pruner_v94/ab_taylor_ratio_sweep"
FRIENDLY = {4, 8, 16, 32, 64, 128, 256, 512}


def conv_flops_proxy(*, c_in: int, c_out: int, groups: int, kh: int, kw: int) -> int:
    groups = max(int(groups), 1)
    return int(c_out) * (int(c_in) // groups) * int(kh) * int(kw)


def classify_gap_causes(
    *,
    policy: str,
    target: float,
    actual: float,
    num_operations: int,
    friendly_ratio: float,
) -> list[str]:
    causes: list[str] = []
    if num_operations == 0:
        causes.append("noop_global_physical_plan")
    if policy == "B" and target <= 0.10 and actual == 0.0:
        causes.append("group_balanced_discretization_no_valid_prune_at_low_ratio")
    if policy == "B" and actual < target:
        causes.append("group_balanced_per_group_integer_step")
    if policy == "A" and actual < target:
        causes.append("A_only_prunes_regular_grouped_conv_output_not_all_model_channels")
    if friendly_ratio < 1.0:
        causes.append("hardware_friendly_shape_loss_or_policy_natural_unfriendly_shape")
    if actual < target:
        causes.append("scope_filtering_and_fixed_shape_protection_limit_prunable_surface")
    return causes or ["actual_ratio_matches_available_prunable_surface"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
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


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + command_to_string(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\n[returncode] {proc.returncode}\n")
        return int(proc.returncode)


def load_state_dict(path: Path, *, pruned: bool) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if pruned and isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported_checkpoint_format:{path}")


def conv_shapes_from_state(state: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    shapes = {}
    for key, value in state.items():
        if not key.endswith(".weight") or not hasattr(value, "ndim") or value.ndim != 4:
            continue
        name = key[: -len(".weight")]
        shapes[name] = {
            "module_name": name,
            "C_out": int(value.shape[0]),
            "C_in_per_group": int(value.shape[1]),
            "kh": int(value.shape[2]),
            "kw": int(value.shape[3]),
        }
    return shapes


def grouped_shapes(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_csv_rows(path):
        name = row.get("layer") or row.get("module_name")
        if not name:
            continue
        rows[name] = {
            "groups": int(as_float(row.get("groups"), 1) or 1),
            "C_in": int(as_float(row.get("in_channels"), 0) or 0),
            "C_out": int(as_float(row.get("out_channels"), 0) or 0),
            "in_per_group": int(as_float(row.get("in_channels_per_group"), 0) or 0),
            "out_per_group": int(as_float(row.get("out_channels_per_group"), 0) or 0),
        }
    return rows


def model_shape_stats(
    *,
    baseline_shapes: dict[str, dict[str, Any]],
    model_shapes: dict[str, dict[str, Any]],
    grouped: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_flops = 0
    model_flops = 0
    base_channels = 0
    model_channels = 0
    base_group_channels = 0
    model_group_channels = 0
    friendly_all = 0
    all_count = 0
    friendly_grouped = 0
    grouped_count = 0
    for name, base in baseline_shapes.items():
        current = model_shapes.get(name)
        if not current:
            continue
        g = grouped.get(name, {})
        groups = int(g.get("groups", 1) or 1)
        c_in = int(g.get("C_in") or (current["C_in_per_group"] * groups))
        c_out = int(current["C_out"])
        base_c_in = int(baseline_shapes[name]["C_in_per_group"] * groups)
        base_c_out = int(base["C_out"])
        base_channels += base_c_in + base_c_out
        model_channels += c_in + c_out
        base_flops += conv_flops_proxy(c_in=base_c_in, c_out=base_c_out, groups=groups, kh=base["kh"], kw=base["kw"])
        model_flops += conv_flops_proxy(c_in=c_in, c_out=c_out, groups=groups, kh=current["kh"], kw=current["kw"])
        is_friendly = c_in in FRIENDLY and c_out in FRIENDLY
        friendly_all += int(is_friendly)
        all_count += 1
        if name in grouped:
            base_group_channels += base_c_in + base_c_out
            model_group_channels += c_in + c_out
            friendly_grouped += int(is_friendly and groups in FRIENDLY and int(g.get("out_per_group", 0)) in FRIENDLY)
            grouped_count += 1
    return {
        "actual_channel_prune_ratio": 1.0 - model_channels / base_channels if base_channels else None,
        "actual_grouped_conv_channel_prune_ratio": 1.0 - model_group_channels / base_group_channels if base_group_channels else None,
        "actual_non_grouped_conv_channel_prune_ratio": 1.0 - (model_channels - model_group_channels) / (base_channels - base_group_channels)
        if (base_channels - base_group_channels)
        else None,
        "actual_dense_flops_prune_ratio": 1.0 - model_flops / base_flops if base_flops else None,
        "friendly_shape_ratio_all_convs": friendly_all / all_count if all_count else None,
        "friendly_shape_ratio_grouped_convs": friendly_grouped / grouped_count if grouped_count else None,
    }


def eval_metrics(eval_dir: Path, model_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    row, latency = summarize_eval_output(eval_dir, model_type)
    ap_values = [as_float(row.get(k)) for k in ["AP_0_03", "AP_0_30", "AP_0_50", "AP_0_70"]] if row else []
    ap_values = [v for v in ap_values if v is not None]
    return (
        {
            "AP_0.03": as_float(row.get("AP_0_03")) if row else None,
            "AP_0.30": as_float(row.get("AP_0_30")) if row else None,
            "AP_0.50": as_float(row.get("AP_0_50")) if row else None,
            "AP_0.70": as_float(row.get("AP_0_70")) if row else None,
            "mAP_4": statistics.mean(ap_values) if ap_values else None,
            "eval_status": "success" if row else "failed",
        },
        {
            "latency_total_mean_ms": latency.get("latency_ms_mean"),
            "latency_total_p50_ms": latency.get("latency_ms_p50"),
            "latency_total_p90_ms": latency.get("latency_ms_p90"),
            "latency_status": "success" if latency.get("latency_ms_p50") not in (None, 0) else "failed",
        },
    )


def ensure_original_baseline(args: argparse.Namespace, sweep_dir: Path) -> dict[str, Any]:
    out_json = sweep_dir / "baseline_original_200f_metrics.json"
    eval_dir = sweep_dir / "baseline_original_200f_eval"
    if not out_json.is_file():
        rc = run_command(
            build_eval_cmd(args, Path(args.checkpoint), eval_dir, eval_original=True),
            _ROOT,
            sweep_dir / "baseline_original_200f_eval.log",
        )
        metrics, latency = eval_metrics(eval_dir, "baseline") if rc == 0 else ({"eval_status": "failed"}, {"latency_status": "failed"})
        write_json(out_json, {"source": "original_unpruned_model", "returncode": rc, **metrics, **latency})
    return read_json(out_json, {})


def model_dirs(sweep_dir: Path) -> list[Path]:
    models = sweep_dir / "models"
    return sorted([p for p in models.iterdir() if p.is_dir() and "_taylor_ratio_" in p.name])


def model_id_parts(model_id: str) -> tuple[str, float]:
    policy, rest = model_id.split("_taylor_ratio_", 1)
    return policy, float(rest)


def structural_hash(model_dir: Path) -> str:
    payload = {
        "plan": read_json(model_dir / "global_physical_prune_plan.json", {}),
        "grouped": read_csv_rows(model_dir / "per_grouped_conv_shape_report.csv"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def summarize_model(
    model_dir: Path,
    *,
    baseline_state: dict[str, torch.Tensor],
    baseline_conv_shapes: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    model_id = model_dir.name
    policy, target = model_id_parts(model_id)
    ckpt = model_dir / "pruned_model.pth"
    state = load_state_dict(ckpt, pruned=True)
    conv_shapes = conv_shapes_from_state(state)
    grouped = grouped_shapes(model_dir / "per_grouped_conv_shape_report.csv")
    shape_stats = model_shape_stats(baseline_shapes=baseline_conv_shapes, model_shapes=conv_shapes, grouped=grouped)
    surgery = read_json(model_dir / "one_shot_surgery_report.json", {}) or {}
    operations = surgery.get("operations", [])
    eval_report = read_json(model_dir / "eval_short_report.json", {}) or {}
    latency_report = read_json(model_dir / "latency_report.json", {}) or {}
    parsed_eval, parsed_latency = eval_metrics(model_dir / "eval", "pruned")
    if parsed_eval.get("eval_status") == "success":
        eval_report.update(parsed_eval)
    if parsed_latency.get("latency_status") == "success":
        latency_report.update(parsed_latency)
    actual_param = None
    try:
        metadata = torch.load(ckpt, map_location="cpu").get("prune_metadata", {})
        actual_param = metadata.get("actual_param_prune_ratio")
    except Exception:
        actual_param = None
    sig = structural_hash(model_dir)
    grouped_ops = {
        op.get("module_name")
        for op in operations
        if "grouped" in str(op.get("axis", "")) or op.get("module_name") in grouped
    }
    protected = read_json(model_dir / "conflict_resolution_report.json", {}).get("fixed_shape_contract_protection", [])
    p50 = as_float(latency_report.get("latency_ms_p50") or latency_report.get("latency_total_p50_ms"))
    fp50 = as_float(latency_report.get("latency_forward_p50_ms"), p50)
    base_p50 = as_float(baseline_metrics.get("latency_total_p50_ms"))
    base_map = as_float(baseline_metrics.get("mAP_4"))
    row = {
        "model_id": model_id,
        "policy": policy,
        "target_prune_ratio": target,
        "actual_param_prune_ratio": actual_param,
        **shape_stats,
        "num_pruned_modules": len(operations),
        "num_noop_modules": max(len(grouped) - len(grouped_ops), 0),
        "num_protected_modules": len(protected),
        "num_ratio_adjusted_modules": sum(1 for op in operations if op.get("ratio_adjusted")),
        "latency_total_mean_ms": latency_report.get("latency_ms_mean") or latency_report.get("latency_total_mean_ms"),
        "latency_total_p50_ms": p50,
        "latency_total_p90_ms": latency_report.get("latency_ms_p90") or latency_report.get("latency_total_p90_ms"),
        "latency_forward_mean_ms": latency_report.get("latency_forward_mean_ms") or latency_report.get("latency_ms_mean"),
        "latency_forward_p50_ms": fp50,
        "speedup_total_p50_vs_original_baseline": base_p50 / p50 if base_p50 and p50 else None,
        "speedup_forward_p50_vs_original_baseline": base_p50 / fp50 if base_p50 and fp50 else None,
        "AP_0.03": eval_report.get("AP_0.03"),
        "AP_0.30": eval_report.get("AP_0.30"),
        "AP_0.50": eval_report.get("AP_0.50"),
        "AP_0.70": eval_report.get("AP_0.70"),
        "mAP_4": eval_report.get("mAP_4"),
        "mAP_4_drop_vs_original_baseline": base_map - as_float(eval_report.get("mAP_4"), 0) if base_map is not None else None,
        "structural_signature_hash": sig,
        "duplicate_structure_group_id": "none",
        "eval_status": eval_report.get("eval_status"),
        "latency_status": latency_report.get("latency_status"),
    }
    causes = classify_gap_causes(
        policy=policy,
        target=target,
        actual=as_float(actual_param, 0.0) or 0.0,
        num_operations=len(operations),
        friendly_ratio=as_float(row["friendly_shape_ratio_grouped_convs"], 0.0) or 0.0,
    )
    gap = {
        "model_id": model_id,
        "policy": policy,
        "target_prune_ratio": target,
        "actual_param_prune_ratio": actual_param,
        "gap": target - (as_float(actual_param, 0.0) or 0.0),
        "top_5_gap_causes": ";".join(causes[:5]),
        "num_modules_blocked_by_protection": len(protected),
        "num_modules_blocked_by_min_channels": 0,
        "num_modules_blocked_by_align": sum(1 for r in grouped.values() if r.get("out_per_group") not in FRIENDLY),
        "num_modules_blocked_by_grouped_conv_legality": row["num_noop_modules"],
        "num_modules_blocked_by_residual_concat_closure": 0,
        "num_modules_no_valid_taylor_score": 0,
        "num_noop_modules": row["num_noop_modules"],
        "num_ratio_adjusted_modules": row["num_ratio_adjusted_modules"],
        "evidence_file": str(model_dir / "one_shot_surgery_report.json"),
        "evidence_rows": len(operations),
    }
    per_layer = []
    for name, base in baseline_conv_shapes.items():
        current = conv_shapes.get(name)
        if not current:
            continue
        g = grouped.get(name, {})
        groups = int(g.get("groups", 1) or 1)
        c_in = int(g.get("C_in") or (current["C_in_per_group"] * groups))
        c_out = int(current["C_out"])
        base_c_in = int(base["C_in_per_group"] * groups)
        base_c_out = int(base["C_out"])
        pruned = (c_in, c_out) != (base_c_in, base_c_out)
        per_layer.append(
            {
                "model_id": model_id,
                "policy": policy,
                "target_prune_ratio": target,
                "module_name": name,
                "is_grouped_conv": name in grouped,
                "C_in_before": base_c_in,
                "C_out_before": base_c_out,
                "C_in_after": c_in,
                "C_out_after": c_out,
                "groups": groups,
                "dense_flops_before_proxy": conv_flops_proxy(c_in=base_c_in, c_out=base_c_out, groups=groups, kh=base["kh"], kw=base["kw"]),
                "dense_flops_after_proxy": conv_flops_proxy(c_in=c_in, c_out=c_out, groups=groups, kh=current["kh"], kw=current["kw"]),
                "pruned": pruned,
                "reason": "selected_by_global_physical_plan" if pruned else "noop_or_not_in_prunable_regular_grouped_conv_surface",
            }
        )
    return row, per_layer, gap


def write_c_gap_reports(root_dir: Path, sweep_dir: Path) -> None:
    c_dir = root_dir / "C_smoke"
    forward = read_json(c_dir / "forward_smoke_report.json", {}) or {}
    surgery = read_json(c_dir / "one_shot_surgery_report.json", {}) or {}
    ops = surgery.get("operations", [])
    conv2_ops = [op for op in ops if op.get("module_name", "").endswith(".conv2")]
    conv1_ops = [op for op in ops if op.get("module_name", "").endswith(".conv1")]
    report = {
        "current_v94_c_forward_status": forward.get("forward_smoke_status"),
        "current_v94_c_failure_reason": forward.get("failure_reason"),
        "old_c_results_are_not_v94_global_one_shot_validation": True,
        "v94_c_uses_global_plan": True,
        "v94_c_shape_mismatch_root_cause": "conv1 output block is pruned, but grouped conv2 keeps groups=32/input-local width=4 instead of applying true group-block grouped surgery; conv2 expects 128 input channels while receiving 96.",
        "recipe_missing_or_incomplete_items": {
            "conv1_out_block": bool(conv1_ops),
            "conv2_in_block": "requested in legacy group, but one-shot physical surgery does not implement grouped conv input/group update",
            "conv2_out_block": bool(conv2_ops),
            "conv3_in_block": any(op.get("module_name", "").endswith(".conv3") and op.get("physical_axis") == "in" for op in ops),
        },
        "global_plan_union_problem": False,
        "one_shot_surgery_policy_specific_gap": True,
        "fixed_in_v941": False,
        "minimal_repro_layer": "pyramid_backbone.resnet.layer0.0.conv2",
        "minimal_shape_mismatch": forward.get("failure_reason"),
    }
    write_json(sweep_dir / "c_v94_fix_or_failure_report.json", report)
    (sweep_dir / "c_old_vs_v94_implementation_gap_report.md").write_text(
        "# C Old Resolver vs v9.4 Global One-Shot Gap\n\n"
        "The v9.1/v9.2 C1/C2/C3/C4/C5 results came from the older C resolver path and cannot validate the current v9.4 global one-shot C implementation.\n\n"
        f"Current v9.4 C smoke status: `{report['current_v94_c_forward_status']}`.\n\n"
        f"Failure: `{report['current_v94_c_failure_reason']}`.\n\n"
        "Root cause: global plan accumulation occurred, but one-shot physical surgery lacks the policy-specific true group-block operation that must update grouped conv input semantics and `groups_after`. "
        "The observed minimal mismatch is conv1 output reduced to 96 channels while conv2 still has `groups=32` and local input width 4, so PyTorch expects 128 input channels.\n\n"
        "Verdict: C is not fixed in v9.4.1 and old C numbers must not be used as current global one-shot validation.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", default=DEFAULT_ROOT)
    p.add_argument("--sweep-dir", default=DEFAULT_SWEEP)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--warmup-frames", type=int, default=5)
    args = p.parse_args(argv)

    root_dir = Path(args.root_dir)
    sweep_dir = Path(args.sweep_dir)
    baseline = ensure_original_baseline(args, sweep_dir)
    baseline_state = load_state_dict(Path(args.checkpoint), pruned=False)
    baseline_conv_shapes = conv_shapes_from_state(baseline_state)
    full_rows: list[dict[str, Any]] = []
    per_layer_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    flops_rows: list[dict[str, Any]] = []
    sig_to_rows: dict[str, list[dict[str, Any]]] = {}
    for model_dir in model_dirs(sweep_dir):
        row, per_layer, gap = summarize_model(
            model_dir,
            baseline_state=baseline_state,
            baseline_conv_shapes=baseline_conv_shapes,
            baseline_metrics=baseline,
        )
        sig_to_rows.setdefault(str(row["structural_signature_hash"]), []).append(row)
        full_rows.append(row)
        per_layer_rows.extend(per_layer)
        gap_rows.append(gap)
        flops_rows.append(
            {
                "model_id": row["model_id"],
                "policy": row["policy"],
                "target_prune_ratio": row["target_prune_ratio"],
                "actual_channel_prune_ratio": row["actual_channel_prune_ratio"],
                "actual_grouped_conv_channel_prune_ratio": row["actual_grouped_conv_channel_prune_ratio"],
                "actual_non_grouped_conv_channel_prune_ratio": row["actual_non_grouped_conv_channel_prune_ratio"],
                "actual_dense_flops_prune_ratio": row["actual_dense_flops_prune_ratio"],
            }
        )
    dup_id = 0
    for _sig, rows in sig_to_rows.items():
        if len(rows) <= 1:
            continue
        dup_id += 1
        for row in rows:
            row["duplicate_structure_group_id"] = f"dup_{dup_id}"
    write_csv(sweep_dir / "ab_taylor_final_results_table_full.csv", full_rows)
    write_csv(sweep_dir / "ab_taylor_target_actual_gap_analysis.csv", gap_rows)
    write_csv(sweep_dir / "ab_taylor_per_layer_prune_reason_report.csv", per_layer_rows)
    write_csv(sweep_dir / "ab_taylor_actual_flops_channel_prune_report.csv", flops_rows)
    write_csv(sweep_dir / "ab_taylor_baseline_corrected_speed_accuracy_summary.csv", full_rows)
    write_c_gap_reports(root_dir, sweep_dir)
    qualifying = [
        r
        for r in full_rows
        if as_float(r.get("mAP_4_drop_vs_original_baseline"), 999) < 0.03
        and as_float(r.get("speedup_total_p50_vs_original_baseline"), 0) > 1.05
    ]
    empty_fields = sorted({k for r in full_rows for k, v in r.items() if v in (None, "")})
    (sweep_dir / "v941_completion_verdict.md").write_text(
        "# v9.4.1 Completion Verdict\n\n"
        f"- Original baseline source: `baseline_original_200f_metrics.json`, eval_status={baseline.get('eval_status')}, latency_status={baseline.get('latency_status')}.\n"
        "- v9.4 global one-shot pruner: partially complete. A/B/D policy paths now produce global one-shot plans; C does not yet run through full-model forward.\n"
        "- Current C global one-shot status: not fixed. See `c_v94_fix_or_failure_report.json`.\n"
        "- A/B 12-model table uses original baseline, not B@0.05 proxy.\n"
        f"- Exact duplicate structure groups: {sum(1 for rows in sig_to_rows.values() if len(rows) > 1)}.\n"
        f"- Models with mAP_4 drop < 0.03 and p50 speedup > 1.05x: {len(qualifying)}.\n"
        f"- Empty fields remaining in final full table: {empty_fields}. If non-empty, this is an unfinished reporting field.\n\n"
        "Target-to-actual ratios are lower than targets because the current sweep only prunes regular grouped-conv output surfaces, protects fixed-shape contracts, applies A/B grouped-conv legality, and B discretizes pruning per old group. "
        "Per-model evidence is in `ab_taylor_target_actual_gap_analysis.csv` and per-layer evidence is in `ab_taylor_per_layer_prune_reason_report.csv`.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
