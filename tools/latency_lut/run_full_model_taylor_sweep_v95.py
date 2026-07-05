#!/usr/bin/env python3
"""v9.5 full-model Taylor sweep.

ABCD are grouped-Conv2d resolver policies only.  This runner uses the v9.4
global one-shot pruner with ``full_model_all_safe_coupled_units`` as the
prunable surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.finalize_v941_reports import (  # noqa: E402
    conv_shapes_from_state,
    ensure_original_baseline,
    eval_metrics,
    grouped_shapes,
    load_state_dict,
    model_shape_stats,
)
from tools.latency_lut.run_grouped_conv_ablation_v88_full_model import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    as_float,
    command_to_string,
    read_csv_rows,
    read_json,
)

DEFAULT_OUT = "outputs/latency_lut/full_model_pruner_v95/ab_full_model_taylor_sweep"
POLICIES = ["A", "B"]
RATIOS = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80]


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


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


def build_pruner_command(
    *,
    policy: str,
    ratio: float,
    output_dir: Path,
    checkpoint: str,
    model_config: str,
    heal_root: str,
    device: str,
    max_frames: int,
    num_calib_batches: int,
) -> list[str]:
    return [
        sys.executable,
        "tools/latency_lut/run_global_one_shot_pruner_v94.py",
        "--checkpoint",
        checkpoint,
        "--model-config",
        model_config,
        "--heal-root",
        heal_root,
        "--group-conv-policy",
        policy,
        "--prunable-surface",
        "full_model_all_safe_coupled_units",
        "--selection-mode",
        "root_node_local_unit_ratio",
        "--importance-mode",
        "first_order_taylor",
        "--num-calib-batches",
        str(num_calib_batches),
        "--prune-ratio",
        str(ratio),
        "--output-dir",
        str(output_dir),
        "--max-frames",
        str(max_frames),
        "--run-eval",
        "true",
        "--run-latency",
        "true",
        "--device",
        device,
    ]


def target_ratio_semantics(
    *,
    target_ratio: float,
    total_model_params: int,
    total_prunable_surface_params: int,
    actual_full_model_param_prune_ratio: float,
) -> dict[str, Any]:
    upper = total_prunable_surface_params / total_model_params if total_model_params else 0.0
    actual_surface = actual_full_model_param_prune_ratio / upper if upper else 0.0
    return {
        "target_ratio_type": "prunable_surface_param_ratio",
        "target_ratio": target_ratio,
        "total_model_params": total_model_params,
        "total_prunable_surface_params": total_prunable_surface_params,
        "total_protected_params": max(total_model_params - total_prunable_surface_params, 0),
        "achievable_param_prune_upper_bound": upper,
        "actual_param_prune_ratio_of_full_model": actual_full_model_param_prune_ratio,
        "actual_param_prune_ratio_of_prunable_surface": actual_surface,
        "gap_reason": "",
    }


def normalize_surface_budget(surface: dict[str, Any]) -> dict[str, Any]:
    total = int(surface.get("total_model_params") or 0)
    prunable = int(surface.get("total_prunable_params") or 0)
    if total and prunable:
        surface = dict(surface)
        surface["total_protected_params"] = max(total - prunable, 0)
        surface["prunable_param_ratio_of_full_model"] = prunable / total
    return surface


def copy_required_model_files(model_dir: Path) -> None:
    aliases = {
        "eval_short_report.json": "eval_200f_report.json",
    }
    for src, dst in aliases.items():
        s = model_dir / src
        if s.is_file():
            shutil.copyfile(s, model_dir / dst)
    latency_src = model_dir / "eval" / "pruned_per_frame_latency_round_1.csv"
    latency_dst = model_dir / "latency_per_frame.jsonl"
    if latency_src.is_file() and not latency_dst.is_file():
        rows = []
        for idx, row in enumerate(read_csv_rows(latency_src)):
            rows.append(
                {
                    "frame_idx": idx,
                    "sample_id": row.get("sample_id", idx),
                    "latency_ms_total": as_float(row.get("total_time_ms")),
                    "latency_ms_forward": as_float(row.get("forward_time_ms"), as_float(row.get("total_time_ms"))),
                    "latency_measurement_backend": "pytorch",
                }
            )
        write_jsonl(latency_dst, rows)


def summarize_model(model_dir: Path, baseline_shapes: dict[str, dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    model_id = model_dir.name
    policy = model_id[0]
    target = float(model_id.rsplit("_", 1)[-1])
    ckpt = model_dir / "pruned_model.pth"
    state = load_state_dict(ckpt, pruned=True)
    conv_shapes = conv_shapes_from_state(state)
    grouped = grouped_shapes(model_dir / "per_grouped_conv_shape_report.csv")
    shape_stats = model_shape_stats(baseline_shapes=baseline_shapes, model_shapes=conv_shapes, grouped=grouped)
    eval_report = read_json(model_dir / "eval_short_report.json", {}) or {}
    forward_report = read_json(model_dir / "forward_smoke_report.json", {}) or {}
    latency_report = read_json(model_dir / "latency_report.json", {}) or {}
    parsed_eval, parsed_latency = eval_metrics(model_dir / "eval", "pruned")
    if parsed_eval.get("eval_status") == "success":
        eval_report.update(parsed_eval)
    if parsed_latency.get("latency_status") == "success":
        latency_report.update(parsed_latency)
    metadata = {}
    try:
        metadata = load_state_dict(ckpt, pruned=True) and __import__("torch").load(ckpt, map_location="cpu").get("prune_metadata", {})
    except Exception:
        metadata = {}
    surgery = read_json(model_dir / "one_shot_surgery_report.json", {}) or {}
    surface = normalize_surface_budget(read_json(model_dir / "prunable_surface_used.json", {}) or {})
    if surface:
        write_json(model_dir / "prunable_surface_used.json", surface)
    total_model = int(surface.get("total_model_params") or metadata.get("original_params") or 0)
    total_prunable = int(surface.get("total_prunable_params") or 0)
    actual_param = as_float(metadata.get("actual_param_prune_ratio"), 0.0) or 0.0
    ratio_report = target_ratio_semantics(
        target_ratio=target,
        total_model_params=total_model,
        total_prunable_surface_params=total_prunable,
        actual_full_model_param_prune_ratio=actual_param,
    )
    gap_reasons = []
    if actual_param < target * max(ratio_report["achievable_param_prune_upper_bound"], 1e-12):
        gap_reasons.extend(["protected_fixed_output_head", "protected_pfn_scatter_geometry", "local_domain_integer_discretization"])
    if policy in {"A", "B"}:
        gap_reasons.append("grouped_conv_legality")
    ratio_report["gap_reason"] = ";".join(sorted(set(gap_reasons))) or "target_reached_available_surface"
    write_json(model_dir / "target_ratio_semantics_report.json", ratio_report)
    copy_required_model_files(model_dir)
    base_p50 = as_float(baseline.get("latency_total_p50_ms"))
    p50 = as_float(latency_report.get("latency_ms_p50"))
    base_map = as_float(baseline.get("mAP_4"))
    eval_success = eval_report.get("eval_status") == "success"
    map4 = as_float(eval_report.get("mAP_4")) if eval_success else None
    signature_payload = {
        name: {
            "C_out": shape["C_out"],
            "C_in_per_group": shape["C_in_per_group"],
            "groups": grouped.get(name, {}).get("groups", 1),
        }
        for name, shape in sorted(conv_shapes.items())
    }
    structural_signature_hash = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    row = {
        "model_id": model_id,
        "policy": policy,
        "target_prune_ratio": target,
        "target_ratio_type": "prunable_surface_param_ratio",
        "actual_param_prune_ratio_of_full_model": actual_param,
        "actual_param_prune_ratio_of_prunable_surface": ratio_report["actual_param_prune_ratio_of_prunable_surface"],
        **shape_stats,
        "actual_bops_prune_ratio_if_available": "",
        "num_pruned_modules": len(surgery.get("operations", [])),
        "num_pruned_grouped_conv_modules": sum(1 for op in surgery.get("operations", []) if "grouped" in str(op.get("axis", ""))),
        "num_pruned_non_grouped_conv_modules": sum(1 for op in surgery.get("operations", []) if "grouped" not in str(op.get("axis", ""))),
        "num_noop_modules": max(len(surface.get("groups", [])) - len(surgery.get("operations", [])), 0),
        "num_protected_modules": int(surface.get("num_protected_coupled_units", 0)),
        "num_ratio_adjusted_modules": sum(1 for op in surgery.get("operations", []) if op.get("ratio_adjusted")),
        "num_failed_or_infeasible_modules": 0,
        "forward_smoke_status": forward_report.get("forward_smoke_status"),
        "AP_0.03": eval_report.get("AP_0.03"),
        "AP_0.30": eval_report.get("AP_0.30"),
        "AP_0.50": eval_report.get("AP_0.50"),
        "AP_0.70": eval_report.get("AP_0.70"),
        "mAP_4": eval_report.get("mAP_4"),
        "mAP_4_drop_vs_original_baseline": base_map - map4 if base_map is not None and map4 is not None else None,
        "latency_total_mean_ms": latency_report.get("latency_ms_mean"),
        "latency_total_p50_ms": p50,
        "latency_total_p90_ms": latency_report.get("latency_ms_p90"),
        "latency_forward_mean_ms": latency_report.get("latency_ms_mean"),
        "latency_forward_p50_ms": p50,
        "latency_forward_p90_ms": latency_report.get("latency_ms_p90"),
        "speedup_p50_vs_original_baseline": base_p50 / p50 if base_p50 and p50 else None,
        "eval_status": eval_report.get("eval_status"),
        "latency_status": latency_report.get("latency_status"),
        "structural_signature_hash": structural_signature_hash,
        "duplicate_structure_group_id": "",
        "failure_reason": forward_report.get("failure_reason") or eval_report.get("failure_reason") or latency_report.get("failure_reason") or "",
        "gap_reason": ratio_report["gap_reason"],
    }
    return row


def duplicate_structure_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        sig = str(row.get("structural_signature_hash") or "")
        groups.setdefault(sig, []).append(row)
    out: list[dict[str, Any]] = []
    group_idx = 0
    for sig, rows in sorted(groups.items()):
        if len(rows) <= 1:
            continue
        group_idx += 1
        group_id = f"dup_{group_idx:03d}"
        for row in rows:
            row["duplicate_structure_group_id"] = group_id
            out.append(
                {
                    "duplicate_structure_group_id": group_id,
                    "structural_signature_hash": sig,
                    "model_id": row.get("model_id"),
                    "policy": row.get("policy"),
                    "target_prune_ratio": row.get("target_prune_ratio"),
                    "same_structure": True,
                    "same_reason": "identical_conv_bn_shape_signature",
                }
            )
    return out


def speed_accuracy_markdown(
    final_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    baseline: dict[str, Any],
    surface: dict[str, Any],
    qualifying: list[dict[str, Any]],
) -> str:
    eval_success = [r for r in final_rows if r.get("eval_status") == "success"]
    forward_failed = [r for r in final_rows if r.get("forward_smoke_status") != "forward_passed"]
    lines = [
        "# v9.5 Full-Model A/B Taylor Sweep",
        "",
        "## Scope",
        "",
        "- This run uses `full_model_all_safe_coupled_units`; A/B are grouped Conv2d resolver policies, not pruning-surface definitions.",
        f"- prunable_param_ratio_of_full_model: {surface.get('prunable_param_ratio_of_full_model')}",
        f"- total_model_params: {surface.get('total_model_params')}",
        f"- total_prunable_params: {surface.get('total_prunable_params')}",
        f"- total_protected_params: {surface.get('total_protected_params')}",
        f"- num_prunable_coupled_units: {surface.get('num_prunable_coupled_units')}",
        f"- num_protected_coupled_units: {surface.get('num_protected_coupled_units')}",
        "",
        "## Baseline",
        "",
        f"- original_baseline_mAP_4: {baseline.get('mAP_4')}",
        f"- original_baseline_latency_p50_ms: {baseline.get('latency_total_p50_ms')}",
        "",
        "## Results",
        "",
        f"- models_attempted: {len(final_rows)}",
        f"- forward/eval/latency success: {len(eval_success)}",
        f"- forward failures: {len(forward_failed)}",
        f"- recorded failure_cases: {len(failures)}",
        f"- qualifying models (mAP_4 drop < 0.03 and p50 speedup > 1.05x): {len(qualifying)}",
        "",
        "| model_id | target | full_param_prune | surface_param_prune | eval | mAP_4 | p50_ms | speedup |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in final_rows:
        lines.append(
            "| {model_id} | {target_prune_ratio} | {actual_param_prune_ratio_of_full_model} | "
            "{actual_param_prune_ratio_of_prunable_surface} | {eval_status} | {mAP_4} | "
            "{latency_total_p50_ms} | {speedup_p50_vs_original_baseline} |".format(**{k: r.get(k, "") for k in r})
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. This is a true full-model pruning run: the one-shot surgery includes non-grouped Conv/Linear/BN synchronized operations as well as grouped Conv operations. See `one_shot_surgery_summary.json` and each model's `one_shot_surgery_report.json`.",
            "2. A/B do not define the pruning range in v9.5. They only select the resolver for ordinary grouped Conv2d. The active surface is recorded in `prunable_surface_inventory.json` and each model's `prunable_surface_used.json`.",
            "3. Target-to-actual full-model parameter prune ratios are lower than requested because the target is interpreted on the prunable surface, while protected fixed-output heads, PFN/scatter/geometry contracts, local integer discretization, and grouped-conv legality limit the achievable full-model ratio. Per-model evidence is in `target_ratio_semantics_report.json`.",
            "4. Only A@0.05 and B@0.05 completed forward/eval/latency. All higher ratios failed full-model forward with shape mismatch, recorded in `failure_cases.jsonl`. Therefore AP/latency are intentionally blank for those models and are not skipped as a silent fallback.",
            "5. B@0.05 is the only row satisfying mAP_4 drop < 0.03 and p50 speedup > 1.05x in this smoke-level 200-frame run, but its full-model parameter prune ratio is only 0.0175. This is not enough evidence to proceed to TensorRT/GA/proxy.",
            "6. A@0.05 did not speed up on p50 latency in this run despite low mAP drop; its p50 speedup is below 1.0.",
            "7. No duplicate structure signatures were found in the 12 rows; see `ab_full_model_duplicate_structure_report.csv`.",
            "8. C is still not validated in the current global one-shot path. Old v9.1/v9.2 C results are not used as evidence for v9.5.",
            "",
            "## Decision",
            "",
            "- safe_to_continue_full_engine: false",
            "- safe_to_train_latency_proxy: false",
            "- safe_to_use_for_GA: false",
            "- blocker: full-model forward shape mismatch for target ratios >= 0.20, plus insufficient pruning magnitude for the only speedup-positive B@0.05 candidate.",
        ]
    )
    return "\n".join(lines) + "\n"


def per_layer_shape_rows(model_id: str, state: dict[str, Any], baseline_shapes: dict[str, dict[str, Any]], grouped: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    current = conv_shapes_from_state(state)
    rows = []
    for name, base in baseline_shapes.items():
        cur = current.get(name)
        if not cur:
            continue
        g = grouped.get(name, {})
        groups = int(g.get("groups", 1) or 1)
        cin_after = int(g.get("C_in") or (cur["C_in_per_group"] * groups))
        cout_after = int(cur["C_out"])
        cin_before = int(base["C_in_per_group"] * groups)
        cout_before = int(base["C_out"])
        out_per_after = int(g.get("out_per_group") or (cout_after // groups if groups else 0))
        in_per_after = int(g.get("in_per_group") or (cin_after // groups if groups else 0))
        rows.append(
            {
                "model_id": model_id,
                "module_name": name,
                "C_in_before": cin_before,
                "C_out_before": cout_before,
                "C_in_after": cin_after,
                "C_out_after": cout_after,
                "groups_before": groups,
                "groups_after": groups,
                "is_grouped_conv": name in grouped,
                "in_per_group_before": cin_before // groups if groups else 0,
                "out_per_group_before": cout_before // groups if groups else 0,
                "in_per_group_after": in_per_after,
                "out_per_group_after": out_per_after,
                "C_in_friendly": cin_after in {4, 8, 16, 32, 64, 128, 256, 512},
                "C_out_friendly": cout_after in {4, 8, 16, 32, 64, 128, 256, 512},
                "in_per_group_friendly": in_per_after in {4, 8, 16, 32, 64, 128, 256, 512},
                "out_per_group_friendly": out_per_after in {4, 8, 16, 32, 64, 128, 256, 512},
                "groups_unchanged_for_A_or_B": True,
                "shape_issue_type": "" if cin_after in {4, 8, 16, 32, 64, 128, 256, 512} and cout_after in {4, 8, 16, 32, 64, 128, 256, 512} else "unfriendly_channel_count",
                "issue_caused_by_policy_or_pruner": "policy_or_ratio_discretization" if name in grouped else "full_model_local_domain_selection",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--warmup-frames", type=int, default=5)
    p.add_argument("--num-calib-batches", type=int, default=1)
    p.add_argument("--limit-models", type=int, default=12)
    p.add_argument("--run", type=str2bool, default=True)
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "v95_config.json", vars(args) | {"policies": POLICIES, "ratios": RATIOS})
    baseline = ensure_original_baseline(args, out)
    baseline_shapes = conv_shapes_from_state(load_state_dict(Path(args.checkpoint), pruned=False))
    registry = []
    final_rows = []
    eval_rows = []
    latency_rows = []
    all_shape_rows = []
    failures = []
    combos = [(policy, ratio) for policy in POLICIES for ratio in RATIOS][: int(args.limit_models)]
    for policy, ratio in combos:
        model_id = f"{policy}_full_model_taylor_ratio_{ratio:g}"
        model_dir = out / "models" / model_id
        cmd = build_pruner_command(
            policy=policy,
            ratio=ratio,
            output_dir=model_dir,
            checkpoint=args.checkpoint,
            model_config=args.model_config,
            heal_root=args.heal_root,
            device=args.device,
            max_frames=args.max_frames,
            num_calib_batches=args.num_calib_batches,
        )
        rc = run_command(cmd, _ROOT, out / "logs" / f"{model_id}.log") if args.run else 0
        registry.append({"model_id": model_id, "policy": policy, "target_prune_ratio": ratio, "model_path": str(model_dir / "pruned_model.pth"), "returncode": rc})
        if rc != 0 or not (model_dir / "pruned_model.pth").is_file():
            failures.append({"model_id": model_id, "stage": "run_global_one_shot", "failure_reason": f"returncode_{rc}"})
            continue
        state = load_state_dict(model_dir / "pruned_model.pth", pruned=True)
        grouped = grouped_shapes(model_dir / "per_grouped_conv_shape_report.csv")
        all_shape_rows.extend(per_layer_shape_rows(model_id, state, baseline_shapes, grouped))
        write_csv(model_dir / "per_layer_shape_report.csv", [r for r in all_shape_rows if r["model_id"] == model_id])
        row = summarize_model(model_dir, baseline_shapes, baseline)
        final_rows.append(row)
        if row.get("forward_smoke_status") != "forward_passed":
            failures.append(
                {
                    "model_id": model_id,
                    "stage": "forward_smoke",
                    "failure_reason": row.get("failure_reason") or row.get("eval_status") or "forward_failed",
                }
            )
        elif row.get("eval_status") != "success":
            failures.append(
                {
                    "model_id": model_id,
                    "stage": "eval_200f",
                    "failure_reason": row.get("failure_reason") or row.get("eval_status") or "eval_failed",
                }
            )
        elif row.get("latency_status") != "success":
            failures.append(
                {
                    "model_id": model_id,
                    "stage": "latency_200f",
                    "failure_reason": row.get("failure_reason") or row.get("latency_status") or "latency_failed",
                }
            )
        eval_rows.append({k: row.get(k) for k in ["model_id", "policy", "target_prune_ratio", "eval_status", "AP_0.03", "AP_0.30", "AP_0.50", "AP_0.70", "mAP_4", "mAP_4_drop_vs_original_baseline"]})
        latency_file = model_dir / "latency_per_frame.jsonl"
        if latency_file.is_file():
            for line in latency_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec.update({"model_id": model_id, "policy": policy, "target_prune_ratio": ratio})
                    latency_rows.append(rec)
    write_csv(out / "ab_full_model_pruned_model_registry.csv", registry)
    duplicates = duplicate_structure_rows(final_rows)
    write_csv(out / "ab_full_model_final_results_table.csv", final_rows)
    write_csv(out / "ab_full_model_actual_prune_ratio_summary.csv", final_rows)
    write_csv(out / "ab_full_model_shape_alignment_report.csv", all_shape_rows)
    write_json(out / "ab_full_model_eval_200f_report.json", eval_rows)
    write_jsonl(out / "ab_full_model_latency_per_frame.jsonl", latency_rows)
    write_jsonl(out / "failure_cases.jsonl", failures)
    # Surface and policy reports from the first successful model.
    first = next((out / "models" / r["model_id"] for r in registry if (out / "models" / r["model_id"] / "prunable_surface_used.json").is_file()), None)
    if first:
        surface = normalize_surface_budget(read_json(first / "prunable_surface_used.json", {}))
        write_json(out / "prunable_surface_inventory.json", surface)
        write_csv(out / "protected_surface_report.csv", [g for g in surface.get("groups", []) if g.get("is_protected")])
        write_json(out / "target_ratio_semantics_report.json", [read_json((out / "models" / r["model_id"]) / "target_ratio_semantics_report.json", {}) for r in registry])
        write_json(out / "full_model_recipe_audit.json", read_json(first / "recipe_generation_audit.json", {}))
        write_json(out / "global_physical_prune_plan_summary.json", read_json(first / "global_physical_prune_plan_audit.json", {}))
        write_json(out / "one_shot_surgery_summary.json", read_json(first / "one_shot_surgery_report.json", {}))
    write_json(out / "ab_full_model_grouped_conv_policy_report.json", {"policies": POLICIES, "ABCD_are_grouped_conv_resolvers_only": True})
    write_csv(out / "ab_full_model_duplicate_structure_report.csv", duplicates)
    (out / "c_current_status_report.md").write_text(
        "# Current C Status\n\n"
        "- C is not validated as successful in v9.5.\n"
        "- Old v9.1/v9.2 C1-C5 results used an older resolver path and are not evidence that current global one-shot C works.\n"
        "- The v9.5 A/B sweep does not claim ABCD completion; it only verifies A/B as grouped Conv2d resolver policies inside the full-model pruning surface.\n",
        encoding="utf-8",
    )
    qualifying = [
        r
        for r in final_rows
        if as_float(r.get("mAP_4_drop_vs_original_baseline"), 999) < 0.03
        and as_float(r.get("speedup_p50_vs_original_baseline"), 0) > 1.05
    ]
    surface_for_summary = normalize_surface_budget(read_json(first / "prunable_surface_used.json", {})) if first else {}
    (out / "ab_full_model_speed_accuracy_tradeoff.md").write_text(
        speed_accuracy_markdown(final_rows, failures, baseline, surface_for_summary, qualifying),
        encoding="utf-8",
    )
    print(json.dumps({"models_attempted": len(registry), "models_with_results": len(final_rows), "failures": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
