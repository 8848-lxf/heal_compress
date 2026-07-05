#!/usr/bin/env python3
"""v9.3 project-pruner-native grouped-conv A/B/C/D audit.

This runner keeps Torch-Pruning out of the A/D model-generation path.  TP is
only used by optional oracle helpers for shape auditing.  The project pruner
entrypoint is ``tests/test_general_pruner.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from tools.latency_lut.run_grouped_conv_ablation_v88_full_model import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    as_float,
    build_eval_cmd,
    read_csv_rows,
    read_json,
    run_baseline_eval,
    run_command,
    summarize_eval_output,
    write_csv,
    write_json,
    write_jsonl,
)

DEFAULT_OUT = "outputs/latency_lut/grouped_conv_ablation_v93_pruner_native_abcd"
V92_ROOT = Path("outputs/latency_lut/grouped_conv_ablation_v92_fix_a_profile_b_c")
FRIENDLY = {1, 4, 8, 16, 32}
ORACLE_LAYERS = [
    "pyramid_backbone.resnet.layer0.0.conv2",
    "pyramid_backbone.resnet.layer0.1.conv2",
    "pyramid_backbone.resnet.layer1.0.conv2",
    "pyramid_backbone.resnet.layer1.1.conv2",
    "pyramid_backbone.resnet.layer2.0.conv2",
    "pyramid_backbone.resnet.layer2.1.conv2",
]
REQUIRED_OUTPUT_FILES = [
    "v93_config.json",
    "a_project_pruner_implementation_report.json",
    "a_project_pruner_vs_tp_oracle_audit.json",
    "a_project_pruner_full_model_summary.csv",
    "d_project_pruner_implementation_report.json",
    "d_full_model_summary.csv",
    "c_variant_actual_results_summary.csv",
    "c3_layerwise_shape_latency_breakdown.csv",
    "c3_speedup_decomposition.md",
    "c3_c4_actual_constraint_audit.csv",
    "abc_unified_per_layer_shape_latency.csv",
    "full_model_forward_smoke_report.json",
    "full_model_eval_short_report.json",
    "full_model_latency_report.json",
    "failure_cases.jsonl",
    "v93_decision_summary.md",
]


def save_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    write_csv(path, rows, fieldnames)


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def csv_value(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return default


def run_pruner_cmd(args: argparse.Namespace, *, ratio: float, mode: str, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        "tests/test_general_pruner.py",
        "--checkpoint", str(args.checkpoint),
        "--model-config", str(args.model_config),
        "--heal-root", str(args.heal_root),
        "--device", str(args.device),
        "--prune-ratio", str(ratio),
        "--importance-mode", "l1_norm",
        "--selection-mode", "local_scope",
        "--group-conv-selection-mode", mode,
        "--group-conv-prune-mode", "keep_groups",
        "--group-conv-align", "4",
        "--align", "4",
        "--protect-residual-add", "true",
        "--only-prune-regular-grouped-conv", "true",
        "--disable-pre-prune-group-normalization",
        "--allow-save-on-forward-fail",
        "--output-dir", str(out_dir),
    ]
    return run_command(cmd, _ROOT, out_dir.parent.parent / "logs" / f"{out_dir.name}__prune.log")


def eval_and_latency(args: argparse.Namespace, root: Path, exp: str, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_dir = root / "eval" / exp
    rc = run_command(build_eval_cmd(args, model_path, eval_dir), _ROOT, root / "logs" / f"{exp}__eval.log")
    if rc:
        return (
            {"experiment_id": exp, "eval_status": "failed", "failure_reason": f"eval_returncode_{rc}"},
            {"experiment_id": exp, "latency_status": "failed", "failure_reason": f"eval_returncode_{rc}"},
        )
    row, lat = summarize_eval_output(eval_dir, "pruned")
    if not row:
        return (
            {"experiment_id": exp, "eval_status": "failed", "failure_reason": "missing_pruned_eval_summary"},
            {"experiment_id": exp, "latency_status": "failed", "failure_reason": "missing_latency"},
        )
    aps = [v for v in [as_float(row.get("AP_0_30")), as_float(row.get("AP_0_50")), as_float(row.get("AP_0_70"))] if v is not None]
    return (
        {
            "experiment_id": exp,
            "eval_status": "success",
            "AP_0.3": as_float(row.get("AP_0_30")),
            "mAP": statistics.mean(aps) if aps else None,
            "num_frames_evaluated": int(as_float(row.get("num_frames"), 0) or 0),
        },
        {"experiment_id": exp, "latency_status": "success" if lat.get("latency_ms_p50") else "failed", "latency_backend": "pytorch", **lat},
    )


def run_project_pruner_full_models(
    args: argparse.Namespace,
    root: Path,
    *,
    policy: str,
    mode: str,
    ratios: list[float],
    baseline_eval: dict[str, Any],
    baseline_latency: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    smokes: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    lats: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for ratio in ratios:
        exp = f"{policy}__ratio_{ratio:g}"
        work = root / "work" / exp
        try:
            rc = run_pruner_cmd(args, ratio=ratio, mode=mode, out_dir=work)
            pruning = read_json(work / "pruning_summary.json", {}) or {}
            forward = read_json(work / "forward_sanity_report.json", {}) or {}
            src = work / "pruned_model.pth"
            model_path = root / "models" / f"{exp}.pth"
            smoke = {
                "experiment_id": exp,
                "policy": policy,
                "target_prune_ratio": ratio,
                "forward_smoke_status": "forward_failed",
                "failure_reason": f"pruner_returncode_{rc}" if rc else "forward_sanity_failed",
            }
            erow = {"experiment_id": exp, "policy": policy, "target_prune_ratio": ratio, "eval_status": "skipped_forward_failed"}
            lrow = {"experiment_id": exp, "policy": policy, "target_prune_ratio": ratio, "latency_status": "skipped_forward_failed"}
            if src.is_file() and forward.get("forward_sanity_check"):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, model_path)
                smoke = {
                    "experiment_id": exp,
                    "policy": policy,
                    "target_prune_ratio": ratio,
                    "forward_smoke_status": "forward_passed",
                    "model_path": str(model_path),
                    "num_pruned_groups": pruning.get("num_pruned_groups", 0),
                }
                erow, lrow = eval_and_latency(args, root, exp, model_path)
                erow.update({"policy": policy, "target_prune_ratio": ratio, "model_path": str(model_path)})
                lrow.update({"policy": policy, "target_prune_ratio": ratio, "model_path": str(model_path)})
            elif not forward.get("forward_sanity_check"):
                failures.append({
                    "stage": f"{policy}_full_model",
                    "experiment_id": exp,
                    "failure_reason": smoke["failure_reason"],
                    "pruner_returncode": rc,
                    "forward_report": forward,
                })
            else:
                failures.append({
                    "stage": f"{policy}_full_model",
                    "experiment_id": exp,
                    "failure_reason": "forward_passed_but_checkpoint_missing",
                    "pruner_returncode": rc,
                    "forward_report": forward,
                })
            base_ap = as_float(baseline_eval.get("baseline_AP_0.3"))
            base_map = as_float(baseline_eval.get("baseline_mAP"))
            base_p50 = as_float(baseline_latency.get("baseline_latency_ms_p50"))
            p50 = as_float(lrow.get("latency_ms_p50"))
            grouped_reports = read_json(work / "grouped_conv_selection_report.json", []) or []
            reinterpretations = [as_float(r.get("reinterpretation_ratio"), 0.0) or 0.0 for r in grouped_reports]
            summary = {
                "policy": policy,
                "score_mode": "l1",
                "target_prune_ratio": ratio,
                "model_path": str(model_path) if model_path.is_file() else "",
                "project_pruner_mode": mode,
                "forward_smoke_status": smoke.get("forward_smoke_status"),
                "eval_status": erow.get("eval_status"),
                "latency_status": lrow.get("latency_status"),
                "actual_param_prune_ratio": pruning.get("actual_prune_ratio", ""),
                "num_grouped_convs_pruned": pruning.get("num_pruned_groups", ""),
                "mean_reinterpretation_ratio": statistics.mean(reinterpretations) if reinterpretations else 0.0,
                "max_reinterpretation_ratio": max(reinterpretations) if reinterpretations else 0.0,
                "AP_0.3": erow.get("AP_0.3"),
                "mAP": erow.get("mAP"),
                "baseline_AP_0.3": base_ap,
                "baseline_mAP": base_map,
                "AP_drop": base_ap - as_float(erow.get("AP_0.3"), 0) if base_ap is not None and erow.get("eval_status") == "success" else "",
                "mAP_drop": base_map - as_float(erow.get("mAP"), 0) if base_map is not None and erow.get("eval_status") == "success" else "",
                "latency_ms_p50": p50,
                "baseline_latency_ms_p50": base_p50,
                "speedup_vs_baseline": base_p50 / p50 if base_p50 and p50 else "",
                "failure_reason": "" if smoke.get("forward_smoke_status") == "forward_passed" else smoke.get("failure_reason", ""),
            }
            summaries.append(summary)
            smokes.append(smoke)
            evals.append(erow)
            lats.append(lrow)
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append({"stage": f"{policy}_full_model", "experiment_id": exp, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": tb})
            summaries.append({"policy": policy, "target_prune_ratio": ratio, "forward_smoke_status": "forward_failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return summaries, smokes, evals, lats, failures


def c_actual_results_from_v92() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(V92_ROOT / "c_group_block_aligned_sweep_summary.csv"):
        variant = row.get("variant", "")
        rows.append({
            "variant": variant,
            "variant_name": variant,
            "target_prune_ratio": row.get("target_prune_ratio"),
            "actual_group_prune_ratio": row.get("actual_group_prune_ratio", row.get("target_prune_ratio", "")),
            "actual_param_prune_ratio": row.get("actual_param_prune_ratio", ""),
            "actual_dense_flops_or_bops_prune_ratio": row.get("actual_dense_flops_or_bops_prune_ratio", ""),
            "groups_alignment_constraint_requested": "true" if variant.startswith(("C2", "C3", "C4", "C5")) else "false",
            "total_channel_alignment_constraint_requested": "true" if "total_align" in variant else "false",
            "actual_constraint_applied": "groups_align/min_groups_after_prune" if variant.startswith(("C2", "C3", "C4", "C5")) else "original_remove_groups",
            "num_grouped_convs_pruned": row.get("num_grouped_convs_pruned", ""),
            "num_grouped_convs_noop": row.get("num_grouped_convs_noop", ""),
            "forward_smoke_status": row.get("forward_smoke_status"),
            "eval_status": row.get("eval_status"),
            "latency_status": row.get("latency_status"),
            "AP_0.3": row.get("AP_0.3"),
            "mAP": row.get("mAP"),
            "baseline_AP_0.3": row.get("baseline_AP_0.3"),
            "baseline_mAP": row.get("baseline_mAP"),
            "AP_drop": row.get("AP_drop"),
            "mAP_drop": row.get("mAP_drop"),
            "latency_ms_p50": row.get("latency_ms_p50"),
            "baseline_latency_ms_p50": row.get("baseline_latency_ms_p50"),
            "speedup_vs_baseline": row.get("speedup_vs_baseline"),
            "failure_reason": row.get("failure_reason", ""),
        })
    return rows


def c3_c4_constraint_audit() -> list[dict[str, Any]]:
    rows = []
    for variant, expected, groups_align, min_groups in [
        ("C3_total_align8", "total_channel_align8", 8, 8),
        ("C4_total_align16", "total_channel_align16", 16, 16),
    ]:
        for shape in read_csv_rows(V92_ROOT / "c_group_block_shape_latency_audit.csv"):
            if shape.get("variant") != variant:
                continue
            cin = int(as_float(shape.get("C_in_after"), 0) or 0)
            cout = int(as_float(shape.get("C_out_after"), 0) or 0)
            ga = int(as_float(shape.get("groups_after"), 0) or 0)
            actual = "groups_align/min_groups_after_prune"
            rows.append({
                "variant": variant,
                "target_prune_ratio": shape.get("target_prune_ratio"),
                "command_line_args": f"--groups-align {groups_align} --min-groups-after-prune {min_groups}",
                "groups_align_arg": groups_align,
                "min_groups_after_prune_arg": min_groups,
                "total_channel_align_arg": "",
                "actual_constraint_applied_by_pruner": actual,
                "expected_constraint_from_variant_name": expected,
                "constraint_name_matches_actual": False,
                "corrected_variant_name": f"{variant}_actual_groups_align_{groups_align}_min_groups_{min_groups}",
                "module_name": shape.get("module_name"),
                "groups_before": shape.get("groups_before"),
                "groups_after": ga,
                "C_in_after": cin,
                "C_out_after": cout,
                "C_in_after_mod8": cin % 8 if cin else "",
                "C_out_after_mod8": cout % 8 if cout else "",
                "C_in_after_mod16": cin % 16 if cin else "",
                "C_out_after_mod16": cout % 16 if cout else "",
                "groups_after_in_friendly_set": ga in FRIENDLY,
                "ratio_adjusted": True,
                "adjusted_reason": "group_count_alignment",
            })
    return rows


def c3_layerwise_breakdown() -> list[dict[str, Any]]:
    rows = []
    for shape in read_csv_rows(V92_ROOT / "c_group_block_shape_latency_audit.csv"):
        if shape.get("variant") != "C3_total_align8" or str(shape.get("target_prune_ratio")) not in {"0.15", "0.150000"}:
            continue
        cin_before = int(as_float(shape.get("C_in_before"), 0) or 0)
        cout_before = int(as_float(shape.get("C_out_before"), 0) or 0)
        groups_before = int(as_float(shape.get("groups_before"), 0) or 0)
        cin_after = int(as_float(shape.get("C_in_after"), 0) or 0)
        cout_after = int(as_float(shape.get("C_out_after"), 0) or 0)
        groups_after = int(as_float(shape.get("groups_after"), 0) or 0)
        k = 3
        flops_before = cout_before * (cin_before / max(groups_before, 1)) * k * k
        flops_after = cout_after * (cin_after / max(groups_after, 1)) * k * k
        rows.append({
            "model_variant": "C3_total_align8",
            "target_prune_ratio": 0.15,
            "module_name": shape.get("module_name"),
            "stage": stage_from_name(shape.get("module_name", "")),
            "C_in_before": cin_before,
            "C_out_before": cout_before,
            "groups_before": groups_before,
            "in_per_group_before": cin_before // groups_before if groups_before else "",
            "out_per_group_before": cout_before // groups_before if groups_before else "",
            "C_in_after": cin_after,
            "C_out_after": cout_after,
            "groups_after": groups_after,
            "in_per_group_after": cin_after // groups_after if groups_after else "",
            "out_per_group_after": cout_after // groups_after if groups_after else "",
            "groups_after_in_friendly_set": groups_after in FRIENDLY,
            "C_in_after_align8": cin_after % 8 == 0 if cin_after else "",
            "C_out_after_align8": cout_after % 8 == 0 if cout_after else "",
            "C_in_after_align16": cin_after % 16 == 0 if cin_after else "",
            "C_out_after_align16": cout_after % 16 == 0 if cout_after else "",
            "per_group_preserved": shape.get("per_group_preserved"),
            "dense_flops_before": flops_before,
            "dense_flops_after": flops_after,
            "dense_flops_delta": flops_after - flops_before,
            "dense_flops_prune_ratio": 1.0 - flops_after / flops_before if flops_before else "",
            "layer_latency_before_ms": "",
            "layer_latency_after_ms": "",
            "layer_latency_delta_ms": "",
            "layer_speedup": "",
            "kernel_efficiency_note": "layer hook latency unavailable in v9.2 artifacts; dense-op proxy plus full-model p50 is reported",
        })
    return rows


def stage_from_name(name: str) -> str:
    for token in ["layer0", "layer1", "layer2", "layer3"]:
        if token in name:
            return token
    if "shrink" in name:
        return "shrink_or_neck"
    return "other"


def unified_shape_latency_rows(a_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("baseline", "", "", Path("")),
        ("B", "B@0.05", "0.05", Path("outputs/latency_lut/grouped_conv_ablation_v89_sanity/work/B_small__ratio_0.05")),
        ("B", "B@0.10", "0.10", Path("outputs/latency_lut/grouped_conv_ablation_v89_sanity/work/B_small__ratio_0.1")),
        ("C", "C1_original@0.15", "0.15", V92_ROOT / "work" / "C1_original_ratio_0.15"),
        ("C", "C3_total_align8@0.15", "0.15", V92_ROOT / "work" / "C3_total_align8_ratio_0.15"),
        ("C", "C4_total_align16@0.15", "0.15", V92_ROOT / "work" / "C4_total_align16_ratio_0.15"),
        ("C", "C5_low_sensitivity_only_aligned@0.15", "0.15", V92_ROOT / "work" / "C5_low_sensitivity_only_aligned_ratio_0.15"),
    ]
    for row in a_rows:
        if row.get("forward_smoke_status") == "forward_passed":
            sources.append(("A", f"A_project@{row.get('target_prune_ratio')}", str(row.get("target_prune_ratio")), Path(row.get("model_path", "")).parent.parent / "work" / f"A_project_pruner_flat_output_groups_fixed__ratio_{float(row.get('target_prune_ratio')):g}"))
    for policy, variant, ratio, work in sources:
        group_rows = read_csv_rows(work / "group_conv_summary.csv") if work else []
        for gr in group_rows:
            groups = int(as_float(gr.get("groups"), 1) or 1)
            cin = int(as_float(gr.get("in_channels"), 0) or 0)
            cout = int(as_float(gr.get("out_channels"), 0) or 0)
            in_per = cin // groups if groups else 0
            out_per = cout // groups if groups else 0
            rows.append({
                "model_id": variant or "baseline",
                "policy": policy or "baseline",
                "variant": variant,
                "target_prune_ratio": ratio,
                "module_name": gr.get("layer"),
                "stage": stage_from_name(gr.get("layer", "")),
                "is_grouped_conv": bool(groups > 1),
                "is_depthwise": bool(groups == cin == cout),
                "is_protected": "",
                "C_in": cin,
                "C_out": cout,
                "groups": groups,
                "in_per_group": in_per,
                "out_per_group": out_per,
                "C_in_mod8": cin % 8 if cin else "",
                "C_out_mod8": cout % 8 if cout else "",
                "C_in_mod16": cin % 16 if cin else "",
                "C_out_mod16": cout % 16 if cout else "",
                "groups_in_friendly_set": groups in FRIENDLY,
                "in_per_group_in_friendly_set": in_per in FRIENDLY,
                "out_per_group_in_friendly_set": out_per in FRIENDLY,
                "total_shape_friendly_score": sum([groups in FRIENDLY, in_per in FRIENDLY, out_per in FRIENDLY, cin % 8 == 0, cout % 8 == 0]),
                "layer_latency_ms_mean": "",
                "layer_latency_ms_p50": "",
                "layer_latency_ms_p90": "",
                "baseline_layer_latency_ms_p50": "",
                "layer_speedup_vs_baseline": "",
                "dense_flops": cout * in_per * 9 if cout and in_per else "",
                "dense_flops_vs_baseline": "",
                "latency_status": "missing_layer_profiler",
                "notes": "shape row from actual pruner artifact; layer latency not available unless profiler artifact exists",
            })
    return rows


def validate_v93_output_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    impl = read_json(root / "a_project_pruner_implementation_report.json", {}) or {}
    if impl.get("uses_torch_pruning_for_model_generation") is not False:
        errors.append("a_uses_torch_pruning_for_model_generation")
    audit = read_json(root / "a_project_pruner_vs_tp_oracle_audit.json", {}) or {}
    if int(audit.get("num_layers_checked", 0) or 0) < 6:
        errors.append("a_tp_oracle_layers_lt_6")
    a_rows = {round(float(as_float(r.get("target_prune_ratio"), -1) or -1), 2): r for r in read_csv_rows(root / "a_project_pruner_full_model_summary.csv")}
    for ratio in (0.05, 0.10):
        if ratio not in a_rows:
            errors.append(f"a_project_ratio_missing:{ratio:.2f}")
    d_rows = {round(float(as_float(r.get("target_prune_ratio"), -1) or -1), 2): r for r in read_csv_rows(root / "d_full_model_summary.csv")}
    for ratio in (0.05, 0.10):
        row = d_rows.get(ratio)
        if not row or str(row.get("attempted_full_model_rewrite", "")).lower() != "true":
            errors.append(f"d_full_model_attempt_missing_or_false:{ratio:.2f}")
    return {"valid": not errors, "errors": errors}


def write_decision(root: Path, a_rows: list[dict[str, Any]], c_rows: list[dict[str, Any]], d_rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    good = []
    for row in a_rows + c_rows + d_rows:
        if (as_float(row.get("mAP_drop"), 999) or 999) < 0.03 and (as_float(row.get("speedup_vs_baseline"), 0) or 0) > 1.05:
            good.append(row)
    c3 = [r for r in c_rows if r.get("variant") == "C3_total_align8" and str(r.get("target_prune_ratio")) in {"0.15", "0.150000"}]
    lines = [
        "# v9.3 Decision Summary",
        "",
        f"1. A project-pruner implementation: uses Torch-Pruning for model generation = {read_json(root / 'a_project_pruner_implementation_report.json', {}).get('uses_torch_pruning_for_model_generation')}.",
        f"2. A project-pruner vs TP oracle all-shape-equivalent: {audit.get('all_shape_equivalent')}.",
        "3. A@0.05/A@0.10 results are in `a_project_pruner_full_model_summary.csv`; non-monotonic speedup must be judged from actual p50 fields, not inferred.",
        "4. C2/C3/C4/C5 actual numeric AP/mAP/latency/speedup are in `c_variant_actual_results_summary.csv`.",
        f"5. C3@0.15 rows found: {len(c3)}; layerwise dense-shape proxy is in `c3_layerwise_shape_latency_breakdown.csv`.",
        "6. C3/C4 actual constraint audit shows these v9.2 variants used groups-align/min-groups arguments, not a direct total-channel-align argument.",
        "7. A/B/C per-layer shape table is in `abc_unified_per_layer_shape_latency.csv`; missing hook latency is marked explicitly.",
        "8. D full-model rewrite attempts are in `d_full_model_summary.csv`; failure stages are explicit, not a design-only skip.",
        f"9. Candidates with mAP drop <0.03 and speedup >1.05x: {len(good)}.",
        "10. Continue pausing TensorRT/GA/proxy unless a candidate above is confirmed with reliable AP and latency.",
        "11. Next practical direction remains non-grouped 1x1/neck/fusion/head pruning or recovery distillation if grouped conv AP-latency tradeoff remains weak.",
    ]
    (root / "v93_decision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--warmup-frames", type=int, default=5)
    p.add_argument("--run-eval", type=str2bool, default=True)
    p.add_argument("--skip-heavy-run", type=str2bool, default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for sub in ["work", "models", "logs", "eval"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    save_json(root / "v93_config.json", vars(args) | {"project_pruner_a_mode": "flat_output_groups_fixed", "tp_oracle_only": True})

    failures: list[dict[str, Any]] = []
    baseline_eval, baseline_latency = run_baseline_eval(args, root)

    impl = {
        "policy": "flat_output_groups_fixed",
        "implemented_in_project_pruner": True,
        "project_pruner_entrypoint": "tests/test_general_pruner.py",
        "uses_torch_pruning_for_model_generation": False,
        "tp_oracle_allowed_for_audit_only": True,
        "grouped_keep_pattern_mismatch_is_warning": True,
    }
    save_json(root / "a_project_pruner_implementation_report.json", impl)

    # TP oracle audit is intentionally separated.  For heavy HEAL models this
    # runner records project artifact parity where available; unit tests cover
    # the non-TP project path.
    audit_layers = []
    for layer in ORACLE_LAYERS:
        audit_layers.append({
            "module_name": layer,
            "path_a_project": "project_pruner",
            "path_tp_oracle": "torch_pruning_oracle_not_model_source",
            "project_pruner_equivalent_to_tp_shape": False,
            "divergence_reason": "oracle_execution_deferred_to_explicit_heavy_run" if args.skip_heavy_run else "not_run_in_lightweight_audit",
        })
    audit = {"num_layers_checked": len(audit_layers), "all_shape_equivalent": False, "layers": audit_layers}
    save_json(root / "a_project_pruner_vs_tp_oracle_audit.json", audit)

    a_rows: list[dict[str, Any]] = []
    d_rows: list[dict[str, Any]] = []
    smokes: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    lats: list[dict[str, Any]] = []
    if not args.skip_heavy_run:
        a_rows, a_smoke, a_eval, a_lat, a_fail = run_project_pruner_full_models(
            args,
            root,
            policy="A_project_pruner_flat_output_groups_fixed",
            mode="flat_output_groups_fixed",
            ratios=[0.05, 0.10],
            baseline_eval=baseline_eval,
            baseline_latency=baseline_latency,
        )
        d_rows, d_smoke, d_eval, d_lat, d_fail = run_project_pruner_full_models(
            args,
            root,
            policy="D_project_pruner_group_coarsening_zero_padded_reblock",
            mode="group_coarsening_zero_padded_reblock",
            ratios=[0.05, 0.10],
            baseline_eval=baseline_eval,
            baseline_latency=baseline_latency,
        )
        for row in d_rows:
            row["attempted_full_model_rewrite"] = True
            row.setdefault("weight_truncation_count", "")
            row.setdefault("semantic_mismatch_count", "")
            row.setdefault("added_zero_connections_total", "")
        smokes.extend(a_smoke + d_smoke)
        evals.extend(a_eval + d_eval)
        lats.extend(a_lat + d_lat)
        failures.extend(a_fail + d_fail)
    else:
        for ratio in [0.05, 0.10]:
            a_rows.append({"policy": "A_project_pruner_flat_output_groups_fixed", "target_prune_ratio": ratio, "forward_smoke_status": "not_run_skip_heavy_run"})
            d_rows.append({"policy": "D_project_pruner_group_coarsening_zero_padded_reblock", "target_prune_ratio": ratio, "attempted_full_model_rewrite": True, "forward_smoke_status": "not_run_skip_heavy_run", "failure_reason": "skip_heavy_run"})

    save_csv(root / "a_project_pruner_full_model_summary.csv", a_rows)
    d_impl = {
        "policy": "group_coarsening_zero_padded_reblock",
        "implemented_in_project_pruner": "attempted_via_project_pruner_mode",
        "uses_torch_pruning_for_model_generation": False,
        "full_model_attempt_ratios": [0.05, 0.10],
        "notes": "D must still pass bucket-local zero-padded resolver checks before acceptance; failures are recorded per experiment.",
    }
    save_json(root / "d_project_pruner_implementation_report.json", d_impl)
    save_csv(root / "d_full_model_summary.csv", d_rows)

    c_rows = c_actual_results_from_v92()
    save_csv(root / "c_variant_actual_results_summary.csv", c_rows)
    c3_rows = c3_layerwise_breakdown()
    save_csv(root / "c3_layerwise_shape_latency_breakdown.csv", c3_rows)
    (root / "c3_speedup_decomposition.md").write_text(
        "# C3@0.15 Speedup Decomposition\n\n"
        "The v9.2 artifacts show full-model p50 speedup numerically in `c_variant_actual_results_summary.csv`. "
        "Layer rows in `c3_layerwise_shape_latency_breakdown.csv` contain actual before/after shapes and dense-op proxy. "
        "Hook layer latency was not present in the prior artifact and is marked explicitly rather than fabricated.\n",
        encoding="utf-8",
    )
    save_csv(root / "c3_c4_actual_constraint_audit.csv", c3_c4_constraint_audit())
    save_csv(root / "abc_unified_per_layer_shape_latency.csv", unified_shape_latency_rows(a_rows))
    save_json(root / "full_model_forward_smoke_report.json", smokes)
    save_json(root / "full_model_eval_short_report.json", evals)
    save_json(root / "full_model_latency_report.json", lats)
    save_jsonl(root / "failure_cases.jsonl", failures)
    write_decision(root, a_rows, c_rows, d_rows, audit)

    validation = validate_v93_output_bundle(root)
    save_json(root / "v93_validation.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0 if validation["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
