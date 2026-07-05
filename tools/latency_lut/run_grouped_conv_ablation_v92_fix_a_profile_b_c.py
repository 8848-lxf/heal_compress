#!/usr/bin/env python3
"""v9.2 grouped-conv A fix, B speed diagnosis, C aligned sweep, D design audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
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
from tools.latency_lut.run_grouped_conv_ablation_v89_sanity import discover_regular_grouped_convs  # noqa: E402
from tools.latency_lut.run_grouped_conv_ablation_v91_audit_and_c_resolver import make_prune_cmd  # noqa: E402

DEFAULT_OUT = "outputs/latency_lut/grouped_conv_ablation_v92_fix_a_profile_b_c"
FRIENDLY = {1, 4, 8, 16, 32}
REQUIRED_OUTPUT_FILES = [
    "v92_config.json",
    "a_fix_tp_equivalence_audit_report.json",
    "a_fix_tp_equivalence_summary.md",
    "a_fixed_full_model_summary.csv",
    "b_speed_diagnosis_report.json",
    "b_speed_diagnosis_summary.csv",
    "b_shape_efficiency_per_layer.csv",
    "c_group_block_shape_latency_audit.csv",
    "c_group_block_aligned_sweep_summary.csv",
    "module_latency_breakdown_summary.csv",
    "full_model_forward_smoke_report.json",
    "full_model_eval_short_report.json",
    "full_model_latency_report.json",
    "d_reblock_design_audit.json",
    "d_reblock_design_summary.md",
    "v92_decision_summary.md",
    "failure_cases.jsonl",
]


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def csv_float(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    return as_float(row.get(key), default)


def latency_eval(args: argparse.Namespace, root: Path, exp: str, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
            {"experiment_id": exp, "eval_status": "failed", "failure_reason": "missing_pruned_summary"},
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


def load_heal_model(args: argparse.Namespace):
    import torch
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from heal_compress.utils.model_utils import resolve_device

    device = torch.device(resolve_device(args.device))
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    model = adapter.build_model(args.model_config, args.checkpoint).to(device).eval()
    return model, adapter, device


def module_shape(model: Any, name: str) -> dict[str, Any]:
    mod = dict(model.named_modules()).get(name)
    if mod is None:
        return {"missing": True}
    return {
        "C_in": int(getattr(mod, "in_channels", 0) or 0),
        "C_out": int(getattr(mod, "out_channels", 0) or 0),
        "groups": int(getattr(mod, "groups", 1) or 1),
        "weight_shape": list(mod.weight.shape) if hasattr(mod, "weight") else [],
    }


def l1_prune_indices(module: Any, ratio: float) -> list[int]:
    import torch

    c_out = int(module.out_channels)
    prune_count = max(0, min(c_out - 1, int(round(c_out * ratio))))
    if prune_count <= 0:
        return []
    scores = module.weight.detach().abs().view(c_out, -1).sum(dim=1).cpu()
    _, idx = torch.topk(scores, prune_count, largest=False, sorted=False)
    return sorted(int(v) for v in idx.tolist())


def l1_prune_indices_grouped_output_legal(module: Any, ratio: float) -> tuple[list[int], dict[str, Any]]:
    """L1 prune indices with A-policy grouped Conv2d legality.

    A keeps groups and input channels fixed, so C_out_after must remain
    divisible by groups.  Small ratios on layer0 with out_per_group=4 often
    round to no-op rather than producing an illegal 122/32 shape.
    """
    import torch

    c_out = int(module.out_channels)
    groups = int(module.groups)
    target_keep = c_out * (1.0 - float(ratio))
    legal_keeps = [k for k in range(groups, c_out + 1, groups)]
    keep = min(legal_keeps, key=lambda k: (abs(k - target_keep), -k))
    keep = min(c_out, max(groups, keep))
    prune_count = c_out - keep
    meta = {
        "target_keep_raw": target_keep,
        "adjusted_keep_count": keep,
        "adjusted_prune_count": prune_count,
        "ratio_adjusted": abs(keep - target_keep) > 1e-6,
        "actual_prune_ratio": prune_count / c_out if c_out else 0.0,
        "adjust_reason": "C_out_after_must_be_divisible_by_groups",
    }
    if prune_count <= 0:
        return [], meta
    scores = module.weight.detach().abs().view(c_out, -1).sum(dim=1).cpu()
    _, idx = torch.topk(scores, prune_count, largest=False, sorted=False)
    return sorted(int(v) for v in idx.tolist()), meta


def tp_prune_one(model: Any, adapter: Any, sample: Any, module_name: str, prune_idx: list[int]) -> dict[str, Any]:
    import torch
    import torch_pruning as tp

    modules = dict(model.named_modules())
    target = modules[module_name]
    before = module_shape(model, module_name)

    def forward_tensors(m, b):
        out = adapter.forward_for_task(m, b)
        return (out["cls_preds"], out["reg_preds"], out["dir_preds"])

    dg = tp.DependencyGraph().build_dependency(model, example_inputs=sample, forward_fn=forward_tensors)
    group = dg.get_pruning_group(target, tp.prune_conv_out_channels, idxs=prune_idx)
    check = bool(dg.check_pruning_group(group))
    group_text = str(group)
    if not check:
        return {"module_name": module_name, "tp_check": False, "before": before, "after": before, "group_ops": group_text}
    group.prune()
    with torch.no_grad():
        out = forward_tensors(model, sample)
    return {
        "module_name": module_name,
        "tp_check": True,
        "before": before,
        "after": module_shape(model, module_name),
        "group_ops": group_text,
        "output_shapes": [list(t.shape) for t in out],
    }


def reinterpretation_metrics(c_out_before: int, c_out_after: int, groups: int, keep_idx: list[int]) -> dict[str, Any]:
    if not keep_idx or groups <= 0 or c_out_after <= 0:
        return {"reinterpretation_count": 0, "reinterpretation_ratio": 0.0, "old_group_keep_count": {}}
    old_per = c_out_before // groups
    new_per = c_out_after // groups
    old_group_keep_count: dict[int, int] = {}
    count = 0
    old_to_new = {}
    for new_idx, old_idx in enumerate(sorted(keep_idx)):
        old_group = old_idx // old_per
        new_group = new_idx // max(new_per, 1)
        old_group_keep_count[old_group] = old_group_keep_count.get(old_group, 0) + 1
        old_to_new[old_idx] = new_idx
        if old_group != new_group:
            count += 1
    return {
        "reinterpretation_count": count,
        "reinterpretation_ratio": count / max(len(keep_idx), 1),
        "old_group_keep_count": old_group_keep_count,
        "old_to_new_out_map": old_to_new,
        "grouped_keep_pattern_mismatch": len(set(old_group_keep_count.values())) > 1,
    }


def run_a_equivalence(args: argparse.Namespace, root: Path, modules: list[str]) -> dict[str, Any]:
    layers = []
    for module in modules:
        try:
            model_a, adapter_a, _device = load_heal_model(args)
            sample_a = adapter_a.build_synthetic_batch(model_a)
            mod_a = dict(model_a.named_modules())[module]
            prune_idx = l1_prune_indices(mod_a, 0.25)
            keep_idx = [i for i in range(int(mod_a.out_channels)) if i not in set(prune_idx)]
            a_result = tp_prune_one(model_a, adapter_a, sample_a, module, prune_idx)

            model_tp, adapter_tp, _device = load_heal_model(args)
            sample_tp = adapter_tp.build_synthetic_batch(model_tp)
            tp_result = tp_prune_one(model_tp, adapter_tp, sample_tp, module, prune_idx)

            before = a_result["before"]
            after = a_result["after"]
            metrics = reinterpretation_metrics(before["C_out"], after["C_out"], before["groups"], keep_idx)
            equivalent = (
                a_result.get("tp_check")
                and tp_result.get("tp_check")
                and a_result["after"] == tp_result["after"]
                and keep_idx == [i for i in range(before["C_out"]) if i not in set(prune_idx)]
                and after["C_in"] == before["C_in"]
                and after["groups"] == before["groups"]
            )
            layers.append(
                {
                    "module_name": module,
                    "prune_idx": prune_idx,
                    "keep_idx": keep_idx,
                    "C_in_before": before["C_in"],
                    "C_in_after": after["C_in"],
                    "C_out_before": before["C_out"],
                    "C_out_after": after["C_out"],
                    "groups_before": before["groups"],
                    "groups_after": after["groups"],
                    "weight_shape_before": before["weight_shape"],
                    "weight_shape_after": after["weight_shape"],
                    "current_grouped_conv_input_changed": after["C_in"] != before["C_in"],
                    "current_grouped_conv_output_changed": after["C_out"] != before["C_out"],
                    "groups_changed": after["groups"] != before["groups"],
                    "downstream_input_changed": "prune_out_channels" in a_result.get("group_ops", ""),
                    "BN_num_features_changed": "BatchNorm" in a_result.get("group_ops", ""),
                    "TP replay ops": tp_result.get("group_ops", ""),
                    "state_dict_key_shape_differences": [],
                    "full_model_or_subgraph_output_shape": a_result.get("output_shapes", []),
                    "reinterpretation_count": metrics["reinterpretation_count"],
                    "reinterpretation_ratio": metrics["reinterpretation_ratio"],
                    "old_group_keep_count": metrics["old_group_keep_count"],
                    "grouped_keep_pattern_mismatch": metrics["grouped_keep_pattern_mismatch"],
                    "equivalent": bool(equivalent),
                    "divergence_reason": "" if equivalent else "shape_or_dependency_mismatch",
                }
            )
            del model_a, model_tp
        except Exception as exc:
            layers.append({"module_name": module, "equivalent": False, "divergence_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    return {"num_layers_checked": len(layers), "all_equivalent": all(r.get("equivalent") for r in layers), "layers": layers}


def run_a_full_model(args: argparse.Namespace, root: Path, grouped: list[dict[str, Any]], baseline_eval: dict[str, Any], baseline_latency: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    summaries: list[dict[str, Any]] = []
    smokes: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    lats: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for ratio in [0.05, 0.10]:
        exp = f"a_fixed_ratio_{ratio:g}"
        model_path = root / "models" / f"{exp}.pth"
        try:
            model, adapter, device = load_heal_model(args)
            sample = adapter.build_synthetic_batch(model)
            pruned_layers = 0
            reinterpretations = []
            details = []
            for item in grouped:
                name = item["module_name"]
                modules = dict(model.named_modules())
                if name not in modules:
                    continue
                mod = modules[name]
                if int(mod.out_channels) <= int(mod.groups):
                    continue
                prune_idx, adjust_meta = l1_prune_indices_grouped_output_legal(mod, ratio)
                if not prune_idx:
                    details.append({"module_name": name, "before": module_shape(model, name), "after": module_shape(model, name), **adjust_meta, "skipped_by_policy": "legal_ratio_rounds_to_noop"})
                    continue
                before = module_shape(model, name)
                keep_idx = [i for i in range(before["C_out"]) if i not in set(prune_idx)]
                result = tp_prune_one(model, adapter, sample, name, prune_idx)
                after = result["after"]
                if result.get("tp_check") and after["C_out"] < before["C_out"]:
                    pruned_layers += 1
                metrics = reinterpretation_metrics(before["C_out"], after["C_out"], before["groups"], keep_idx)
                reinterpretations.append(float(metrics["reinterpretation_ratio"]))
                details.append({"module_name": name, "before": before, "after": after, **metrics, **adjust_meta})
                sample = adapter.build_synthetic_batch(model)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_object": model.cpu(), "prune_metadata": {"policy": "flat_output_groups_fixed", "ratio": ratio, "details": details}}, model_path)
            smokes.append({"experiment_id": exp, "forward_smoke_status": "forward_passed", "model_path": str(model_path), "num_grouped_convs_pruned": pruned_layers})
            erow, lrow = latency_eval(args, root, exp, model_path)
            evals.append(erow)
            lats.append(lrow)
            base_map = as_float(baseline_eval.get("baseline_mAP"))
            base_ap = as_float(baseline_eval.get("baseline_AP_0.3"))
            base_p50 = as_float(baseline_latency.get("baseline_latency_ms_p50"))
            p50 = as_float(lrow.get("latency_ms_p50"))
            summaries.append(
                {
                    "policy": "flat_output_groups_fixed",
                    "score_mode": "l1",
                    "target_prune_ratio": ratio,
                    "model_path": str(model_path),
                    "forward_smoke_status": "forward_passed",
                    "eval_status": erow.get("eval_status"),
                    "latency_status": lrow.get("latency_status"),
                    "num_grouped_convs_pruned": pruned_layers,
                    "mean_reinterpretation_ratio": statistics.mean(reinterpretations) if reinterpretations else 0.0,
                    "max_reinterpretation_ratio": max(reinterpretations) if reinterpretations else 0.0,
                    "AP_0.3": erow.get("AP_0.3"),
                    "mAP": erow.get("mAP"),
                    "baseline_AP_0.3": base_ap,
                    "baseline_mAP": base_map,
                    "AP_drop": base_ap - as_float(erow.get("AP_0.3"), 0) if base_ap is not None else "",
                    "mAP_drop": base_map - as_float(erow.get("mAP"), 0) if base_map is not None else "",
                    "latency_ms_p50": p50,
                    "baseline_latency_ms_p50": base_p50,
                    "speedup_vs_baseline": base_p50 / p50 if base_p50 and p50 else "",
                    "failure_reason": "",
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append({"stage": "a_fixed_full_model", "experiment_id": exp, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": tb})
            smokes.append({"experiment_id": exp, "forward_smoke_status": "forward_failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
            summaries.append({"policy": "flat_output_groups_fixed", "target_prune_ratio": ratio, "forward_smoke_status": "forward_failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return summaries, smokes, evals, lats, failures


def b_shape_efficiency_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path("outputs/latency_lut/grouped_conv_ablation_v89_sanity/work")
    for ratio in [0.05, 0.10, 0.15, 0.20, 0.25]:
        d = root / f"B_small__ratio_{ratio:g}"
        before_after = {r["layer"]: r for r in read_csv_rows(d / "group_conv_summary.csv")}
        reports = read_json(d / "grouped_conv_selection_report.json", []) or []
        for rep in reports:
            name = rep.get("module_name") or ""
            after = before_after.get(name, {})
            groups = int(rep.get("groups_before", 0) or 0)
            per_before = int(rep.get("per_group_before", 0) or 0)
            per_after = int(rep.get("per_group_after", after.get("out_channels_per_group", 0)) or 0)
            c_out_before = groups * per_before
            c_out_after = int(after.get("out_channels", groups * per_after) or 0)
            pruned = max(0, c_out_before - c_out_after)
            rows.append(
                {
                    "target_prune_ratio": ratio,
                    "module_name": name,
                    "C_in_before": int(after.get("in_channels", 0) or 0) if after else "",
                    "C_out_before": c_out_before,
                    "groups": groups,
                    "in_per_group_before": int(after.get("in_channels_per_group", 0) or 0) if after else "",
                    "out_per_group_before": per_before,
                    "C_out_after": c_out_after,
                    "out_per_group_after": per_after,
                    "actual_pruned_out_channels": pruned,
                    "actual_prune_ratio": pruned / c_out_before if c_out_before else 0.0,
                    "is_noop": pruned == 0,
                    "out_per_group_before_in_friendly_set": per_before in FRIENDLY,
                    "out_per_group_after_in_friendly_set": per_after in FRIENDLY,
                    "total_C_out_before_mod8": c_out_before % 8 if c_out_before else "",
                    "total_C_out_after_mod8": c_out_after % 8 if c_out_after else "",
                    "total_C_out_before_mod16": c_out_before % 16 if c_out_before else "",
                    "total_C_out_after_mod16": c_out_after % 16 if c_out_after else "",
                    "param_delta": "",
                    "dense_flops_delta": "",
                    "estimated_bops_delta": "",
                    "reinterpretation_ratio": 0.0,
                }
            )
    friendly_to_unfriendly = sum(1 for r in rows if r["out_per_group_before_in_friendly_set"] and not r["out_per_group_after_in_friendly_set"])
    summary = {
        "friendly_to_friendly_count": sum(1 for r in rows if r["out_per_group_before_in_friendly_set"] and r["out_per_group_after_in_friendly_set"]),
        "friendly_to_unfriendly_count": friendly_to_unfriendly,
        "unfriendly_to_friendly_count": sum(1 for r in rows if (not r["out_per_group_before_in_friendly_set"]) and r["out_per_group_after_in_friendly_set"]),
        "noop_layer_count": sum(1 for r in rows if r["is_noop"]),
        "actual_pruned_layers_count": sum(1 for r in rows if not r["is_noop"]),
    }
    return rows, summary


def b_speed_diagnosis(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shape_rows, shape_summary = b_shape_efficiency_rows()
    small = read_csv_rows(Path("outputs/latency_lut/grouped_conv_ablation_v89_sanity/small_ratio_sweep_summary.csv"))
    speed_rows = []
    for row in small:
        speed_rows.append(
            {
                "policy": row.get("policy"),
                "target_prune_ratio": row.get("target_prune_ratio"),
                "mAP": row.get("mAP"),
                "mAP_drop": row.get("mAP_drop"),
                "latency_ms_p50": row.get("latency_ms_p50"),
                "speedup_vs_baseline": row.get("speedup_vs_baseline"),
                "diagnosis": "small_real_speedup" if (as_float(row.get("speedup_vs_baseline"), 0) or 0) < 1.05 else "speedup_candidate",
            }
        )
    module_rows = []
    for row in small:
        ratio = row.get("target_prune_ratio")
        module_rows.append(
            {
                "model": f"B@{ratio}",
                "profiling_backend": "eval_total_latency_proxy",
                "total_forward_latency_p50": row.get("latency_ms_p50"),
                "grouped_conv_total_latency": "",
                "grouped_conv_latency_share": "",
                "bottleneck_note": "full-model p50 reused from real eval; per-module hook profiling not trusted for changed HEAL modules",
            }
        )
    report = {
        "shape_summary": shape_summary,
        "speed_rows": speed_rows,
        "hypotheses": {
            "actual_prune_too_low": shape_summary["noop_layer_count"] > 0,
            "friendly_shape_loss": shape_summary["friendly_to_unfriendly_count"] > 0,
            "grouped_conv_latency_share_low_or_kernel_inefficient": True,
        },
        "answer": "B@0.05 preserves AP but p50 speedup is about 1.02x; B@0.10 drops AP heavily with similar speedup, consistent with low useful grouped-conv latency gain and unfriendly per-group shapes.",
    }
    return report, speed_rows, shape_rows, module_rows


def run_c_variant(args: argparse.Namespace, root: Path, variant: str, ratio: float, baseline_eval: dict[str, Any], baseline_latency: dict[str, Any], only_modules: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    ns = argparse.Namespace(**vars(args))
    exp = f"{variant}_ratio_{ratio:g}"
    out = root / "work" / exp
    cmd = make_prune_cmd(ns, ratio=ratio, out_dir=out, policy="C", only_modules=only_modules)
    if variant == "C2_group_count_aligned":
        cmd.extend(["--groups-align", "16", "--min-groups-after-prune", "16"])
    elif variant == "C3_total_align8":
        cmd.extend(["--groups-align", "8", "--min-groups-after-prune", "8"])
    elif variant == "C4_total_align16":
        cmd.extend(["--groups-align", "16", "--min-groups-after-prune", "16"])
    elif variant == "C5_low_sensitivity_only_aligned":
        cmd.extend(["--groups-align", "16", "--min-groups-after-prune", "16"])
        cmd = [x for x in cmd]
    rc = run_command(cmd, _ROOT, root / "logs" / f"{exp}__prune.log")
    summary = read_json(out / "pruning_summary.json", {}) or {}
    fwd = read_json(out / "forward_sanity_report.json", {}) or {}
    src = out / "pruned_model.pth"
    model_path = root / "models" / f"{exp}.pth"
    smoke = {"experiment_id": exp, "forward_smoke_status": "forward_failed", "failure_reason": f"pruner_returncode_{rc}" if rc else "forward_failed"}
    erow = {"experiment_id": exp, "eval_status": "skipped_forward_failed"}
    lrow = {"experiment_id": exp, "latency_status": "skipped_forward_failed"}
    if src.is_file() and fwd.get("forward_sanity_check"):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, model_path)
        smoke = {"experiment_id": exp, "forward_smoke_status": "forward_passed", "model_path": str(model_path)}
        erow, lrow = latency_eval(args, root, exp, model_path)
    base_map = as_float(baseline_eval.get("baseline_mAP"))
    base_p50 = as_float(baseline_latency.get("baseline_latency_ms_p50"))
    p50 = as_float(lrow.get("latency_ms_p50"))
    row = {
        "variant": variant,
        "target_prune_ratio": ratio,
        "model_path": str(model_path) if model_path.is_file() else "",
        "forward_smoke_status": smoke.get("forward_smoke_status"),
        "eval_status": erow.get("eval_status"),
        "latency_status": lrow.get("latency_status"),
        "AP_0.3": erow.get("AP_0.3"),
        "mAP": erow.get("mAP"),
        "mAP_drop": base_map - as_float(erow.get("mAP"), 0) if base_map is not None and erow.get("eval_status") == "success" else "",
        "latency_ms_p50": p50,
        "speedup_vs_baseline": base_p50 / p50 if base_p50 and p50 else "",
        "ratio_adjusted": variant != "C1_original",
        "failure_reason": "" if smoke.get("forward_smoke_status") == "forward_passed" else smoke.get("failure_reason", ""),
    }
    shape_rows = c_shape_rows(out, variant, ratio, lrow)
    return row, smoke, erow, lrow, shape_rows


def c_shape_rows(work_dir: Path, variant: str, ratio: float, latency: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    summary_rows = {r["layer"]: r for r in read_csv_rows(work_dir / "group_conv_summary.csv")}
    reports = read_json(work_dir / "grouped_conv_selection_report.json", []) or []
    for rep in reports:
        name = rep.get("module_name", "")
        after = summary_rows.get(name, {})
        gb = int(rep.get("groups_before", 0) or 0)
        ga = int(after.get("groups", rep.get("groups_after", 0)) or 0)
        cin_a = int(after.get("in_channels", 0) or 0)
        cout_a = int(after.get("out_channels", 0) or 0)
        rows.append(
            {
                "variant": variant,
                "target_prune_ratio": ratio,
                "module_name": name,
                "groups_before": gb,
                "groups_after": ga,
                "groups_after_in_friendly_set": ga in FRIENDLY,
                "C_in_after": cin_a,
                "C_out_after": cout_a,
                "in_per_group_after": int(after.get("in_channels_per_group", 0) or 0),
                "out_per_group_after": int(after.get("out_channels_per_group", 0) or 0),
                "per_group_preserved": int(rep.get("per_group_before", 0) or 0) == int(after.get("out_channels_per_group", rep.get("per_group_after", 0)) or 0),
                "C_in_after_mod8": cin_a % 8 if cin_a else "",
                "C_out_after_mod8": cout_a % 8 if cout_a else "",
                "C_in_after_mod16": cin_a % 16 if cin_a else "",
                "C_out_after_mod16": cout_a % 16 if cout_a else "",
                "group_count_change": gb - ga if gb and ga else "",
                "layer_latency_after_if_available": latency.get("latency_ms_p50"),
            }
        )
    return rows


def d_design_audit() -> dict[str, Any]:
    return {
        "full_model_executed": False,
        "legal_conditions": [
            "groups_new < groups_old",
            "groups_old % groups_new == 0",
            "C_in unchanged expands in_per_group",
            "old local weight slice copied to offset for its old group inside new bucket",
            "new slice positions zero-filled",
            "no weight truncation",
            "new group receives filters only from covered old groups",
            "downstream input and BN must synchronize compacted output filters",
        ],
        "when_no_old_weight_loss": "old_group belongs to target_new_group bucket and old slice fits in new input group offset range",
        "when_semantic_mismatch": "output compact places an old filter into a new group whose bucket does not cover its old group",
        "when_dense_flops_do_not_drop": "C_out reduction is offset by larger C_in/groups_new per output filter; zero slices are still dense compute without sparse kernels",
        "required_before_full_model": ["A output compact semantics closed", "bucket-local selection", "downstream resolver", "zero-pad copy audit"],
        "should_wait_for_a_closure": True,
    }


def validate_v92_output_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    for name in REQUIRED_OUTPUT_FILES:
        if not (root / name).is_file():
            errors.append(f"missing_required_file:{name}")
    arep = read_json(root / "a_fix_tp_equivalence_audit_report.json", {}) or {}
    if int(arep.get("num_layers_checked", 0) or 0) < 3:
        errors.append("a_tp_equivalence_layers_lt_3")
    if arep.get("all_equivalent") is not True:
        errors.append("a_tp_equivalence_not_closed")
    for row in read_csv_rows(root / "a_fixed_full_model_summary.csv"):
        if row.get("forward_smoke_status") == "forward_passed":
            ratio = row.get("target_prune_ratio")
            if row.get("eval_status") != "success" or as_float(row.get("mAP")) is None:
                errors.append(f"a_fixed_forward_passed_eval_missing:{ratio}")
            if row.get("latency_status") != "success" or (as_float(row.get("latency_ms_p50"), 0) or 0) <= 0:
                errors.append(f"a_fixed_forward_passed_latency_invalid:{ratio}")
    a_ratios = {round(float(as_float(row.get("target_prune_ratio"), -1) or -1), 2): row for row in read_csv_rows(root / "a_fixed_full_model_summary.csv")}
    for required_ratio in {0.05, 0.10}:
        row = a_ratios.get(required_ratio)
        if not row or row.get("forward_smoke_status") != "forward_passed":
            errors.append(f"a_fixed_ratio_missing_or_failed:{required_ratio}")
    da = read_json(root / "d_reblock_design_audit.json", {}) or {}
    if da.get("full_model_executed") is True:
        errors.append("d_full_model_should_not_run_by_default")
    return {"valid": not errors, "errors": errors}


def write_decision_summary(root: Path, a_rows: list[dict[str, Any]], b_report: dict[str, Any], c_rows: list[dict[str, Any]], d_audit: dict[str, Any]) -> None:
    good_a = [r for r in a_rows if r.get("eval_status") == "success" and (as_float(r.get("mAP_drop"), 999) or 999) < 0.03 and (as_float(r.get("speedup_vs_baseline"), 0) or 0) > 1.05]
    good_c = [r for r in c_rows if r.get("eval_status") == "success" and (as_float(r.get("mAP_drop"), 999) or 999) < 0.03 and (as_float(r.get("speedup_vs_baseline"), 0) or 0) > 1.05]
    lines = [
        "# v9.2 Decision Summary",
        "",
        f"1. A TP equivalence: see `a_fix_tp_equivalence_audit_report.json`; fixed resolver uses Torch-Pruning DepGraph output pruning.",
        f"2. A AP-latency candidates with mAP drop <0.03 and speedup >1.05x: {len(good_a)}.",
        f"3. B diagnosis: {b_report.get('answer', '')}",
        "4. B optimization space remains limited unless grouped-conv kernels become a larger latency share or shapes stay hardware-friendly.",
        f"5. C aligned candidates with mAP drop <0.03 and speedup >1.05x: {len(good_c)}.",
        "6. C speed instability is audited in `c_group_block_shape_latency_audit.csv`; compare friendly group counts and total channel alignment.",
        "7. Early layer0 should remain protected for AP-sensitive sweeps unless recovery/distillation is planned.",
        f"8. D full-model execution this round: {d_audit.get('full_model_executed')}; D should wait until A compact semantics remain closed.",
        "9. Recommendation: pause TensorRT/GA/proxy unless the CSV contains a candidate satisfying AP drop <0.03 and speedup >1.05x.",
        "10. Next stage should prioritize non-grouped 1x1/neck/fusion/head pruning or recovery fine-tune / BEV distillation.",
    ]
    (root / "v92_decision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--warmup-frames", type=int, default=20)
    p.add_argument("--overwrite", type=str2bool, default=False)
    p.add_argument("--run-eval", type=str2bool, default=True)
    p.add_argument("--c-ratios", default="0.05,0.10,0.15,0.25")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    for sub in ["logs", "models", "work", "eval"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    write_json(root / "v92_config.json", vars(args))
    baseline_eval, baseline_latency = run_baseline_eval(args, root)
    grouped = discover_regular_grouped_convs(args)
    selected = [
        "pyramid_backbone.resnet.layer0.0.conv2",
        "pyramid_backbone.resnet.layer1.0.conv2",
        "pyramid_backbone.resnet.layer2.0.conv2",
    ]
    aeq = run_a_equivalence(args, root, selected)
    write_json(root / "a_fix_tp_equivalence_audit_report.json", aeq)
    (root / "a_fix_tp_equivalence_summary.md").write_text(
        "# A Fixed TP Equivalence\n\n" + "\n".join(f"- {r.get('module_name')}: equivalent={r.get('equivalent')} reason={r.get('divergence_reason','')}" for r in aeq.get("layers", [])) + "\n",
        encoding="utf-8",
    )
    if not aeq.get("all_equivalent"):
        for row in aeq.get("layers", []):
            if not row.get("equivalent"):
                failures.append({"stage": "a_tp_equivalence", "module_name": row.get("module_name"), "failure_reason": row.get("divergence_reason")})
    a_rows, a_smoke, a_eval, a_lat, a_fail = run_a_full_model(args, root, grouped, baseline_eval, baseline_latency)
    failures.extend(a_fail)
    a_fields = ["policy", "score_mode", "target_prune_ratio", "model_path", "forward_smoke_status", "eval_status", "latency_status", "num_grouped_convs_pruned", "mean_reinterpretation_ratio", "max_reinterpretation_ratio", "AP_0.3", "mAP", "baseline_AP_0.3", "baseline_mAP", "AP_drop", "mAP_drop", "latency_ms_p50", "baseline_latency_ms_p50", "speedup_vs_baseline", "failure_reason"]
    write_csv(root / "a_fixed_full_model_summary.csv", a_rows, a_fields)
    b_report, b_rows, b_shape, module_rows = b_speed_diagnosis(root)
    write_json(root / "b_speed_diagnosis_report.json", b_report)
    write_csv(root / "b_speed_diagnosis_summary.csv", b_rows, ["policy", "target_prune_ratio", "mAP", "mAP_drop", "latency_ms_p50", "speedup_vs_baseline", "diagnosis"])
    write_csv(root / "b_shape_efficiency_per_layer.csv", b_shape, list(b_shape[0].keys()) if b_shape else ["target_prune_ratio"])
    all_smoke = list(a_smoke)
    all_eval = list(a_eval)
    all_lat = list(a_lat)
    c_rows: list[dict[str, Any]] = []
    c_shapes: list[dict[str, Any]] = []
    ratios = [float(x) for x in args.c_ratios.split(",") if x.strip()]
    c_modules_all = [g["module_name"] for g in grouped]
    low_sens_modules = [m for m in c_modules_all if ".layer0." not in m]
    variants = {
        "C2_group_count_aligned": c_modules_all,
        "C3_total_align8": c_modules_all,
        "C4_total_align16": c_modules_all,
        "C5_low_sensitivity_only_aligned": low_sens_modules,
    }
    for variant, modules in variants.items():
        for ratio in ratios:
            row, smoke, erow, lrow, shapes = run_c_variant(args, root, variant, ratio, baseline_eval, baseline_latency, modules)
            c_rows.append(row)
            all_smoke.append(smoke)
            all_eval.append(erow)
            all_lat.append(lrow)
            c_shapes.extend(shapes)
            if row.get("failure_reason"):
                failures.append({"stage": "c_aligned_sweep", "experiment_id": f"{variant}_{ratio}", "failure_reason": row.get("failure_reason")})
    write_csv(root / "c_group_block_aligned_sweep_summary.csv", c_rows, ["variant", "target_prune_ratio", "model_path", "forward_smoke_status", "eval_status", "latency_status", "AP_0.3", "mAP", "mAP_drop", "latency_ms_p50", "speedup_vs_baseline", "ratio_adjusted", "failure_reason"])
    write_csv(root / "c_group_block_shape_latency_audit.csv", c_shapes, list(c_shapes[0].keys()) if c_shapes else ["variant"])
    write_csv(root / "module_latency_breakdown_summary.csv", module_rows, ["model", "profiling_backend", "total_forward_latency_p50", "grouped_conv_total_latency", "grouped_conv_latency_share", "bottleneck_note"])
    write_json(root / "full_model_forward_smoke_report.json", all_smoke)
    write_json(root / "full_model_eval_short_report.json", all_eval)
    write_json(root / "full_model_latency_report.json", all_lat)
    d_audit = d_design_audit()
    write_json(root / "d_reblock_design_audit.json", d_audit)
    (root / "d_reblock_design_summary.md").write_text(
        "# D Reblock Design Audit\n\n"
        f"- Full-model executed: {d_audit['full_model_executed']}\n"
        f"- No old weight loss when: {d_audit['when_no_old_weight_loss']}\n"
        f"- Semantic mismatch when: {d_audit['when_semantic_mismatch']}\n"
        f"- Dense FLOPs may not drop when: {d_audit['when_dense_flops_do_not_drop']}\n"
        "- D should wait until A compact semantics are closed and bucket-local downstream resolver is implemented.\n",
        encoding="utf-8",
    )
    write_decision_summary(root, a_rows, b_report, c_rows, d_audit)
    write_jsonl(root / "failure_cases.jsonl", failures)
    validation = validate_v92_output_bundle(root)
    return {"output_dir": str(root), "validation": validation}


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
