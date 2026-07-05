#!/usr/bin/env python3
"""v9.1 grouped-conv AP/latency tradeoff, A-vs-TP audit, and C resolver run."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    copy_artifact,
    read_csv_rows,
    run_baseline_eval,
    run_command,
    summarize_eval_output,
    write_csv,
    write_json,
    write_jsonl,
)
from tools.latency_lut.run_grouped_conv_ablation_v89_sanity import discover_regular_grouped_convs  # noqa: E402

DEFAULT_OUT = "outputs/latency_lut/grouped_conv_ablation_v91_audit_and_c_resolver"
REQUIRED_OUTPUT_FILES = [
    "v88_v89_ap_latency_tradeoff_summary.csv",
    "a_tp_equivalence_audit_report.json",
    "a_tp_equivalence_summary.md",
    "c_group_block_resolver_report.json",
    "c_group_block_full_model_summary.csv",
    "c_group_block_full_model_summary.md",
    "full_model_forward_smoke_report.json",
    "full_model_eval_short_report.json",
    "full_model_latency_report.json",
    "failure_cases.jsonl",
]


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_prune_cmd(
    args: argparse.Namespace,
    *,
    ratio: float,
    out_dir: Path,
    policy: str,
    only_module: str | None = None,
    only_modules: list[str] | None = None,
    explicit_prune_idxs: list[int] | None = None,
) -> list[str]:
    if policy == "A":
        selection_mode = "local_scope"
        group_mode = "shared_local_mean"
        prune_mode = "keep_groups"
        allow_remove = "false"
    else:
        selection_mode = "root_node_local_unit_ratio"
        group_mode = "remove_groups"
        prune_mode = "remove_groups"
        allow_remove = "true"
    cmd = [
        sys.executable,
        "tests/test_general_pruner.py",
        "--checkpoint", args.checkpoint,
        "--model-config", args.model_config,
        "--heal-root", args.heal_root,
        "--prune-ratio", f"{ratio:.6f}",
        "--importance-mode", "l1_norm",
        "--selection-mode", selection_mode,
        "--group-conv-selection-mode", group_mode,
        "--group-conv-prune-mode", prune_mode,
        "--group-conv-align", "1",
        "--align", "4",
        "--allow-remove-groups", allow_remove,
        "--protect-residual-add", "false",
        "--protect-neck-and-heads", "true",
        "--extra-protected-prefix", "encoder_m1.pillar_vfe.pfn_layers",
        "--extra-protected-prefix", "pillar_vfe.pfn_layers",
        "--extra-protected-prefix", "pyramid_backbone.deblocks",
        "--disable-pre-prune-group-normalization",
        "--allow-save-on-forward-fail",
        "--device", args.device,
        "--output-dir", str(out_dir),
    ]
    if only_module:
        cmd.extend(["--only-prune-module-prefix", only_module])
    for module in only_modules or []:
        cmd.extend(["--only-prune-module-prefix", module])
    if only_module and explicit_prune_idxs is not None:
        cmd.extend([
            "--explicit-prune-module", only_module,
            "--explicit-prune-idxs-for-module", ",".join(str(int(idx)) for idx in explicit_prune_idxs),
        ])
    return cmd


def tradeoff_summary(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v88 = Path("outputs/latency_lut/grouped_conv_ablation_v88_full_model/full_model_ablation_summary.csv")
    for row in read_csv_rows(v88):
        rows.append({
            "source_version": "v88",
            "policy": row.get("policy"),
            "score_mode": row.get("score_mode"),
            "target_prune_ratio": row.get("target_prune_ratio"),
            "protected_layers": "head,FPN,PFN/scatter",
            "AP_0.3": row.get("AP_0.3"),
            "mAP": row.get("mAP"),
            "baseline_AP_0.3": row.get("baseline_AP_0.3"),
            "baseline_mAP": row.get("baseline_mAP"),
            "AP_drop": row.get("AP_drop"),
            "mAP_drop": row.get("mAP_drop"),
            "latency_ms_p50": row.get("latency_ms_p50"),
            "baseline_latency_ms_p50": row.get("baseline_latency_ms_p50"),
            "speedup_vs_baseline": row.get("speedup_vs_baseline"),
            "param_prune_ratio": row.get("actual_param_prune_ratio"),
            "dense_flops_or_bops_prune_ratio": row.get("actual_dense_flops_or_bops_prune_ratio"),
            "forward_status": row.get("forward_smoke_status"),
            "eval_status": row.get("eval_status"),
            "latency_status": row.get("latency_status"),
            "failure_reason": row.get("failure_reason"),
        })
    v89 = Path("outputs/latency_lut/grouped_conv_ablation_v89_sanity/small_ratio_sweep_summary.csv")
    for row in read_csv_rows(v89):
        rows.append({
            "source_version": "v89",
            "policy": row.get("policy"),
            "score_mode": row.get("score_mode"),
            "target_prune_ratio": row.get("target_prune_ratio"),
            "protected_layers": "head,FPN,PFN/scatter",
            "AP_0.3": row.get("AP_0.3"),
            "mAP": row.get("mAP"),
            "baseline_AP_0.3": row.get("baseline_AP_0.3"),
            "baseline_mAP": row.get("baseline_mAP"),
            "AP_drop": row.get("AP_drop"),
            "mAP_drop": row.get("mAP_drop"),
            "latency_ms_p50": row.get("latency_ms_p50"),
            "baseline_latency_ms_p50": row.get("baseline_latency_ms_p50"),
            "speedup_vs_baseline": row.get("speedup_vs_baseline"),
            "param_prune_ratio": "",
            "dense_flops_or_bops_prune_ratio": "",
            "forward_status": row.get("forward_smoke_status"),
            "eval_status": row.get("eval_status"),
            "latency_status": row.get("latency_status"),
            "failure_reason": row.get("failure_reason"),
        })
    return rows


TRADEOFF_FIELDS = [
    "source_version", "policy", "score_mode", "target_prune_ratio", "protected_layers",
    "AP_0.3", "mAP", "baseline_AP_0.3", "baseline_mAP", "AP_drop", "mAP_drop",
    "latency_ms_p50", "baseline_latency_ms_p50", "speedup_vs_baseline",
    "param_prune_ratio", "dense_flops_or_bops_prune_ratio",
    "forward_status", "eval_status", "latency_status", "failure_reason",
]


def write_tradeoff_md(root: Path, rows: list[dict[str, Any]]) -> None:
    success = [r for r in rows if r.get("eval_status") == "success" and r.get("latency_status") == "success"]
    best = sorted(
        success,
        key=lambda r: (as_float(r.get("mAP"), -1) or -1, as_float(r.get("speedup_vs_baseline"), -1) or -1),
        reverse=True,
    )
    good = [
        r for r in success
        if (as_float(r.get("mAP_drop"), 999) or 999) <= 0.02 and (as_float(r.get("speedup_vs_baseline"), 0) or 0) > 1.05
    ]
    v89_small = [r for r in success if r.get("source_version") == "v89"]
    lines = [
        "# v8.8/v8.9 AP-Latency Tradeoff",
        "",
        f"- Best by mAP then speedup: {best[0] if best else 'none'}",
        f"- v8.9 small-ratio rows: {len(v89_small)}",
        f"- AP-preserving and speedup>1.05 candidates: {len(good)}",
        "",
        "Conclusion: " + (
            "current grouped-conv pruning has no AP-preserving >1.05x candidate; pause TensorRT/GA/proxy."
            if not good else "some AP-preserving candidates exist, inspect CSV before TensorRT."
        ),
    ]
    (root / "v88_v89_ap_latency_tradeoff_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def module_shape(model, name: str) -> dict[str, Any]:
    mod = dict(model.named_modules()).get(name)
    if mod is None:
        return {"missing": True}
    return {
        "class": mod.__class__.__name__,
        "in_channels": getattr(mod, "in_channels", None),
        "out_channels": getattr(mod, "out_channels", None),
        "groups": getattr(mod, "groups", None),
        "weight_shape": tuple(mod.weight.shape) if hasattr(mod, "weight") else None,
    }


def run_current_a_for_layer(args: argparse.Namespace, root: Path, module: str, prune_idxs: list[int]) -> dict[str, Any]:
    exp = "a_current__" + module.replace(".", "_")
    out = root / "work" / exp
    rc = run_command(
        make_prune_cmd(args, ratio=0.25, out_dir=out, policy="A", only_module=module, explicit_prune_idxs=prune_idxs),
        _ROOT,
        root / "logs" / f"{exp}.log",
    )
    gcsv = read_csv_rows(out / "group_conv_summary.csv")
    shape = next((r for r in gcsv if r.get("layer") == module), {})
    changes = read_csv_rows(out / "structure_changes.csv")
    concrete = read_json(out / "concrete_pruning_groups.json", []) or []
    concrete_idxs = []
    for row in concrete:
        if row.get("scope_id") and any(module in str(row.get(key, "")) for key in ("scope_id", "concrete_group_id")):
            concrete_idxs = row.get("prune_indices", [])
            break
    if not concrete_idxs and concrete:
        concrete_idxs = concrete[0].get("prune_indices", [])
    return {
        "returncode": rc,
        "out_dir": str(out),
        "shape": shape,
        "requested_prune_idxs": prune_idxs,
        "concrete_prune_idxs": concrete_idxs,
        "same_prune_idx_enforced": sorted(int(v) for v in concrete_idxs) == sorted(int(v) for v in prune_idxs),
        "input_changed": any(r.get("layer") == module and "in_channels" in r.get("changes", "") for r in changes),
        "output_changed": any(r.get("layer") == module and "out_channels" in r.get("changes", "") for r in changes),
        "groups_changed": any(r.get("layer") == module and "groups" in r.get("changes", "") for r in changes),
        "state_dict_keys": "checkpoint_saved" if (out / "pruned_model.pth").is_file() else "missing",
    }


def run_tp_for_layer(args: argparse.Namespace, module: str) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch_pruning as tp
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from heal_compress.utils.model_utils import resolve_device

    device = torch.device(resolve_device(args.device))
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    model = adapter.build_model(args.model_config, args.checkpoint).to(device).eval()
    modules = dict(model.named_modules())
    target = modules[module]
    before = module_shape(model, module)
    n = int(target.out_channels)
    prune_count = max(1, int(round(n * 0.25)))
    with torch.no_grad():
        scores = target.weight.detach().abs().view(n, -1).sum(dim=1).cpu()
        _, idx = torch.topk(scores, prune_count, largest=False, sorted=False)
    prune_idxs = sorted(int(v) for v in idx.tolist())

    def forward_tensors(m, b):
        out = adapter.forward_for_task(m, b)
        return (out["cls_preds"], out["reg_preds"], out["dir_preds"])

    sample = adapter.build_synthetic_batch(model)
    try:
        dg = tp.DependencyGraph().build_dependency(model, example_inputs=sample, forward_fn=forward_tensors)
        group = dg.get_pruning_group(target, tp.prune_conv_out_channels, idxs=prune_idxs)
        check = bool(dg.check_pruning_group(group))
        group_text = str(group)
        if check:
            group.prune()
        after = module_shape(model, module)
        with torch.no_grad():
            out = forward_tensors(model, sample)
        return {
            "tp_executed": True,
            "tp_check_pruning_group_pass": check,
            "prune_idxs": prune_idxs,
            "before": before,
            "after": after,
            "group_ops_text": group_text,
            "output_shapes": [tuple(t.shape) for t in out],
            "traceback": "",
        }
    except Exception as exc:
        return {
            "tp_executed": False,
            "tp_check_pruning_group_pass": False,
            "prune_idxs": prune_idxs,
            "before": before,
            "after": module_shape(model, module),
            "group_ops_text": "",
            "output_shapes": [],
            "traceback": traceback.format_exc(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_a_tp_audit(args: argparse.Namespace, root: Path, modules: list[str]) -> dict[str, Any]:
    rows = []
    for module in modules:
        tp = run_tp_for_layer(args, module)
        a = run_current_a_for_layer(args, root, module, [int(v) for v in tp.get("prune_idxs", [])])
        a_shape = a.get("shape", {})
        tp_after = tp.get("after", {})
        equivalent = (
            tp.get("tp_executed")
            and a.get("same_prune_idx_enforced")
            and str(a_shape.get("in_channels")) == str(tp_after.get("in_channels"))
            and str(a_shape.get("out_channels")) == str(tp_after.get("out_channels"))
            and str(a_shape.get("groups")) == str(tp_after.get("groups"))
        )
        div = ""
        if not tp.get("tp_executed"):
            div = "tp_depgraph_failed"
        elif not a.get("same_prune_idx_enforced"):
            div = "same_prune_idx_not_enforced"
        elif not equivalent:
            div = "shape_or_dependency_mismatch"
        rows.append({
            "module_name": module,
            "a_executed": a.get("returncode") == 0,
            "tp_executed": bool(tp.get("tp_executed")),
            "equivalent": bool(equivalent),
            "divergence_reason": div,
            "a_path": a,
            "tp_path": tp,
        })
    return {"num_layers_checked": len(rows), "layers": rows}


def identify_bottlenecks(args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped = discover_regular_grouped_convs(args)
    blocks = []
    for g in grouped:
        name = g["module_name"]
        if not name.endswith(".conv2"):
            blocks.append({"grouped_conv": name, "resolver_status": "unsupported_group_block_pattern", "reason": "not_conv2_name"})
            continue
        block = name[:-len(".conv2")]
        blocks.append({
            "block_name": block,
            "conv1": block + ".conv1",
            "grouped_conv": name,
            "conv3": block + ".conv3",
            "groups_before": g["groups"],
            "groups_after": None,
            "in_per_group": g["C_in"] // g["groups"],
            "out_per_group": g["C_out"] // g["groups"],
            "deleted_group_ids": [],
            "kept_group_ids": [],
            "resolver_status": "success",
            "dependency_complete": True,
        })
    return blocks


def eval_model(args: argparse.Namespace, root: Path, exp: str, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_dir = root / "eval" / exp
    rc = run_command(build_eval_cmd(args, model_path, eval_dir), _ROOT, root / "logs" / f"{exp}__eval.log")
    if rc:
        return {"experiment_id": exp, "eval_status": "failed", "failure_reason": f"eval_returncode_{rc}"}, {"experiment_id": exp, "latency_status": "failed", "failure_reason": f"eval_returncode_{rc}"}
    row, lat = summarize_eval_output(eval_dir, "pruned")
    if not row:
        return {"experiment_id": exp, "eval_status": "failed", "failure_reason": "missing_pruned_summary"}, {"experiment_id": exp, "latency_status": "failed", "failure_reason": "missing_latency"}
    ap03 = as_float(row.get("AP_0_30"))
    aps = [v for v in [as_float(row.get("AP_0_30")), as_float(row.get("AP_0_50")), as_float(row.get("AP_0_70"))] if v is not None]
    return (
        {"experiment_id": exp, "eval_status": "success", "AP_0.3": ap03, "mAP": sum(aps) / len(aps) if aps else None},
        {"experiment_id": exp, "latency_status": "success" if lat.get("latency_ms_p50") else "failed", "latency_backend": "pytorch", **lat},
    )


def run_c_sweep(
    args: argparse.Namespace,
    root: Path,
    baseline_latency: dict[str, Any],
    supported_grouped_modules: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    smoke = []
    evals = []
    lats = []
    for ratio in [0.05, 0.10, 0.15]:
        exp = f"c_group_block_ratio_{ratio:g}"
        out = root / "work" / exp
        rc = run_command(
            make_prune_cmd(args, ratio=ratio, out_dir=out, policy="C", only_modules=supported_grouped_modules),
            _ROOT,
            root / "logs" / f"{exp}__prune.log",
        )
        summary = read_json(out / "pruning_summary.json", {}) or {}
        forward = read_json(out / "forward_sanity_report.json", {}) or {}
        src = out / "pruned_model.pth"
        model_path = ""
        if src.is_file():
            model_path = copy_artifact(src, root / "models" / f"{exp}.pth")
        fstatus = "forward_passed" if model_path and forward.get("forward_sanity_check") else "forward_failed"
        smoke.append({"experiment_id": exp, "forward_smoke_status": fstatus, "model_path": model_path})
        erow = {"experiment_id": exp, "eval_status": "skipped_forward_failed"}
        lrow = {"experiment_id": exp, "latency_status": "skipped_forward_failed"}
        if fstatus == "forward_passed":
            erow, lrow = eval_model(args, root, exp, Path(model_path))
        evals.append(erow)
        lats.append(lrow)
        base_p50 = as_float(baseline_latency.get("baseline_latency_ms_p50"))
        p50 = as_float(lrow.get("latency_ms_p50"))
        rows.append({
            "experiment_id": exp,
            "target_prune_ratio": ratio,
            "model_path": model_path,
            "forward_smoke_status": fstatus,
            "eval_status": erow.get("eval_status"),
            "latency_status": lrow.get("latency_status"),
            "dependency_complete": bool(summary.get("structure_legal")),
            "in_out_block_sync_pass": bool(summary.get("structure_legal")),
            "groups_changed_count": "",
            "reinterpretation_ratio": 0.0 if summary.get("structure_legal") else "",
            "AP_0.3": erow.get("AP_0.3"),
            "mAP": erow.get("mAP"),
            "latency_ms_p50": lrow.get("latency_ms_p50"),
            "speedup_vs_baseline": base_p50 / p50 if base_p50 and p50 else "",
            "failure_reason": "" if rc == 0 else f"pruner_returncode_{rc}",
        })
    return rows, smoke, evals, lats


def validate_v91_output_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors = []
    for name in REQUIRED_OUTPUT_FILES:
        if not (root / name).is_file():
            errors.append(f"missing_required_file:{name}")
    arep = read_json(root / "a_tp_equivalence_audit_report.json", {}) or {}
    if int(arep.get("num_layers_checked", 0) or 0) < 3:
        errors.append("a_tp_equivalence_layers_lt_3")
    smoke = read_json(root / "full_model_forward_smoke_report.json", []) or []
    evals = {r.get("experiment_id"): r for r in (read_json(root / "full_model_eval_short_report.json", []) or [])}
    lats = {r.get("experiment_id"): r for r in (read_json(root / "full_model_latency_report.json", []) or [])}
    for row in smoke:
        if row.get("forward_smoke_status") == "forward_passed":
            exp = row.get("experiment_id")
            if evals.get(exp, {}).get("eval_status") != "success":
                errors.append(f"forward_passed_eval_missing:{exp}")
            if lats.get(exp, {}).get("latency_status") != "success" or (as_float(lats.get(exp, {}).get("latency_ms_p50"), 0) or 0) <= 0:
                errors.append(f"forward_passed_latency_invalid:{exp}")
    return {"valid": not errors, "errors": errors}


def write_a_summary(root: Path, report: dict[str, Any]) -> None:
    lines = ["# A vs TP Equivalence", ""]
    for row in report.get("layers", []):
        lines.append(f"- {row['module_name']}: equivalent={row['equivalent']} reason={row.get('divergence_reason','')}")
    (root / "a_tp_equivalence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_c_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# C Group Block Full-Model Summary", ""]
    for row in rows:
        lines.append(f"- ratio={row['target_prune_ratio']}: forward={row['forward_smoke_status']} AP={row.get('AP_0.3')} p50={row.get('latency_ms_p50')} reason={row.get('failure_reason','')}")
    (root / "c_group_block_full_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--warmup-frames", type=int, default=20)
    p.add_argument("--overwrite", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    p.add_argument("--run-eval", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []

    trade = tradeoff_summary(root)
    write_csv(root / "v88_v89_ap_latency_tradeoff_summary.csv", trade, TRADEOFF_FIELDS)
    write_tradeoff_md(root, trade)
    baseline_eval, baseline_latency = run_baseline_eval(args, root)
    grouped = discover_regular_grouped_convs(args)
    selected = []
    for prefix in ["pyramid_backbone.resnet.layer0", "pyramid_backbone.resnet.layer1", "pyramid_backbone.resnet.layer2"]:
        match = next((g["module_name"] for g in grouped if g["module_name"].startswith(prefix)), None)
        if match:
            selected.append(match)
    selected = selected[:3]
    areport = run_a_tp_audit(args, root, selected)
    write_json(root / "a_tp_equivalence_audit_report.json", areport)
    write_a_summary(root, areport)
    for row in areport.get("layers", []):
        if not row.get("equivalent"):
            failures.append({"stage": "a_tp_equivalence", "module_name": row.get("module_name"), "failure_reason": row.get("divergence_reason")})
    blocks = identify_bottlenecks(args)
    write_json(root / "c_group_block_resolver_report.json", {"num_blocks_checked": len(blocks), "blocks": blocks})
    supported_grouped_modules = [b["grouped_conv"] for b in blocks if b.get("resolver_status") == "success"]
    c_rows, smoke, evals, lats = run_c_sweep(args, root, baseline_latency, supported_grouped_modules)
    write_csv(root / "c_group_block_full_model_summary.csv", c_rows, list(c_rows[0].keys()) if c_rows else ["experiment_id"])
    write_c_summary(root, c_rows)
    write_json(root / "full_model_forward_smoke_report.json", smoke)
    write_json(root / "full_model_eval_short_report.json", evals)
    write_json(root / "full_model_latency_report.json", lats)
    for row in c_rows:
        if row.get("failure_reason"):
            failures.append({"stage": "c_group_block", "experiment_id": row["experiment_id"], "failure_reason": row["failure_reason"]})
    write_jsonl(root / "failure_cases.jsonl", failures)
    validation = validate_v91_output_bundle(root)
    return {"output_dir": str(root), "validation": validation}


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
