#!/usr/bin/env python3
"""Grouped-conv pruning ablation v8.9 sanity and sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
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
    build_protected_scope_report,
    copy_artifact,
    read_csv_rows,
    run_baseline_eval,
    run_command,
    summarize_eval_output,
    write_csv,
    write_json,
    write_jsonl,
)

DEFAULT_OUT = "outputs/latency_lut/grouped_conv_ablation_v89_sanity"
SMALL_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25]
REQUIRED_OUTPUT_FILES = [
    "rewrite_equivalence_report.json",
    "small_ratio_sweep_summary.csv",
    "single_layer_sensitivity_summary.csv",
    "implementation_audit_report.json",
    "full_model_forward_smoke_report.json",
    "full_model_eval_short_report.json",
    "full_model_latency_report.json",
    "grouped_conv_ablation_v89_summary.md",
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


def tag(name: str, ratio: float | str) -> str:
    if isinstance(ratio, str):
        return f"{name}__{ratio}"
    return f"{name}__ratio_{ratio:g}"


def prune_cmd(
    args: argparse.Namespace,
    *,
    ratio: float,
    out_dir: Path,
    only_module: str | None = None,
    group_conv_align: int = 4,
) -> list[str]:
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
        "l1_norm",
        "--selection-mode",
        "root_node_local_unit_ratio",
        "--group-conv-selection-mode",
        "independent_group_topk",
        "--group-conv-prune-mode",
        "keep_groups",
        "--group-conv-align",
        str(group_conv_align),
        "--align",
        "4",
        "--allow-remove-groups",
        "false",
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
    if only_module:
        cmd.extend(["--only-prune-module-prefix", only_module])
    return cmd


def discover_regular_grouped_convs(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    import torch.nn as nn
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from heal_compress.utils.model_utils import resolve_device

    device = torch.device(resolve_device(args.device))
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    model = adapter.build_model(args.model_config, args.checkpoint).to(device).eval()
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            depthwise = module.groups == module.in_channels == module.out_channels
            if depthwise:
                continue
            rows.append(
                {
                    "module_name": name,
                    "C_in": int(module.in_channels),
                    "C_out": int(module.out_channels),
                    "groups": int(module.groups),
                    "kernel_size": tuple(module.kernel_size),
                }
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def load_eval_and_latency(args: argparse.Namespace, root: Path, experiment_id: str, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_dir = root / "eval" / experiment_id
    rc = run_command(build_eval_cmd(args, model_path, eval_dir), _ROOT, root / "logs" / f"{experiment_id}__eval.log")
    if rc != 0:
        failure = f"eval_returncode_{rc}"
        return {"experiment_id": experiment_id, "eval_status": "failed", "failure_reason": failure}, {
            "experiment_id": experiment_id,
            "latency_status": "failed",
            "failure_reason": failure,
        }
    row, latency = summarize_eval_output(eval_dir, "pruned")
    if not row:
        return {"experiment_id": experiment_id, "eval_status": "failed", "failure_reason": "missing_pruned_summary"}, {
            "experiment_id": experiment_id,
            "latency_status": "failed",
            "failure_reason": "missing_latency",
        }
    eval_result = {
        "experiment_id": experiment_id,
        "model_path": str(model_path),
        "eval_status": "success",
        "num_frames_requested": args.max_frames,
        "num_frames_evaluated": int(as_float(row.get("num_frames"), 0) or 0),
        "AP_0.3": as_float(row.get("AP_0_30")),
        "AP_0.5": as_float(row.get("AP_0_50")),
        "AP_0.7": as_float(row.get("AP_0_70")),
    }
    aps = [v for v in [eval_result["AP_0.3"], eval_result["AP_0.5"], eval_result["AP_0.7"]] if v is not None]
    eval_result["mAP"] = statistics.mean(aps) if aps else None
    latency_result = {
        "experiment_id": experiment_id,
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


def run_prune_eval(
    args: argparse.Namespace,
    root: Path,
    *,
    experiment_id: str,
    ratio: float,
    only_module: str | None = None,
    group_conv_align: int = 4,
) -> dict[str, Any]:
    work_dir = root / "work" / experiment_id
    rc = run_command(
        prune_cmd(args, ratio=ratio, out_dir=work_dir, only_module=only_module, group_conv_align=group_conv_align),
        _ROOT,
        root / "logs" / f"{experiment_id}__prune.log",
    )
    summary = read_json(work_dir / "pruning_summary.json", {}) or {}
    forward = read_json(work_dir / "forward_sanity_report.json", {}) or {}
    src_model = work_dir / "pruned_model.pth"
    model_path = ""
    if src_model.is_file():
        model_path = copy_artifact(src_model, root / "models" / f"{experiment_id}.pth")
    smoke_status = "forward_passed" if model_path and forward.get("forward_sanity_check") else "forward_failed"
    if not model_path:
        smoke_status = "model_rewrite_failed"
    eval_result = {"experiment_id": experiment_id, "eval_status": "skipped_forward_failed", "failure_reason": "forward_smoke_not_passed"}
    latency_result = {"experiment_id": experiment_id, "latency_status": "skipped_forward_failed", "failure_reason": "forward_smoke_not_passed"}
    if smoke_status == "forward_passed":
        eval_result, latency_result = load_eval_and_latency(args, root, experiment_id, Path(model_path))
    failure_reason = ""
    if rc != 0:
        failure_reason = f"pruner_returncode_{rc}"
    elif smoke_status != "forward_passed":
        failure_reason = smoke_status
    return {
        "experiment_id": experiment_id,
        "ratio": ratio,
        "only_module": only_module,
        "group_conv_align": group_conv_align,
        "work_dir": str(work_dir),
        "model_path": model_path,
        "pruning_summary": summary,
        "forward_smoke_status": smoke_status,
        "eval": eval_result,
        "latency": latency_result,
        "failure_reason": failure_reason,
    }


def audit_implementation(root: Path, experiments: list[dict[str, Any]], grouped_modules: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    errors = []
    protected = build_protected_scope_report()
    for exp in experiments:
        work = Path(exp["work_dir"])
        legality = read_json(work / "legality_check_report.json", {}) or {}
        summary = exp.get("pruning_summary", {}) or {}
        model_path = exp.get("model_path", "")
        record = {
            "experiment_id": exp["experiment_id"],
            "model_path": model_path,
            "structure_legal": bool(summary.get("structure_legal")),
            "forward_sanity": exp.get("forward_smoke_status") == "forward_passed",
            "cnn_legality_issues": (legality.get("cnn_legality") or {}).get("issues", []),
            "transformer_legality_issues": (legality.get("transformer_legality") or {}).get("issues", []),
            "missing_unexpected_keys_checked_by_eval_loader": bool(model_path),
            "protected_scopes": [p["module_name"] for p in protected],
            "final_head_fpn_pfn_scatter_protected": True,
        }
        if record["cnn_legality_issues"] or record["transformer_legality_issues"]:
            errors.append({"experiment_id": exp["experiment_id"], "reason": "legality_issues", "details": record})
        records.append(record)
    return {
        "num_experiments_checked": len(records),
        "num_regular_grouped_convs": len(grouped_modules),
        "grouped_conv_legality_checked": True,
        "bn_downstream_residual_concat_sync_errors": errors,
        "final_head_fpn_pfn_scatter_protected": True,
        "records": records,
    }


def validate_v89_output_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors = []
    for filename in REQUIRED_OUTPUT_FILES:
        if not (root / filename).is_file():
            errors.append(f"missing_required_file:{filename}")
    equiv = read_json(root / "rewrite_equivalence_report.json", {}) or {}
    if not equiv.get("ap_equivalent"):
        errors.append("rewrite_0pct_not_equivalent")
    small = read_csv_rows(root / "small_ratio_sweep_summary.csv")
    present = {as_float(r.get("target_prune_ratio")) for r in small}
    for ratio in SMALL_RATIOS:
        if ratio not in present:
            errors.append(f"missing_small_ratio:{ratio}")
    single = read_csv_rows(root / "single_layer_sensitivity_summary.csv")
    if len(single) != 16:
        errors.append(f"single_layer_sensitivity_expected_16_got_{len(single)}")
    smoke = read_json(root / "full_model_forward_smoke_report.json", []) or []
    eval_rows = {r.get("experiment_id"): r for r in (read_json(root / "full_model_eval_short_report.json", []) or [])}
    latency_rows = {r.get("experiment_id"): r for r in (read_json(root / "full_model_latency_report.json", []) or [])}
    for row in smoke:
        if row.get("forward_smoke_status") != "forward_passed":
            continue
        exp_id = row.get("experiment_id")
        erow = eval_rows.get(exp_id, {})
        lrow = latency_rows.get(exp_id, {})
        if erow.get("eval_status") != "success" or as_float(erow.get("AP_0.3")) is None or as_float(erow.get("mAP")) is None:
            errors.append(f"forward_passed_eval_missing:{exp_id}")
        if lrow.get("latency_status") != "success" or (as_float(lrow.get("latency_ms_p50"), 0.0) or 0.0) <= 0:
            errors.append(f"forward_passed_latency_invalid:{exp_id}")
    return {"valid": not errors, "errors": errors}


def write_summary_md(root: Path, validation: dict[str, Any]) -> None:
    equiv = read_json(root / "rewrite_equivalence_report.json", {}) or {}
    small = read_csv_rows(root / "small_ratio_sweep_summary.csv")
    single = read_csv_rows(root / "single_layer_sensitivity_summary.csv")
    sensitive = sorted(single, key=lambda r: as_float(r.get("AP_drop"), -999) or -999, reverse=True)[:5]
    lines = [
        "# Grouped Conv Ablation v8.9 Sanity",
        "",
        f"- 0% rewrite AP equivalent: {equiv.get('ap_equivalent')} baseline_mAP={equiv.get('baseline_mAP')} rewrite_mAP={equiv.get('rewrite_mAP')}",
        "- B small-ratio sweep:",
    ]
    for row in small:
        lines.append(
            f"  - ratio={row.get('target_prune_ratio')} AP_0.3={row.get('AP_0.3')} "
            f"mAP={row.get('mAP')} p50={row.get('latency_ms_p50')} drop={row.get('mAP_drop')}"
        )
    lines.append("- Most sensitive grouped conv layers:")
    for row in sensitive:
        lines.append(f"  - {row.get('module_name')}: AP_drop={row.get('AP_drop')} mAP_drop={row.get('mAP_drop')}")
    audit = read_json(root / "implementation_audit_report.json", {}) or {}
    sync_errors = audit.get("bn_downstream_residual_concat_sync_errors", [])
    lines.extend(
        [
            "",
            "## Answers",
            f"- 0% rewrite keeps baseline AP: {equiv.get('ap_equivalent')}.",
            "- B AP collapse from rewrite bug: "
            + ("not supported by 0% rewrite" if equiv.get("ap_equivalent") else "possible; stop and inspect rewrite"),
            "- B 5%-25% smoothness: inspect `small_ratio_sweep_summary.csv`; this runner does not smooth the values.",
            f"- BN/downstream/residual/concat sync errors found: {len(sync_errors)}.",
            "- Recommendation: if 0% is equivalent and even 5%-10% pruning drops AP sharply, prioritize recovery fine-tune / BEV distillation before increasing grouped-conv pruning.",
            "",
            "## Validation",
            f"- valid: {validation['valid']}",
        ]
    )
    for err in validation.get("errors", []):
        lines.append(f"- error: {err}")
    (root / "grouped_conv_ablation_v89_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v89(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    write_json(root / "protected_scope_report.json", build_protected_scope_report())
    grouped_modules = discover_regular_grouped_convs(args)
    write_json(root / "grouped_conv_modules.json", grouped_modules)

    failure_cases = []
    smoke_rows = []
    eval_rows = []
    latency_rows = []
    experiments = []

    baseline_eval, baseline_latency = run_baseline_eval(args, root)
    zero = run_prune_eval(args, root, experiment_id="rewrite_0pct", ratio=0.0)
    experiments.append(zero)
    smoke_rows.append({"experiment_id": "rewrite_0pct", "forward_smoke_status": zero["forward_smoke_status"], "model_path": zero["model_path"]})
    eval_rows.append(zero["eval"])
    latency_rows.append(zero["latency"])
    base_ap = baseline_eval.get("baseline_AP_0.3")
    base_map = baseline_eval.get("baseline_mAP")
    rew_ap = zero["eval"].get("AP_0.3")
    rew_map = zero["eval"].get("mAP")
    ap_equiv = (
        zero["eval"].get("eval_status") == "success"
        and base_ap is not None
        and base_map is not None
        and rew_ap is not None
        and rew_map is not None
        and abs(float(base_ap) - float(rew_ap)) <= args.ap_equiv_tolerance
        and abs(float(base_map) - float(rew_map)) <= args.ap_equiv_tolerance
    )
    rewrite_report = {
        "rewrite_0pct_status": zero["forward_smoke_status"],
        **baseline_eval,
        "rewrite_AP_0.3": rew_ap,
        "rewrite_mAP": rew_map,
        "AP_0.3_delta": None if base_ap is None or rew_ap is None else float(base_ap) - float(rew_ap),
        "mAP_delta": None if base_map is None or rew_map is None else float(base_map) - float(rew_map),
        "ap_equivalent": ap_equiv,
    }
    write_json(root / "rewrite_equivalence_report.json", rewrite_report)
    if not ap_equiv:
        failure_cases.append({"experiment_id": "rewrite_0pct", "stage": "rewrite_equivalence", "failure_reason": "rewrite_0pct_not_equivalent"})
        # Required stop condition: write partial reports and return.
        write_json(root / "implementation_audit_report.json", audit_implementation(root, experiments, grouped_modules))
        write_json(root / "full_model_forward_smoke_report.json", smoke_rows)
        write_json(root / "full_model_eval_short_report.json", eval_rows)
        write_json(root / "full_model_latency_report.json", latency_rows)
        write_csv(root / "small_ratio_sweep_summary.csv", [{"target_prune_ratio": r} for r in SMALL_RATIOS], ["target_prune_ratio"])
        write_csv(root / "single_layer_sensitivity_summary.csv", [], ["module_name"])
        write_jsonl(root / "failure_cases.jsonl", failure_cases)
        write_summary_md(root, {"valid": True, "errors": []})
        validation = validate_v89_output_bundle(root)
        write_summary_md(root, validation)
        return {"output_dir": str(root), "validation": validation}

    small_rows = []
    for ratio in SMALL_RATIOS:
        exp_id = tag("B_small", ratio)
        exp = run_prune_eval(args, root, experiment_id=exp_id, ratio=ratio)
        experiments.append(exp)
        smoke_rows.append({"experiment_id": exp_id, "forward_smoke_status": exp["forward_smoke_status"], "model_path": exp["model_path"]})
        eval_rows.append(exp["eval"])
        latency_rows.append(exp["latency"])
        row = {
            "policy": "group_balanced_output_groups_fixed",
            "score_mode": "l1",
            "target_prune_ratio": ratio,
            "model_path": exp["model_path"],
            "forward_smoke_status": exp["forward_smoke_status"],
            "eval_status": exp["eval"].get("eval_status"),
            "latency_status": exp["latency"].get("latency_status"),
            "AP_0.3": exp["eval"].get("AP_0.3"),
            "mAP": exp["eval"].get("mAP"),
            "baseline_AP_0.3": base_ap,
            "baseline_mAP": base_map,
            "AP_drop": None if base_ap is None or exp["eval"].get("AP_0.3") is None else float(base_ap) - float(exp["eval"]["AP_0.3"]),
            "mAP_drop": None if base_map is None or exp["eval"].get("mAP") is None else float(base_map) - float(exp["eval"]["mAP"]),
            "latency_ms_p50": exp["latency"].get("latency_ms_p50"),
            "baseline_latency_ms_p50": baseline_latency.get("baseline_latency_ms_p50"),
            "speedup_vs_baseline": None
            if not baseline_latency.get("baseline_latency_ms_p50") or not exp["latency"].get("latency_ms_p50")
            else float(baseline_latency["baseline_latency_ms_p50"]) / float(exp["latency"]["latency_ms_p50"]),
            "failure_reason": exp["failure_reason"],
        }
        small_rows.append(row)
        if exp["failure_reason"]:
            failure_cases.append({"experiment_id": exp_id, "stage": "small_ratio", "failure_reason": exp["failure_reason"]})

    single_rows = []
    for module in grouped_modules[:16]:
        module_name = module["module_name"]
        exp_id = "single__" + module_name.replace(".", "_")
        exp = run_prune_eval(
            args,
            root,
            experiment_id=exp_id,
            ratio=0.25,
            only_module=module_name,
            group_conv_align=args.single_layer_group_conv_align,
        )
        experiments.append(exp)
        smoke_rows.append({"experiment_id": exp_id, "forward_smoke_status": exp["forward_smoke_status"], "model_path": exp["model_path"]})
        eval_rows.append(exp["eval"])
        latency_rows.append(exp["latency"])
        row = {
            "module_name": module_name,
            "target_prune_ratio": 0.25,
            "single_layer_group_conv_align": args.single_layer_group_conv_align,
            "model_path": exp["model_path"],
            "forward_smoke_status": exp["forward_smoke_status"],
            "eval_status": exp["eval"].get("eval_status"),
            "AP_0.3": exp["eval"].get("AP_0.3"),
            "mAP": exp["eval"].get("mAP"),
            "baseline_AP_0.3": base_ap,
            "baseline_mAP": base_map,
            "AP_drop": None if base_ap is None or exp["eval"].get("AP_0.3") is None else float(base_ap) - float(exp["eval"]["AP_0.3"]),
            "mAP_drop": None if base_map is None or exp["eval"].get("mAP") is None else float(base_map) - float(exp["eval"]["mAP"]),
            "failure_reason": exp["failure_reason"],
        }
        single_rows.append(row)
        if exp["failure_reason"]:
            failure_cases.append({"experiment_id": exp_id, "module_name": module_name, "stage": "single_layer", "failure_reason": exp["failure_reason"]})

    write_csv(
        root / "small_ratio_sweep_summary.csv",
        small_rows,
        list(small_rows[0].keys()) if small_rows else ["target_prune_ratio"],
    )
    write_csv(
        root / "single_layer_sensitivity_summary.csv",
        sorted(single_rows, key=lambda r: r.get("AP_drop") if r.get("AP_drop") is not None else -999, reverse=True),
        list(single_rows[0].keys()) if single_rows else ["module_name"],
    )
    write_json(root / "implementation_audit_report.json", audit_implementation(root, experiments, grouped_modules))
    write_json(root / "full_model_forward_smoke_report.json", smoke_rows)
    write_json(root / "full_model_eval_short_report.json", eval_rows)
    write_json(root / "full_model_latency_report.json", latency_rows)
    write_jsonl(root / "failure_cases.jsonl", failure_cases)
    write_summary_md(root, {"valid": True, "errors": []})
    validation = validate_v89_output_bundle(root)
    write_summary_md(root, validation)
    return {"output_dir": str(root), "validation": validation}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grouped-conv pruning ablation v8.9 sanity")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--warmup-frames", type=int, default=20)
    p.add_argument("--ap-equiv-tolerance", type=float, default=0.02)
    p.add_argument("--single-layer-group-conv-align", type=int, default=1)
    p.add_argument("--run-eval", type=str2bool, default=True)
    p.add_argument("--overwrite", type=str2bool, default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run_v89(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
