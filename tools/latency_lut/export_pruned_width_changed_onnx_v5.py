from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.run_full_engine_candidate_benchmark import (
    DEPLOY_MODE,
    FIXED_K,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_REPO,
    DEFAULT_TRT_ROOT,
    _apply_runtime_env,
    _context,
    _export_single_engine_onnx,
    _prepare_candidate_checkpoint,
)
from tools.latency_lut.v5_pruned_mixed_common import audit_width_changed_onnx, load_json, write_json


def _copy_if_exists(src: str | Path | None, dst: Path) -> str | None:
    if not src:
        return None
    p = Path(src)
    if not p.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    return str(dst)


def _copy_prune_artifacts(prepared: dict[str, Any], candidate_dir: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    export_report = Path(str(prepared.get("export_report") or ""))
    export_dir = export_report.parent if export_report.is_file() else None
    if export_dir and export_dir.is_dir():
        for name in (
            "prune_replay.json",
            "pruning_summary.json",
            "selection_summary.json",
            "dependency_scopes.json",
            "concrete_pruning_groups.json",
            "coupled_channel_units.json",
            "atomic_prune_units.json",
            "structure_changes.json",
            "structure_changes.csv",
            "scope_channel_importance.csv",
            "group_importance.csv",
            "root_node_local_domains.json",
            "domain_selection_summary.json",
            "unit_importance.csv",
            "pruned_model.pth",
        ):
            copied = _copy_if_exists(export_dir / name, candidate_dir / name)
            if copied:
                out[name] = copied
    _copy_if_exists(prepared.get("prune_plan"), candidate_dir / "prune_plan.json")
    _copy_if_exists(prepared.get("export_report"), candidate_dir / "export_pruned_model_report.json")
    return out


def _resolved_units_from_structure(structure_changes: str | Path | None) -> list[dict[str, Any]]:
    p = Path(structure_changes) if structure_changes else None
    if not p or not p.is_file():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    units: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("layer") or row.get("layer_name") or row.get("module") or "")
        before = row.get("before") or {}
        after = row.get("after") or {}
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        c_in_old = before.get("in_channels") or before.get("C_in")
        c_out_old = before.get("out_channels") or before.get("C_out")
        c_in_new = after.get("in_channels") or after.get("C_in")
        c_out_new = after.get("out_channels") or after.get("C_out")
        if c_in_old == c_in_new and c_out_old == c_out_new:
            continue
        units.append(
            {
                "unit_id": name,
                "root_node": name,
                "C_in_original": c_in_old,
                "C_out_original": c_out_old,
                "C_in_new": c_in_new,
                "C_out_new": c_out_new,
                "C_in_aligned8": c_in_new,
                "C_out_aligned8": c_out_new,
                "is_grouped_conv": bool((before.get("groups") or after.get("groups") or 1) != 1),
                "groups_old": before.get("groups", 1),
                "groups_new": after.get("groups", before.get("groups", 1)),
            }
        )
    return units


def export_one(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    cid = str(candidate["candidate_id"])
    candidate_dir = Path(args.output_dir) / cid
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_json(candidate_dir / "candidate.json", candidate)
    runner_args = SimpleNamespace(
        candidate=str(candidate_dir / "candidate.json"),
        output=str(candidate_dir / "export_probe_result.json"),
        val_subset_size=1,
        deploy_mode=DEPLOY_MODE,
        fixed_k=FIXED_K,
        trtexec=args.trtexec,
        device=int(args.device),
        precision_profile="FP16",
        config=args.config,
        checkpoint=args.checkpoint,
        heal_repo=args.heal_repo,
        plugin=args.plugin,
        trt_root=args.trt_root,
        output_root=str(candidate_dir / "quant_deploy"),
        trace_report=args.trace_report,
        rebuild=True,
        onnx=None,
        engine_dir=str(candidate_dir / "engine_unused"),
        warmup=1,
        repeat=1,
        timeout=int(args.timeout),
    )
    ctx = _context(runner_args, candidate, cid, candidate_dir / "export_probe_result.json")
    _apply_runtime_env(ctx)
    prepared = _prepare_candidate_checkpoint(ctx, candidate)
    if not prepared.get("success"):
        report = {
            "candidate_id": cid,
            "success": False,
            "status": prepared.get("status"),
            "failed_stage": prepared.get("failed_stage") or "physical_prune",
            "error": prepared.get("error"),
            "candidate_dir": str(candidate_dir),
            "prepared": prepared,
        }
        write_json(candidate_dir / "pruned_model_export_report.json", report)
        return report
    ctx.checkpoint = Path(str(prepared["checkpoint"])).expanduser()
    export_report = _export_single_engine_onnx(ctx)
    if not export_report.get("success") or not ctx.onnx_path.is_file():
        report = {
            "candidate_id": cid,
            "success": False,
            "status": "onnx_export_failed",
            "failed_stage": "onnx_export",
            "error": export_report.get("error") or f"ONNX was not produced: {ctx.onnx_path}",
            "candidate_dir": str(candidate_dir),
            "prepared": prepared,
            "export_report": export_report,
        }
        write_json(candidate_dir / "pruned_model_export_report.json", report)
        return report
    artifact_paths = _copy_prune_artifacts(prepared, candidate_dir)
    width_onnx = candidate_dir / "width_changed.onnx"
    shutil.copy2(ctx.onnx_path, width_onnx)
    structure_audit = audit_width_changed_onnx(cid, width_onnx, candidate=candidate)
    structure_audit.update(
        {
            "pruned_checkpoint": str(ctx.checkpoint),
            "source_onnx": str(ctx.onnx_path),
            "candidate_dir": str(candidate_dir),
            "prepared": prepared,
            "export_report": export_report,
        }
    )
    write_json(candidate_dir / "structure_audit.json", structure_audit)
    resolved_units = _resolved_units_from_structure(candidate_dir / "structure_changes.json")
    updated_candidate = dict(candidate)
    updated_candidate["resolved_units"] = resolved_units
    updated_candidate["structure_audit_path"] = str(candidate_dir / "structure_audit.json")
    updated_candidate["width_changed_onnx"] = str(width_onnx)
    updated_candidate["prune_artifacts"] = artifact_paths
    write_json(candidate_dir / "candidate.json", updated_candidate)
    if (
        structure_audit["is_baseline_topology"]
        or not structure_audit["is_width_changed_subnet"]
        or structure_audit["num_changed_conv_layers"] <= 0
        or float(structure_audit.get("param_keep_ratio") or 1.0) >= 1.0
        or not structure_audit["uses_physical_pruning"]
    ):
        structure_audit["success"] = False
        structure_audit["status"] = "width_changed_structure_audit_failed"
        structure_audit["failed_stage"] = "structure_audit"
    else:
        structure_audit["success"] = True
        structure_audit["status"] = "width_changed_onnx_exported"
        structure_audit["failed_stage"] = None
    write_json(candidate_dir / "pruned_model_export_report.json", structure_audit)
    return structure_audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(args.candidates, {"candidates": []})
    candidates = list(payload.get("candidates") or [])
    if args.limit is not None:
        candidates = candidates[: int(args.limit)]
    reports: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        cid = str(candidate.get("candidate_id"))
        out_report = Path(args.output_dir) / cid / "pruned_model_export_report.json"
        if args.resume and out_report.is_file():
            reports.append(load_json(out_report))
            continue
        print(f"[pruned-onnx-v5] {idx}/{len(candidates)} {cid}")
        reports.append(export_one(args, candidate))
    summary = {
        "candidates_requested": len(candidates),
        "width_changed_onnx_exported": sum(1 for row in reports if row.get("success")),
        "failed": sum(1 for row in reports if not row.get("success")),
        "failed_by_stage": {},
        "reports": reports,
    }
    for row in reports:
        if not row.get("success"):
            stage = str(row.get("failed_stage") or row.get("status") or "unknown")
            summary["failed_by_stage"][stage] = int(summary["failed_by_stage"].get(stage, 0)) + 1
    write_json(Path(args.output_dir) / "export_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal-repo", "--heal_repo", dest="heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--trt-root", "--trt_root", dest="trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec", default="${TENSORRT_ROOT}/targets/x86_64-linux-gnu/bin/trtexec")
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--trace-report", "--trace_report", dest="trace_report", default=str(Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/tracer_reports/lidar_pyramid/coupled_channel_groups.json")))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
