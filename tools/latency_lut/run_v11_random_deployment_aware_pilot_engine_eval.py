#!/usr/bin/env python3
"""Run exactly one random deployment-aware high-INT8 pilot TensorRT engine/eval."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.latency_lut import run_v11_mixed_precision_lut_dataset_builder as builder


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _candidate_rows(dryrun_dir: Path) -> list[dict[str, Any]]:
    summary = _read_json(dryrun_dir / "onnx_qdq_profile_gate_dryrun_summary.json", {})
    rows: list[dict[str, Any]] = []
    for row in summary.get("profiles", []):
        if not isinstance(row, Mapping):
            continue
        if not row.get("qdq_insert_success"):
            continue
        if row.get("unmatched_int8_precision_groups"):
            continue
        if int(row.get("ambiguous_mapping_count") or 0) != 0:
            continue
        if "grouped_conv_int8_per_group_shape_not_supported" in (row.get("fallback_reasons") or {}):
            continue
        subnet_dir = dryrun_dir / "subnets" / str(row.get("subnet_id", ""))
        manifest = _read_json(subnet_dir / "pruning_manifest.json", {})
        grouped_rows = _read_json(subnet_dir / "grouped_conv_int8_eligibility_report.json", [])
        grouped_unsupported = sum(1 for item in grouped_rows if isinstance(item, Mapping) and not item.get("int8_shape_supported"))
        if grouped_unsupported != 0:
            continue
        enriched = dict(row)
        enriched["actual_param_prune_ratio"] = float(manifest.get("actual_param_prune_ratio") or manifest.get("achieved_global_param_prune_ratio") or 0.0)
        enriched["target_global_prune_bin"] = manifest.get("target_global_prune_bin", "")
        enriched["grouped_conv_unsupported_int8_shape_count"] = grouped_unsupported
        rows.append(enriched)
    return rows


def select_candidate(dryrun_dir: Path, subnet_id: str = "", profile_id: str = "") -> dict[str, Any]:
    rows = _candidate_rows(dryrun_dir)
    if subnet_id and profile_id:
        for row in rows:
            if row.get("subnet_id") == subnet_id and row.get("profile_id") == profile_id:
                return row
        raise ValueError(f"requested pilot candidate not gate-clean:{subnet_id}/{profile_id}")
    if not rows:
        raise ValueError("no gate-clean pilot candidates found")
    rows.sort(
        key=lambda row: (
            not (0.40 <= float(row.get("actual_param_prune_ratio") or 0.0) <= 0.70),
            str(row.get("profile_id", "")) != "profile_003",
            -int(row.get("requested_int8_layer_count") or 0),
            -float(row.get("requested_int8_group_ratio") or 0.0),
            str(row.get("subnet_id", "")),
        )
    )
    return rows[0]


def _pilot_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        build_engines=True,
        eval_engines=True,
        overwrite_profiles=True,
        allow_precision_mismatch_eval=False,
        calib_train_frames=int(args.calib_train_frames),
        warmup_frames=int(args.warmup_frames),
        eval_frames=int(args.eval_frames),
        smoke_frames=int(args.smoke_frames),
        ap_thresholds=str(args.ap_thresholds),
        fixed_k=int(args.fixed_k),
        trt_root=str(args.trt_root),
        trtexec=str(args.trtexec),
        trt_build_timeout_seconds=int(args.trt_build_timeout_seconds),
        plugin=str(args.plugin),
        heal_root=str(args.heal_root),
        model_config=str(args.model_config),
        checkpoint=str(args.checkpoint),
        num_workers=int(args.num_workers),
        device=str(args.device),
        require_onnx_origin_map=True,
        export_layer_info_during_build=False,
        enable_synthetic_trt_eval=False,
        output_dir=str(args.output_dir),
    )


def _latency_value(report: Mapping[str, Any], key: str, stat: str) -> Any:
    latency = report.get("latency_summary") or {}
    if isinstance(latency, Mapping):
        flat_prefix = str(key).removesuffix("_latency_ms")
        flat_key = f"{flat_prefix}_{stat}_ms"
        if flat_key in latency:
            return latency.get(flat_key)
    group = latency.get(key) if isinstance(latency, Mapping) else {}
    if isinstance(group, Mapping):
        return group.get(stat)
    return None


def write_report(output_dir: Path, selected: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    build = result.get("build") or {}
    structure = result.get("structure") or {}
    precision = result.get("precision") or {}
    smoke = result.get("smoke") or {}
    eval_report = result.get("eval") or {}
    ap = eval_report.get("ap") or {}
    qdq = _read_json(output_dir / "qdq_insert_report.json", {})
    lines = [
        "# Pilot Engine Eval Report",
        "",
        f"- selected_subnet_profile: {selected.get('subnet_id')}/{selected.get('profile_id')}",
        f"- selection_reason: high_int8 profile with gate-clean ONNX/QDQ/canonical mapping, actual_param_prune_ratio={float(selected.get('actual_param_prune_ratio') or 0.0):.6f}",
        f"- requested_int8_layer_count: {selected.get('requested_int8_layer_count')}",
        f"- requested_int8_group_ratio: {selected.get('requested_int8_group_ratio')}",
        f"- inserted_qdq_nodes_count: {len(qdq.get('inserted_qdq_nodes') or [])}",
        f"- actual_int8_realized_count: {precision.get('int8_realized_layer_count', 0)}",
        f"- fused_int8_with_fp16_boundary_count: {precision.get('int8_compute_fp16_boundary_count', 0)}",
        f"- true_precision_mismatch_count: {precision.get('precision_realization_mismatch_count', 0)}",
        f"- grouped_conv_unsupported_shape_count: {selected.get('grouped_conv_unsupported_int8_shape_count', 0)}",
        f"- engine_build_success: {build.get('build_success')}",
        f"- engine_structure_check_passed: {structure.get('structure_check_passed')}",
        f"- engine_precision_realization_check_passed: {precision.get('precision_realization_passed')}",
        f"- trt_smoke_success: {smoke.get('success')}",
        f"- eval_success_200frames: {eval_report.get('eval_success')}",
        f"- evaluated_frames: {eval_report.get('evaluated_frames')}",
        f"- synthetic_used: {eval_report.get('synthetic_used')}",
        f"- validation_dataloader_used: {eval_report.get('validation_dataloader_used')}",
        f"- skipped_frames: {eval_report.get('skipped_frames')}",
        f"- eval_failure_reason: {eval_report.get('failure_reason', '')}",
        f"- AP@0.03: {ap.get('AP@0.03')}",
        f"- AP@0.30: {ap.get('AP@0.30')}",
        f"- AP@0.50: {ap.get('AP@0.50')}",
        f"- AP@0.70: {ap.get('AP@0.70')}",
        f"- mAP: {ap.get('mAP')}",
        f"- forward_latency_mean_ms: {_latency_value(eval_report, 'forward_latency_ms', 'mean')}",
        f"- forward_latency_p50_ms: {_latency_value(eval_report, 'forward_latency_ms', 'p50')}",
        f"- forward_latency_p90_ms: {_latency_value(eval_report, 'forward_latency_ms', 'p90')}",
        f"- pipeline_status: {result.get('status')}",
        f"- next_step_recommendation: {'enter_4x4_small_build' if result.get('status') == 'eval_success' else 'fix_pilot_blocker_before_4x4'}",
    ]
    (output_dir / "pilot_engine_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    dryrun_dir = Path(args.dryrun_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_candidate(dryrun_dir, args.subnet_id, args.profile_id)
    subnet_dir = dryrun_dir / "subnets" / str(selected["subnet_id"])
    source_profile_dir = subnet_dir / str(selected["profile_id"])
    manifest = _read_json(subnet_dir / "pruning_manifest.json", {})
    profile = _read_json(source_profile_dir / "mixed_precision_profile.json", {})
    if not profile:
        raise FileNotFoundError(source_profile_dir / "mixed_precision_profile.json")
    groups = builder._precision_groups_from_json(subnet_dir / "precision_coupling_groups.json")
    profile["profile_id"] = str(selected["profile_id"])
    profile["subnet_id"] = str(selected["subnet_id"])
    profile["structure_hash"] = str(manifest.get("structure_hash", ""))
    profile = builder._finalize_profile_counts(profile)
    selected_payload = {
        **selected,
        "source_subnet_dir": str(subnet_dir),
        "source_profile_dir": str(source_profile_dir),
        "pilot_output_dir": str(output_dir),
        "pilot_label_only": True,
    }
    _write_json(output_dir / "selected_subnet_profile.json", selected_payload)
    ctx = {
        "args": _pilot_args(args),
        "subnet_dir": subnet_dir,
        "subnet_id": str(selected["subnet_id"]),
        "subnet_index": int(str(selected["subnet_id"]).split("_")[-1]),
        "structure_hash": str(manifest.get("structure_hash", "")),
        "groups": groups,
        "profile": profile,
        "profile_id": str(selected["profile_id"]),
        "profile_index": int(str(selected["profile_id"]).split("_")[-1]),
        "profile_dir": output_dir,
        "existing_hashes": set(),
    }
    result = builder.run_one_profile_pipeline(ctx)
    build = result.get("build") or {}
    if build.get("command"):
        (output_dir / "build_command.txt").write_text(" ".join(str(item) for item in build["command"]) + "\n", encoding="utf-8")
    if (output_dir / "eval_report.json").is_file():
        shutil.copyfile(output_dir / "eval_report.json", output_dir / "eval_report_200frames.json")
    _write_json(output_dir / "pilot_pipeline_result.json", result)
    write_report(output_dir, selected_payload, result)
    print(json.dumps({"success": result.get("status") == "eval_success", "status": result.get("status"), "selected": selected_payload, "output_dir": str(output_dir)}, indent=2, default=str))
    return 0 if result.get("status") == "eval_success" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dryrun-dir", default="outputs/latency_lut/v11_random_deployment_aware_subnets_dryrun_v2")
    parser.add_argument("--output-dir", default="outputs/latency_lut/v11_random_deployment_aware_pilot_engine_eval")
    parser.add_argument("--subnet-id", default="")
    parser.add_argument("--profile-id", default="")
    parser.add_argument("--calib-train-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--eval-frames", type=int, default=200)
    parser.add_argument("--smoke-frames", type=int, default=5)
    parser.add_argument("--ap-thresholds", default="0.03,0.30,0.50,0.70")
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=29696)
    parser.add_argument("--trt-root", default="${TENSORRT_ROOT}")
    parser.add_argument("--trtexec", default="")
    parser.add_argument("--trt-build-timeout-seconds", type=int, default=900)
    parser.add_argument("--plugin", default="quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
    parser.add_argument("--heal-root", default="../../HEAL")
    parser.add_argument("--model-config", default="${MODEL_ROOT}/lidar_pyramid/config.yaml")
    parser.add_argument("--checkpoint", default="${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        args = parse_args(argv)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "pilot_pipeline_result.json", {"success": False, "status": "pilot_runner_failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
        print(json.dumps({"success": False, "status": "pilot_runner_failed", "failure": f"{type(exc).__name__}: {exc}", "output_dir": str(output_dir)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
