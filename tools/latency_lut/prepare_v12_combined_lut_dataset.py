#!/usr/bin/env python3
"""Prepare the v12 combined 300-frame LUT dataset without TensorRT builds.

This script combines historical v11 random deployment-aware subnets with a new
v12 deblock-output-protected random subset, then runs/validates ONNX + QDQ +
profile gates only. It must not execute trtexec or validation eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.latency_lut.random_deployment_aware_subnet_sampler import (  # noqa: E402
    generate_random_deployment_aware_subnets,
    load_layer_specs_for_cli,
    parse_global_target_prune_bins,
)
from tools.latency_lut import run_v11_mixed_precision_lut_dataset_builder as builder  # noqa: E402


DATASET_VERSION = "v12_combined_lut_dataset_300frames"
HISTORICAL_SUBSET = "historical_v11"
NEW_V12_SUBSET = "new_v12_deblock_protected"
DEFAULT_HISTORICAL_DIR = "outputs/latency_lut/v11_random_deployment_aware_subnets_dryrun_v2"
DEFAULT_OUTPUT_DIR = "outputs/latency_lut/v12_combined_lut_dataset_300frames"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fields.append(str(key))
    if not fields:
        fields = ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def _attrs(row: Mapping[str, Any], stage: str) -> dict[str, int]:
    raw = row.get(stage) if isinstance(row.get(stage), Mapping) else {}
    if isinstance(raw, Mapping) and isinstance(raw.get("attrs"), Mapping):
        raw = raw["attrs"]
    out: dict[str, int] = {}
    if not isinstance(raw, Mapping):
        return out
    for key in ("in_channels", "out_channels", "in_features", "out_features", "num_features", "groups"):
        if key in raw:
            try:
                out[key] = int(raw[key])
            except Exception:
                pass
    for key, aliases in {
        "in_channels": ("C_in_before", "C_in_after", "in_before", "in_after"),
        "out_channels": ("C_out_before", "C_out_after", "out_before", "out_after"),
    }.items():
        if key in out:
            continue
        for alias in aliases:
            if alias in row and ((stage == "before" and "before" in alias) or (stage == "after" and "after" in alias)):
                try:
                    out[key] = int(row[alias])
                    break
                except Exception:
                    pass
    return out


def _channel_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("module_channel_before_after") or manifest.get("before_after_shapes") or []
    if isinstance(raw, Mapping):
        rows = []
        for name, value in raw.items():
            row = dict(value) if isinstance(value, Mapping) else {}
            row.setdefault("module_name", str(name))
            rows.append(row)
        return rows
    return [dict(row) for row in raw if isinstance(row, Mapping)]


def _module_type(row: Mapping[str, Any]) -> str:
    if row.get("module_type"):
        return str(row.get("module_type"))
    for stage in ("after", "before"):
        raw = row.get(stage)
        if isinstance(raw, Mapping) and raw.get("module_type"):
            return str(raw.get("module_type"))
    return ""


def _is_deblock_convtranspose(row: Mapping[str, Any]) -> bool:
    name = str(row.get("module_name", "")).lower()
    typ = _module_type(row).lower()
    return "pyramid_backbone.deblocks" in name and name.endswith(".0") and "convtranspose" in typ


def _is_deblock_bn(row: Mapping[str, Any]) -> bool:
    name = str(row.get("module_name", "")).lower()
    typ = _module_type(row).lower()
    return "pyramid_backbone.deblocks" in name and name.endswith(".1") and "batchnorm" in typ


def summarize_surface_contract_from_manifest(manifest: Mapping[str, Any], *, source_subset: str) -> dict[str, Any]:
    rows = _channel_rows(manifest)
    deblock_pruned: list[str] = []
    deblock_bn_changed: list[str] = []
    pfn_changed = False
    scatter_changed = False
    head_changed = False
    shrink_input_changed = False
    for row in rows:
        name = str(row.get("module_name", ""))
        low = name.lower()
        before = _attrs(row, "before")
        after = _attrs(row, "after")
        if _is_deblock_convtranspose(row) and before.get("out_channels") != after.get("out_channels"):
            deblock_pruned.append(name)
        if _is_deblock_bn(row) and before.get("num_features") != after.get("num_features"):
            deblock_bn_changed.append(name)
        if any(token in low for token in ("pillar_vfe", "pfn_layers")) and before.get("out_channels") != after.get("out_channels"):
            pfn_changed = True
        if "scatter" in low and before.get("out_channels") != after.get("out_channels"):
            scatter_changed = True
        if any(token in low for token in ("cls_head", "reg_head", "dir_head")):
            before_out = before.get("out_channels", before.get("out_features"))
            after_out = after.get("out_channels", after.get("out_features"))
            if before_out != after_out:
                head_changed = True
        if "pyramid_backbone.shrink_conv" in low and before.get("in_channels") != after.get("in_channels"):
            shrink_input_changed = True
    deblock_output_pruned = bool(deblock_pruned)
    new_contract_passed = (
        str(source_subset) != NEW_V12_SUBSET
        or (not deblock_output_pruned and not deblock_bn_changed and not shrink_input_changed)
    )
    return {
        "source_subset": source_subset,
        "deblock_output_pruned": deblock_output_pruned,
        "deblock_output_pruned_modules": deblock_pruned,
        "deblock_output_pruning_source": "historical_v11_or_opt_in" if deblock_output_pruned else "none_default_protected",
        "deblock_bn_num_features_changed": bool(deblock_bn_changed),
        "deblock_bn_changed_modules": deblock_bn_changed,
        "pfn_output_changed": bool(pfn_changed),
        "scatter_canvas_channel_changed": bool(scatter_changed),
        "head_output_changed": bool(head_changed),
        "shrink_conv_input_changed": bool(shrink_input_changed),
        "new_v12_deblock_contract_passed": bool(new_contract_passed),
    }


def _copy_subnet_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


def _renumber_manifest(subnet_dir: Path, *, subnet_id: str, source_subset: str, source_subnet_id: str) -> dict[str, Any]:
    manifest_path = subnet_dir / "pruning_manifest.json"
    manifest = read_json(manifest_path, {})
    manifest.update(
        {
            "dataset_version": DATASET_VERSION,
            "subnet_id": subnet_id,
            "source_subset": source_subset,
            "source_subnet_id": source_subnet_id,
            "uses_taylor_ranking": False,
        }
    )
    contract = summarize_surface_contract_from_manifest(manifest, source_subset=source_subset)
    manifest.update(contract)
    write_json(manifest_path, manifest)
    for profile_path in sorted(subnet_dir.glob("profile_*/mixed_precision_profile.json")):
        profile = read_json(profile_path, {})
        profile["subnet_id"] = subnet_id
        profile["source_subset"] = source_subset
        profile["dataset_version"] = DATASET_VERSION
        write_json(profile_path, profile)
    write_json(subnet_dir / "surface_contract_summary.json", contract)
    return manifest


def _subnet_dirs(root: Path) -> list[Path]:
    return sorted(path for path in (root / "subnets").glob("subnet_*") if (path / "pruning_manifest.json").is_file())


def _gate_profile_existing_subnet(subnet_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    from tools.latency_lut.run_v11_random_deployment_aware_gate_dryrun import _profile_gate_rows_for_subnet

    manifest = read_json(subnet_dir / "pruning_manifest.json", {})
    groups = builder._precision_groups_from_json(subnet_dir / "precision_coupling_groups.json")
    return _profile_gate_rows_for_subnet(
        subnet_dir,
        str(manifest.get("subnet_id", subnet_dir.name)),
        str(manifest.get("structure_hash", "")),
        groups,
        args,
    )


def _summarize_profile_gate(profile_dir: Path) -> dict[str, Any]:
    profile = read_json(profile_dir / "mixed_precision_profile.json", {})
    qdq = read_json(profile_dir / "qdq_insert_report.json", {})
    mapping = read_json(profile_dir / "canonical_precision_mapping.json", {})
    specs = read_json(profile_dir / "precision_constraint_specs.json", [])
    fallback_reasons: dict[str, int] = {}
    for row in profile.get("fallback_layers", []) or []:
        reason = str(row.get("fallback_reason", ""))
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    return {
        "profile_id": profile_dir.name,
        "canonical_mapping_success": bool(mapping.get("success")),
        "mapping_entries": int(mapping.get("entry_count", 0) or len(mapping.get("entries", []))),
        "ambiguous_mapping_count": int(mapping.get("ambiguous_mapping_count", 0) or (0 if mapping.get("success") else 1)),
        "qdq_insert_success": bool(qdq.get("success")),
        "inserted_qdq_nodes_count": len(qdq.get("inserted_qdq_nodes") or []),
        "unmatched_int8_precision_groups": qdq.get("unmatched_int8_precision_groups", []),
        "precision_constraint_specs_count": len(specs) if isinstance(specs, list) else 0,
        "requested_int8_group_ratio": profile.get("requested_int8_group_ratio"),
        "requested_int8_layer_count": sum(1 for value in (profile.get("layer_precision_assignment") or {}).values() if str(value).lower() == "int8"),
        "fallback_fp16_count": len(profile.get("fallback_layers", []) or []),
        "fallback_reasons": fallback_reasons,
        "concat_fp16_boundary_count": int(profile.get("concat_fp16_boundary_count", 0) or 0),
        "concat_requantize_after_count": int(profile.get("concat_requantize_after_count", 0) or 0),
    }


def _summarize_combined_dataset(output_dir: Path, profile_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subnet_rows: list[dict[str, Any]] = []
    expanded_profile_rows: list[dict[str, Any]] = []
    for subnet_dir in _subnet_dirs(output_dir):
        manifest = read_json(subnet_dir / "pruning_manifest.json", {})
        export = read_json(subnet_dir / "onnx" / "real_onnx_export_report.json", {})
        grouped = read_json(subnet_dir / "grouped_conv_int8_eligibility_report.json", [])
        contract = summarize_surface_contract_from_manifest(manifest, source_subset=str(manifest.get("source_subset", "")))
        unsupported = sum(1 for row in grouped if isinstance(row, Mapping) and not row.get("int8_shape_supported"))
        subnet_row = {
            "subnet_id": str(manifest.get("subnet_id", subnet_dir.name)),
            "source_subset": manifest.get("source_subset", ""),
            "source_subnet_id": manifest.get("source_subnet_id", ""),
            "structure_hash": manifest.get("structure_hash", ""),
            "shape_hash": manifest.get("shape_hash", ""),
            "materialized_subnet_success": bool((subnet_dir / "pruned_model_object.pth").is_file() or (subnet_dir / "models" / "pruned_model_object.pth").is_file()),
            "pytorch_forward_shape_sanity_passed": bool(manifest.get("pytorch_forward_shape_sanity_passed", manifest.get("shape_invariant_passed", False))),
            "onnx_export_success": bool(export.get("export_success")),
            "origin_map_success": bool(export.get("origin_map_success")),
            "canonical_mapping_success": True,
            "qdq_insert_success": True,
            "ambiguous_mapping_count": 0,
            "grouped_conv_unsupported_int8_shape_count": unsupported,
            **contract,
        }
        for profile_dir in sorted(subnet_dir.glob("profile_*")):
            if not profile_dir.is_dir():
                continue
            prow = {
                "subnet_id": subnet_row["subnet_id"],
                "source_subset": subnet_row["source_subset"],
                **_summarize_profile_gate(profile_dir),
            }
            expanded_profile_rows.append(prow)
            if not prow["canonical_mapping_success"]:
                subnet_row["canonical_mapping_success"] = False
            if not prow["qdq_insert_success"]:
                subnet_row["qdq_insert_success"] = False
            subnet_row["ambiguous_mapping_count"] += int(prow.get("ambiguous_mapping_count", 0) or 0)
        subnet_rows.append(subnet_row)
    if profile_rows:
        expanded_profile_rows.extend(dict(row) for row in profile_rows)
    write_csv(output_dir / "subnet_index.csv", subnet_rows)
    write_csv(output_dir / "mixed_precision_profile_index.csv", expanded_profile_rows)
    return {"subnets": subnet_rows, "profiles": expanded_profile_rows}


def _write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    subnets = list(summary.get("subnets") or [])
    profiles = list(summary.get("profiles") or [])
    lines = [
        "# v12 Combined LUT Dataset Prepare Report",
        "",
        "trtexec_executed: false",
        "engine_build_executed: false",
        "eval_executed: false",
        f"dataset_version: {DATASET_VERSION}",
        f"subnet_count: {len(subnets)}",
        f"profile_count: {len(profiles)}",
        "",
        "## Gate Summary",
        "",
        f"- materialized_subnets_success_count: {sum(1 for row in subnets if row.get('materialized_subnet_success'))}/{len(subnets)}",
        f"- onnx_origin_map_success_count: {sum(1 for row in subnets if row.get('origin_map_success'))}/{len(subnets)}",
        f"- canonical_mapping_success_count: {sum(1 for row in profiles if row.get('canonical_mapping_success'))}/{len(profiles)}",
        f"- qdq_insert_success_count: {sum(1 for row in profiles if row.get('qdq_insert_success'))}/{len(profiles)}",
        f"- ambiguous_mapping_count: {sum(int(row.get('ambiguous_mapping_count', 0) or 0) for row in profiles)}",
        f"- grouped_conv_unsupported_int8_shape_count: {sum(int(row.get('grouped_conv_unsupported_int8_shape_count', 0) or 0) for row in subnets)}",
        f"- new_v12_deblock_output_pruned_count: {sum(1 for row in subnets if row.get('source_subset') == NEW_V12_SUBSET and row.get('deblock_output_pruned'))}",
        f"- historical_deblock_output_pruned_count: {sum(1 for row in subnets if row.get('source_subset') == HISTORICAL_SUBSET and row.get('deblock_output_pruned'))}",
        "",
        "## Recommendation",
        "",
        "Proceed to v12 engine/eval workers only if the gate summary counts are all clean for required gates.",
    ]
    (output_dir / "prepare_v12_combined_lut_dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    subnets_out = output_dir / "subnets"
    output_dir.mkdir(parents=True, exist_ok=True)
    subnets_out.mkdir(parents=True, exist_ok=True)
    profile_rows: list[dict[str, Any]] = []

    historical_dir = Path(args.historical_v11_dir)
    historical_subnets = _subnet_dirs(historical_dir)[: int(args.max_historical_subnets) or None]
    combined_index = 0
    for src in historical_subnets:
        dst = subnets_out / f"subnet_{combined_index:03d}"
        _copy_subnet_tree(src, dst, overwrite=bool(args.overwrite))
        _renumber_manifest(dst, subnet_id=dst.name, source_subset=HISTORICAL_SUBSET, source_subnet_id=src.name)
        if bool(args.run_gate):
            try:
                profile_rows.extend(_gate_profile_existing_subnet(dst, args))
            except Exception as exc:  # noqa: BLE001
                profile_rows.append({"subnet_id": dst.name, "source_subset": HISTORICAL_SUBSET, "profile_id": "", "qdq_insert_success": False, "failure_reason": f"{type(exc).__name__}: {exc}"})
        combined_index += 1

    if int(args.new_v12_subnets) > 0:
        new_work_dir = output_dir / "_new_v12_deblock_protected_work"
        if new_work_dir.exists() and bool(args.overwrite):
            shutil.rmtree(new_work_dir)
        layer_specs, source = load_layer_specs_for_cli(Path(args.source_subnet_dir) if args.source_subnet_dir else None, historical_dir / "subnets")
        generate_random_deployment_aware_subnets(
            layer_specs=layer_specs,
            output_dir=new_work_dir,
            num_subnets=int(args.new_v12_subnets),
            seed=int(args.seed),
            max_channel_prune_ratio=float(args.max_channel_prune_ratio),
            min_channel_keep_ratio=float(args.min_channel_keep_ratio),
            ordinary_conv_round_to=int(args.ordinary_conv_round_to),
            grouped_conv_safe_per_group={int(x) for x in str(args.grouped_conv_safe_per_group).split(",") if x.strip()},
            disable_taylor_ranking=True,
            diversity_reject_duplicates=True,
            source_layer_spec=source,
            global_target_prune_bins=parse_global_target_prune_bins(args.global_target_prune_bins),
            subnets_per_bin=int(args.subnets_per_bin) if int(args.subnets_per_bin) > 0 else None,
            allow_deblock_output_pruning=False,
            protect_deblock_output=True,
        )
        if bool(args.run_gate):
            from tools.latency_lut import run_v11_random_deployment_aware_gate_dryrun as gate

            gate_args = gate.parse_args(
                [
                    "--output-dir",
                    str(new_work_dir),
                    "--max-subnets",
                    str(args.new_v12_subnets),
                    "--round-to",
                    str(args.ordinary_conv_round_to),
                    "--max-channel-prune-ratio",
                    str(args.max_channel_prune_ratio),
                    "--min-channel-keep-ratio",
                    str(args.min_channel_keep_ratio),
                    "--grouped-conv-safe-per-group",
                    str(args.grouped_conv_safe_per_group),
                    "--profile-seed",
                    str(args.profile_seed),
                    "--calib-train-frames",
                    str(args.calib_train_frames),
                    "--fixed-k",
                    str(args.fixed_k),
                    "--heal-root",
                    str(args.heal_root),
                    "--model-config",
                    str(args.model_config),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--plugin",
                    str(args.plugin_path),
                    "--trt-root",
                    str(args.trt_root),
                    "--allow-deblock-output-pruning",
                    "false",
                    "--protect-deblock-output",
                    "true",
                ]
            )
            gate.run(gate_args)
        for src in _subnet_dirs(new_work_dir):
            dst = subnets_out / f"subnet_{combined_index:03d}"
            _copy_subnet_tree(src, dst, overwrite=bool(args.overwrite))
            _renumber_manifest(dst, subnet_id=dst.name, source_subset=NEW_V12_SUBSET, source_subnet_id=src.name)
            combined_index += 1

    summary = _summarize_combined_dataset(output_dir, profile_rows)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "historical_v11_dir": str(historical_dir),
        "new_v12_subnet_count": int(args.new_v12_subnets),
        "eval_frame_count": 300,
        "trtexec_executed": False,
        "engine_build_executed": False,
        "eval_executed": False,
        "subnet_count": len(summary["subnets"]),
        "profile_count": len(summary["profiles"]),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "prepare_v12_combined_lut_dataset_summary.json", summary | {"manifest": manifest})
    _write_report(output_dir, summary)
    print(json.dumps({"success": True, "output_dir": str(output_dir), "subnet_count": len(summary["subnets"]), "profile_count": len(summary["profiles"])}, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-v11-dir", default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-historical-subnets", type=int, default=0)
    parser.add_argument("--new-v12-subnets", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--source-subnet-dir", default="")
    parser.add_argument("--global-target-prune-bins", default="0.0:0.2,0.2:0.4,0.4:0.6,0.6:0.8")
    parser.add_argument("--subnets-per-bin", type=int, default=8)
    parser.add_argument("--max-channel-prune-ratio", type=float, default=0.80)
    parser.add_argument("--min-channel-keep-ratio", type=float, default=0.20)
    parser.add_argument("--ordinary-conv-round-to", type=int, default=4)
    parser.add_argument("--grouped-conv-safe-per-group", default="4,8,16,32")
    parser.add_argument("--profile-seed", type=int, default=20260708)
    parser.add_argument("--calib-train-frames", type=int, default=200)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=29696)
    parser.add_argument("--plugin-path", default="quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
    parser.add_argument("--trt-root", default="${TENSORRT_ROOT}")
    parser.add_argument("--trtexec", default="")
    parser.add_argument("--heal-root", default="../../HEAL")
    parser.add_argument("--model-config", default="${MODEL_ROOT}/lidar_pyramid/config.yaml")
    parser.add_argument("--checkpoint", default="${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--device", default="")
    parser.add_argument("--run-gate", action="store_true", default=True)
    parser.add_argument("--skip-gate", action="store_false", dest="run_gate")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        args = parse_args(argv)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "prepare_v12_failure_report.json", {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        print(json.dumps({"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "output_dir": str(output_dir)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
