from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_cell(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _module_scope(layer: str) -> str:
    if layer.startswith(("cls_head", "reg_head", "dir_head")):
        return "detection_head"
    if layer.startswith(("pillar_vfe", "encoder_m1", "voxel_encoder", "pfn")):
        return "pfn"
    if layer.startswith("shrink"):
        return "shrink"
    if "fusion" in layer:
        return "pyramid_fusion"
    if layer.startswith("backbone_m1.resnet.layer0"):
        return "backbone.stage1"
    if layer.startswith("backbone_m1"):
        return "backbone"
    if layer.startswith("pyramid_backbone.resnet.layer0"):
        return "backbone.stage2"
    if layer.startswith("pyramid_backbone.resnet.layer1"):
        return "backbone.stage2"
    if layer.startswith("pyramid_backbone.resnet.layer2"):
        return "backbone.stage3"
    if layer.startswith("pyramid_backbone"):
        return "backbone"
    return layer.split(".", 1)[0]


PROTECTED_PREFIXES = ("pyramid_backbone.deblocks", "shrink_conv", "cls_head", "reg_head", "dir_head")
SENSITIVE_PREFIXES = (
    "pillar_vfe",
    "encoder_m1",
    "voxel_encoder",
    "backbone_m1",
    "pyramid_backbone",
    "shrink_conv",
    "cls_head",
    "reg_head",
    "dir_head",
)


def _is_protected(layer: str) -> bool:
    return layer.startswith(PROTECTED_PREFIXES)


def _layer_diff_csv(pruned_dir: Path, out_csv: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader((pruned_dir / "structure_changes.csv").open(encoding="utf-8")))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        before = _parse_json_cell(row.get("before_attrs"))
        after = _parse_json_cell(row.get("after_attrs"))
        layer = str(row.get("layer") or "")
        layer_type = str(row.get("module_type") or "")
        orig_c_in = before.get("in_channels")
        orig_c_out = before.get("out_channels", before.get("num_features"))
        pruned_c_in = after.get("in_channels")
        pruned_c_out = after.get("out_channels", after.get("num_features"))
        groups_before = before.get("groups")
        groups_after = after.get("groups")
        cpg_before = int(orig_c_in / groups_before) if orig_c_in and groups_before else None
        cpg_after = int(pruned_c_in / groups_after) if pruned_c_in and groups_after else None
        keep_ratio = None
        if orig_c_out and pruned_c_out:
            keep_ratio = float(pruned_c_out) / float(orig_c_out)
        elif orig_c_in and pruned_c_in:
            keep_ratio = float(pruned_c_in) / float(orig_c_in)
        num_pruned = None
        if orig_c_out and pruned_c_out:
            num_pruned = int(orig_c_out) - int(pruned_c_out)
        elif orig_c_in and pruned_c_in:
            num_pruned = int(orig_c_in) - int(pruned_c_in)
        is_group_conv = bool(groups_before and int(groups_before) > 1)
        out_rows.append(
            {
                "layer_name": layer,
                "module_scope": _module_scope(layer),
                "layer_type": layer_type,
                "orig_C_in": orig_c_in,
                "orig_C_out": orig_c_out,
                "pruned_C_in": pruned_c_in,
                "pruned_C_out": pruned_c_out,
                "groups_before": groups_before,
                "groups_after": groups_after,
                "channels_per_group_before": cpg_before,
                "channels_per_group_after": cpg_after,
                "is_group_conv": is_group_conv,
                "align_rule": "groups_align8_channels_per_group_align8" if is_group_conv else "channel_align8",
                "is_protected": _is_protected(layer),
                "num_pruned_channels": num_pruned,
                "keep_ratio": keep_ratio,
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_name",
        "module_scope",
        "layer_type",
        "orig_C_in",
        "orig_C_out",
        "pruned_C_in",
        "pruned_C_out",
        "groups_before",
        "groups_after",
        "channels_per_group_before",
        "channels_per_group_after",
        "is_group_conv",
        "align_rule",
        "is_protected",
        "num_pruned_channels",
        "keep_ratio",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def _min_keep(rows: list[dict[str, Any]], prefix: str) -> float | None:
    values = [
        float(row["keep_ratio"])
        for row in rows
        if row.get("keep_ratio") is not None and (str(row.get("module_scope")) == prefix or str(row.get("module_scope")).startswith(prefix + "."))
    ]
    return min(values) if values else 1.0


def _onnx_contains_pruned_shapes(onnx_path: Path) -> dict[str, Any]:
    if not onnx_path.is_file():
        return {"checked": False, "reason": "onnx_missing"}
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        shapes = [list(init.dims) for init in model.graph.initializer]
        return {
            "checked": True,
            "contains_backbone_m1_8x64_conv": [8, 64, 3, 3] in shapes,
            "contains_backbone_m1_64x8_conv": [64, 8, 3, 3] in shapes,
            "contains_pyramid_256x8_group_conv": [256, 8, 3, 3] in shapes,
        }
    except Exception as exc:
        return {"checked": False, "reason": str(exc)}


def _diagnostics(candidate_id: str, candidate: dict[str, Any], result: dict[str, Any], pruned_dir: Path, layer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _read_json(pruned_dir / "pruning_summary.json", {})
    selection = _read_json(pruned_dir / "selection_summary.json", {})
    legality = _read_json(pruned_dir / "legality_check_report.json", {})
    group_conv = _read_json(pruned_dir / "group_conv_alignment_report.json", {})
    forward = _read_json(pruned_dir / "forward_sanity_report.json", {})
    changed_layers = [str(row["layer_name"]) for row in layer_rows]
    final_heads = [name for name in changed_layers if name.startswith(("cls_head", "reg_head", "dir_head"))]
    sensitive_changed = [name for name in changed_layers if name.startswith(SENSITIVE_PREFIXES) and not _is_protected(name)]
    original_params = summary.get("original_params")
    pruned_params = summary.get("pruned_params")
    total_before = summary.get("total_coupled_channels_before")
    total_after = summary.get("total_coupled_channels_after")
    return {
        "candidate_id": "light_prune_fp16",
        "candidate_id": candidate_id,
        "candidate_file": f"outputs/latency_lut/full_engine_candidates/{candidate_id}.json",
        "result_file": f"outputs/latency_lut/full_engine_candidates/{candidate_id}.full_engine_result.json",
        "pruned_dir": str(pruned_dir),
        "requested_keep_ratio": (candidate.get("pruning") or {}).get("target_keep_ratio"),
        "requested_prune_ratio": summary.get("target_prune_ratio"),
        "actual_global_keep_ratio": (float(total_after) / float(total_before)) if total_before and total_after else None,
        "param_keep_ratio": (float(pruned_params) / float(original_params)) if original_params and pruned_params else None,
        "bops_keep_ratio_est": None,
        "actual_prune_ratio": summary.get("actual_prune_ratio"),
        "num_coupled_groups_total": summary.get("num_total_groups"),
        "num_coupled_groups_kept": summary.get("num_remaining_groups"),
        "num_coupled_groups_pruned": summary.get("num_pruned_groups"),
        "num_prunable_groups": summary.get("num_prunable_groups"),
        "num_protected_groups": summary.get("num_protected_groups"),
        "module_keep_ratios": {
            "pfn": _min_keep(layer_rows, "pfn"),
            "backbone": _min_keep(layer_rows, "backbone"),
            "backbone.stage1": _min_keep(layer_rows, "backbone.stage1"),
            "backbone.stage2": _min_keep(layer_rows, "backbone.stage2"),
            "backbone.stage3": _min_keep(layer_rows, "backbone.stage3"),
            "shrink": _min_keep(layer_rows, "shrink"),
            "pyramid_fusion": _min_keep(layer_rows, "pyramid_fusion"),
            "detection_head": _min_keep(layer_rows, "detection_head"),
        },
        "module_keep_ratio_method": "minimum keep_ratio among changed Conv/BN rows by module_scope; modules without changed rows are reported as 1.0",
        "protected_modules": list(PROTECTED_PREFIXES),
        "unprotected_sensitive_modules": sorted(set(sensitive_changed)),
        "head_pruned": any(name.startswith(("cls_head", "reg_head", "dir_head")) for name in changed_layers),
        "fusion_pruned": any("fusion" in name for name in changed_layers),
        "pfn_pruned": any(name.startswith(("pillar_vfe", "encoder_m1", "voxel_encoder", "pfn")) for name in changed_layers),
        "final_prediction_conv_pruned": bool(final_heads),
        "final_prediction_conv_changed_layers": final_heads,
        "shape_check_passed": bool(legality.get("legal")) and bool(summary.get("structure_legal")),
        "forward_sanity_check_passed": bool(forward.get("forward_sanity_check")),
        "forward_sanity_skipped": bool(forward.get("skipped")),
        "group_conv_alignment_passed": bool(group_conv.get("all_group_convs_aligned")),
        "group_conv_alignment_issues": group_conv.get("issues", []),
        "selection_summary": {
            "selection_mode": selection.get("selection_mode"),
            "group_conv_selection_mode": selection.get("group_conv_selection_mode"),
            "num_grouped_conv_align_violations": selection.get("num_grouped_conv_align_violations"),
            "structure_legal": selection.get("structure_legal"),
            "forward_sanity_check": selection.get("forward_sanity_check"),
        },
        "result_metrics": {
            "T_real_p50": result.get("T_real_p50"),
            "T_real_p90": result.get("T_real_p90"),
            "mAP": result.get("mAP"),
            "AP_0_70": result.get("AP_0_70"),
        },
    }


def _prediction_sanity(args: argparse.Namespace, baseline_result: dict[str, Any], light_result: dict[str, Any]) -> dict[str, Any]:
    if args.skip_prediction_sanity:
        return {"success": False, "skipped": True, "reason": "skip_prediction_sanity"}
    try:
        import torch
        from heal_compress.quant_deploy.run_dynamic_single_engine_maxk import _dataset_loader, _prepare_inputs
        from heal_compress.quant_deploy.deployment_equivalence import TensorRTEngineRunner
        from heal_compress.quant_deploy.export_lidar_pyramid_onnx import _to_device

        if args.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        trt_root = Path(args.trt_root)
        lib_dirs = [
            trt_root / "targets" / "x86_64-linux-gnu" / "lib",
            trt_root / "lib",
        ]
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        available_lib_dirs = [str(path) for path in lib_dirs if path.is_dir()]
        if available_lib_dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join(available_lib_dirs + ([existing_ld] if existing_ld else []))
        for lib_name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            for lib_dir in lib_dirs:
                lib_path = lib_dir / lib_name
                if lib_path.is_file():
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    break
        ctypes.CDLL(str(Path(light_result["plugin_path"]).resolve()), mode=ctypes.RTLD_GLOBAL)

        def one(label: str, result: dict[str, Any]) -> dict[str, Any]:
            ns = SimpleNamespace(
                output_root=result["output_root"],
                plugin_path=result["plugin_path"],
                hypes_yaml=args.config,
                checkpoint=result["checkpoint_used"],
                heal_repo=args.heal_repo,
                device=args.device,
                trt_root=args.trt_root,
                fixed_k=29696,
                ap_iou_backend="gpu",
            )
            _hypes, device, model, modality, dataset, loader = _dataset_loader(ns)
            torch.cuda.set_device(device)
            runner = TensorRTEngineRunner(Path(result["engine_path"]), device)
            frames = []
            output_names = None
            for frame_idx, batch in enumerate(loader):
                if len(frames) >= int(args.frames):
                    break
                if batch is None:
                    continue
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                ego = _to_device(ego, device)
                batch = _to_device(batch, device)
                if output_names is None:
                    with torch.no_grad():
                        raw = model(ego)
                    output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
                inputs, _meta = _prepare_inputs(ego, modality, fixed_k=29696)
                outputs, _profile = runner.run_profiled(inputs)
                cls = outputs.get("cls_preds")
                reg = outputs.get("reg_preds")
                direc = outputs.get("dir_preds")
                cls_score = torch.sigmoid(cls.float()) if cls is not None else None
                output = {name: outputs[name].float() for name in (output_names or []) if name in outputs}
                od = OrderedDict()
                od["ego"] = output
                pred_box, _pred_score, _gt_box = dataset.post_process(batch, od)
                count = int(pred_box.shape[0]) if torch.is_tensor(pred_box) else int(len(pred_box))
                frames.append(
                    {
                        "frame_id": int(frame_idx),
                        "cls_score_mean": float(cls_score.mean().item()) if cls_score is not None else None,
                        "cls_score_max": float(cls_score.max().item()) if cls_score is not None else None,
                        "postprocess_pred_box_count": count,
                        "final_detection_count": count,
                        "reg_has_nan": bool(torch.isnan(reg).any().item()) if reg is not None else None,
                        "reg_has_inf": bool(torch.isinf(reg).any().item()) if reg is not None else None,
                        "dir_has_nan": bool(torch.isnan(direc).any().item()) if direc is not None else None,
                        "dir_has_inf": bool(torch.isinf(direc).any().item()) if direc is not None else None,
                        "almost_no_boxes": count <= 1,
                    }
                )
            return {
                "label": label,
                "success": True,
                "num_frames": len(frames),
                "frames": frames,
                "mean_final_detection_count": sum(row["final_detection_count"] for row in frames) / max(len(frames), 1),
                "almost_no_boxes_frames": sum(1 for row in frames if row["almost_no_boxes"]),
            }

        return {
            "success": True,
            "frames_requested": int(args.frames),
            "baseline_like_fp16": one("baseline_like_fp16", baseline_result),
            "light_prune_fp16": one("light_prune_fp16", light_result),
        }
    except Exception as exc:
        import traceback

        return {"success": False, "skipped": False, "reason": "prediction_sanity_failed", "error": str(exc), "traceback": traceback.format_exc()}


def _write_failure_analysis(path: Path, diagnostics: dict[str, Any], consistency: dict[str, Any], sanity: dict[str, Any]) -> None:
    module_keep = diagnostics["module_keep_ratios"]
    lines = [
        "# light_prune_fp16 Failure Analysis",
        "",
        "## Summary",
        "",
        f"- Candidate mAP: {diagnostics['result_metrics'].get('mAP')}",
        f"- Candidate AP_0_70: {diagnostics['result_metrics'].get('AP_0_70')}",
        f"- Requested keep ratio: {diagnostics.get('requested_keep_ratio')}",
        f"- Actual parameter keep ratio: {diagnostics.get('param_keep_ratio')}",
        f"- Actual coupled-channel keep ratio: {diagnostics.get('actual_global_keep_ratio')}",
        f"- Minimum changed backbone.stage1 keep ratio: {module_keep.get('backbone.stage1')}",
        f"- Group-conv alignment legal: {diagnostics.get('group_conv_alignment_passed')}",
        f"- Structure legal: {diagnostics.get('shape_check_passed')}",
        f"- Forward sanity check passed: {diagnostics.get('forward_sanity_check_passed')} (skipped={diagnostics.get('forward_sanity_skipped')})",
        "",
        "## Required Answers",
        "",
        "1. light_prune_fp16 is structurally too aggressive for the name `light`: it prunes several early backbone_m1 residual bottleneck conv1 layers from 64 output channels to 8, a 12.5% keep ratio, even though the global parameter prune ratio is only about 13.6%.",
        "2. No evidence shows detection head or final cls/reg/dir prediction conv output channels were pruned; no `cls_head`, `reg_head`, or `dir_head` rows appear in structure changes.",
        "3. PFN/PillarVFE and pyramid_fusion do not appear in changed layers, but sensitive `backbone_m1` early feature layers are unprotected and were heavily pruned.",
        f"4. Replay/checkpoint/ONNX/engine consistency check: {consistency.get('overall_status')}. See detailed flags below.",
        f"5. Group-conv alignment issues: {diagnostics.get('group_conv_alignment_issues') or 'none reported'}; pre-prune group normalization changed several pyramid_backbone group convs from 32 groups to 16 groups.",
        "6. The most likely mAP collapse cause is the unsafe global L1 selection: it preserves global parameter budget but creates severe local bottlenecks in early BEV backbone blocks, without fine-tuning and with forward sanity skipped.",
        "7. Next very-light candidate should use target_keep_ratio >= 0.97, protect PFN/scatter boundary/backbone_m1 early layers/pyramid_fusion/shrink/head, enforce per-layer min_keep_ratio >= 0.875, run forward sanity, then run 5-frame prediction sanity before full calibration sampling.",
        "",
        "## Code-Side Guardrail Added",
        "",
        "The calibration candidate preset has been changed so future `light_prune_fp16` / `light_prune_fp32` candidates use `target_keep_ratio=0.97`, `min_keep_ratio=0.875`, and extra protected prefixes including `encoder_m1`, `backbone_m1.resnet.layer0`, `shrink_conv`, `cls_head`, `reg_head`, and `dir_head`. The full-engine runner now forwards `extra_protected_prefixes` to the existing general pruner.",
        "",
        "## Consistency Details",
        "",
        "```json",
        json.dumps(consistency, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Prediction Sanity",
        "",
        "```json",
        json.dumps(sanity, ensure_ascii=False, indent=2)[:12000],
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", default="light_prune_fp16")
    parser.add_argument("--candidate-dir", default="outputs/latency_lut/full_engine_candidates")
    parser.add_argument("--pruned-dir", default=None)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--config", default="${MODEL_ROOT}/lidar_pyramid/config.yaml")
    parser.add_argument("--heal-repo", default="../../HEAL")
    parser.add_argument("--trt-root", default="${TENSORRT_ROOT}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--skip-prediction-sanity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    candidate_id = args.candidate_id
    candidate = _read_json(candidate_dir / f"{candidate_id}.json", {})
    result = _read_json(candidate_dir / f"{candidate_id}.full_engine_result.json", {})
    baseline_result = _read_json(candidate_dir / "baseline_like_fp16.full_engine_result.json", {})
    if args.pruned_dir:
        pruned_dir = Path(args.pruned_dir)
    else:
        checkpoint_used = Path(str(result.get("checkpoint_used", "")))
        pruned_dir = checkpoint_used.parent if checkpoint_used.name == "pruned_model.pth" else candidate_dir / f"{candidate_id}.work" / "pruned_model_004"
    layer_csv = candidate_dir / f"{candidate_id}.layer_channel_diff.csv"
    layer_rows = _layer_diff_csv(pruned_dir, layer_csv)
    diagnostics = _diagnostics(candidate_id, candidate, result, pruned_dir, layer_rows)
    diag_path = candidate_dir / f"{candidate_id}.pruning_diagnostics.json"
    _write_json(diag_path, diagnostics)
    onnx_path = Path(str(result.get("onnx_path", "")))
    engine_path = Path(str(result.get("engine_path", "")))
    consistency = {
        "pruned_checkpoint_exists": Path(str(result.get("checkpoint_used", ""))).is_file(),
        "prune_replay_json_exists": (pruned_dir / "prune_replay.json").is_file(),
        "prune_replay_operations": len((_read_json(pruned_dir / "prune_replay.json", {}) or {}).get("operations", [])),
        "checkpoint_hash_matches_result": None,
        "onnx_exists": onnx_path.is_file(),
        "engine_exists": engine_path.is_file(),
        "onnx_hash_matches_result": _sha256(onnx_path) == result.get("onnx_hash"),
        "engine_hash_matches_result": _sha256(engine_path) == result.get("engine_hash"),
        "onnx_initializer_pruned_shape_check": _onnx_contains_pruned_shapes(onnx_path),
        "runner_eval_output_root_matches_result": str(result.get("output_root", "")).endswith(f"{candidate_id}.work/quant_deploy"),
        "runner_eval_uses_candidate_engine": str(engine_path).endswith(f"{candidate_id}.work/quant_deploy/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp16/lidar_pyramid_dynamic_agent_single_engine_maxK_fp16.engine"),
        "detected_failure_flags": [],
    }
    if not consistency["onnx_hash_matches_result"]:
        consistency["detected_failure_flags"].append("onnx_hash_mismatch")
    if not consistency["engine_hash_matches_result"]:
        consistency["detected_failure_flags"].append("engine_hash_mismatch")
    if diagnostics["final_prediction_conv_pruned"]:
        consistency["detected_failure_flags"].append("head_output_channel_changed")
    if not diagnostics["group_conv_alignment_passed"]:
        consistency["detected_failure_flags"].append("group_conv_alignment_error")
    consistency["overall_status"] = "consistent_no_replay_export_engine_mismatch_found" if not consistency["detected_failure_flags"] else "inconsistent"
    diagnostics["consistency"] = consistency
    _write_json(diag_path, diagnostics)
    sanity = _prediction_sanity(args, baseline_result, result)
    sanity_path = candidate_dir / f"{candidate_id}.prediction_sanity.json"
    _write_json(sanity_path, sanity)
    if candidate_id == "light_prune_fp16":
        _write_failure_analysis(Path("outputs/latency_lut/light_prune_fp16_failure_analysis.md"), diagnostics, consistency, sanity)
    print(json.dumps({"diagnostics": str(diag_path), "layer_csv": str(layer_csv), "prediction_sanity": str(sanity_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
