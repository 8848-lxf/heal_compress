from __future__ import annotations

import argparse
import ctypes
import sys
import time
import traceback
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bucketed_padded_agent_latency import pad_voxel_tensors_to_fixed_k
from deployment_equivalence import TensorRTEngineRunner, _load_model_context, _record_len_value
from dynamic_single_engine_maxk_common import (
    FIXED_K,
    calibration_cache_path,
    engine_path,
    numeric_summary,
    onnx_path,
    single_engine_input_names,
)
from evaluate_lidar_pyramid_trt_ap import IOU_THRESHOLDS, _calculate_tp_fp, _mean, _percentile, _timed
from export_lidar_pyramid_onnx import _extract_inputs, _to_device
from exportable_lidar_pyramid_fixed_k_scatter_plugin import safe_voxel_num_points_for_fixed_k
from latency_decomposition import LATENCY_FIELDS, summarize_latency_rows
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, DEFAULT_TRT_ROOT, ensure_quant_deploy_run_dirs, read_json, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamic_agent_single_engine_maxK TensorRT engines on val split.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--precision", default="all", choices=["all", "fp32", "fp16", "int8"])
    parser.add_argument("--calibration_frames", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--eval_frames", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--fixed_k", type=int, default=FIXED_K)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--ap_iou_backend", choices=["gpu", "cpu"], default="gpu")
    return parser.parse_args(argv)


def _copy_outputs_to_cpu(outputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float, int, int]:
    start = time.perf_counter()
    copied = {name: tensor.detach().cpu() for name, tensor in outputs.items()}
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    return copied, elapsed, len(copied), sum(int(tensor.numel() * tensor.element_size()) for tensor in copied.values())


def _dataset_loader(args: argparse.Namespace):
    hypes, device, model, modality = _load_model_context(args)
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    return hypes, device, model, modality, dataset, loader


def _prepare_inputs(ego: dict[str, Any], modality: str, *, fixed_k: int = FIXED_K) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    start = time.perf_counter()
    original_tensors, _agent_modalities = _extract_inputs(ego, modality)
    voxel_features, voxel_coords, voxel_num_points, _record_len, pairwise_t_matrix = original_tensors
    n_agents = int(_record_len_value(ego))
    original_num_voxels = int(voxel_features.shape[0])
    if original_num_voxels > int(fixed_k):
        raise ValueError(f"num_voxels={original_num_voxels} exceeds fixed_K={int(fixed_k)}")
    tensors_by_name = {
        "voxel_features": voxel_features.float(),
        "voxel_coords": voxel_coords.to(torch.int32),
        "voxel_num_points": voxel_num_points.to(torch.int32),
        "pairwise_t_matrix": pairwise_t_matrix[:, :n_agents, :n_agents, :, :].float(),
    }
    padded, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, int(fixed_k))
    padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_mask).to(torch.int32)
    padded["valid_voxel_mask"] = valid_mask.float()
    inputs = {name: padded[name] for name in single_engine_input_names()}
    padding_ms = (time.perf_counter() - start) * 1000.0
    meta = {
        "record_len": int(n_agents),
        "N": int(n_agents),
        "original_num_voxels": original_num_voxels,
        "fixed_K": int(fixed_k),
        "padding_voxel_count": int(int(fixed_k) - original_num_voxels),
        "padding_ratio": float((int(fixed_k) - original_num_voxels) / int(fixed_k)),
        "valid_voxel_count": int(valid_mask.sum().item()),
        "padding_ms": padding_ms,
    }
    return inputs, meta


def _report_paths(dirs: dict[str, Path], precision: str, frames: int, calibration_frames: int | None) -> tuple[Path, Path]:
    if precision == "int8":
        suffix = f"int8_train_calib{int(calibration_frames)}_val{int(frames)}"
    else:
        suffix = f"{precision}_val{int(frames)}"
    return (
        dirs["evaluation"] / f"dynamic_single_engine_maxK_{suffix}.json",
        dirs["benchmark"] / f"dynamic_single_engine_maxK_{suffix}.json",
    )


def _engine_for(dirs: dict[str, Path], precision: str, calibration_frames: int | None, fixed_k: int = FIXED_K) -> Path:
    if precision == "int8":
        return engine_path(dirs, "int8", int(calibration_frames), fixed_k=int(fixed_k))
    return engine_path(dirs, precision, fixed_k=int(fixed_k))


def _baseline_multi_engine_fp16(dirs: dict[str, Path], frames: int) -> float | None:
    report = read_json(dirs["evaluation"] / f"trt_fp16_ap_report_dynamic_agent_dim_fixed_k_plugin_{int(frames)}.json", default={}) or {}
    value = report.get("mAP", report.get("map"))
    return float(value) if value is not None else None


def evaluate_one(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    *,
    precision: str,
    frames: int,
    calibration_frames: int | None = None,
) -> dict[str, Any]:
    plugin_path = Path(args.plugin_path).expanduser()
    ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    hypes, device, model, modality, dataset, loader = _dataset_loader(args)
    del hypes
    if device.type != "cuda":
        raise RuntimeError("TensorRT dynamic single-engine maxK evaluation requires CUDA.")
    torch.cuda.set_device(device)
    engine = _engine_for(dirs, precision, calibration_frames, fixed_k=int(args.fixed_k))
    runner = TensorRTEngineRunner(engine, device)
    output_names: list[str] | None = None
    result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
    rows: list[dict[str, Any]] = []
    forward_times: list[float] = []
    post_times: list[float] = []
    total_times: list[float] = []
    skipped: list[dict[str, Any]] = []
    lines: list[str] = []
    actual = 0
    for frame_idx, batch in enumerate(loader):
        if actual >= int(frames):
            break
        if batch is None:
            skipped.append({"frame_id": int(frame_idx), "reason": "batch_is_none"})
            continue
        try:
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            ego = _to_device(ego, device)
            batch = _to_device(batch, device)
            if output_names is None:
                with torch.no_grad():
                    raw = model(ego)
                output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
            inputs, input_meta = _prepare_inputs(ego, modality, fixed_k=int(args.fixed_k))
            outputs, profile = runner.run_profiled(inputs)
            _cpu_outputs, d2h_ms, d2h_copies, d2h_bytes = _copy_outputs_to_cpu(outputs)
            output = {name: outputs[name].float() for name in (output_names or [])}

            def _postprocess():
                od = OrderedDict()
                od["ego"] = output
                return dataset.post_process(batch, od)

            (pred_box, pred_score, gt_box), post_ms = _timed(_postprocess, device)
            for thr in IOU_THRESHOLDS:
                _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr, args.ap_iou_backend, device)
            fwd_ms = float(profile.get("total_runner_ms", 0.0)) + d2h_ms
            forward_times.append(fwd_ms)
            post_times.append(post_ms)
            total_times.append(fwd_ms + post_ms)
            row = {
                "frame_id": int(frame_idx),
                "sample_idx": int(actual),
                "strategy": "dynamic_agent_single_engine_maxK",
                "precision": precision,
                "calibration_split": "train" if precision == "int8" else None,
                "evaluation_split": "val",
                "calibration_frames": int(calibration_frames) if precision == "int8" else None,
                "calibration_eval_overlap": False,
                "single_engine": True,
                "engine_count": 1,
                "record_len": int(input_meta["record_len"]),
                "N_runtime_shape": int(input_meta["N"]),
                "fixed_K": int(args.fixed_k),
                "original_num_voxels": int(input_meta["original_num_voxels"]),
                "padding_voxel_count": int(input_meta["padding_voxel_count"]),
                "padding_ratio": float(input_meta["padding_ratio"]),
                "valid_voxel_count": int(input_meta["valid_voxel_count"]),
                "padding_ms": float(input_meta["padding_ms"]),
                "shape_setup_ms": float(profile.get("set_input_shape_ms", 0.0)),
                "dtype_cast_ms": profile.get("dtype_cast_ms", 0.0),
                "contiguous_ms": profile.get("contiguous_ms", 0.0),
                "h2d_copy_ms": profile.get("h2d_copy_ms", 0.0),
                "input_device_copy_ms": profile.get("input_device_copy_ms", 0.0),
                "set_input_shape_ms": profile.get("set_input_shape_ms", 0.0),
                "output_shape_query_ms": profile.get("output_shape_query_ms", 0.0),
                "bind_address_ms": profile.get("bind_address_ms", 0.0),
                "execute_ms": profile.get("execute_async_ms", 0.0),
                "execute_async_ms": profile.get("execute_async_ms", 0.0),
                "synchronize_ms": profile.get("synchronize_ms", 0.0),
                "d2h_copy_ms": d2h_ms,
                "output_wrap_ms": profile.get("output_wrap_ms", 0.0),
                "forward_ms": fwd_ms,
                "total_runner_ms": fwd_ms,
                "postprocess_ms": post_ms,
                "d2h_copies": d2h_copies,
                "bytes_d2h": d2h_bytes,
                "input_shapes": profile.get("input_shapes", {}),
                "output_shapes": profile.get("output_shapes", {}),
                "engine_path": str(engine),
                "no_bucket_router": True,
                "no_N_engine_router": True,
                "valid_voxel_mask enabled": True,
                "pointpillar_scatter_plugin enabled": True,
            }
            rows.append(row)
            actual += 1
            lines.append(
                f"frame={frame_idx} precision={precision} calib={calibration_frames} record_len={row['record_len']} "
                f"K={row['original_num_voxels']} forward_ms={fwd_ms:.3f} execute_ms={row['execute_ms']:.3f}"
            )
        except Exception as exc:
            skipped.append({"frame_id": int(frame_idx), "reason": "exception", "error": str(exc)})
            lines.append(f"frame={frame_idx} skipped error={exc}")
            lines.append(traceback.format_exc())
    from opencood.utils import eval_utils

    ap: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        key = f"AP@{thr:.2f}"
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_value = 0.0
        ap[key] = round(float(ap_value), 4)
    latency = summarize_latency_rows(rows, fields=[*LATENCY_FIELDS, "execute_ms", "forward_ms", "padding_ms", "shape_setup_ms"])
    overall = latency.get("overall") or {}
    report = {
        "success": True,
        "strategy": "dynamic_agent_single_engine_maxK",
        "precision": precision,
        "calibration_split": "train" if precision == "int8" else None,
        "evaluation_split": "val",
        "calibration_frames": int(calibration_frames) if precision == "int8" else None,
        "eval_frames": int(frames),
        "num_frames": int(frames),
        "actual_frames": actual,
        "calibration_eval_overlap": False,
        "single_engine": True,
        "engine_count": 1,
        "AP@0.30": ap.get("AP@0.30", 0.0),
        "AP@0.50": ap.get("AP@0.50", 0.0),
        "AP@0.70": ap.get("AP@0.70", 0.0),
        "mAP": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
        "map": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
        "mAP drop vs single-engine FP16 same val set": None,
        "mAP drop vs multi-engine FP16 same val set": None,
        "AP@0.70 drop vs FP16": None,
        "execute_ms": overall.get("execute_ms"),
        "forward_ms": overall.get("forward_ms"),
        "total_runner_ms": overall.get("total_runner_ms"),
        "shape_setup_ms": overall.get("shape_setup_ms"),
        "padding_ms": overall.get("padding_ms"),
        "forward_mean_ms": _mean(forward_times),
        "forward_p50_ms": _percentile(forward_times, 50),
        "forward_p90_ms": _percentile(forward_times, 90),
        "forward_p95_ms": _percentile(forward_times, 95),
        "forward_p99_ms": _percentile(forward_times, 99),
        "FPS": float(1000.0 / _percentile(forward_times, 50)) if forward_times else None,
        "fps": float(1000.0 / _percentile(forward_times, 50)) if forward_times else None,
        "postprocess_mean_ms": _mean(post_times),
        "total_mean_ms": _mean(total_times),
        "record_len distribution": dict(Counter(str(row["record_len"]) for row in rows)),
        "N runtime shape distribution": dict(Counter(str(row["N_runtime_shape"]) for row in rows)),
        "original_num_voxels distribution": numeric_summary([row["original_num_voxels"] for row in rows]),
        "padding_ratio distribution": numeric_summary([row["padding_ratio"] for row in rows]),
        "fixed_K": int(args.fixed_k),
        "skipped_samples": skipped,
        "skip_reasons": dict(Counter(str(item.get("reason", "unknown")) for item in skipped)),
        "engine_path": str(engine),
        "onnx_path": str(onnx_path(dirs, fixed_k=int(args.fixed_k))),
        "plugin_path": str(plugin_path),
        "calibration_cache_path": str(calibration_cache_path(dirs, int(calibration_frames), fixed_k=int(args.fixed_k))) if precision == "int8" else None,
        "no_bucket_router": True,
        "no_N_engine_router": True,
        "valid_voxel_mask enabled": True,
        "pointpillar_scatter_plugin enabled": True,
        "latency_summary": latency,
        "frames": rows,
        "output_names": output_names or [],
    }
    eval_path, bench_path = _report_paths(dirs, precision, frames, calibration_frames)
    save_json(report, eval_path)
    save_json(
        {
            "strategy": "dynamic_agent_single_engine_maxK",
            "precision": precision,
            "calibration_split": report.get("calibration_split"),
            "evaluation_split": "val",
            "calibration_frames": report.get("calibration_frames"),
            "eval_frames": int(frames),
            "single_engine": True,
            "engine_count": 1,
            "execute_ms": report["execute_ms"],
            "forward_ms": report["forward_ms"],
            "total_runner_ms": report["total_runner_ms"],
            "shape_setup_ms": report["shape_setup_ms"],
            "padding_ms": report["padding_ms"],
            "FPS": report["FPS"],
            "engine_path": str(engine),
        },
        bench_path,
    )
    log_suffix = f"{precision}_val{frames}" if precision != "int8" else f"int8_train_calib{calibration_frames}_val{frames}"
    (dirs["logs_evaluation"] / f"evaluate_dynamic_single_engine_maxK_{log_suffix}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _update_drops(dirs: dict[str, Path], frames: int, reports: list[dict[str, Any]]) -> None:
    fp16 = next((row for row in reports if row.get("precision") == "fp16" and int(row.get("eval_frames") or 0) == int(frames)), None)
    fp16_map = float(fp16["mAP"]) if fp16 and fp16.get("mAP") is not None else None
    fp16_ap70 = float(fp16["AP@0.70"]) if fp16 and fp16.get("AP@0.70") is not None else None
    multi_fp16 = _baseline_multi_engine_fp16(dirs, frames)
    for report in reports:
        if int(report.get("eval_frames") or 0) != int(frames):
            continue
        if fp16_map is not None:
            report["mAP drop vs single-engine FP16 same val set"] = round(fp16_map - float(report["mAP"]), 4)
        if multi_fp16 is not None:
            report["mAP drop vs multi-engine FP16 same val set"] = round(multi_fp16 - float(report["mAP"]), 4)
        if fp16_ap70 is not None:
            report["AP@0.70 drop vs FP16"] = round(fp16_ap70 - float(report["AP@0.70"]), 4)
        eval_path, bench_path = _report_paths(
            dirs,
            str(report["precision"]),
            int(frames),
            int(report["calibration_frames"]) if report.get("precision") == "int8" else None,
        )
        save_json(report, eval_path)
        bench = read_json(bench_path, default={}) or {}
        bench.update(
            {
                "mAP_drop_vs_single_engine_FP16_same_val_set": report.get("mAP drop vs single-engine FP16 same val set"),
                "mAP_drop_vs_multi_engine_FP16_same_val_set": report.get("mAP drop vs multi-engine FP16 same val set"),
                "AP@0.70_drop_vs_FP16": report.get("AP@0.70 drop vs FP16"),
            }
        )
        save_json(bench, bench_path)


def evaluate_all(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    reports: list[dict[str, Any]] = []
    precisions = ["fp32", "fp16", "int8"] if args.precision == "all" else [args.precision]
    for frames in args.eval_frames:
        for precision in precisions:
            if precision == "int8":
                for calibration_frames in args.calibration_frames:
                    reports.append(evaluate_one(args, dirs, precision="int8", frames=int(frames), calibration_frames=int(calibration_frames)))
            else:
                reports.append(evaluate_one(args, dirs, precision=precision, frames=int(frames), calibration_frames=None))
        _update_drops(dirs, int(frames), reports)
    return {"success": all(row.get("success") for row in reports), "reports": reports}


def main(argv: list[str] | None = None) -> int:
    report = evaluate_all(parse_args(argv))
    print(report)
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
