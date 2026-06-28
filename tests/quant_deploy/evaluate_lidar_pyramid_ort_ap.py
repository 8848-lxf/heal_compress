from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment_equivalence import TensorRTEngineRunner, _load_model_context, _record_len_value
from evaluate_lidar_pyramid_trt_ap import IOU_THRESHOLDS, _calculate_tp_fp, _mean, _percentile, _timed
from export_lidar_pyramid_onnx import (
    INPUT_NAMES,
    _extract_inputs,
    _input_names_for_export_mode,
    _prepare_export_tensors,
    _tensor_output_names,
    _to_device,
)
from exportable_lidar_pyramid import FixedLidarPyramidExportWrapper
from exportable_lidar_pyramid_dynamic_agent import ExportableLidarPyramidDynamicAgent
from exportable_lidar_pyramid_padded_agent import ExportableLidarPyramidPaddedAgent
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    ensure_quant_deploy_run_dirs,
    read_json,
    save_json,
    write_summary_files,
)


BACKEND_ORDER = [
    ("pytorch_original", "PyTorch original"),
    ("pytorch_export_wrapper", "PyTorch export wrapper"),
    ("onnxruntime_fp32", "ONNXRuntime FP32"),
    ("tensorrt_fp32", "TensorRT FP32"),
    ("tensorrt_fp16", "TensorRT FP16"),
]


def _to_numpy_float(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tensor_to_list(value: torch.Tensor | None, limit: int = 20) -> list[Any]:
    if value is None:
        return []
    tensor = value.detach().float().cpu()
    if tensor.ndim == 0:
        return [float(tensor.item())]
    return tensor[:limit].tolist()


def tensor_stats(value: Any) -> dict[str, Any]:
    arr = _to_numpy_float(value)
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(getattr(value, "dtype", arr.dtype)),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "shape": list(arr.shape),
        "dtype": str(getattr(value, "dtype", arr.dtype)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr, dtype=np.float64)),
        "std": float(np.std(arr, dtype=np.float64)),
    }


def _nearest_tail_percentile(values: np.ndarray, pct: float) -> float | None:
    flat = np.sort(values.reshape(-1))
    if flat.size == 0:
        return None
    index = int(np.ceil((pct / 100.0) * (flat.size - 1)))
    index = min(max(index, 0), flat.size - 1)
    return float(flat[index])


def tensor_error_summary(reference: Any, candidate: Any) -> dict[str, Any]:
    ref = _to_numpy_float(reference)
    cand = _to_numpy_float(candidate)
    same_shape = ref.shape == cand.shape
    report: dict[str, Any] = {
        "same_shape": bool(same_shape),
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "max_abs_error": None,
        "mean_abs_error": None,
        "p95_abs_error": None,
        "p99_abs_error": None,
        "relative_error": None,
    }
    if not same_shape:
        return report
    diff = np.abs(ref - cand)
    if diff.size == 0:
        report.update({"max_abs_error": 0.0, "mean_abs_error": 0.0, "p95_abs_error": 0.0, "p99_abs_error": 0.0, "relative_error": 0.0})
        return report
    mean_abs = float(np.mean(diff, dtype=np.float64))
    ref_mean_abs = float(np.mean(np.abs(ref), dtype=np.float64))
    report.update(
        {
            "max_abs_error": float(np.max(diff)),
            "mean_abs_error": mean_abs,
            "p95_abs_error": _nearest_tail_percentile(diff, 95.0),
            "p99_abs_error": _nearest_tail_percentile(diff, 99.0),
            "relative_error": float(mean_abs / max(ref_mean_abs, 1.0e-12)),
        }
    )
    return report


def output_layout_item(name: str, tensor: torch.Tensor, anchor_num: int, num_bins: int) -> dict[str, Any]:
    shape = list(tensor.shape)
    expected_channels: int | None = None
    expected_shape = None
    permute = None
    channel_semantics = None
    if name == "cls_preds":
        expected_channels = anchor_num
        expected_shape = "[B, anchor_num, H, W]"
        permute = "cls_preds.permute(0, 2, 3, 1).reshape(B, -1)"
        channel_semantics = "one score channel per anchor yaw"
    elif name == "reg_preds":
        expected_channels = anchor_num * 7
        expected_shape = "[B, anchor_num*7, H, W]"
        permute = "reg_preds.permute(0, 2, 3, 1).contiguous().view(B, -1, 7)"
        channel_semantics = "7 box deltas per anchor yaw"
    elif name == "dir_preds":
        expected_channels = anchor_num * num_bins
        expected_shape = "[B, anchor_num*num_bins, H, W]"
        permute = "dir_preds.permute(0, 2, 3, 1).contiguous().reshape(B, -1, num_bins)"
        channel_semantics = "direction logits per anchor yaw and direction bin"
    layout_match = len(shape) == 4 and expected_channels is not None and int(shape[1]) == int(expected_channels)
    return {
        "name": name,
        "shape": shape,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "stride": list(tensor.stride()),
        "postprocess_expected_shape": expected_shape,
        "expected_channels": expected_channels,
        "actual_channels": int(shape[1]) if len(shape) > 1 else None,
        "layout_semantics_match": bool(layout_match),
        "permute_in_postprocess": permute,
        "reshape_in_postprocess": permute,
        "channel_semantics": channel_semantics,
    }


def _ap_value(report: dict[str, Any], key: str) -> float | None:
    aliases = {
        "AP@0.30": ("AP@0.30", "ap_0_3"),
        "AP@0.50": ("AP@0.50", "ap_0_5"),
        "AP@0.70": ("AP@0.70", "ap_0_7"),
        "mAP": ("mAP", "map"),
    }
    for alias in aliases[key]:
        if alias in report and report[alias] is not None:
            return float(report[alias])
    return None


def _mean_ap_from_report(report: dict[str, Any]) -> float | None:
    value = _ap_value(report, "mAP")
    if value is not None:
        return value
    vals = [_ap_value(report, key) for key in ("AP@0.30", "AP@0.50", "AP@0.70")]
    valid = [v for v in vals if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def build_five_way_ap_table(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = _mean_ap_from_report(reports.get("pytorch_original") or {}) or 0.0
    rows: list[dict[str, Any]] = []
    for key, label in BACKEND_ORDER:
        report = reports.get(key) or {}
        map_value = _mean_ap_from_report(report)
        rows.append(
            {
                "backend_key": key,
                "backend": label,
                "AP@0.30": _ap_value(report, "AP@0.30"),
                "AP@0.50": _ap_value(report, "AP@0.50"),
                "AP@0.70": _ap_value(report, "AP@0.70"),
                "mAP": map_value,
                "mAP_drop_vs_PyTorch": round(baseline - map_value, 4) if map_value is not None else None,
                "actual_frames": report.get("actual_frames"),
                "success": report.get("success"),
            }
        )
    return rows


def classify_precision_drop_branch(reports: dict[str, dict[str, Any]], tolerance: float = 0.03) -> dict[str, Any]:
    pytorch_map = _mean_ap_from_report(reports.get("pytorch_original") or {})
    ort_map = _mean_ap_from_report(reports.get("onnxruntime_fp32") or {})
    trt_map = _mean_ap_from_report(reports.get("tensorrt_fp32") or {})
    if pytorch_map is None:
        return {"suspected_area": "insufficient PyTorch baseline AP", "pytorch_map": None}
    ort_close = ort_map is not None and abs(pytorch_map - ort_map) <= tolerance
    trt_close = trt_map is not None and abs(pytorch_map - trt_map) <= tolerance
    if ort_close and not trt_close:
        suspected = "TensorRT runtime/engine output or TRT evaluator"
    elif not ort_close:
        suspected = "ONNX export/fixed_static wrapper/ORT-TRT shared evaluator"
    elif trt_close:
        suspected = "AP is aligned across PyTorch, ONNXRuntime, and TensorRT FP32"
    else:
        suspected = "undetermined"
    return {
        "pytorch_map": pytorch_map,
        "onnxruntime_fp32_map": ort_map,
        "tensorrt_fp32_map": trt_map,
        "tolerance": tolerance,
        "onnxruntime_fp32_close_to_pytorch": bool(ort_close),
        "tensorrt_fp32_close_to_pytorch": bool(trt_close),
        "suspected_area": suspected,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lidar_pyramid ONNXRuntime AP and five-way deployment diagnostics.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--onnx_path", default=None)
    parser.add_argument("--fp32_engine_path", default=None)
    parser.add_argument("--fp16_engine_path", default=None)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ap_iou_backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--top_drop_frames", type=int, default=5)
    parser.add_argument("--pyramid_forward_export_mode", default="fixed_static", choices=["fixed_static", "dynamic_agent_dim", "padded_agent_static"])
    parser.add_argument("--max_cav", type=int, default=2)
    return parser.parse_args(argv)


def _default_onnx_path(dirs: dict[str, Path], summary: dict[str, Any]) -> Path:
    if summary.get("onnx_path"):
        candidate = Path(summary["onnx_path"])
        if candidate.exists():
            return candidate
    return dirs["onnx_fp32"] / "lidar_pyramid_fp32_dynamic.onnx"


def _default_engine_path(dirs: dict[str, Path], precision: str) -> Path:
    return dirs[f"engine_{precision}"] / f"lidar_pyramid_{precision}.engine"


def _make_output_dict(output_names: list[str], outputs: tuple[Any, ...] | dict[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        return {name: outputs[name] for name in output_names if name in outputs and torch.is_tensor(outputs[name])}
    return {name: value for name, value in zip(output_names, outputs)}


def _wrapper_for_mode(model: torch.nn.Module, modality: str, output_names: list[str], mode: str, max_cav: int) -> torch.nn.Module:
    if mode == "dynamic_agent_dim":
        return ExportableLidarPyramidDynamicAgent(model, modality, output_names)
    if mode == "padded_agent_static":
        return ExportableLidarPyramidPaddedAgent(model, modality, output_names, max_cav=max_cav)
    return FixedLidarPyramidExportWrapper(model, modality, output_names)


def _input_names_and_tensors(ego: dict[str, Any], modality: str, mode: str, max_cav: int) -> tuple[list[str], tuple[torch.Tensor, ...]]:
    original_tensors, _agent_modalities = _extract_inputs(ego, modality)
    return _input_names_for_export_mode(mode), _prepare_export_tensors(original_tensors, export_mode=mode, max_cav=max_cav)


def _run_onnx_session(session: Any, tensors_by_name: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    inputs = {name: tensor.detach().cpu().numpy() for name, tensor in tensors_by_name.items()}
    output_values = session.run(None, inputs)
    output_names = [output.name for output in session.get_outputs()]
    return {name: torch.from_numpy(value).to(device=device) for name, value in zip(output_names, output_values)}


def _create_ort_session(ort: Any, onnx_path: Path, providers: list[str]) -> tuple[Any, str]:
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers), "default"
    except Exception as exc:
        if "MatMulBnFusion_Gemm" not in str(exc) and "ShapeInferenceError" not in str(exc):
            raise
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers), "graph_optimization_disabled_after_MatMulBnFusion_Gemm_failure"


def _empty_result_stat() -> dict[float, dict[str, Any]]:
    return {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}


def _empty_backend_state() -> dict[str, Any]:
    return {"result_stat": _empty_result_stat(), "forward_times": [], "post_times": [], "total_times": [], "actual_frames": 0}


def _finalize_ap_report(
    backend_key: str,
    backend_label: str,
    state: dict[str, Any],
    *,
    num_frames: int,
    skipped_frames: int,
    error: str | None = None,
) -> dict[str, Any]:
    try:
        from opencood.utils import eval_utils

        ap: dict[str, float] = {}
        for thr in IOU_THRESHOLDS:
            key = f"AP@{thr:.2f}"
            stat = state["result_stat"][thr]
            if stat["gt"] > 0 and stat["score"]:
                ap_value, _, _ = eval_utils.calculate_ap(state["result_stat"], thr)
            else:
                ap_value = 0.0
            ap[key] = round(float(ap_value), 4)
        success = error is None
    except Exception as exc:
        ap = {f"AP@{thr:.2f}": 0.0 for thr in IOU_THRESHOLDS}
        success = False
        error = error or str(exc)
    return {
        "backend": backend_key,
        "backend_label": backend_label,
        "success": success,
        "num_frames": int(num_frames),
        "actual_frames": int(state.get("actual_frames", 0)),
        "skipped_frames": int(skipped_frames),
        "AP@0.30": ap.get("AP@0.30", 0.0),
        "AP@0.50": ap.get("AP@0.50", 0.0),
        "AP@0.70": ap.get("AP@0.70", 0.0),
        "mAP": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
        "forward_mean_ms": _mean(state["forward_times"]),
        "forward_p50_ms": _percentile(state["forward_times"], 50),
        "forward_p90_ms": _percentile(state["forward_times"], 90),
        "postprocess_mean_ms": _mean(state["post_times"]),
        "postprocess_p50_ms": _percentile(state["post_times"], 50),
        "total_mean_ms": _mean(state["total_times"]),
        "total_p50_ms": _percentile(state["total_times"], 50),
        "error": error,
    }


def _frame_tp_fp_summary(pred_box: Any, pred_score: Any, gt_box: Any, backend: str, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for thr in IOU_THRESHOLDS:
        stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []}}
        _calculate_tp_fp(pred_box, pred_score, gt_box, stat, thr, backend, device)
        tp = int(sum(stat[thr]["tp"]))
        fp = int(sum(stat[thr]["fp"]))
        gt = int(stat[thr]["gt"])
        result[f"{thr:.2f}"] = {
            "tp": tp,
            "fp": fp,
            "fn": max(gt - tp, 0),
            "gt": gt,
            "num_scores": len(stat[thr]["score"]),
        }
    return result


def _postprocess_output(dataset: Any, batch: dict[str, Any], output: dict[str, torch.Tensor], device: torch.device) -> tuple[Any, Any, Any]:
    od = OrderedDict()
    od["ego"] = output
    return dataset.post_process(batch, od)


def _postprocess_stage_summary(
    dataset: Any,
    batch: dict[str, Any],
    output: dict[str, torch.Tensor],
    pred_box: torch.Tensor | None,
    pred_score: torch.Tensor | None,
    gt_box: torch.Tensor | None,
    *,
    topk: int,
) -> dict[str, Any]:
    cav_content = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
    post = dataset.post_processor
    threshold = float(post.params["target_args"]["score_threshold"])
    anchor_num = int(getattr(post, "anchor_num", post.params.get("anchor_args", {}).get("num", 2)))
    num_bins = int(post.params.get("dir_args", {}).get("num_bins", 2))
    raw = {name: tensor_stats(output[name]) for name in ("cls_preds", "reg_preds", "dir_preds") if name in output}

    cls = output["cls_preds"]
    prob = torch.sigmoid(cls.permute(0, 2, 3, 1)).reshape(1, -1)
    flat_scores = prob.reshape(-1)
    score_count = int(flat_scores.numel())
    k = min(int(topk), score_count)
    if k > 0:
        top_scores, top_indices = torch.topk(flat_scores, k=k)
    else:
        top_scores = torch.empty(0, device=cls.device)
        top_indices = torch.empty(0, dtype=torch.long, device=cls.device)
    mask = torch.gt(prob, threshold).view(1, -1)
    reg = output["reg_preds"]
    anchor_box = cav_content["anchor_box"]
    if reg.ndim == 4:
        decoded = post.delta_to_boxes3d(reg, anchor_box)
    else:
        decoded = reg.view(1, -1, 7)
    decoded_flat = decoded[0]
    selected_decoded = decoded_flat[top_indices] if top_indices.numel() else torch.empty((0, 7), device=cls.device)

    dir_summary: dict[str, Any] = {}
    if "dir_preds" in output:
        dir_preds = output["dir_preds"]
        dir_logits = dir_preds.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
        dir_argmax = torch.argmax(dir_logits, dim=-1).reshape(-1)
        counts = torch.bincount(dir_argmax.detach().cpu(), minlength=num_bins)
        dir_summary = {
            "argmax_distribution": {str(i): int(counts[i].item()) for i in range(num_bins)},
            "topk_argmax": dir_argmax[top_indices.to(dir_argmax.device)].detach().cpu().tolist() if top_indices.numel() else [],
        }

    pred_count = 0 if pred_box is None else int(pred_box.shape[0])
    score_tensor = pred_score.detach().float().reshape(-1) if torch.is_tensor(pred_score) else torch.empty(0)
    gt_count = 0 if gt_box is None else int(gt_box.shape[0])
    return {
        "raw_output_stats": raw,
        "score_threshold": threshold,
        "sigmoid_scores": tensor_stats(flat_scores),
        "score_above_threshold_count": int(mask.sum().item()),
        "decoded_boxes_all_count": int(decoded_flat.shape[0]),
        "num_boxes_before_nms": int(mask.sum().item()),
        "topk_scores": _tensor_to_list(top_scores, limit=topk),
        "topk_anchor_indices": top_indices.detach().cpu().tolist(),
        "decoded_boxes_at_topk_anchor_indices": _tensor_to_list(selected_decoded, limit=topk),
        "dir_preds_argmax": dir_summary,
        "num_boxes_after_nms": pred_count,
        "final_pred_score_stats": tensor_stats(score_tensor),
        "max_score": float(score_tensor.max().item()) if score_tensor.numel() else None,
        "mean_score": float(score_tensor.mean().item()) if score_tensor.numel() else None,
        "final_pred_box_tensor": _tensor_to_list(pred_box, limit=topk) if pred_box is not None else [],
        "final_pred_score": _tensor_to_list(score_tensor, limit=topk),
        "gt_box_tensor": _tensor_to_list(gt_box, limit=topk) if gt_box is not None else [],
        "gt_box_count": gt_count,
        "postprocess_expected": {
            "anchor_num": anchor_num,
            "num_bins": num_bins,
            "cls_preds": "[B, anchor_num, H, W]",
            "reg_preds": "[B, anchor_num*7, H, W]",
            "dir_preds": "[B, anchor_num*num_bins, H, W]",
        },
    }


def _decoded_box_error_against_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_boxes = torch.tensor(reference.get("decoded_boxes_at_topk_anchor_indices") or [], dtype=torch.float32)
    cand_boxes = torch.tensor(candidate.get("decoded_boxes_at_topk_anchor_indices") or [], dtype=torch.float32)
    if ref_boxes.numel() == 0 or cand_boxes.numel() == 0:
        return {"available": False}
    limit = min(ref_boxes.shape[0], cand_boxes.shape[0])
    ref_boxes = ref_boxes[:limit]
    cand_boxes = cand_boxes[:limit]
    return {
        "available": True,
        "center_xyz": tensor_error_summary(ref_boxes[:, 0:3], cand_boxes[:, 0:3]),
        "size_hwl": tensor_error_summary(ref_boxes[:, 3:6], cand_boxes[:, 3:6]),
        "yaw": tensor_error_summary(ref_boxes[:, 6], cand_boxes[:, 6]),
    }


def _layout_report_for_outputs(outputs_by_backend: dict[str, dict[str, torch.Tensor]], anchor_num: int, num_bins: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "postprocess_expected_layout": {
            "cls_preds": "[B, anchor_num, H, W]",
            "reg_preds": "[B, anchor_num*7, H, W]",
            "dir_preds": "[B, anchor_num*num_bins, H, W]",
            "anchor_num": anchor_num,
            "num_bins": num_bins,
            "post_processor": "dataset.post_process -> dataset.post_processor.post_process",
            "trt_decode_or_nms_is_separate": False,
        },
        "backends": {},
    }
    for backend, outputs in outputs_by_backend.items():
        report["backends"][backend] = {
            name: output_layout_item(name, tensor, anchor_num=anchor_num, num_bins=num_bins)
            for name, tensor in outputs.items()
            if name in {"cls_preds", "reg_preds", "dir_preds"}
        }
    return report


def _update_summary(output_root: Path, fields: dict[str, Any]) -> None:
    dirs = ensure_quant_deploy_run_dirs(output_root)
    summary = read_json(dirs["summary"] / "summary_all.json", default={}) or {}
    summary.update(fields)
    write_summary_files(summary, dirs)


def run_five_way_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    summary = read_json(dirs["summary"] / "summary_all.json", default={}) or {}
    onnx_path = Path(args.onnx_path) if args.onnx_path else _default_onnx_path(dirs, summary)
    fp32_engine_path = Path(args.fp32_engine_path) if args.fp32_engine_path else _default_engine_path(dirs, "fp32")
    fp16_engine_path = Path(args.fp16_engine_path) if args.fp16_engine_path else _default_engine_path(dirs, "fp16")
    log_lines: list[str] = []
    try:
        import onnxruntime as ort
        from opencood.data_utils.datasets import build_dataset

        hypes, device, model, modality = _load_model_context(args)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dataset = build_dataset(hypes, visualize=True, train=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_batch_test)

        first_batch = next(batch for batch in loader if batch is not None)
        first_batch = _to_device(first_batch, device)
        first_ego = first_batch["ego"] if isinstance(first_batch, dict) and "ego" in first_batch else first_batch
        with torch.no_grad():
            first_raw = model(first_ego)
        output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in first_raw and torch.is_tensor(first_raw[name])]
        if not output_names:
            output_names = _tensor_output_names(first_raw)
        export_mode = getattr(args, "pyramid_forward_export_mode", "fixed_static")
        wrapper = _wrapper_for_mode(model, modality, output_names, export_mode, int(args.max_cav)).to(device).eval()
        providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in ort.get_available_providers()]
        providers = providers or ort.get_available_providers()
        ort_session, ort_session_mode = _create_ort_session(ort, onnx_path, providers)
        trt_fp32_runner = TensorRTEngineRunner(fp32_engine_path, device) if fp32_engine_path.exists() else None
        trt_fp16_runner = TensorRTEngineRunner(fp16_engine_path, device) if fp16_engine_path.exists() else None

        states = {key: _empty_backend_state() for key, _label in BACKEND_ORDER}
        per_frame: list[dict[str, Any]] = []
        top5_candidates: list[dict[str, Any]] = []
        output_layout_report: dict[str, Any] | None = None
        skipped = 0
        actual = 0

        # Rebuild the loader after consuming the first batch.
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_batch_test)
        for frame_idx, batch in enumerate(loader):
            if actual >= int(args.num_frames):
                break
            if batch is None:
                skipped += 1
                continue
            try:
                batch = _to_device(batch, device)
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                input_names, tensors = _input_names_and_tensors(ego, modality, export_mode, int(args.max_cav))
                tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
                outputs_by_backend: dict[str, dict[str, torch.Tensor]] = {}
                forward_ms: dict[str, float] = {}

                with torch.no_grad():
                    outputs_by_backend["pytorch_original"], forward_ms["pytorch_original"] = _timed(lambda: _make_output_dict(output_names, model(ego)), device)
                    wrapper_tuple, forward_ms["pytorch_export_wrapper"] = _timed(lambda: wrapper(*tensors), device)
                    outputs_by_backend["pytorch_export_wrapper"] = _make_output_dict(output_names, wrapper_tuple)
                    outputs_by_backend["onnxruntime_fp32"], forward_ms["onnxruntime_fp32"] = _timed(
                        lambda: _run_onnx_session(ort_session, tensors_by_name, device),
                        device,
                    )
                    if trt_fp32_runner is not None:
                        outputs_by_backend["tensorrt_fp32"], forward_ms["tensorrt_fp32"] = _timed(lambda: trt_fp32_runner.run(tensors_by_name), device)
                    if trt_fp16_runner is not None:
                        outputs_by_backend["tensorrt_fp16"], forward_ms["tensorrt_fp16"] = _timed(lambda: trt_fp16_runner.run(tensors_by_name), device)

                if output_layout_report is None:
                    anchor_num = int(getattr(dataset.post_processor, "anchor_num", 2))
                    num_bins = int(dataset.post_processor.params.get("dir_args", {}).get("num_bins", 2))
                    output_layout_report = _layout_report_for_outputs(outputs_by_backend, anchor_num=anchor_num, num_bins=num_bins)
                    output_layout_report["onnxruntime_providers"] = providers
                    output_layout_report["onnxruntime_session_mode"] = ort_session_mode
                    output_layout_report["onnx_path"] = str(onnx_path)
                    output_layout_report["trt_fp32_engine_path"] = str(fp32_engine_path)
                    output_layout_report["trt_fp16_engine_path"] = str(fp16_engine_path)

                post_outputs: dict[str, dict[str, Any]] = {}
                stage_summaries: dict[str, dict[str, Any]] = {}
                for backend_key, _label in BACKEND_ORDER:
                    if backend_key not in outputs_by_backend:
                        continue
                    output = {name: outputs_by_backend[backend_key][name].float() for name in output_names if name in outputs_by_backend[backend_key]}
                    (pred_box, pred_score, gt_box), post_ms = _timed(lambda out=output: _postprocess_output(dataset, batch, out, device), device)
                    for thr in IOU_THRESHOLDS:
                        _calculate_tp_fp(pred_box, pred_score, gt_box, states[backend_key]["result_stat"], thr, args.ap_iou_backend, device)
                    states[backend_key]["forward_times"].append(forward_ms[backend_key])
                    states[backend_key]["post_times"].append(post_ms)
                    states[backend_key]["total_times"].append(forward_ms[backend_key] + post_ms)
                    states[backend_key]["actual_frames"] += 1
                    stage = _postprocess_stage_summary(
                        dataset,
                        batch,
                        output,
                        pred_box,
                        pred_score,
                        gt_box,
                        topk=int(args.topk),
                    )
                    stage_summaries[backend_key] = stage
                    post_outputs[backend_key] = {
                        "num_boxes_before_nms": stage["num_boxes_before_nms"],
                        "num_boxes_after_nms": stage["num_boxes_after_nms"],
                        "max_score": stage["max_score"],
                        "mean_score": stage["mean_score"],
                        "tp_fp_fn": _frame_tp_fp_summary(pred_box, pred_score, gt_box, args.ap_iou_backend, device),
                    }

                errors: dict[str, dict[str, Any]] = {}
                reference = outputs_by_backend.get("pytorch_export_wrapper", outputs_by_backend["pytorch_original"])
                for backend_key, outputs in outputs_by_backend.items():
                    if backend_key == "pytorch_export_wrapper":
                        continue
                    errors[backend_key + "_vs_pytorch_fixed"] = {
                        name: tensor_error_summary(reference[name], outputs[name])
                        for name in output_names
                        if name in reference and name in outputs
                    }
                if "pytorch_export_wrapper" in outputs_by_backend:
                    errors["pytorch_export_wrapper_vs_original"] = {
                        name: tensor_error_summary(outputs_by_backend["pytorch_original"][name], outputs_by_backend["pytorch_export_wrapper"][name])
                        for name in output_names
                    }

                record_len = _record_len_value(ego)
                py_tp = post_outputs.get("pytorch_original", {}).get("tp_fp_fn", {})
                trt_tp = post_outputs.get("tensorrt_fp32", {}).get("tp_fp_fn", {})
                drop_proxy = 0.0
                for thr_key in ("0.30", "0.50", "0.70"):
                    py_item = py_tp.get(thr_key, {})
                    trt_item = trt_tp.get(thr_key, {})
                    drop_proxy += max(float(py_item.get("tp", 0)) - float(trt_item.get("tp", 0)), 0.0)
                    drop_proxy += max(float(trt_item.get("fp", 0)) - float(py_item.get("fp", 0)), 0.0) * 0.1
                    drop_proxy += max(float(trt_item.get("fn", 0)) - float(py_item.get("fn", 0)), 0.0)

                frame_report = {
                    "frame_id": frame_idx,
                    "record_len": record_len,
                    "is_record_len_1": record_len == 1,
                    "is_record_len_2": record_len == 2,
                    "output_errors_vs_pytorch_fixed": errors,
                    "postprocess_counts": post_outputs,
                    "drop_proxy_trt_fp32_vs_pytorch": drop_proxy,
                }
                per_frame.append(frame_report)
                top5_candidates.append({"rank_score": drop_proxy, "frame_id": frame_idx, "record_len": record_len, "stage_summaries": stage_summaries})
                actual += 1
                log_lines.append(f"frame={frame_idx} record_len={record_len} drop_proxy={drop_proxy:.3f}")
                if actual % 16 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            except Exception as exc:
                skipped += 1
                log_lines.append(f"frame={frame_idx} skipped error={exc}")
                log_lines.append(traceback.format_exc())

        reports = {
            key: _finalize_ap_report(key, label, states[key], num_frames=args.num_frames, skipped_frames=skipped)
            for key, label in BACKEND_ORDER
        }
        five_way_table = build_five_way_ap_table(reports)
        branch = classify_precision_drop_branch(reports)

        top5 = sorted(top5_candidates, key=lambda item: float(item.get("rank_score", 0.0)), reverse=True)[: int(args.top_drop_frames)]
        for item in top5:
            reference_stage = item["stage_summaries"].get("pytorch_original") or item["stage_summaries"].get("pytorch_export_wrapper") or {}
            for backend_key, stage in item["stage_summaries"].items():
                if backend_key == "pytorch_original":
                    continue
                stage["decoded_box_error_vs_pytorch_original_topk"] = _decoded_box_error_against_reference(reference_stage, stage)

        ort_report = reports["onnxruntime_fp32"]
        result = {
            "success": True,
            "onnx_path": str(onnx_path),
            "fp32_engine_path": str(fp32_engine_path),
            "fp16_engine_path": str(fp16_engine_path),
            "num_frames": int(args.num_frames),
            "actual_frames": actual,
            "skipped_frames": skipped,
            "reports": reports,
            "five_way_ap_table": five_way_table,
            "branch_classification": branch,
            "pyramid_forward_export_mode": export_mode,
            "onnxruntime_session_mode": ort_session_mode,
        }
        suffix = "" if export_mode == "fixed_static" else f"_{export_mode}"
        save_json(ort_report, dirs["evaluation"] / f"ort_fp32_ap_report{suffix}.json")
        save_json(result, dirs["evaluation"] / f"five_way_ap_report{suffix}.json")
        save_json({"frames": per_frame, "actual_frames": actual, "skipped_frames": skipped}, dirs["debug"] / f"per_frame_ap_and_error_report{suffix}.json")
        save_json({"frames": top5}, dirs["debug"] / f"postprocess_stage_diff_top5{suffix}.json")
        save_json(output_layout_report or {}, dirs["debug"] / f"output_layout_report{suffix}.json")
        if export_mode == "fixed_static":
            save_json(ort_report, dirs["evaluation"] / "ort_fp32_ap_report.json")
            save_json(result, dirs["evaluation"] / "five_way_ap_report.json")
            save_json({"frames": per_frame, "actual_frames": actual, "skipped_frames": skipped}, dirs["debug"] / "per_frame_ap_and_error_report.json")
            save_json({"frames": top5}, dirs["debug"] / "postprocess_stage_diff_top5.json")
            save_json(output_layout_report or {}, dirs["debug"] / "output_layout_report.json")
        _update_summary(
            Path(args.output_root),
            {
                "ort_fp32_ap_report": ort_report,
                "five_way_ap_table": five_way_table,
                "precision_drop_branch": branch,
                "per_frame_ap_and_error_report": str(dirs["debug"] / "per_frame_ap_and_error_report.json"),
                "postprocess_stage_diff_top5": str(dirs["debug"] / "postprocess_stage_diff_top5.json"),
                "output_layout_report": str(dirs["debug"] / "output_layout_report.json"),
                "trt_precision_equivalence_bug_open": not bool(branch.get("tensorrt_fp32_close_to_pytorch")),
            },
        )
    except Exception as exc:
        result = {"success": False, "error": str(exc), "traceback": traceback.format_exc()}
        save_json(result, dirs["evaluation"] / "five_way_ap_report.json")
        log_lines.append(result["traceback"])
    (dirs["logs_evaluation"] / "evaluate_ort_five_way_ap.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    result = run_five_way_evaluation(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_scalar))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
