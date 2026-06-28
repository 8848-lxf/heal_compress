from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment_equivalence import TensorRTEngineRunner, _load_model_context
from export_lidar_pyramid_onnx import (
    INPUT_NAMES,
    _extract_inputs,
    _input_names_for_export_mode,
    _prepare_export_tensors,
    _tensor_output_names,
    _to_device,
)
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    ensure_quant_deploy_run_dirs,
    save_csv,
    save_json,
)


IOU_THRESHOLDS = (0.30, 0.50, 0.70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lidar_pyramid TensorRT engine AP via HEAL post_process.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    parser.add_argument("--engine_path", default=None)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ap_iou_backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--pyramid_forward_export_mode", default="fixed_static", choices=["fixed_static", "dynamic_agent_dim", "padded_agent_static"])
    parser.add_argument("--max_cav", type=int, default=2)
    return parser.parse_args(argv)


def _default_engine_path(dirs: dict[str, Path], precision: str) -> Path:
    return dirs[f"engine_{precision}"] / f"lidar_pyramid_{precision}.engine"


def _timed(fn, device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat: dict[float, dict[str, Any]], threshold: float, backend: str, device: torch.device) -> None:
    if backend == "cpu":
        from opencood.utils import eval_utils

        eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, threshold)
        return
    try:
        from tests.test_baseline_eval import calculate_tp_fp_for_threshold

        calculate_tp_fp_for_threshold(pred_box, pred_score, gt_box, result_stat, threshold, "gpu", device)
    except Exception:
        from opencood.utils import eval_utils

        eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, threshold)


def evaluate_engine(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    precision = args.precision.lower()
    engine_path = Path(args.engine_path) if args.engine_path else _default_engine_path(dirs, precision)
    out_dir = dirs[f"evaluation_{precision}"]
    log_path = dirs["logs_evaluation"] / f"evaluate_ap_{precision}.log"
    if not engine_path.exists():
        result = {"precision": precision, "success": False, "error": f"engine file does not exist: {engine_path}"}
        save_json(result, out_dir / f"eval_metrics_{precision}.json")
        save_csv([result], out_dir / f"eval_metrics_{precision}.csv")
        log_path.write_text(result["error"] + "\n", encoding="utf-8")
        return result

    lines: list[str] = []
    try:
        hypes, device, model, modality = _load_model_context(args)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        from opencood.data_utils.datasets import build_dataset
        from opencood.utils import eval_utils

        dataset = build_dataset(hypes, visualize=True, train=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_batch_test)
        result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
        total_times: list[float] = []
        forward_times: list[float] = []
        post_times: list[float] = []
        actual = 0
        skipped = 0
        output_names: list[str] | None = None
        trt_runner = TensorRTEngineRunner(engine_path, device)

        for frame_idx, batch in enumerate(loader):
            if actual >= int(args.num_frames):
                break
            if batch is None:
                skipped += 1
                continue
            try:
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                ego = _to_device(ego, device)
                batch = _to_device(batch, device)
                if output_names is None:
                    with torch.no_grad():
                        raw = model(ego)
                    output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
                    if not output_names:
                        output_names = _tensor_output_names(raw)
                original_tensors, _agent_modalities = _extract_inputs(ego, modality)
                input_names = _input_names_for_export_mode(args.pyramid_forward_export_mode)
                tensors = _prepare_export_tensors(original_tensors, export_mode=args.pyramid_forward_export_mode, max_cav=int(args.max_cav))
                tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
                trt_outputs, fwd_ms = _timed(lambda: trt_runner.run(tensors_by_name), device)
                output = {name: trt_outputs[name].float() for name in output_names}

                def _postprocess():
                    od = OrderedDict()
                    od["ego"] = output
                    return dataset.post_process(batch, od)

                (pred_box, pred_score, gt_box), post_ms = _timed(_postprocess, device)
                for thr in IOU_THRESHOLDS:
                    _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr, args.ap_iou_backend, device)
                forward_times.append(fwd_ms)
                post_times.append(post_ms)
                total_times.append(fwd_ms + post_ms)
                actual += 1
                lines.append(f"frame={frame_idx} precision={precision} forward_ms={fwd_ms:.3f} post_ms={post_ms:.3f}")
                if actual % 128 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception as exc:
                skipped += 1
                lines.append(f"frame={frame_idx} skipped error={exc}")
                lines.append(traceback.format_exc())

        ap: dict[str, float] = {}
        for thr in IOU_THRESHOLDS:
            key = f"AP@{thr:.2f}"
            if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
                ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
            else:
                ap_value = 0.0
            ap[key] = round(float(ap_value), 4)
        result = {
            "precision": precision,
            "success": True,
            "engine_path": str(engine_path),
            "num_frames": int(args.num_frames),
            "actual_frames": actual,
            "skipped_frames": skipped,
            "ap_0_3": ap.get("AP@0.30", 0.0),
            "ap_0_5": ap.get("AP@0.50", 0.0),
            "ap_0_7": ap.get("AP@0.70", 0.0),
            "map": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
            "forward_mean_ms": _mean(forward_times),
            "forward_p50_ms": _percentile(forward_times, 50),
            "forward_p90_ms": _percentile(forward_times, 90),
            "postprocess_mean_ms": _mean(post_times),
            "postprocess_p50_ms": _percentile(post_times, 50),
            "total_mean_ms": _mean(total_times),
            "total_p50_ms": _percentile(total_times, 50),
            "output_names": output_names or [],
            "pyramid_forward_export_mode": args.pyramid_forward_export_mode,
        }
    except Exception as exc:
        result = {
            "precision": precision,
            "success": False,
            "engine_path": str(engine_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        lines.append(result["traceback"])

    suffix = "" if args.pyramid_forward_export_mode == "fixed_static" else f"_{args.pyramid_forward_export_mode}"
    save_json(result, out_dir / f"eval_metrics_{precision}{suffix}.json")
    save_csv([result], out_dir / f"eval_metrics_{precision}{suffix}.csv")
    if args.pyramid_forward_export_mode == "fixed_static":
        save_json(result, out_dir / f"eval_metrics_{precision}.json")
        save_csv([result], out_dir / f"eval_metrics_{precision}.csv")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    result = evaluate_engine(parse_args(argv))
    print(result)
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
