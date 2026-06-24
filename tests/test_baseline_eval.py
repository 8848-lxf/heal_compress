#!/usr/bin/env python3
"""Baseline evaluation of original HEAL LiDAROnly model on DAIR-V2X full val set.

Outputs AP@0.03 / AP@0.30 / AP@0.50 / AP@0.70 and timing statistics
(total/forward/postprocess: mean, p50).

Example commands
================
Evaluate lidar_pyramid original model (auto GPU, exclude 5/6/7):
  python tests/test_baseline_eval.py \
      --model-name lidar_pyramid \
      --model-config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml \
      --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth

Evaluate lidar_cobevt original model on specific GPU:
  python tests/test_baseline_eval.py \
      --model-name lidar_cobevt \
      --model-config /home/lixingfeng/UniAD_examine/HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_cobevt.yaml \
      --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth \
      --device cuda:2

Evaluate with CPU AP backend:
  python tests/test_baseline_eval.py \
      --model-name lidar_pyramid \
      --model-config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml \
      --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
      --ap-iou-backend cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup: add HEAL and Auto_Search to sys.path
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_HEAL_COMPRESS_ROOT = _THIS_DIR.parent          # heal_compress/
_UNIAD_EXAMINE = _HEAL_COMPRESS_ROOT.parent     # UniAD_examine/

HEAL_REPO = str(_UNIAD_EXAMINE / "HEAL")
AUTO_SEARCH = str(_UNIAD_EXAMINE / "Auto_Search")

for _p in [str(_HEAL_COMPRESS_ROOT), HEAL_REPO, AUTO_SEARCH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Now we can import
from utils.model_utils import auto_select_gpu, resolve_device  # heal_compress
from coop_lidar_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
from coop_lidar_compress.utils.io import ensure_dir, save_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IOU_THRESHOLDS = (0.03, 0.30, 0.50, 0.70)

DEFAULT_CONFIGS = {
    "lidar_pyramid": {
        "config": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth",
    },
    "lidar_cobevt": {
        "config": "HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_cobevt.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth",
    },
    "lidar_attfuse": {
        "config": "HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_attfuse.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_attfuse/net_epoch_bestval_at33.pth",
    },
    "lidar_v2xvit": {
        "config": "HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_v2xvit.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth",
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("baseline_eval")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "eval_log.txt", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def timed_call(fn: Callable[[], Any], device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - t0) * 1000.0


def maybe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def maybe_median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------
def calculate_tp_fp_for_threshold(
    det_boxes, det_score, gt_boxes, result_stat, iou_thresh: float,
    backend: str, device: torch.device,
) -> None:
    gt = 0 if gt_boxes is None else int(gt_boxes.shape[0])
    if det_boxes is None or det_score is None or int(det_boxes.shape[0]) == 0:
        result_stat[iou_thresh]["gt"] += gt
        return
    if backend == "gpu":
        _calculate_tp_fp_gpu_bev(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, device)
    elif backend == "cpu":
        from opencood.utils import eval_utils
        eval_utils.caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh)
    else:
        raise ValueError(f"Unknown AP IoU backend: {backend}")


def _calculate_tp_fp_gpu_bev(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, device):
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
    from opencood.utils import box_utils

    gt = 0 if gt_boxes is None else int(gt_boxes.shape[0])
    det_boxes = det_boxes.to(device=device, dtype=torch.float32)
    det_score = det_score.to(device=device, dtype=torch.float32).reshape(-1)
    order = torch.argsort(det_score, descending=True)
    det_score = det_score[order]
    det_boxes = det_boxes[order]
    result_stat[iou_thresh]["score"] += det_score.detach().cpu().tolist()

    if gt == 0:
        n = int(det_boxes.shape[0])
        result_stat[iou_thresh]["fp"] += [1] * n
        result_stat[iou_thresh]["tp"] += [0] * n
        result_stat[iou_thresh]["gt"] += 0
        return

    gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
    det7 = box_utils._corners_to_nms_boxes_torch(det_boxes).contiguous()
    gt7 = box_utils._corners_to_nms_boxes_torch(gt_boxes).contiguous()
    iou_matrix = boxes_iou_bev(det7, gt7)
    available = torch.ones((gt,), dtype=torch.bool, device=device)
    fp, tp = [], []
    for i in range(iou_matrix.shape[0]):
        if not available.any():
            fp.append(1); tp.append(0); continue
        ious = iou_matrix[i].clone()
        ious[~available] = -1.0
        max_iou, gt_idx = torch.max(ious, dim=0)
        if float(max_iou.item()) < iou_thresh:
            fp.append(1); tp.append(0)
        else:
            fp.append(0); tp.append(1); available[gt_idx] = False
    result_stat[iou_thresh]["fp"] += fp
    result_stat[iou_thresh]["tp"] += tp
    result_stat[iou_thresh]["gt"] += gt


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------
def build_dataset(adapter: HEALLiDARAdapter, model_config: str, batch_size: int = 1, num_workers: int = 4):
    from opencood.data_utils.datasets import build_dataset as heal_build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(model_config))
    hypes = adapter._absolutize_dataset_paths(hypes)
    dataset = heal_build_dataset(hypes, visualize=True, train=False)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": dataset.collate_batch_test,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = lambda _: torch.set_num_threads(1)
    loader = DataLoader(dataset, **kwargs)
    return hypes, dataset, loader


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(adapter: HEALLiDARAdapter, model_config: str, checkpoint: str, device: torch.device, logger: logging.Logger):
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(model_config))
    model = train_utils.create_model(hypes)

    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("load_state_dict missing keys (%d): %s", len(missing), missing[:10])
    if unexpected:
        logger.warning("load_state_dict unexpected keys (%d): %s", len(unexpected), unexpected[:10])

    model = model.to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    logger.info("Model loaded: params=%d (%.2f MB), device=%s", param_count, param_mb, device)
    return model, param_count, param_mb


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(
    model, dataset, loader, device: torch.device, logger: logging.Logger,
    ap_iou_backend: str = "gpu", verbose: bool = True,
) -> dict[str, Any]:
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
    total_times, forward_times, post_times = [], [], []
    total_frames = len(dataset)
    actual = 0
    skipped = 0

    model.eval()
    for frame_idx, batch_data in enumerate(loader):
        if batch_data is None:
            skipped += 1
            continue
        try:
            batch_data = train_utils.to_device(batch_data, device)
            with torch.no_grad():
                output, fwd_ms = timed_call(lambda: model(batch_data["ego"]), device)

                def _postprocess():
                    od = OrderedDict()
                    od["ego"] = output
                    return dataset.post_process(batch_data, od)

                (pred_box, pred_score, gt_box), post_ms = timed_call(_postprocess, device)

            for thr in IOU_THRESHOLDS:
                calculate_tp_fp_for_threshold(pred_box, pred_score, gt_box, result_stat, thr, ap_iou_backend, device)

            total_ms = fwd_ms + post_ms
            total_times.append(total_ms)
            forward_times.append(fwd_ms)
            post_times.append(post_ms)
            actual += 1

            if verbose:
                logger.info(
                    "[%d/%d] total=%.1fms forward=%.1fms postprocess=%.1fms",
                    frame_idx + 1, total_frames, total_ms, fwd_ms, post_ms,
                )

            del output, pred_box, pred_score, gt_box, batch_data
            if (frame_idx + 1) % 256 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:
            skipped += 1
            logger.exception("frame %d skipped: %s", frame_idx, exc)

    # Compute AP
    ap = {}
    for thr in IOU_THRESHOLDS:
        key = f"AP@{thr:.2f}"
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_val, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_val = 0.0
        ap[key] = round(float(ap_val), 4)

    summary = {
        "total_frames": total_frames,
        "actual_frames": actual,
        "skipped_frames": skipped,
        "AP@0.03": ap.get("AP@0.03", 0.0),
        "AP@0.30": ap.get("AP@0.30", 0.0),
        "AP@0.50": ap.get("AP@0.50", 0.0),
        "AP@0.70": ap.get("AP@0.70", 0.0),
        "total_time_mean_ms": round(maybe_mean(total_times), 3),
        "total_time_p50_ms": round(maybe_median(total_times), 3),
        "forward_time_mean_ms": round(maybe_mean(forward_times), 3),
        "forward_time_p50_ms": round(maybe_median(forward_times), 3),
        "postprocess_time_mean_ms": round(maybe_mean(post_times), 3),
        "postprocess_time_p50_ms": round(maybe_median(post_times), 3),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Baseline evaluation of HEAL LiDAROnly model")
    p.add_argument("--model-name", default="lidar_pyramid", help="Model name (lidar_pyramid, lidar_cobevt, etc.)")
    p.add_argument("--model-config", default=None, help="Path to HEAL YAML config. Auto-resolved if --model-name is a known preset.")
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pth). Auto-resolved from presets if omitted.")
    p.add_argument("--heal-repo", default=HEAL_REPO)
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    p.add_argument("--ap-iou-backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default=None, help="Output directory. Defaults to tests/outputs/baseline_eval/<model_name>")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--no-verbose", dest="verbose", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve model config and checkpoint from presets if needed
    if args.model_config is None or args.checkpoint is None:
        preset = DEFAULT_CONFIGS.get(args.model_name)
        if preset is None:
            raise ValueError(f"Unknown model preset '{args.model_name}'. Provide --model-config and --checkpoint explicitly.")
        if args.model_config is None:
            args.model_config = str(_UNIAD_EXAMINE / preset["config"])
        if args.checkpoint is None:
            args.checkpoint = str(_UNIAD_EXAMINE / preset["checkpoint"])

    # Output directory
    if args.output_dir is None:
        args.output_dir = str(_THIS_DIR / "outputs" / "baseline_eval" / args.model_name)
    out_dir = Path(args.output_dir)
    ensure_dir(str(out_dir))

    logger = setup_logger(out_dir)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, default=str))

    # GPU selection
    device_str = resolve_device(args.device)
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif args.ap_iou_backend == "gpu":
        logger.warning("Device is CPU but --ap-iou-backend=gpu; switching to cpu backend")
        args.ap_iou_backend = "cpu"
    logger.info("Using device: %s", device)

    # Build adapter
    adapter = HEALLiDARAdapter(heal_repo=args.heal_repo)

    # Load model
    model, param_count, param_mb = load_model(
        adapter, args.model_config, args.checkpoint, device, logger,
    )

    # Build dataset
    logger.info("Building dataset...")
    hypes, dataset, loader = build_dataset(adapter, args.model_config, args.batch_size, args.num_workers)
    logger.info("Dataset: %d frames", len(dataset))

    # Evaluate
    logger.info("Starting evaluation...")
    summary = evaluate(model, dataset, loader, device, logger, args.ap_iou_backend, args.verbose)
    summary["model_name"] = args.model_name
    summary["checkpoint"] = args.checkpoint
    summary["param_count"] = param_count
    summary["param_size_mb"] = round(param_mb, 2)

    # Save results
    save_json(summary, str(out_dir / "baseline_eval_summary.json"))
    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION RESULTS: %s", args.model_name)
    logger.info("  AP@0.03 = %.4f", summary["AP@0.03"])
    logger.info("  AP@0.30 = %.4f", summary["AP@0.30"])
    logger.info("  AP@0.50 = %.4f", summary["AP@0.50"])
    logger.info("  AP@0.70 = %.4f", summary["AP@0.70"])
    logger.info("  Total:       mean=%.1fms  p50=%.1fms", summary["total_time_mean_ms"], summary["total_time_p50_ms"])
    logger.info("  Forward:     mean=%.1fms  p50=%.1fms", summary["forward_time_mean_ms"], summary["forward_time_p50_ms"])
    logger.info("  Postprocess: mean=%.1fms  p50=%.1fms", summary["postprocess_time_mean_ms"], summary["postprocess_time_p50_ms"])
    logger.info("  Params: %d (%.2f MB)", param_count, param_mb)
    logger.info("Results saved to: %s", out_dir / "baseline_eval_summary.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
