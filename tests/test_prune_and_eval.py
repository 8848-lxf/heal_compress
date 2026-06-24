#!/usr/bin/env python3
"""Minimal end-to-end structured pruning + evaluation closed loop for HEAL LiDAROnly.

Pipeline steps:
  1. Auto-select GPU (excluding 5/6/7 unless --device specified)
  2. Load real HEAL model + full validation dataset
  3. Dynamic forward dependency tracing -> coupled channel groups
  4. First-order Taylor importance estimation (gradient * weight)
  5. 25% structured physical channel pruning via model-specific spec
  6. Structure legality check + real-batch forward sanity check
  7. Save pruned_model.pth
  8. Evaluate pruned model on full val set: AP@0.03/0.30/0.50/0.70 + timing

Example commands
================
Prune lidar_pyramid 25% and evaluate (auto GPU, exclude 5/6/7):
  python tests/test_prune_and_eval.py \
      --model-name lidar_pyramid \
      --prune-ratio 0.25

Prune lidar_pyramid on a specific GPU:
  python tests/test_prune_and_eval.py \
      --model-name lidar_pyramid \
      --prune-ratio 0.25 \
      --device cuda:2

Prune with explicit config and checkpoint:
  python tests/test_prune_and_eval.py \
      --model-config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml \
      --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth \
      --prune-ratio 0.25

Prune with more calibration batches:
  python tests/test_prune_and_eval.py \
      --model-name lidar_pyramid \
      --prune-ratio 0.25 \
      --num-calib-batches 32

Skip evaluation (only prune and save):
  python tests/test_prune_and_eval.py \
      --model-name lidar_pyramid \
      --prune-ratio 0.25 \
      --skip-eval
"""
from __future__ import annotations

import argparse
import copy
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
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_HEAL_COMPRESS_ROOT = _THIS_DIR.parent
_UNIAD_EXAMINE = _HEAL_COMPRESS_ROOT.parent

HEAL_REPO = str(_UNIAD_EXAMINE / "HEAL")
AUTO_SEARCH = str(_UNIAD_EXAMINE / "Auto_Search")

for _p in [str(_HEAL_COMPRESS_ROOT), HEAL_REPO, AUTO_SEARCH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.model_utils import auto_select_gpu, resolve_device
from coop_lidar_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
from coop_lidar_compress.tracing.dynamic_trace import DynamicDependencyTracer
from coop_lidar_compress.tracing.coupled_groups import CoupledGroupBuilder, CoupledChannelGroup
from coop_lidar_compress.pruning.lidar_pyramid_structured import (
    build_lidar_pyramid_prune_spec,
    apply_lidar_pyramid_prune_spec,
    check_lidar_pyramid_structural_legality,
    prune_spec_scope_channel_counts,
    parameter_count,
    parameter_size_mb,
)
from coop_lidar_compress.utils.io import ensure_dir, save_json, save_yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IOU_THRESHOLDS = (0.03, 0.30, 0.50, 0.70)

DEFAULT_CONFIGS = {
    "lidar_pyramid": {
        "config": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth",
        "hypes_yaml": "opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pyramid.yaml",
    },
    "lidar_cobevt": {
        "config": "HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_cobevt.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth",
        "hypes_yaml": "opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_cobevt.yaml",
    },
    "lidar_attfuse": {
        "config": "HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_attfuse.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_attfuse/net_epoch_bestval_at33.pth",
        "hypes_yaml": "opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_attfuse.yaml",
    },
    "lidar_v2xvit": {
        "config": "HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_v2xvit.yaml",
        "checkpoint": "Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth",
        "hypes_yaml": "opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_v2xvit.yaml",
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prune_and_eval")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "prune_and_eval_log.txt", mode="a", encoding="utf-8")
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


def build_calib_loader(adapter: HEALLiDARAdapter, model_config: str, batch_size: int = 1, num_workers: int = 2):
    """Build a calibration dataloader (train split with label_dict for loss computation)."""
    from opencood.data_utils.datasets import build_dataset as heal_build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(model_config))
    hypes = adapter._absolutize_dataset_paths(hypes)
    dataset = heal_build_dataset(hypes, visualize=False, train=False)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": dataset.collate_batch_train,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = lambda _: torch.set_num_threads(1)
    return DataLoader(dataset, **kwargs)


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
    return model, hypes


def count_params(model: nn.Module) -> tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return n, mb


# ---------------------------------------------------------------------------
# Step 3: Dynamic forward dependency tracing + coupled channel groups
# (used only for importance computation — pruning uses model-specific spec)
# ---------------------------------------------------------------------------
def trace_and_build_groups(
    model: nn.Module, adapter: HEALLiDARAdapter, device: torch.device, logger: logging.Logger,
    output_dir: Path,
) -> tuple[Any, list[CoupledChannelGroup]]:
    """Run dynamic forward tracing and build coupled channel groups."""
    logger.info("Step 3: Dynamic forward dependency tracing...")
    protected = adapter.get_protected_layers(model)
    logger.info("  Protected layers (%d): %s", len(protected), protected[:10])

    sample_batch = adapter.build_synthetic_batch(model)

    tracer = DynamicDependencyTracer(
        model, protected_layers=protected, forward_fn=adapter.forward_for_task,
    )
    graph = tracer.trace(sample_batch)
    logger.info("  Traced graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges))
    if graph.manual_warnings:
        for w in graph.manual_warnings:
            logger.warning("  Trace warning: %s", w)

    graph.save(str(output_dir / "dependency_graph.yaml"), str(output_dir / "dependency_graph.json"))

    logger.info("Step 3b: Building coupled channel groups...")
    builder = CoupledGroupBuilder(graph)
    groups = builder.build()
    logger.info("  Built %d coupled channel groups", len(groups))

    CoupledGroupBuilder.save(groups, str(output_dir / "coupled_groups.yaml"))

    for g in groups[:5]:
        members_str = ", ".join(f"{m.layer_name}({m.role})" for m in g.members[:6])
        extra = f"...+{len(g.members)-6}" if len(g.members) > 6 else ""
        logger.info("  Group %s [protected=%s]: %s%s", g.group_id, g.protected, members_str, extra)
    if len(groups) > 5:
        logger.info("  ... and %d more groups", len(groups) - 5)

    return graph, groups


# ---------------------------------------------------------------------------
# Step 5: First-order Taylor importance estimation
# ---------------------------------------------------------------------------
def compute_importance(
    model: nn.Module, adapter: HEALLiDARAdapter, model_config: str,
    groups: list[CoupledChannelGroup],
    device: torch.device, logger: logging.Logger, output_dir: Path,
    num_calib_batches: int = 16,
) -> dict[str, list[float]]:
    """Compute first-order Taylor importance scores for coupled channel groups."""
    from coop_lidar_compress.pruning.importance.group_importance import compute_group_importance

    logger.info("Step 5: Computing first-order Taylor importance (calib_batches=%d)...", num_calib_batches)

    calib_loader = build_calib_loader(adapter, model_config, batch_size=1, num_workers=2)

    def loss_fn(outputs, batch):
        return adapter.compute_task_loss(outputs, batch)

    def forward_fn(m, batch):
        return adapter.forward_for_task(m, batch)

    result = compute_group_importance(
        model=model,
        dataloader=calib_loader,
        loss_fn=loss_fn,
        coupled_channel_groups=groups,
        device=device,
        num_calib_batches=num_calib_batches,
        forward_fn=forward_fn,
        output_dir=str(output_dir / "importance"),
        logger=logger,
        importance_mode="first_order_taylor",
    )

    logger.info("  Importance mode: %s", result.importance_mode)
    logger.info("  Calibration batches used: %d", result.summary.get("actual_used_calib_batches", 0))
    logger.info("  Prunable groups: %d", result.summary.get("num_prunable_groups", 0))
    logger.info("  Protected groups: %d", result.summary.get("num_protected_groups", 0))

    return result.channel_scores


# ---------------------------------------------------------------------------
# Step 6: Model-specific structured physical pruning
# ---------------------------------------------------------------------------
def prune_model(
    model: nn.Module, channel_scores: dict[str, list[float]],
    prune_ratio: float, logger: logging.Logger, output_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    """Apply structured channel pruning using model-specific prune spec.

    Uses build_lidar_pyramid_prune_spec + apply_lidar_pyramid_prune_spec which
    correctly handles Bottleneck grouped convolutions, residual connections,
    BasicBlock backbone, deblocks, shrink conv, and detection head inputs.
    """
    logger.info("Step 6: Structured pruning (ratio=%.2f) via model-specific spec...", prune_ratio)

    before_params, before_mb = count_params(model)

    # Build pruning spec with Taylor importance if available, else L1 fallback
    spec = build_lidar_pyramid_prune_spec(
        model=model,
        prune_ratio=prune_ratio,
        align=16,
        min_channels=16,
        importance_by_layer=channel_scores if channel_scores else None,
    )
    save_json(spec, str(output_dir / "prune_spec.json"))

    # Log spec summary
    orig_ch, kept_ch = prune_spec_scope_channel_counts(spec)
    if orig_ch > 0:
        logger.info("  Spec channel reduction: %d -> %d (%.1f%% removed)",
                     orig_ch, kept_ch, (1.0 - kept_ch / orig_ch) * 100)

    scopes = spec.get("scopes", [])
    logger.info("  Pruning scopes: %s", scopes)

    # Log backbone details
    basic_resnet = spec.get("basic_resnet_backbone")
    if basic_resnet and basic_resnet.get("stages"):
        for stage in basic_resnet["stages"]:
            n_blocks = len(stage.get("blocks", []))
            prunable = stage.get("output_prunable", False)
            logger.info("  basic_resnet stage%d: %d blocks, output_prunable=%s",
                        stage["stage_index"], n_blocks, prunable)

    resnet = spec.get("resnet_backbone")
    if resnet and resnet.get("stages"):
        for stage in resnet["stages"]:
            n_blocks = len(stage.get("blocks", []))
            prunable = stage.get("output_prunable", False)
            logger.info("  pyramid_resnet stage%d: %d blocks, output_prunable=%s",
                        stage["stage_index"], n_blocks, prunable)

    for entry in spec.get("deblocks", []):
        logger.info("  deblock%d: %d -> %d channels", entry["index"],
                     entry["original_out_channels"], len(entry["keep_indices"]))

    # Apply the spec (physically modifies model weights in-place)
    report = apply_lidar_pyramid_prune_spec(model, spec)
    num_operations = len(report.get("operations", []))

    after_params, after_mb = count_params(model)
    actual_prune_ratio = 1.0 - (after_params / max(before_params, 1))

    prune_report = {
        "prune_ratio_target": prune_ratio,
        "prune_ratio_actual_params": round(actual_prune_ratio, 4),
        "params_before": before_params,
        "params_after": after_params,
        "size_before_mb": round(before_mb, 2),
        "size_after_mb": round(after_mb, 2),
        "num_operations": num_operations,
        "legality_issues": report.get("issues", []),
        "spec_format": spec.get("format", ""),
    }
    save_json(prune_report, str(output_dir / "prune_report.json"))

    logger.info("  Applied %d surgery operations", num_operations)
    logger.info("  Params: %d -> %d (%.1f%% reduction)", before_params, after_params, actual_prune_ratio * 100)
    logger.info("  Size: %.2f MB -> %.2f MB", before_mb, after_mb)

    if report.get("issues"):
        logger.warning("  Legality issues from apply_spec: %d", len(report["issues"]))
        for issue in report["issues"][:5]:
            logger.warning("    %s", issue)

    return model, prune_report


# ---------------------------------------------------------------------------
# Step 7: Structure legality check + forward sanity check
# ---------------------------------------------------------------------------
def legality_and_sanity_check(
    model: nn.Module, adapter: HEALLiDARAdapter, device: torch.device,
    logger: logging.Logger, output_dir: Path,
) -> dict[str, Any]:
    """Check structural legality and run forward pass on a real batch."""
    logger.info("Step 7: Structure legality check...")

    # Use the model-specific legality checker
    report = check_lidar_pyramid_structural_legality(model)
    issues = report.get("issues", [])

    if issues:
        logger.error("  LEGALITY CHECK FAILED: %d issues found", len(issues))
        for issue in issues[:10]:
            logger.error("    %s", issue)
    else:
        logger.info("  Legality check PASSED: all layer dimensions valid")

    # Forward sanity check with synthetic batch
    logger.info("Step 7b: Forward sanity check (synthetic batch)...")
    model.eval()
    try:
        sample = adapter.build_synthetic_batch(model)
        with torch.no_grad():
            output = adapter.forward_for_task(model, sample)
        if isinstance(output, dict):
            output_keys = list(output.keys())
            has_output = True
        else:
            output_keys = [str(type(output).__name__)]
            has_output = output is not None
        logger.info("  Forward sanity check PASSED: output_keys=%s", output_keys[:5])
    except Exception as exc:
        has_output = False
        output_keys = []
        logger.error("  Forward sanity check FAILED: %s", exc)
        issues.append({"layer": "model", "issue": "forward_failed", "error": str(exc)})

    check_result = {
        "legality_passed": len([i for i in issues if i.get("issue") != "forward_failed"]) == 0,
        "forward_passed": has_output,
        "issues": issues,
        "output_keys": output_keys,
    }
    save_json(check_result, str(output_dir / "legality_check.json"))
    return check_result


# ---------------------------------------------------------------------------
# Step 8: Save pruned model
# ---------------------------------------------------------------------------
def save_pruned_model(
    model: nn.Module, output_dir: Path, prune_report: dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Save pruned model checkpoint."""
    ckpt_path = str(output_dir / "pruned_model.pth")
    torch.save({
        "model": model.state_dict(),
        "prune_metadata": {
            "target_prune_ratio": prune_report["prune_ratio_target"],
            "actual_param_prune_ratio": prune_report["prune_ratio_actual_params"],
            "params_before": prune_report["params_before"],
            "params_after": prune_report["params_after"],
        },
    }, ckpt_path)
    size_mb = Path(ckpt_path).stat().st_size / (1024 * 1024)
    logger.info("Step 8: Pruned model saved to %s (%.2f MB)", ckpt_path, size_mb)
    return ckpt_path


# ---------------------------------------------------------------------------
# Step 9: Evaluate pruned model on full val set
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


def evaluate_pruned(
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

    ap = {}
    for thr in IOU_THRESHOLDS:
        key = f"AP@{thr:.2f}"
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_val, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_val = 0.0
        ap[key] = round(float(ap_val), 4)

    return {
        "total_frames": total_frames,
        "actual_frames": actual,
        "skipped_frames": skipped,
        **ap,
        "total_time_mean_ms": round(maybe_mean(total_times), 3),
        "total_time_p50_ms": round(maybe_median(total_times), 3),
        "forward_time_mean_ms": round(maybe_mean(forward_times), 3),
        "forward_time_p50_ms": round(maybe_median(forward_times), 3),
        "postprocess_time_mean_ms": round(maybe_mean(post_times), 3),
        "postprocess_time_p50_ms": round(maybe_median(post_times), 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="HEAL LiDAROnly structured pruning + evaluation")
    p.add_argument("--model-name", default="lidar_pyramid")
    p.add_argument("--model-config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--heal-repo", default=HEAL_REPO)
    p.add_argument("--device", default="auto")
    p.add_argument("--prune-ratio", type=float, default=0.25)
    p.add_argument("--num-calib-batches", type=int, default=16)
    p.add_argument("--ap-iou-backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation after pruning")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--no-verbose", dest="verbose", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve config and checkpoint from presets
    preset = DEFAULT_CONFIGS.get(args.model_name, {})
    if args.model_config is None:
        if "config" not in preset:
            raise ValueError(f"Unknown model '{args.model_name}'. Provide --model-config explicitly.")
        args.model_config = str(_UNIAD_EXAMINE / preset["config"])
    if args.checkpoint is None:
        if "checkpoint" not in preset:
            raise ValueError(f"Unknown model '{args.model_name}'. Provide --checkpoint explicitly.")
        args.checkpoint = str(_UNIAD_EXAMINE / preset["checkpoint"])

    # Output directory
    if args.output_dir is None:
        ratio_str = f"{int(args.prune_ratio * 100)}"
        args.output_dir = str(_THIS_DIR / "outputs" / f"prune_{args.model_name}_{ratio_str}")
    out_dir = Path(args.output_dir)
    ensure_dir(str(out_dir))

    logger = setup_logger(out_dir)
    logger.info("=" * 60)
    logger.info("HEAL LiDAROnly Structured Pruning Pipeline")
    logger.info("=" * 60)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, default=str))

    # Step 1: GPU selection
    device_str = resolve_device(args.device)
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif args.ap_iou_backend == "gpu":
        logger.warning("Device is CPU but --ap-iou-backend=gpu; switching to cpu backend")
        args.ap_iou_backend = "cpu"
    logger.info("Step 1: Using device: %s", device)

    # Step 2: Load model and dataset
    logger.info("Step 2: Loading model and dataset...")
    adapter_cfg = {
        "model": {
            "hypes_yaml": preset.get("hypes_yaml", ""),
            "heal_repo": args.heal_repo,
        }
    }
    adapter = HEALLiDARAdapter(heal_repo=args.heal_repo, config=adapter_cfg)
    model, hypes = load_model(adapter, args.model_config, args.checkpoint, device, logger)
    orig_params, orig_mb = count_params(model)
    logger.info("  Original model: %d params (%.2f MB)", orig_params, orig_mb)

    # Step 3: Dynamic forward tracing + coupled channel groups
    # (needed for importance computation, not for pruning itself)
    graph, groups = trace_and_build_groups(model, adapter, device, logger, out_dir)

    # Step 5: First-order Taylor importance
    channel_scores = compute_importance(
        model, adapter, args.model_config, groups,
        device, logger, out_dir, args.num_calib_batches,
    )

    # Step 6: Physical pruning using model-specific spec
    model, prune_report = prune_model(
        model, channel_scores, args.prune_ratio, logger, out_dir,
    )

    # Step 7: Legality check + forward sanity check
    check_result = legality_and_sanity_check(model, adapter, device, logger, out_dir)

    if not check_result["legality_passed"]:
        logger.error("PIPELINE ABORTED: legality check failed")
        sys.exit(1)
    if not check_result["forward_passed"]:
        logger.error("PIPELINE ABORTED: forward sanity check failed")
        sys.exit(1)

    # Step 8: Save pruned model
    ckpt_path = save_pruned_model(model, out_dir, prune_report, logger)

    # Step 9: Evaluate pruned model
    if not args.skip_eval:
        logger.info("Step 9: Evaluating pruned model on full val set...")
        hypes_eval, dataset, loader = build_dataset(adapter, args.model_config, args.batch_size, args.num_workers)
        logger.info("  Dataset: %d frames", len(dataset))

        eval_summary = evaluate_pruned(model, dataset, loader, device, logger, args.ap_iou_backend, args.verbose)
        eval_summary["model_name"] = args.model_name
        eval_summary["checkpoint"] = ckpt_path
        eval_summary["prune_ratio_target"] = args.prune_ratio
        eval_summary["prune_ratio_actual_params"] = prune_report["prune_ratio_actual_params"]
        eval_summary["params_before"] = prune_report["params_before"]
        eval_summary["params_after"] = prune_report["params_after"]
        eval_summary["size_before_mb"] = prune_report["size_before_mb"]
        eval_summary["size_after_mb"] = prune_report["size_after_mb"]

        save_json(eval_summary, str(out_dir / "pruned_eval_summary.json"))

        logger.info("=" * 60)
        logger.info("PRUNED MODEL EVALUATION RESULTS: %s", args.model_name)
        logger.info("  Prune ratio: target=%.2f actual=%.4f",
                     args.prune_ratio, prune_report["prune_ratio_actual_params"])
        logger.info("  Params: %d -> %d", prune_report["params_before"], prune_report["params_after"])
        logger.info("  AP@0.03 = %.4f", eval_summary["AP@0.03"])
        logger.info("  AP@0.30 = %.4f", eval_summary["AP@0.30"])
        logger.info("  AP@0.50 = %.4f", eval_summary["AP@0.50"])
        logger.info("  AP@0.70 = %.4f", eval_summary["AP@0.70"])
        logger.info("  Total:       mean=%.1fms  p50=%.1fms",
                     eval_summary["total_time_mean_ms"], eval_summary["total_time_p50_ms"])
        logger.info("  Forward:     mean=%.1fms  p50=%.1fms",
                     eval_summary["forward_time_mean_ms"], eval_summary["forward_time_p50_ms"])
        logger.info("  Postprocess: mean=%.1fms  p50=%.1fms",
                     eval_summary["postprocess_time_mean_ms"], eval_summary["postprocess_time_p50_ms"])
        logger.info("=" * 60)
    else:
        logger.info("Evaluation skipped (--skip-eval)")

    logger.info("Pipeline complete. Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
