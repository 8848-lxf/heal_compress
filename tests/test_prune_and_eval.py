#!/usr/bin/env python3
"""
工具名称：test_prune_and_eval.py

作用：
    加载 HEAL / DAIR-V2X / LiDAROnly / lidar_pyramid 原始模型和剪枝模型，
    在完整验证集上执行推理评估，用于验证结构化剪枝工具生成的模型是否可运行，
    并对比剪枝前后的 AP 和推理耗时。

示例命令：
    cd .

    python tests/test_prune_and_eval.py \
        --original-checkpoint ${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth \
        --pruned-checkpoint tests/outputs/prune_lidar_pyramid_25_l1/pruned_model.pth \
        --rounds 1 \
        --gup-id auto \
        --output-dir tests/outputs/prune_lidar_pyramid_25_l1/eval_full_val

输出：
    tests/outputs/prune_lidar_pyramid_25_l1/eval_full_val/
        eval_log.txt
        baseline_per_frame_latency_round_1.csv
        pruned_per_frame_latency_round_1.csv
        per_round_summary.csv
        ap_results.json
        eval_config.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_UNIAD = _ROOT.parent
if str(_UNIAD) not in sys.path:
    sys.path.insert(0, str(_UNIAD))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
from heal_compress.pruning.grouped_conv import grouped_conv_pruning_fn, merge_grouped_conv_groups
from heal_compress.pruning.pruning_fns import get_pruning_fn
from heal_compress.utils.io_utils import ensure_unique_dir, save_csv, save_json
from heal_compress.utils.model_utils import resolve_device


DEFAULT_ORIGINAL = "${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_CONFIG = "${MODEL_ROOT}/lidar_pyramid/config.yaml"
DEFAULT_HEAL_ROOT = "../../HEAL"
IOU_THRESHOLDS = (0.03, 0.30, 0.50, 0.70)


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("prune_eval")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(output_dir / "eval_log.txt", mode="w", encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def count_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


def timed_call(fn, device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, (time.perf_counter() - t0) * 1000.0


def mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def p50(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def _delta_pct(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return (base - new) / base * 100.0


def build_eval_comparison(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        by_round.setdefault(int(summary["round_id"]), {})[summary["model_type"]] = summary
    for round_id in sorted(by_round):
        baseline = by_round[round_id].get("baseline")
        pruned = by_round[round_id].get("pruned")
        if not baseline or not pruned:
            continue
        row = {
            "round_id": round_id,
            "baseline_checkpoint": baseline["checkpoint"],
            "pruned_checkpoint": pruned["checkpoint"],
            "baseline_model_size_mb": baseline["model_size_mb"],
            "pruned_model_size_mb": pruned["model_size_mb"],
            "actual_prune_ratio": pruned["actual_prune_ratio"],
        }
        for key in ("AP_0_03", "AP_0_30", "AP_0_50", "AP_0_70"):
            row[f"baseline_{key}"] = baseline[key]
            row[f"pruned_{key}"] = pruned[key]
            row[f"delta_{key}"] = round(float(pruned[key]) - float(baseline[key]), 6)
        for key in (
            "total_time_mean_ms",
            "total_time_p50_ms",
            "forward_time_mean_ms",
            "forward_time_p50_ms",
            "postprocess_time_mean_ms",
            "postprocess_time_p50_ms",
        ):
            row[f"baseline_{key}"] = baseline[key]
            row[f"pruned_{key}"] = pruned[key]
            row[f"speedup_{key}_pct"] = round(_delta_pct(float(baseline[key]), float(pruned[key])), 3)
            row[f"delta_{key}"] = round(float(pruned[key]) - float(baseline[key]), 3)
        rows.append(row)
    return rows


def log_eval_comparison(rows: list[dict[str, Any]], logger: logging.Logger) -> None:
    if not rows:
        logger.info("No baseline/pruned pair found; eval comparison skipped")
        return
    for row in rows:
        logger.info("Round %d baseline vs pruned AP comparison:", row["round_id"])
        logger.info(
            "  AP@0.03 %.6f -> %.6f (delta %.6f) | AP@0.30 %.6f -> %.6f (delta %.6f)",
            row["baseline_AP_0_03"], row["pruned_AP_0_03"], row["delta_AP_0_03"],
            row["baseline_AP_0_30"], row["pruned_AP_0_30"], row["delta_AP_0_30"],
        )
        logger.info(
            "  AP@0.50 %.6f -> %.6f (delta %.6f) | AP@0.70 %.6f -> %.6f (delta %.6f)",
            row["baseline_AP_0_50"], row["pruned_AP_0_50"], row["delta_AP_0_50"],
            row["baseline_AP_0_70"], row["pruned_AP_0_70"], row["delta_AP_0_70"],
        )
        logger.info("Round %d timing comparison:", row["round_id"])
        logger.info(
            "  total mean %.3f -> %.3f ms (%+.3f ms, speedup %.3f%%), p50 %.3f -> %.3f ms (speedup %.3f%%)",
            row["baseline_total_time_mean_ms"], row["pruned_total_time_mean_ms"],
            row["delta_total_time_mean_ms"], row["speedup_total_time_mean_ms_pct"],
            row["baseline_total_time_p50_ms"], row["pruned_total_time_p50_ms"],
            row["speedup_total_time_p50_ms_pct"],
        )
        logger.info(
            "  forward mean %.3f -> %.3f ms (%+.3f ms, speedup %.3f%%), p50 %.3f -> %.3f ms (speedup %.3f%%)",
            row["baseline_forward_time_mean_ms"], row["pruned_forward_time_mean_ms"],
            row["delta_forward_time_mean_ms"], row["speedup_forward_time_mean_ms_pct"],
            row["baseline_forward_time_p50_ms"], row["pruned_forward_time_p50_ms"],
            row["speedup_forward_time_p50_ms_pct"],
        )
        logger.info(
            "  postprocess mean %.3f -> %.3f ms (%+.3f ms, speedup %.3f%%), p50 %.3f -> %.3f ms (speedup %.3f%%)",
            row["baseline_postprocess_time_mean_ms"], row["pruned_postprocess_time_mean_ms"],
            row["delta_postprocess_time_mean_ms"], row["speedup_postprocess_time_mean_ms_pct"],
            row["baseline_postprocess_time_p50_ms"], row["pruned_postprocess_time_p50_ms"],
            row["speedup_postprocess_time_p50_ms_pct"],
        )


def resolve_eval_device(gup_id: str | None, legacy_device: str | None) -> torch.device:
    """Resolve GPU selection.

    ``--gup-id`` is intentionally kept with the requested spelling. Values:
    ``auto`` selects the freest allowed GPU, an integer selects ``cuda:<id>``,
    and ``cpu`` forces CPU. ``--device`` is accepted as a compatibility alias.
    """
    selected = gup_id if gup_id not in (None, "") else legacy_device
    if selected in (None, "", "auto"):
        return torch.device(resolve_device("auto"))
    selected = str(selected)
    if selected == "cpu" or selected.startswith("cuda"):
        return torch.device(selected)
    if selected.isdigit():
        return torch.device(f"cuda:{selected}")
    return torch.device(resolve_device(selected))


def build_dataset(adapter: HEALLiDARAdapter, model_config: str, batch_size: int = 1, num_workers: int = 4):
    from opencood.data_utils.datasets import build_dataset as heal_build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(model_config))
    hypes = adapter._absolutize_dataset_paths(hypes)
    dataset = heal_build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_batch_test,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, loader


def load_model(adapter: HEALLiDARAdapter, config: str, checkpoint: str, device: torch.device, logger: logging.Logger) -> tuple[nn.Module, dict[str, Any]]:
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    ckpt = torch.load(checkpoint, map_location="cpu")
    metadata = ckpt.get("prune_metadata", {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_object"), nn.Module):
        model = ckpt["model_object"]
        return model.to(device).eval(), metadata
    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(config))
    model = train_utils.create_model(hypes)
    replay = ckpt.get("prune_replay", []) if isinstance(ckpt, dict) else []
    if isinstance(ckpt, dict):
        replay = _hydrate_grouped_independent_replay(replay, ckpt.get("prune_metadata", {}))
    if replay:
        apply_prune_replay(model, replay, logger)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("%s missing keys: %d", checkpoint, len(missing))
    if unexpected:
        logger.warning("%s unexpected keys: %d", checkpoint, len(unexpected))
    return model.to(device).eval(), metadata


def _hydrate_grouped_independent_replay(replay: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Backfill group_keep_map for older checkpoints saved before replay stored it."""
    if not replay or not isinstance(metadata, dict):
        return replay
    by_layer: dict[str, dict[str, Any]] = {}
    for group_result in metadata.get("applied", []) or []:
        for op in group_result.get("operations", []) or []:
            if op.get("axis") == "grouped_independent_keep" and op.get("group_keep_map"):
                by_layer[str(op.get("layer", ""))] = op
    hydrated: list[dict[str, Any]] = []
    for op in replay:
        new_op = dict(op)
        if new_op.get("axis") == "grouped_independent_keep" and not new_op.get("group_keep_map"):
            source = by_layer.get(str(new_op.get("layer", "")), {})
            if source.get("group_keep_map"):
                new_op["group_keep_map"] = source["group_keep_map"]
                new_op["per_group_after"] = source.get("per_group_after", new_op.get("per_group_after", 0))
        hydrated.append(new_op)
    return hydrated


def apply_prune_replay(model: nn.Module, replay: list[dict[str, Any]], logger: logging.Logger) -> None:
    modules = dict(model.named_modules())
    for op in replay:
        layer = op.get("layer", "")
        module = modules.get(layer)
        if module is None:
            logger.warning("prune_replay layer missing: %s", layer)
            continue
        after = int(op.get("after", 0))
        if after <= 0:
            continue
        axis = op.get("axis", "")
        direction = op.get("direction", "")
        if axis == "grouped_merge":
            factor = int(op.get("merge_factor", 1))
            merge_grouped_conv_groups(module, factor)
            continue
        if axis == "grouped_keep":
            before = int(op.get("before", getattr(module, "out_channels", after)))
            groups = int(op.get("groups", getattr(module, "groups", 1)))
            before_per = before // groups
            after_per = after // groups
            keep = []
            for gi in range(groups):
                start = gi * before_per
                keep.extend(range(start, start + after_per))
            fn = grouped_conv_pruning_fn("keep_groups")
        elif axis == "grouped_remove":
            before = int(op.get("before", getattr(module, "out_channels", after)))
            before_groups = int(op.get("before_groups", getattr(module, "groups", 1)))
            kept_groups = op.get("kept_groups", [])
            before_per = before // before_groups
            keep = []
            for gi in kept_groups:
                start = int(gi) * before_per
                keep.extend(range(start, start + before_per))
            fn = grouped_conv_pruning_fn("remove_groups")
        elif axis == "grouped_independent_keep":
            before = int(op.get("before", getattr(module, "out_channels", after)))
            groups = int(op.get("groups", getattr(module, "groups", 1)))
            group_keep_map = op.get("group_keep_map", {})
            if not group_keep_map:
                raise ValueError(f"grouped_independent_keep replay missing group_keep_map for {layer}")
            before_per = before // groups
            keep = []
            for group_id in range(groups):
                local_keep = group_keep_map.get(str(group_id), group_keep_map.get(group_id, []))
                keep.extend(group_id * before_per + int(local_idx) for local_idx in local_keep)
            fn = grouped_conv_pruning_fn("independent_group_topk")
        elif axis == "grouped_group_balanced_output":
            before = int(op.get("before_out", op.get("before", getattr(module, "out_channels", after))))
            keep_indices = op.get("keep_indices")
            prune_indices = op.get("prune_indices")
            if keep_indices:
                keep = [int(v) for v in keep_indices]
            elif prune_indices:
                prune = {int(v) for v in prune_indices}
                keep = [idx for idx in range(before) if idx not in prune]
            else:
                keep = list(range(after))
            fn = grouped_conv_pruning_fn("group_balanced_output_groups_fixed")
        elif axis == "grouped_flat_output":
            before = int(op.get("before_out", op.get("before", getattr(module, "out_channels", after))))
            after = int(op.get("after_out", op.get("after", after)))
            prune_indices = op.get("prune_indices")
            keep_indices = op.get("keep_indices")
            if keep_indices:
                keep = [int(v) for v in keep_indices]
            elif prune_indices:
                prune = {int(v) for v in prune_indices}
                keep = [idx for idx in range(before) if idx not in prune]
            else:
                # Older replay rows did not store the exact flat keep map. They
                # used sorted compact keep indices, so replay the leading prefix
                # to preserve shape. New v9.3 artifacts store keep_indices.
                keep = list(range(after))
            fn = grouped_conv_pruning_fn("flat_output_groups_fixed")
        else:
            keep_indices = op.get("keep_indices")
            prune_indices = op.get("prune_indices")
            before = int(op.get("before", getattr(module, "out_channels", after)))
            if keep_indices:
                keep = [int(v) for v in keep_indices]
            elif prune_indices:
                prune = {int(v) for v in prune_indices}
                keep = [idx for idx in range(before) if idx not in prune]
            else:
                keep = list(range(after))
            fn = get_pruning_fn(module, direction)
        if fn is None:
            logger.warning("prune_replay no pruning fn: layer=%s direction=%s axis=%s", layer, direction, axis)
            continue
        fn(module, keep)


def calculate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, thr: float) -> None:
    from opencood.utils import eval_utils

    eval_utils.caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, thr)


def evaluate_one_model(
    *,
    model: nn.Module,
    checkpoint: str,
    metadata: dict[str, Any],
    model_type: str,
    dataset: Any,
    loader: Any,
    device: torch.device,
    round_id: int,
    max_frames: int,
    warmup_frames: int,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
    rows: list[dict[str, Any]] = []
    total_times: list[float] = []
    forward_times: list[float] = []
    post_times: list[float] = []
    total_frames = len(dataset)
    actual = 0
    missing = 0
    consecutive_failures = 0
    first_failure = ""
    model.eval()
    logger.info(
        "Start %s round %d: checkpoint=%s size=%.3fMB gpu=%s dataset_frames=%d max_frames=%s warmup=%d",
        model_type,
        round_id,
        checkpoint,
        count_model_size_mb(model),
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        total_frames,
        max_frames or "all",
        warmup_frames,
    )
    iterator = iter(loader)
    frame_idx = -1
    while True:
        if max_frames and actual >= max_frames:
            break
        load_t0 = time.perf_counter()
        try:
            batch_data = next(iterator)
        except StopIteration:
            break
        data_loading_ms = (time.perf_counter() - load_t0) * 1000.0
        frame_idx += 1
        row = {
            "frame_id": frame_idx,
            "round_id": round_id,
            "model_type": model_type,
            "total_time_ms": 0.0,
            "data_loading_time_ms": 0.0,
            "forward_time_ms": 0.0,
            "postprocess_time_ms": 0.0,
            "data_to_gpu_time_ms": 0.0,
            "success": False,
            "skip_reason": "",
        }
        if batch_data is None:
            row["skip_reason"] = "empty_batch"
            rows.append(row)
            missing += 1
            continue
        try:
            batch_data, gpu_ms = timed_call(lambda: train_utils.to_device(batch_data, device), device)
            with torch.no_grad():
                output, fwd_ms = timed_call(lambda: model(batch_data["ego"]), device)

                def postprocess():
                    od = OrderedDict()
                    od["ego"] = output
                    return dataset.post_process(batch_data, od)

                (pred_box, pred_score, gt_box), post_ms = timed_call(postprocess, device)
            if frame_idx >= warmup_frames:
                for thr in IOU_THRESHOLDS:
                    calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr)
                total_ms = data_loading_ms + gpu_ms + fwd_ms + post_ms
                total_times.append(total_ms)
                forward_times.append(fwd_ms)
                post_times.append(post_ms)
                row.update({
                    "total_time_ms": round(total_ms, 3),
                    "data_loading_time_ms": round(data_loading_ms, 3),
                    "forward_time_ms": round(fwd_ms, 3),
                    "postprocess_time_ms": round(post_ms, 3),
                    "data_to_gpu_time_ms": round(gpu_ms, 3),
                    "success": True,
                })
                actual += 1
                consecutive_failures = 0
                logger.info(
                    "%s round %d frame %d total=%.3fms forward=%.3fms postprocess=%.3fms data_to_gpu=%.3fms",
                    model_type, round_id, frame_idx, total_ms, fwd_ms, post_ms, gpu_ms,
                )
            else:
                row["skip_reason"] = "warmup"
            rows.append(row)
            del output, pred_box, pred_score, gt_box, batch_data
            if (frame_idx + 1) % 128 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:
            err = str(exc)
            if not first_failure:
                first_failure = err
                logger.exception("%s round %d frame %d failed: %s", model_type, round_id, frame_idx, exc)
            else:
                logger.error("%s round %d frame %d failed: %s", model_type, round_id, frame_idx, err)
            row["skip_reason"] = str(exc)
            rows.append(row)
            missing += 1
            consecutive_failures += 1
            if actual == 0 and consecutive_failures >= 3:
                logger.error(
                    "%s round %d aborted after %d consecutive failures before any successful frame. First failure: %s",
                    model_type, round_id, consecutive_failures, first_failure,
                )
                break
    ap = {}
    for thr in IOU_THRESHOLDS:
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_val, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_val = 0.0
        ap[f"AP_{str(thr).replace('.', '_')}"] = round(float(ap_val), 6)
    gpu_id = device.index if device.type == "cuda" else ""
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    summary = {
        "round_id": round_id,
        "model_type": model_type,
        "checkpoint": checkpoint,
        "model_size_mb": round(count_model_size_mb(model), 3),
        "target_prune_ratio": metadata.get("target_prune_ratio", 0.0),
        "actual_prune_ratio": metadata.get("actual_prune_ratio", metadata.get("actual_param_prune_ratio", 0.0)),
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "dataset_total_frames": total_frames,
        "actual_frames": actual,
        "missing_frames": missing,
        "total_time_mean_ms": round(mean(total_times), 3),
        "total_time_p50_ms": round(p50(total_times), 3),
        "forward_time_mean_ms": round(mean(forward_times), 3),
        "forward_time_p50_ms": round(p50(forward_times), 3),
        "postprocess_time_mean_ms": round(mean(post_times), 3),
        "postprocess_time_p50_ms": round(p50(post_times), 3),
        "AP_0_03": ap["AP_0_03"],
        "AP_0_30": ap["AP_0_3"],
        "AP_0_50": ap["AP_0_5"],
        "AP_0_70": ap["AP_0_7"],
        "first_failure": first_failure,
    }
    logger.info(
        "%s round %d timing: total mean=%.3fms p50=%.3fms | forward mean=%.3fms p50=%.3fms | postprocess mean=%.3fms p50=%.3fms",
        model_type, round_id,
        summary["total_time_mean_ms"], summary["total_time_p50_ms"],
        summary["forward_time_mean_ms"], summary["forward_time_p50_ms"],
        summary["postprocess_time_mean_ms"], summary["postprocess_time_p50_ms"],
    )
    logger.info("")
    logger.info(
        "%s round %d AP: AP@0.03=%.6f AP@0.30=%.6f AP@0.50=%.6f AP@0.70=%.6f",
        model_type, round_id,
        summary["AP_0_03"], summary["AP_0_30"], summary["AP_0_50"], summary["AP_0_70"],
    )
    return rows, summary


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    out = ensure_unique_dir(args.output_dir)
    args.output_dir = str(out)
    logger = setup_logger(out)
    device = resolve_eval_device(args.gup_id, args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    save_json(vars(args), out / "eval_config.json")
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, default=str))
    logger.info("Using device: %s%s", device, f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    dataset, loader = build_dataset(adapter, args.model_config)
    summaries: list[dict[str, Any]] = []
    ap_results: dict[str, Any] = {}
    for round_id in range(1, args.rounds + 1):
        if args.eval_original:
            model, metadata = load_model(adapter, args.model_config, args.original_checkpoint, device, logger)
            rows, summary = evaluate_one_model(
                model=model,
                checkpoint=args.original_checkpoint,
                metadata=metadata,
                model_type="baseline",
                dataset=dataset,
                loader=loader,
                device=device,
                round_id=round_id,
                max_frames=args.max_frames,
                warmup_frames=args.warmup_frames,
                logger=logger,
            )
            save_csv(rows, out / f"baseline_per_frame_latency_round_{round_id}.csv")
            summaries.append(summary)
            ap_results.setdefault("baseline", []).append({k: summary[k] for k in ("AP_0_03", "AP_0_30", "AP_0_50", "AP_0_70")})
            del model
        if args.eval_pruned:
            model, metadata = load_model(adapter, args.model_config, args.pruned_checkpoint, device, logger)
            rows, summary = evaluate_one_model(
                model=model,
                checkpoint=args.pruned_checkpoint,
                metadata=metadata,
                model_type="pruned",
                dataset=dataset,
                loader=loader,
                device=device,
                round_id=round_id,
                max_frames=args.max_frames,
                warmup_frames=args.warmup_frames,
                logger=logger,
            )
            save_csv(rows, out / f"pruned_per_frame_latency_round_{round_id}.csv")
            summaries.append(summary)
            ap_results.setdefault("pruned", []).append({k: summary[k] for k in ("AP_0_03", "AP_0_30", "AP_0_50", "AP_0_70")})
            del model
    save_csv(summaries, out / "per_round_summary.csv")
    save_json(ap_results, out / "ap_results.json")
    comparison_rows = build_eval_comparison(summaries)
    save_csv(comparison_rows, out / "eval_comparison.csv")
    save_json(comparison_rows, out / "eval_comparison.json")
    log_eval_comparison(comparison_rows, logger)
    return {"summaries": summaries, "ap_results": ap_results, "comparison": comparison_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate original and pruned HEAL lidar_pyramid checkpoints")
    p.add_argument("--original-checkpoint", default=DEFAULT_ORIGINAL)
    p.add_argument("--pruned-checkpoint", default=str(_THIS_DIR / "outputs" / "prune_lidar_pyramid_25_l1" / "pruned_model.pth"))
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--gup-id", default=None, help="GPU id, cuda:N, cpu, or auto. Name kept as requested.")
    p.add_argument("--device", default=None, help="Backward-compatible alias; --gup-id takes precedence.")
    p.add_argument("--output-dir", default=str(_THIS_DIR / "outputs" / "prune_lidar_pyramid_25_l1" / "eval_full_val"))
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--warmup-frames", type=int, default=0)
    p.add_argument("--eval-original", type=str2bool, default=True)
    p.add_argument("--eval-pruned", type=str2bool, default=True)
    args = p.parse_args(argv)
    if args.gup_id is None and args.device is None:
        args.gup_id = "auto"
    return args


def test_prune_eval_script_skips_without_artifacts(tmp_path):
    if not Path(DEFAULT_ORIGINAL).is_file() or not Path(DEFAULT_CONFIG).is_file():
        pytest.skip("real HEAL checkpoint/config not available")
    args = parse_args([
        "--original-checkpoint", DEFAULT_ORIGINAL,
        "--pruned-checkpoint", DEFAULT_ORIGINAL,
        "--model-config", DEFAULT_CONFIG,
        "--max-frames", "1",
        "--eval-pruned", "false",
        "--device", "cpu",
        "--output-dir", str(tmp_path / "eval_smoke"),
    ])
    result = run_eval(args)
    assert "summaries" in result
    assert (tmp_path / "eval_smoke" / "per_round_summary.csv").is_file()


if __name__ == "__main__":
    run_eval(parse_args())
