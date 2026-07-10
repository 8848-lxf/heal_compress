"""Formal real-dataloader pruning evaluation helpers.

The functions here are production copies of the runtime evaluation helpers
that used to live in tests. They do not import from ``tests``.
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
from heal_compress.pruning.grouped_conv import grouped_conv_pruning_fn, merge_grouped_conv_groups
from heal_compress.pruning.pruning_fns import get_pruning_fn
from heal_compress.utils.io_utils import save_csv, save_json
from heal_compress.utils.model_utils import resolve_device


DEFAULT_ORIGINAL = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_CONFIG = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"
DEFAULT_HEAL_ROOT = "/home/lixingfeng/UniAD_examine/HEAL"
IOU_THRESHOLDS = (0.03, 0.30, 0.50, 0.70, 0.90)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
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


def timed_call(fn: Any, device: torch.device) -> tuple[Any, float]:
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


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(q)))))
    return float(ordered[idx])


def build_dataset(adapter: HEALLiDARAdapter, model_config: str, batch_size: int = 1, num_workers: int = 4, *, train: bool = False):
    from opencood.data_utils.datasets import build_dataset as heal_build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(model_config))
    hypes = adapter._absolutize_dataset_paths(hypes)
    dataset = heal_build_dataset(hypes, visualize=True, train=bool(train))
    collate = dataset.collate_batch_train if train and hasattr(dataset, "collate_batch_train") else dataset.collate_batch_test
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, loader


def _hydrate_grouped_independent_replay(replay: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
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
            merge_grouped_conv_groups(module, int(op.get("merge_factor", 1)))
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
            before_per = before // before_groups
            keep = []
            for gi in op.get("kept_groups", []):
                start = int(gi) * before_per
                keep.extend(range(start, start + before_per))
            fn = grouped_conv_pruning_fn("remove_groups")
        elif axis == "grouped_independent_keep":
            before = int(op.get("before", getattr(module, "out_channels", after)))
            groups = int(op.get("groups", getattr(module, "groups", 1)))
            before_per = before // groups
            group_keep_map = op.get("group_keep_map", {})
            if not group_keep_map:
                raise ValueError(f"grouped_independent_keep replay missing group_keep_map for {layer}")
            keep = []
            for group_id in range(groups):
                local_keep = group_keep_map.get(str(group_id), group_keep_map.get(group_id, []))
                keep.extend(group_id * before_per + int(local_idx) for local_idx in local_keep)
            fn = grouped_conv_pruning_fn("independent_group_topk")
        elif axis in {"grouped_group_balanced_output", "grouped_flat_output"}:
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
            mode = "group_balanced_output_groups_fixed" if axis == "grouped_group_balanced_output" else "flat_output_groups_fixed"
            fn = grouped_conv_pruning_fn(mode)
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


def load_model(adapter: HEALLiDARAdapter, config: str, checkpoint: str, device: torch.device, logger: logging.Logger) -> tuple[nn.Module, dict[str, Any]]:
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = ckpt.get("prune_metadata", {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_object"), nn.Module):
        return ckpt["model_object"].to(device).eval(), metadata
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


def calculate_tp_fp(det_boxes: Any, det_score: Any, gt_boxes: Any, result_stat: dict[float, dict[str, Any]], thr: float) -> None:
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
    gpu_times: list[float] = []
    total_frames = len(dataset)
    actual = 0
    missing = 0
    consecutive_failures = 0
    first_failure = ""
    model.eval()
    iterator = iter(loader)
    frame_idx = -1
    logger.info(
        "Start %s round %d: checkpoint=%s size=%.3fMB dataset_frames=%d max_frames=%s warmup=%d",
        model_type,
        round_id,
        checkpoint,
        count_model_size_mb(model),
        total_frames,
        max_frames or "all",
        warmup_frames,
    )
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
            "unaccounted_time_ms": 0.0,
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

                def postprocess() -> Any:
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
                gpu_times.append(gpu_ms)
                row.update(
                    {
                        "total_time_ms": round(total_ms, 3),
                        "data_loading_time_ms": round(data_loading_ms, 3),
                        "forward_time_ms": round(fwd_ms, 3),
                        "postprocess_time_ms": round(post_ms, 3),
                        "data_to_gpu_time_ms": round(gpu_ms, 3),
                        "unaccounted_time_ms": round(total_ms - gpu_ms - fwd_ms - post_ms, 3),
                        "success": True,
                    }
                )
                actual += 1
                consecutive_failures = 0
            else:
                row["skip_reason"] = "warmup"
            rows.append(row)
            del output, pred_box, pred_score, gt_box, batch_data
            if (frame_idx + 1) % 128 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if not first_failure:
                first_failure = err
                logger.exception("%s round %d frame %d failed: %s", model_type, round_id, frame_idx, exc)
            row["skip_reason"] = err
            rows.append(row)
            missing += 1
            consecutive_failures += 1
            if actual == 0 and consecutive_failures >= 3:
                break
    ap: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_val, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_val = 0.0
        ap[f"AP_{str(thr).replace('.', '_')}"] = round(float(ap_val), 6)
    summary = {
        "round_id": round_id,
        "model_type": model_type,
        "checkpoint": checkpoint,
        "model_size_mb": round(count_model_size_mb(model), 3),
        "target_prune_ratio": metadata.get("target_prune_ratio", 0.0),
        "actual_prune_ratio": metadata.get("actual_prune_ratio", metadata.get("actual_param_prune_ratio", 0.0)),
        "gpu_id": device.index if device.type == "cuda" else "",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dataset_total_frames": total_frames,
        "actual_frames": actual,
        "missing_frames": missing,
        "total_time_mean_ms": round(mean(total_times), 3),
        "total_time_p50_ms": round(p50(total_times), 3),
        "total_time_p90_ms": round(percentile(total_times, 0.90), 3),
        "total_time_p95_ms": round(percentile(total_times, 0.95), 3),
        "total_time_p99_ms": round(percentile(total_times, 0.99), 3),
        "forward_time_mean_ms": round(mean(forward_times), 3),
        "forward_time_p50_ms": round(p50(forward_times), 3),
        "forward_time_p90_ms": round(percentile(forward_times, 0.90), 3),
        "forward_time_p95_ms": round(percentile(forward_times, 0.95), 3),
        "forward_time_p99_ms": round(percentile(forward_times, 0.99), 3),
        "postprocess_time_mean_ms": round(mean(post_times), 3),
        "postprocess_time_p50_ms": round(p50(post_times), 3),
        "data_to_gpu_time_mean_ms": round(mean(gpu_times), 3),
        "data_to_gpu_time_p50_ms": round(p50(gpu_times), 3),
        "unaccounted_time_mean_ms": 0.0,
        "AP_0_03": ap.get("AP_0_03", 0.0),
        "AP_0_30": ap.get("AP_0_3", 0.0),
        "AP_0_50": ap.get("AP_0_5", 0.0),
        "AP_0_70": ap.get("AP_0_7", 0.0),
        "AP_0_90": ap.get("AP_0_9", 0.0),
        "first_failure": first_failure,
    }
    return rows, summary


def resolve_eval_device(gup_id: str | None, legacy_device: str | None) -> torch.device:
    selected = gup_id if gup_id not in (None, "") else legacy_device
    if selected in (None, "", "auto"):
        return torch.device(resolve_device("auto"))
    selected = str(selected)
    if selected == "cpu" or selected.startswith("cuda"):
        return torch.device(selected)
    if selected.isdigit():
        return torch.device(f"cuda:{selected}")
    return torch.device(resolve_device(selected))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate original and pruned HEAL lidar_pyramid checkpoints")
    p.add_argument("--original-checkpoint", default=DEFAULT_ORIGINAL)
    p.add_argument("--pruned-checkpoint", required=True)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="")
    p.add_argument("--gup-id", default=None)
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--warmup-frames", type=int, default=50)
    p.add_argument("--eval-original", action="store_true")
    p.add_argument("--eval-pruned", action="store_true", default=True)
    p.add_argument("--rounds", type=int, default=1)
    return p.parse_args(argv)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out)
    device = resolve_eval_device(args.gup_id, args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    save_json(vars(args), out / "eval_config.json")
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    dataset, loader = build_dataset(adapter, args.model_config)
    summaries: list[dict[str, Any]] = []
    for round_id in range(1, args.rounds + 1):
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
    save_csv(summaries, out / "per_round_summary.csv")
    return {"summaries": summaries}


def main(argv: list[str] | None = None) -> int:
    report = run_eval(parse_args(argv))
    print(json.dumps({"success": True, "summaries": len(report.get("summaries", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
