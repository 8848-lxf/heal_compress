"""ModelOpt TensorRT evaluation worker for lidar_pyramid engines."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
import traceback
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


IOU_THRESHOLDS = (0.30, 0.50, 0.70)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    rows = sorted(float(v) for v in values)
    pos = (len(rows) - 1) * pct
    low = int(pos)
    high = min(len(rows) - 1, low + 1)
    frac = pos - low
    return rows[low] * (1.0 - frac) + rows[high] * frac


def _latency_distribution(values: list[float], *, prefix: str) -> dict[str, float | int | None]:
    rows = [float(value) for value in values]
    if not rows:
        return {
            f"{prefix}_mean_ms": None,
            f"{prefix}_p50_ms": None,
            f"{prefix}_p90_ms": None,
            f"{prefix}_p95_ms": None,
            f"{prefix}_p99_ms": None,
            f"{prefix}_std_ms": None,
            f"{prefix}_cv": None,
            f"{prefix}_min_ms": None,
            f"{prefix}_max_ms": None,
            f"{prefix}_outlier_count": 0,
        }
    mean = float(statistics.mean(rows))
    std = float(statistics.pstdev(rows)) if len(rows) > 1 else 0.0
    q1 = _percentile(rows, 0.25) or 0.0
    q3 = _percentile(rows, 0.75) or 0.0
    iqr = float(q3 - q1)
    high = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr
    return {
        f"{prefix}_mean_ms": mean,
        f"{prefix}_p50_ms": _percentile(rows, 0.50),
        f"{prefix}_p90_ms": _percentile(rows, 0.90),
        f"{prefix}_p95_ms": _percentile(rows, 0.95),
        f"{prefix}_p99_ms": _percentile(rows, 0.99),
        f"{prefix}_std_ms": std,
        f"{prefix}_cv": float(std / mean) if mean else 0.0,
        f"{prefix}_min_ms": min(rows),
        f"{prefix}_max_ms": max(rows),
        f"{prefix}_outlier_count": sum(1 for value in rows if value < low or value > high),
    }


def _timed(fn: Any, device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def _float_profile(profile: dict[str, Any], key: str) -> float:
    value = profile.get(key, 0.0)
    return float(value) if value is not None else 0.0


def _mean_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        return {}
    numeric_keys = {
        "total_runner_ms",
        "execute_async_ms",
        "synchronize_ms",
        "set_input_shape_ms",
        "bind_address_ms",
        "output_shape_query_ms",
        "h2d_copy_ms",
        "input_device_copy_ms",
        "dtype_cast_ms",
        "contiguous_ms",
        "output_wrap_ms",
    }
    merged: dict[str, Any] = {}
    for key in numeric_keys:
        merged[key] = float(statistics.mean(_float_profile(profile, key) for profile in profiles))
    merged["input_buffer_reallocated"] = any(bool(profile.get("input_buffer_reallocated")) for profile in profiles)
    merged["output_buffer_reallocated"] = any(bool(profile.get("output_buffer_reallocated")) for profile in profiles)
    merged["set_input_shape_calls"] = int(sum(int(profile.get("set_input_shape_calls") or 0) for profile in profiles))
    merged["output_shape_query_calls"] = int(sum(int(profile.get("output_shape_query_calls") or 0) for profile in profiles))
    merged["h2d_copies"] = int(sum(int(profile.get("h2d_copies") or 0) for profile in profiles))
    merged["d2d_input_copies"] = int(sum(int(profile.get("d2d_input_copies") or 0) for profile in profiles))
    merged["bytes_h2d"] = int(sum(int(profile.get("bytes_h2d") or 0) for profile in profiles))
    merged["bytes_d2d_input"] = int(sum(int(profile.get("bytes_d2d_input") or 0) for profile in profiles))
    return merged


def _latency_row_from_profile(
    *,
    frame_id: Any,
    warmup: bool,
    input_prepare_ms: float,
    host_to_device_ms: float,
    profile: dict[str, Any],
    postprocess_ms: float | None,
) -> dict[str, Any]:
    forward_ms = _float_profile(profile, "total_runner_ms")
    shape_binding_ms = (
        _float_profile(profile, "set_input_shape_ms")
        + _float_profile(profile, "bind_address_ms")
        + _float_profile(profile, "output_shape_query_ms")
    )
    reallocated = bool(profile.get("input_buffer_reallocated")) or bool(profile.get("output_buffer_reallocated"))
    row: dict[str, Any] = {
        "frame_id": frame_id,
        "warmup": bool(warmup),
        "forward_ms": forward_ms,
        "input_prepare_ms": float(input_prepare_ms),
        "host_to_device_ms": float(host_to_device_ms),
        "shape_binding_ms": float(shape_binding_ms),
        "buffer_allocation_ms": None if reallocated else 0.0,
        "input_buffer_reallocated": bool(profile.get("input_buffer_reallocated")),
        "output_buffer_reallocated": bool(profile.get("output_buffer_reallocated")),
        "execute_async_ms": _float_profile(profile, "execute_async_ms"),
        "device_sync_ms": _float_profile(profile, "synchronize_ms"),
        "input_device_copy_ms": _float_profile(profile, "input_device_copy_ms"),
        "runner_h2d_copy_ms": _float_profile(profile, "h2d_copy_ms"),
        "dtype_cast_ms": _float_profile(profile, "dtype_cast_ms"),
        "contiguous_ms": _float_profile(profile, "contiguous_ms"),
        "set_input_shape_calls": int(profile.get("set_input_shape_calls") or 0),
        "output_shape_query_calls": int(profile.get("output_shape_query_calls") or 0),
        "h2d_copies": int(profile.get("h2d_copies") or 0),
        "d2d_input_copies": int(profile.get("d2d_input_copies") or 0),
        "bytes_h2d": int(profile.get("bytes_h2d") or 0),
        "bytes_d2d_input": int(profile.get("bytes_d2d_input") or 0),
    }
    if postprocess_ms is not None:
        row["postprocess_ms"] = float(postprocess_ms)
        row["total_ms"] = forward_ms + float(postprocess_ms)
    return row


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output = Path(request["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, "/home/lixingfeng/UniAD_examine")
        sys.path.insert(0, "/home/lixingfeng/UniAD_examine/heal_compress")
        sys.path.insert(0, "/home/lixingfeng/UniAD_examine/HEAL")
        sys.path.insert(0, "/home/lixingfeng/UniAD_examine/heal_compress/tests")
        sys.path.insert(0, "/home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy")
        plugin_path = request.get("plugin_path")
        if plugin_path:
            ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils
        from opencood.utils import eval_utils
        from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner
        from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
        from heal_compress.quantization.config import OnnxExportConfig
        from heal_compress.quantization.export.heal_lidar_pyramid import prepare_signal_maxk_inputs
        from tests.test_baseline_eval import calculate_tp_fp_for_threshold

        device = torch.device(request["device"])
        if device.type != "cuda":
            raise RuntimeError("TensorRT evaluation requires CUDA")
        torch.cuda.set_device(device)
        adapter = HEALLiDARAdapter(heal_repo=request["heal_root"], config={"model": {"hypes_yaml": request["model_config"]}})
        hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(request["model_config"]))
        hypes = adapter._absolutize_dataset_paths(hypes)
        model = adapter.build_model(request["model_config"], request["checkpoint"]).to(device).eval()
        dataset = build_dataset(hypes, visualize=True, train=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(request.get("num_workers", 0)), collate_fn=dataset.collate_batch_test)
        runner = TensorRTEngineRunner(request["engine_path"], device)
        export_config = OnnxExportConfig(
            fixed_k=int(request.get("fixed_k", 29696)),
            min_agents=1,
            opt_agents=2,
            max_agents=2,
        )
        result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
        total_times: list[float] = []
        forward_times: list[float] = []
        post_times: list[float] = []
        rows: list[dict[str, Any]] = []
        skip_reasons: Counter[str] = Counter()
        actual = 0
        warmup_seen = 0
        target = int(request["num_frames"])
        warmup = int(request["warmup_frames"])
        latency_rounds = max(1, int(request.get("latency_rounds", 1)))
        manifest_path = Path(str(request.get("eval_manifest_path", ""))) if request.get("eval_manifest_path") else None
        manifest_payload: dict[str, Any] = {}
        manifest_roles: dict[str, str] = {}
        split_frame_ids: list[str] = []
        if manifest_path is not None:
            if not manifest_path.is_file():
                raise RuntimeError(f"eval_manifest_missing:{manifest_path}")
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            warmup_ids = [str(value) for value in manifest_payload.get("warmup_frame_ids", [])]
            evaluation_ids = [str(value) for value in manifest_payload.get("evaluation_frame_ids", [])]
            if len(warmup_ids) != warmup or len(evaluation_ids) != target:
                raise RuntimeError(
                    f"eval_manifest_count_mismatch:warmup={len(warmup_ids)}!={warmup}:eval={len(evaluation_ids)}!={target}"
                )
            if len(set(warmup_ids + evaluation_ids)) != len(warmup_ids) + len(evaluation_ids):
                raise RuntimeError("eval_manifest_duplicate_frame_ids")
            manifest_roles.update({frame_id: "warmup" for frame_id in warmup_ids})
            manifest_roles.update({frame_id: "evaluation" for frame_id in evaluation_ids})
            split_path = Path(str(hypes.get("validate_dir", "")))
            split_payload = json.loads(split_path.read_text(encoding="utf-8"))
            if not isinstance(split_payload, list):
                raise RuntimeError(f"validation_split_manifest_not_list:{split_path}")
            split_frame_ids = [str(value) for value in split_payload]
            missing = sorted(set(manifest_roles) - set(split_frame_ids))
            if missing:
                raise RuntimeError(f"eval_manifest_ids_missing_from_validation_split:{missing[:8]}")
        evaluated_frame_ids: list[str] = []
        skipped_frame_ids: list[str] = []
        skipped_warmup_frame_ids: list[str] = []
        output_names: list[str] | None = None
        for frame_idx, batch in enumerate(loader):
            if actual >= target:
                break
            frame_id: Any = frame_idx
            role = "warmup" if warmup_seen < warmup else "evaluation"
            if manifest_roles:
                if frame_idx >= len(split_frame_ids):
                    break
                frame_id = split_frame_ids[frame_idx]
                role = manifest_roles.get(str(frame_id), "")
                if not role:
                    continue
            if batch is None:
                skip_reasons["empty_batch"] += 1
                if role == "evaluation":
                    skipped_frame_ids.append(str(frame_id))
                else:
                    skipped_warmup_frame_ids.append(str(frame_id))
                continue
            try:
                batch, host_to_device_ms = _timed(lambda: _move(batch, device), device)
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                if output_names is None:
                    with torch.no_grad():
                        raw = model(ego)
                    output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
                tensors_by_name, input_prepare_ms = _timed(
                    lambda: prepare_signal_maxk_inputs(ego, config=export_config, modality=request.get("modality", "m1")),
                    device,
                )
                round_outputs = None
                round_profiles: list[dict[str, Any]] = []
                for _round_idx in range(latency_rounds):
                    round_outputs, profile = runner.run_profiled(tensors_by_name)
                    round_profiles.append(profile)
                trt_outputs = round_outputs or {}
                profile = _mean_profiles(round_profiles)
                forward_ms = _float_profile(profile, "total_runner_ms")
                output_dict = {name: trt_outputs[name].float() for name in (output_names or ["cls_preds", "reg_preds", "dir_preds"])}

                def postprocess() -> Any:
                    od = OrderedDict()
                    od["ego"] = output_dict
                    return dataset.post_process(batch, od)

                (pred_box, pred_score, gt_box), post_ms = _timed(postprocess, device)
                if role == "warmup":
                    warmup_seen += 1
                    rows.append(
                        _latency_row_from_profile(
                            frame_id=frame_id,
                            warmup=True,
                            input_prepare_ms=input_prepare_ms,
                            host_to_device_ms=host_to_device_ms,
                            profile=profile,
                            postprocess_ms=post_ms,
                        )
                    )
                    continue
                for thr in IOU_THRESHOLDS:
                    calculate_tp_fp_for_threshold(pred_box, pred_score, gt_box, result_stat, thr, "cpu", device)
                actual += 1
                evaluated_frame_ids.append(str(frame_id))
                forward_times.append(forward_ms)
                post_times.append(post_ms)
                total_times.append(forward_ms + post_ms)
                rows.append(
                    _latency_row_from_profile(
                        frame_id=frame_id,
                        warmup=False,
                        input_prepare_ms=input_prepare_ms,
                        host_to_device_ms=host_to_device_ms,
                        profile=profile,
                        postprocess_ms=post_ms,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                skip_reasons[f"{type(exc).__name__}:{exc}"] += 1
                if role == "evaluation":
                    skipped_frame_ids.append(str(frame_id))
                else:
                    skipped_warmup_frame_ids.append(str(frame_id))
                rows.append({"frame_id": frame_id, "success": False, "skip_reason": f"{type(exc).__name__}:{exc}"})
                if actual == 0 and sum(skip_reasons.values()) >= 3:
                    break
        ap: dict[str, float] = {}
        for thr in IOU_THRESHOLDS:
            if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
                ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
            else:
                ap_value = 0.0
            ap[f"AP@{thr:.1f}"] = float(ap_value)
        manifest_complete = (
            not manifest_roles
            or (
                actual == target
                and len(evaluated_frame_ids) == target
                and not skipped_frame_ids
                and warmup_seen == warmup
                and not skipped_warmup_frame_ids
            )
        )
        result = {
            "status": "ok" if actual > 0 and manifest_complete else "evaluation_failed",
            "AP@0.3": ap.get("AP@0.3", 0.0),
            "AP@0.5": ap.get("AP@0.5", 0.0),
            "AP@0.7": ap.get("AP@0.7", 0.0),
            "mAP": float(sum(ap.values()) / len(ap)) if ap else 0.0,
            **_latency_distribution(forward_times, prefix="forward"),
            **_latency_distribution(post_times, prefix="postprocess"),
            **_latency_distribution(total_times, prefix="total"),
            "num_evaluated_frames": actual,
            "num_skipped_frames": int(sum(skip_reasons.values())),
            "skip_reason_counts": dict(skip_reasons),
            "fixed_manifest_enforced": bool(manifest_roles),
            "eval_manifest_path": str(manifest_path) if manifest_path is not None else "",
            "eval_manifest_hash": str(manifest_payload.get("manifest_hash", "")),
            "evaluated_frame_ids": evaluated_frame_ids,
            "skipped_frame_ids": skipped_frame_ids,
            "skipped_warmup_frame_ids": skipped_warmup_frame_ids,
            "latency_rows": rows,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "evaluation_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "output": str(output)}, sort_keys=True))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
