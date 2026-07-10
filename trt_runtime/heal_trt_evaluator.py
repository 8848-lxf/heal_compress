from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .bindings import inspect_engine_bindings
from .engine_runner import TensorRTEngineRunner
from .output_adapter import adapt_outputs_for_postprocess, bind_inputs_for_engine, infer_modality, tensor_output_names, to_device
from .timing import summarize_latency


DEFAULT_CONFIG = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"
DEFAULT_CHECKPOINT = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_HEAL_ROOT = "/home/lixingfeng/UniAD_examine/HEAL"
IOU_THRESHOLDS = (0.03, 0.30, 0.50, 0.70)


def parse_ap_thresholds(value: Any | None) -> tuple[float, ...]:
    if value is None or value == "":
        return tuple(IOU_THRESHOLDS)
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    else:
        parts = list(value)
    thresholds = tuple(float(item) for item in parts)
    if not thresholds:
        raise ValueError("ap thresholds must not be empty")
    if any(thr <= 0.0 or thr > 1.0 for thr in thresholds):
        raise ValueError(f"ap thresholds must be in (0, 1], got {thresholds}")
    return thresholds


def threshold_key(threshold: float) -> str:
    return f"AP@{float(threshold):.2f}"


def _thresholds_from_args(args: Any) -> tuple[float, ...]:
    return parse_ap_thresholds(getattr(args, "ap_thresholds", None))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _add_paths(heal_root: str | Path) -> None:
    root = Path(__file__).resolve().parents[1]
    parent = root.parent
    for item in (str(parent), str(root), str(Path(heal_root).expanduser().resolve())):
        if item not in sys.path:
            sys.path.insert(0, item)


def _load_hypes(config: str | Path, heal_root: str | Path) -> dict[str, Any]:
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(str(Path(config).expanduser()))
    heal_root = Path(heal_root).expanduser()
    for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
        value = hypes.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            hypes[key] = str(heal_root / value)
    return hypes


def _load_model(hypes: dict[str, Any], checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    from opencood.tools import train_utils

    model = train_utils.create_model(hypes)
    state = torch.load(str(Path(checkpoint).expanduser()), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _build_context(args: Any, device: torch.device) -> tuple[Any, Any, Any, str]:
    _add_paths(getattr(args, "heal_root", DEFAULT_HEAL_ROOT))
    from opencood.data_utils.datasets import build_dataset

    hypes = _load_hypes(getattr(args, "model_config", DEFAULT_CONFIG), getattr(args, "heal_root", DEFAULT_HEAL_ROOT))
    model = _load_model(hypes, getattr(args, "checkpoint", DEFAULT_CHECKPOINT), device)
    modality = infer_modality(model)
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(getattr(args, "num_workers", 0)), collate_fn=dataset.collate_batch_test)
    return dataset, loader, model, modality


def _calculate_tp_fp(pred_box: Any, pred_score: Any, gt_box: Any, result_stat: dict[float, dict[str, Any]], thr: float) -> None:
    from opencood.utils import eval_utils

    eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr)


def _calculate_ap(result_stat: dict[float, dict[str, Any]], thresholds: tuple[float, ...]) -> dict[str, float]:
    from opencood.utils import eval_utils

    ap = {}
    for thr in thresholds:
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            value, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            value = 0.0
        ap[threshold_key(thr)] = round(float(value), 6)
    ap["mAP"] = round(sum(ap.values()) / max(len(ap), 1), 6)
    return ap


def _run_eval_loop(
    *,
    engine_path: Path,
    output_dir: Path,
    args: Any,
    smoke: bool,
    frames: int,
    warmup_frames: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("real TensorRT HEAL validation requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dataset, loader, model, modality = _build_context(args, device)
    plugin_path = getattr(args, "plugin", "")
    runner = TensorRTEngineRunner(engine_path, device, plugin_path=plugin_path)
    binding_report = inspect_engine_bindings(engine_path, plugin_path=plugin_path)
    _write_json(output_dir / "output_binding_mapping_report.json", binding_report)
    ap_thresholds = _thresholds_from_args(args)
    result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in ap_thresholds}
    rows: list[dict[str, Any]] = []
    output_names: list[str] | None = None
    completed = 0
    skipped = 0
    first_failure = ""
    for frame_idx, batch in enumerate(loader):
        if completed >= int(frames):
            break
        row = {"frame_id": int(frame_idx), "success": False, "skip_reason": ""}
        total_start = time.perf_counter()
        try:
            if batch is None:
                raise RuntimeError("empty_validation_batch")
            data_start = time.perf_counter()
            batch = to_device(batch, device)
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            data_ms = (time.perf_counter() - data_start) * 1000.0
            if output_names is None:
                with torch.no_grad():
                    raw = model(ego)
                output_names = tensor_output_names(raw)
            inputs = bind_inputs_for_engine(runner.input_names(), ego, modality, fixed_k=int(getattr(args, "fixed_k", 29696)))
            trt_outputs, profile = runner.run_profiled(inputs)

            post_start = time.perf_counter()
            wrapped = adapt_outputs_for_postprocess(output_names, trt_outputs)
            pred_box, pred_score, gt_box = dataset.post_process(batch, wrapped)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            post_ms = (time.perf_counter() - post_start) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0
            if frame_idx >= int(warmup_frames):
                for thr in ap_thresholds:
                    _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr)
                completed += 1
                row.update(
                    {
                        "success": True,
                        "data_to_gpu_latency_ms": round(data_ms, 4),
                        "forward_latency_ms": round(float(profile.get("forward_latency_ms", 0.0)), 4),
                        "postprocess_latency_ms": round(post_ms, 4),
                        "total_latency_ms": round(total_ms, 4),
                        "unaccounted_time_ms": round(total_ms - data_ms - float(profile.get("forward_latency_ms", 0.0)) - post_ms, 4),
                    }
                )
            else:
                row["skip_reason"] = "warmup"
            rows.append(row)
            if smoke and completed >= int(frames):
                break
            if (frame_idx + 1) % 128 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if not first_failure:
                first_failure = f"{type(exc).__name__}: {exc}"
            row["skip_reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if smoke or completed == 0 and skipped >= 3:
                break
    latency = summarize_latency(rows)
    ap = _calculate_ap(result_stat, ap_thresholds) if completed else {threshold_key(thr): None for thr in ap_thresholds} | {"mAP": None}
    return {
        "success": completed > 0,
        "eval_success": completed > 0 and not smoke,
        "smoke_success": completed > 0 if smoke else None,
        "synthetic_used": False,
        "validation_dataloader_used": True,
        "evaluated_frames": completed,
        "skipped_frames": skipped,
        "failure_reason": "" if completed > 0 else first_failure or "no_successful_validation_frames",
        "latency_summary": latency,
        "ap": ap,
        "ap_thresholds": list(ap_thresholds),
        "per_frame_rows": rows,
        "output_names": output_names or [],
    }


def run_trt_smoke(*, engine_path: str | Path, output_dir: str | Path, args: Any, frames: int = 5) -> dict[str, Any]:
    out = Path(output_dir)
    try:
        report = _run_eval_loop(engine_path=Path(engine_path), output_dir=out, args=args, smoke=True, frames=int(frames), warmup_frames=0)
        report.update({"status": "trt_smoke_passed" if report.get("success") else "trt_smoke_failed"})
    except Exception as exc:  # noqa: BLE001
        report = {
            "success": False,
            "status": "trt_smoke_failed",
            "synthetic_used": False,
            "validation_dataloader_used": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    _write_json(out / "trt_smoke_report.json", report)
    return report


def run_real_heal_validation_eval(
    *,
    engine_path: str | Path,
    output_dir: str | Path,
    args: Any,
    warmup_frames: int,
    eval_frames: int,
) -> dict[str, Any]:
    out = Path(output_dir)
    try:
        report = _run_eval_loop(
            engine_path=Path(engine_path),
            output_dir=out,
            args=args,
            smoke=False,
            frames=int(eval_frames),
            warmup_frames=int(warmup_frames),
        )
        report.update({"status": "eval_success" if report.get("eval_success") else "eval_failed"})
    except Exception as exc:  # noqa: BLE001
        report = {
            "eval_success": False,
            "status": "eval_failed",
            "synthetic_used": False,
            "validation_dataloader_used": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "latency_summary": {},
            "ap": {},
            "per_frame_rows": [],
        }
    _write_csv(out / "eval_latency_per_frame.csv", list(report.get("per_frame_rows") or []))
    _write_json(out / "eval_latency_summary.json", report.get("latency_summary", {}))
    _write_json(out / "eval_ap.json", report.get("ap", {}))
    _write_json(out / "eval_report.json", report)
    return report
