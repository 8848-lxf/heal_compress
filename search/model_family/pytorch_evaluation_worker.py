"""Full-validation PyTorch FP32 worker for HEAL DAIR LiDAR model families."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
from torch.utils.data import DataLoader


IOU_THRESHOLDS = (0.30, 0.50, 0.70)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _timed(function: Any, device: torch.device) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    return value, (time.perf_counter() - started) * 1000.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {f"{prefix}_p50_ms": None, f"{prefix}_p90_ms": None, f"{prefix}_p99_ms": None}
    return {
        f"{prefix}_mean_ms": float(statistics.mean(values)),
        f"{prefix}_p50_ms": _percentile(values, 0.50),
        f"{prefix}_p90_ms": _percentile(values, 0.90),
        f"{prefix}_p99_ms": _percentile(values, 0.99),
        f"{prefix}_min_ms": min(values),
        f"{prefix}_max_ms": max(values),
        f"{prefix}_std_ms": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def _gpu_process_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
    }


def _strict_model(config: Path, checkpoint: Path, heal_root: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(str(config))
    model = train_utils.create_model(hypes)
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    incompatibility = model.load_state_dict(state, strict=False)
    missing = list(incompatibility.missing_keys)
    unexpected = list(incompatibility.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"strict_checkpoint_mismatch:missing={missing[:16]}:unexpected={unexpected[:16]}"
        )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {
        "strict_load_passed": True,
        "state_tensor_count": len(state),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": _sha256(config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output = Path(request["output_path"])
    try:
        for path in (
            "/home/lixingfeng/UniAD_examine",
            "/home/lixingfeng/UniAD_examine/heal_compress",
            "/home/lixingfeng/UniAD_examine/HEAL",
            "/home/lixingfeng/UniAD_examine/heal_compress/tests",
        ):
            if path not in sys.path:
                sys.path.insert(0, path)

        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils
        from opencood.utils import eval_utils
        from tests.test_baseline_eval import calculate_tp_fp_for_threshold
        from search.integration.evaluation_worker import _verify_cuda_postprocess_backend

        if str(request.get("conda_env", "")) != "univ2x-opt":
            raise RuntimeError("pytorch_baseline_request_requires_univ2x_opt")
        device = torch.device(str(request.get("device", "cuda:0")))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("pytorch_baseline_requires_cuda")
        torch.cuda.set_device(device)
        torch.set_num_threads(max(1, int(request.get("torch_num_threads", 4))))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        config = Path(request["config_path"]).resolve()
        checkpoint = Path(request["checkpoint_path"]).resolve()
        manifest_path = Path(request["eval_manifest_path"]).resolve()
        heal_root = Path(request["heal_root"]).resolve()
        for required in (config, checkpoint, manifest_path):
            if not required.is_file():
                raise RuntimeError(f"pytorch_baseline_input_missing:{required}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        warmup_ids = [str(value) for value in manifest.get("warmup_frame_ids", [])]
        evaluation_ids = [str(value) for value in manifest.get("evaluation_frame_ids", [])]
        warmup_target = int(request.get("warmup_frames", 200))
        evaluation_target = int(request.get("num_frames", 1789))
        if len(warmup_ids) < warmup_target or len(evaluation_ids) < evaluation_target:
            raise RuntimeError("pytorch_baseline_manifest_insufficient")
        warmup_ids = warmup_ids[:warmup_target]
        evaluation_ids = evaluation_ids[:evaluation_target]
        if not bool(manifest.get("reset_after_warmup", False)):
            raise RuntimeError("pytorch_baseline_requires_warmup_reset_manifest")

        hypes = yaml_utils.load_yaml(str(config))
        for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
            value = hypes.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                hypes[key] = str(heal_root / value)
        split_ids = [str(value) for value in json.loads(Path(hypes["validate_dir"]).read_text(encoding="utf-8"))]
        if evaluation_ids != split_ids[:evaluation_target]:
            raise RuntimeError("pytorch_baseline_manifest_not_exact_validation_order")
        missing_ids = sorted(set(warmup_ids + evaluation_ids) - set(split_ids))
        if missing_ids:
            raise RuntimeError(f"pytorch_baseline_manifest_ids_missing:{missing_ids[:8]}")

        model, checkpoint_audit = _strict_model(config, checkpoint, heal_root, device)
        dataset = build_dataset(hypes, visualize=True, train=False)
        workers = int(request.get("dataloader_num_workers", 8))
        loader_kwargs: dict[str, Any] = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": workers,
            "collate_fn": dataset.collate_batch_test,
            "pin_memory": workers > 0,
        }
        if workers:
            loader_kwargs.update(
                {
                    "prefetch_factor": 2,
                    "persistent_workers": True,
                    "worker_init_fn": _worker_init,
                }
            )
        loader = DataLoader(dataset, **loader_kwargs)
        cuda_postprocess = _verify_cuda_postprocess_backend(device)
        process_start = _gpu_process_snapshot()

        result_stat = {
            threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
            for threshold in IOU_THRESHOLDS
        }
        forward_times: list[float] = []
        postprocess_times: list[float] = []
        total_times: list[float] = []
        transfer_times: list[float] = []
        evaluated: list[str] = []
        warmed: list[str] = []
        skipped: list[str] = []
        skip_reasons: Counter[str] = Counter()

        def run_phase(role: str, ids: list[str]) -> None:
            selected = set(ids)
            completed_ids = warmed if role == "warmup" else evaluated
            for index, raw_batch in enumerate(loader):
                if len(completed_ids) >= len(ids):
                    break
                frame_id = split_ids[index]
                if frame_id not in selected:
                    continue
                try:
                    if raw_batch is None:
                        raise RuntimeError("empty_batch")
                    batch, transfer_ms = _timed(lambda: _move(raw_batch, device), device)
                    ego = batch["ego"]
                    with torch.inference_mode():
                        outputs, forward_ms = _timed(lambda: model(ego), device)

                    def postprocess() -> Any:
                        rows = OrderedDict()
                        rows["ego"] = outputs
                        return dataset.post_process(batch, rows)

                    (pred_box, pred_score, gt_box), postprocess_ms = _timed(postprocess, device)
                    if role == "warmup":
                        warmed.append(frame_id)
                        continue
                    for threshold in IOU_THRESHOLDS:
                        calculate_tp_fp_for_threshold(
                            pred_box,
                            pred_score,
                            gt_box,
                            result_stat,
                            threshold,
                            "gpu",
                            device,
                        )
                    evaluated.append(frame_id)
                    forward_times.append(forward_ms)
                    postprocess_times.append(postprocess_ms)
                    transfer_times.append(transfer_ms)
                    total_times.append(forward_ms + postprocess_ms)
                    if len(evaluated) % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "model": request["model_name"],
                                    "evaluated": len(evaluated),
                                    "target": len(ids),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    reason = f"{type(exc).__name__}:{exc}"
                    skip_reasons[reason] += 1
                    skipped.append(frame_id)
                    raise RuntimeError(f"pytorch_baseline_frame_failed:{frame_id}:{reason}") from exc

        run_phase("warmup", warmup_ids)
        run_phase("evaluation", evaluation_ids)
        if warmed != warmup_ids or evaluated != evaluation_ids or skipped:
            raise RuntimeError(
                f"pytorch_baseline_manifest_incomplete:warmup={len(warmed)}/{len(warmup_ids)}:"
                f"evaluation={len(evaluated)}/{len(evaluation_ids)}:skipped={len(skipped)}"
            )
        ap = {}
        for threshold in IOU_THRESHOLDS:
            value = 0.0
            if result_stat[threshold]["gt"] > 0 and result_stat[threshold]["score"]:
                value, _, _ = eval_utils.calculate_ap(result_stat, threshold)
            ap[f"AP@{threshold:.1f}"] = float(value)
        result = {
            "status": "ok",
            "schema_version": "heal-dair-lidar-pytorch-fp32-baseline-v1",
            "model_name": str(request["model_name"]),
            "precision": "pytorch_fp32",
            **ap,
            "mAP": float(sum(ap.values()) / len(ap)),
            **_distribution(forward_times, "forward"),
            **_distribution(postprocess_times, "postprocess"),
            **_distribution(total_times, "model_postprocess"),
            **_distribution(transfer_times, "host_to_device"),
            "num_warmup_frames": len(warmed),
            "num_evaluated_frames": len(evaluated),
            "num_skipped_frames": len(skipped),
            "warmup_frame_ids": warmed,
            "evaluated_frame_ids": evaluated,
            "skip_reason_counts": dict(skip_reasons),
            "reset_after_warmup": True,
            "eval_manifest_path": str(manifest_path),
            "eval_manifest_hash": str(manifest.get("manifest_hash", "")),
            "dataloader_num_workers": workers,
            "cuda_postprocess_audit": cuda_postprocess,
            "checkpoint_audit": checkpoint_audit,
            "config_path": str(config),
            "checkpoint_path": str(checkpoint),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability(device)),
            "gpu_process_snapshot_start": process_start,
            "gpu_process_snapshot_end": _gpu_process_snapshot(),
            "latency_isolation": str(request.get("latency_isolation", "unverified")),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "evaluation_failed",
            "model_name": str(request.get("model_name", "")),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    _write_json(output, result)
    print(json.dumps({"status": result.get("status"), "output": str(output)}, sort_keys=True), flush=True)
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
