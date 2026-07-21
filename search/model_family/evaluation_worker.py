"""TensorRT AP/latency worker for the HEAL LiDAR V2X-ViT fixed-K contract."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import ctypes
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import torch
from torch.utils.data import DataLoader


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output = Path(request["output_path"])
    try:
        repo_root = Path(__file__).resolve().parents[2]
        for path in (
            "/home/lixingfeng/UniAD_examine",
            "/home/lixingfeng/UniAD_examine/HEAL",
            str(repo_root),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)
        plugin_path = str(request.get("plugin_path", ""))
        if plugin_path:
            ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)

        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils
        from opencood.utils import eval_utils
        from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner
        from tests.test_baseline_eval import calculate_tp_fp_for_threshold
        from adapters.heal_lidar_adapter import HEALLiDARAdapter
        from search.integration.evaluation_worker import (
            IOU_THRESHOLDS,
            _dataloader_worker_init,
            _float_profile,
            _latency_distribution,
            _mean_profiles,
            _move,
            _timed,
            _verify_cuda_postprocess_backend,
        )
        from search.model_family.export import (
            HealLidarBaselineExportPolicy,
            HealV2XViTExportPolicy,
            prepare_heal_lidar_baseline_inputs,
            prepare_v2xvit_fixed_k_inputs,
        )

        device = torch.device(str(request["device"]))
        if device.type != "cuda":
            raise RuntimeError("v2xvit_tensorrt_evaluation_requires_cuda")
        torch.cuda.set_device(device)
        torch_threads = max(1, int(request.get("torch_num_threads", 4)))
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cuda_postprocess = _verify_cuda_postprocess_backend(device)

        adapter = HEALLiDARAdapter(
            heal_repo=request["heal_root"],
            config={"model": {"hypes_yaml": request["model_config"]}},
        )
        hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(request["model_config"]))
        hypes = adapter._absolutize_dataset_paths(hypes)
        dataset = build_dataset(hypes, visualize=True, train=False)
        workers = max(0, int(request.get("dataloader_num_workers", 8)))
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
                    "worker_init_fn": _dataloader_worker_init,
                }
            )
        loader = DataLoader(dataset, **loader_kwargs)
        runner = TensorRTEngineRunner(request["engine_path"], device)
        input_contract = str(request.get("input_contract", "heal_v2xvit_fixed_k"))
        if input_contract == "heal_lidar_baseline_fixed_k":
            policy = HealLidarBaselineExportPolicy(
                fixed_k=int(request["fixed_k"]),
                max_agents=int(request.get("max_agents", 2)),
            )
            prepare_inputs = prepare_heal_lidar_baseline_inputs
        elif input_contract == "heal_v2xvit_fixed_k":
            policy = HealV2XViTExportPolicy(
                fixed_k=int(request["fixed_k"]),
                max_agents=int(request.get("max_agents", 2)),
            )
            prepare_inputs = prepare_v2xvit_fixed_k_inputs
        else:
            raise RuntimeError(f"unsupported_fixed_k_input_contract:{input_contract}")
        manifest_path = Path(request["eval_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        warmup_ids = [str(value) for value in manifest.get("warmup_frame_ids", [])]
        evaluation_ids = [str(value) for value in manifest.get("evaluation_frame_ids", [])]
        warmup_target = int(request["warmup_frames"])
        eval_target = int(request["num_frames"])
        if len(warmup_ids) < warmup_target or len(evaluation_ids) < eval_target:
            raise RuntimeError("v2xvit_eval_manifest_insufficient")
        warmup_ids = warmup_ids[:warmup_target]
        evaluation_ids = evaluation_ids[:eval_target]
        split = json.loads(Path(str(hypes["validate_dir"])).read_text(encoding="utf-8"))
        split_ids = [str(value) for value in split]
        missing = sorted(set(warmup_ids + evaluation_ids) - set(split_ids))
        if missing:
            raise RuntimeError(f"v2xvit_eval_manifest_ids_missing:{missing[:8]}")

        result_stat = {
            threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
            for threshold in IOU_THRESHOLDS
        }
        forward_times: list[float] = []
        post_times: list[float] = []
        total_times: list[float] = []
        latency_rows: list[dict[str, Any]] = []
        evaluated: list[str] = []
        warmed: list[str] = []
        skipped: list[str] = []
        skip_reasons: Counter[str] = Counter()

        def run_phase(role: str, wanted: list[str]) -> None:
            wanted_set = set(wanted)
            for index, batch in enumerate(loader):
                if len(warmed if role == "warmup" else evaluated) >= len(wanted):
                    break
                if index >= len(split_ids):
                    break
                frame_id = split_ids[index]
                if frame_id not in wanted_set:
                    continue
                try:
                    if batch is None:
                        raise RuntimeError("empty_batch")
                    batch, h2d_ms = _timed(lambda: _move(batch, device), device)
                    ego = batch["ego"]
                    prepared, prepare_ms = _timed(
                        lambda: prepare_inputs(ego, policy=policy), device
                    )
                    outputs = None
                    profiles = []
                    for _ in range(max(1, int(request.get("latency_rounds", 1)))):
                        outputs, profile = runner.run_profiled(prepared)
                        profiles.append(profile)
                    profile = _mean_profiles(profiles)
                    engine_outputs = outputs or {}
                    output_dict = {
                        name: engine_outputs[name].float()
                        for name in policy.output_names
                    }

                    def postprocess() -> Any:
                        rows = OrderedDict()
                        rows["ego"] = output_dict
                        return dataset.post_process(batch, rows)

                    (pred_box, pred_score, gt_box), post_ms = _timed(postprocess, device)
                    forward_ms = _float_profile(profile, "total_runner_ms")
                    latency_rows.append(
                        {
                            "frame_id": frame_id,
                            "phase": role,
                            "input_prepare_ms": prepare_ms,
                            "host_to_device_ms": h2d_ms,
                            "forward_ms": forward_ms,
                            "postprocess_ms": post_ms,
                        }
                    )
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
                    post_times.append(post_ms)
                    total_times.append(forward_ms + post_ms)
                except Exception as exc:  # noqa: BLE001
                    reason = f"{type(exc).__name__}:{exc}"
                    skip_reasons[reason] += 1
                    skipped.append(frame_id)
                    raise RuntimeError(f"v2xvit_evaluation_frame_failed:{frame_id}:{reason}") from exc

        run_phase("warmup", warmup_ids)
        run_phase("evaluation", evaluation_ids)
        if warmed != warmup_ids or evaluated != evaluation_ids or skipped:
            raise RuntimeError(
                f"v2xvit_eval_manifest_incomplete:warmup={len(warmed)}/{len(warmup_ids)}:"
                f"eval={len(evaluated)}/{len(evaluation_ids)}:skip={len(skipped)}"
            )
        ap = {}
        for threshold in IOU_THRESHOLDS:
            value = 0.0
            if result_stat[threshold]["gt"] > 0 and result_stat[threshold]["score"]:
                value, _, _ = eval_utils.calculate_ap(result_stat, threshold)
            ap[f"AP@{threshold:.1f}"] = float(value)
        result = {
            "status": "ok",
            **ap,
            "mAP": float(sum(ap.values()) / len(ap)),
            **_latency_distribution(forward_times, prefix="forward"),
            **_latency_distribution(post_times, prefix="postprocess"),
            **_latency_distribution(total_times, prefix="total"),
            "num_warmup_frames": len(warmed),
            "num_evaluated_frames": len(evaluated),
            "num_skipped_frames": len(skipped),
            "warmup_frame_ids": warmed,
            "evaluated_frame_ids": evaluated,
            "skipped_frame_ids": skipped,
            "skip_reason_counts": dict(skip_reasons),
            "fixed_k": policy.fixed_k,
            "max_agents": policy.max_agents,
            "input_contract": input_contract,
            "eval_manifest_path": str(manifest_path),
            "eval_manifest_hash": str(manifest.get("manifest_hash", "")),
            "reset_after_warmup": True,
            "dataloader_num_workers": workers,
            "torch_num_threads": torch_threads,
            "cuda_postprocess_audit": cuda_postprocess,
            "latency_rows": latency_rows,
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "evaluation_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    _write_json(output, result)
    print(json.dumps({"status": result.get("status"), "output": str(output)}, sort_keys=True))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
