"""GPU-only TensorRT evaluator owned by the LiDAR CoBEVT model family."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import traceback
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from .evaluation_worker import (
    IOU_THRESHOLDS,
    _float_profile,
    _latency_distribution,
    _latency_row_from_profile,
    _loader_worker_init,
    _mean_profiles,
    _move,
    _smoke_gpu_ap_iou,
    _timed,
)


def validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    validated = dict(request)
    if str(validated.get("model_family", "")) != "lidar_cobevt":
        raise RuntimeError("cobevt_model_family_required")
    if not str(validated.get("device", "")).startswith("cuda:"):
        raise RuntimeError("cobevt_cuda_device_required")
    if int(validated.get("num_workers", -1)) != 8:
        raise RuntimeError("cobevt_evaluation_num_workers_must_equal_8")
    if str(validated.get("ap_iou_backend", "")).lower() != "gpu":
        raise RuntimeError("cobevt_gpu_ap_iou_backend_required")
    if not bool(validated.get("strict_gpu_ap_iou", False)):
        raise RuntimeError("cobevt_strict_gpu_ap_iou_required")
    if int(validated.get("fixed_k", 0)) <= 0:
        raise RuntimeError("cobevt_fixed_k_must_be_positive")
    return validated


def _execution_rounds(role: str, request: Mapping[str, Any]) -> int:
    key = "warmup_latency_rounds" if str(role) == "warmup" else "latency_rounds"
    fallback = request.get("latency_rounds", 1)
    return max(1, int(request.get(key, fallback)))


def _load_manifest(
    request: Mapping[str, Any],
    *,
    validation_split: Path,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    path = Path(str(request["eval_manifest_path"]))
    if not path.is_file():
        raise RuntimeError(f"eval_manifest_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    warmup_ids = [str(value) for value in payload.get("warmup_frame_ids", [])]
    evaluation_ids = [str(value) for value in payload.get("evaluation_frame_ids", [])]
    if len(warmup_ids) != int(request["warmup_frames"]):
        raise RuntimeError("eval_manifest_warmup_count_mismatch")
    if len(evaluation_ids) != int(request["num_frames"]):
        raise RuntimeError("eval_manifest_evaluation_count_mismatch")
    if not bool(payload.get("reset_after_warmup", False)):
        raise RuntimeError("cobevt_eval_manifest_must_reset_after_warmup")
    if len(set(warmup_ids)) != len(warmup_ids):
        raise RuntimeError("eval_manifest_duplicate_warmup_ids")
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise RuntimeError("eval_manifest_duplicate_evaluation_ids")
    split_payload = json.loads(validation_split.read_text(encoding="utf-8"))
    if not isinstance(split_payload, list):
        raise RuntimeError(f"validation_split_manifest_not_list:{validation_split}")
    split_ids = [str(value) for value in split_payload]
    missing = sorted(set(warmup_ids + evaluation_ids) - set(split_ids))
    if missing:
        raise RuntimeError(f"eval_manifest_ids_missing_from_validation_split:{missing[:8]}")
    return payload, split_ids, warmup_ids, evaluation_ids


def _output_dict(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = ("cls_preds", "reg_preds", "dir_preds")
    missing = [name for name in names if name not in outputs]
    if missing:
        raise RuntimeError(f"cobevt_engine_outputs_missing:{','.join(missing)}")
    selected = {name: outputs[name].float() for name in names}
    for name, tensor in selected.items():
        if tensor.numel() == 0:
            raise RuntimeError(f"cobevt_engine_output_empty:{name}")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"cobevt_engine_output_nonfinite:{name}")
    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    raw_request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output = Path(str(raw_request.get("output_path", Path(args.request).with_suffix(".result.json"))))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = validate_request(raw_request)
        heal_root = Path(str(request["heal_root"])).resolve()
        for entry in (heal_root.parent, heal_root, Path.cwd(), Path.cwd() / "tests"):
            text = str(entry)
            if text not in sys.path:
                sys.path.insert(0, text)
        plugin_path = Path(str(request["plugin_path"]))
        if not plugin_path.is_file():
            raise RuntimeError(f"cobevt_scatter_plugin_missing:{plugin_path}")
        ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
        for value in request.get("additional_plugin_paths", []):
            additional_plugin = Path(str(value)).resolve()
            if not additional_plugin.is_file():
                raise RuntimeError(f"cobevt_additional_plugin_missing:{additional_plugin}")
            ctypes.CDLL(str(additional_plugin), mode=ctypes.RTLD_GLOBAL)

        from adapters.heal_lidar_adapter import HEALLiDARAdapter
        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils
        from opencood.utils import eval_utils
        from quantization.export.heal_lidar_cobevt import prepare_cobevt_maxk_inputs
        from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner
        from tests.test_baseline_eval import calculate_tp_fp_for_threshold

        device = torch.device(str(request["device"]))
        if not torch.cuda.is_available():
            raise RuntimeError("cobevt_cuda_unavailable")
        torch.cuda.set_device(device)
        gpu_ap_iou_smoke = _smoke_gpu_ap_iou(device)

        adapter = HEALLiDARAdapter(
            heal_repo=str(heal_root),
            config={"model": {"hypes_yaml": str(request["model_config"])}},
        )
        config_path = adapter._resolve_heal_path(str(request["model_config"]))
        hypes = yaml_utils.load_yaml(config_path)
        hypes = adapter._absolutize_dataset_paths(hypes)
        dataset = build_dataset(hypes, visualize=True, train=False)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=dataset.collate_batch_test,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=_loader_worker_init,
        )
        manifest, split_ids, warmup_ids, evaluation_ids = _load_manifest(
            request,
            validation_split=Path(str(hypes["validate_dir"])),
        )
        runner = TensorRTEngineRunner(str(request["engine_path"]), device)
        result_stat = {
            threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
            for threshold in IOU_THRESHOLDS
        }
        forward_times: list[float] = []
        postprocess_times: list[float] = []
        total_times: list[float] = []
        latency_rows: list[dict[str, Any]] = []
        skip_reasons: Counter[str] = Counter()
        evaluated_ids: list[str] = []
        warmup_complete: list[str] = []

        phases = (("warmup", set(warmup_ids)), ("evaluation", set(evaluation_ids)))
        for role, selected_ids in phases:
            execution_rounds = _execution_rounds(role, request)
            for index, batch in enumerate(loader):
                if index >= len(split_ids):
                    break
                frame_id = split_ids[index]
                if frame_id not in selected_ids:
                    continue
                try:
                    if batch is None:
                        raise RuntimeError("empty_batch")
                    batch, host_to_device_ms = _timed(lambda: _move(batch, device), device)
                    ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                    inputs, input_prepare_ms = _timed(
                        lambda: prepare_cobevt_maxk_inputs(
                            ego,
                            fixed_k=int(request["fixed_k"]),
                            max_cav=int(request.get("max_cav", 2)),
                        ),
                        device,
                    )
                    outputs: Mapping[str, torch.Tensor] = {}
                    profiles: list[dict[str, Any]] = []
                    for _ in range(execution_rounds):
                        outputs, profile = runner.run_profiled(inputs)
                        profiles.append(profile)
                    profile = _mean_profiles(profiles)
                    typed_outputs = _output_dict(outputs)

                    def postprocess() -> Any:
                        values = OrderedDict()
                        values["ego"] = typed_outputs
                        return dataset.post_process(batch, values)

                    (pred_box, pred_score, gt_box), postprocess_ms = _timed(
                        postprocess, device
                    )
                    row = _latency_row_from_profile(
                        frame_id=frame_id,
                        warmup=role == "warmup",
                        input_prepare_ms=input_prepare_ms,
                        host_to_device_ms=host_to_device_ms,
                        profile=profile,
                        postprocess_ms=postprocess_ms,
                    )
                    latency_rows.append(row)
                    if role == "warmup":
                        warmup_complete.append(frame_id)
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
                    forward_ms = _float_profile(profile, "total_runner_ms")
                    forward_times.append(forward_ms)
                    postprocess_times.append(postprocess_ms)
                    total_times.append(forward_ms + postprocess_ms)
                    evaluated_ids.append(frame_id)
                except Exception as exc:  # noqa: BLE001
                    skip_reasons[f"{type(exc).__name__}:{exc}"] += 1
                    latency_rows.append(
                        {
                            "frame_id": frame_id,
                            "warmup": role == "warmup",
                            "success": False,
                            "skip_reason": f"{type(exc).__name__}:{exc}",
                        }
                    )
                    break
            if skip_reasons:
                break

        ap: dict[str, float] = {}
        for threshold in IOU_THRESHOLDS:
            if result_stat[threshold]["gt"] > 0 and result_stat[threshold]["score"]:
                value, _, _ = eval_utils.calculate_ap(result_stat, threshold)
            else:
                value = 0.0
            ap[f"AP@{threshold:.1f}"] = float(value)
        complete = (
            len(warmup_complete) == len(warmup_ids)
            and len(evaluated_ids) == len(evaluation_ids)
            and not skip_reasons
        )
        result = {
            "status": "ok" if complete else "evaluation_failed",
            "model_family": "lidar_cobevt",
            "AP@0.3": ap["AP@0.3"],
            "AP@0.5": ap["AP@0.5"],
            "AP@0.7": ap["AP@0.7"],
            "mAP": float(sum(ap.values()) / len(ap)),
            **_latency_distribution(forward_times, prefix="forward"),
            **_latency_distribution(postprocess_times, prefix="postprocess"),
            **_latency_distribution(total_times, prefix="total"),
            "num_evaluated_frames": len(evaluated_ids),
            "num_skipped_frames": int(sum(skip_reasons.values())),
            "skip_reason_counts": dict(skip_reasons),
            "warmup_frames": len(warmup_complete),
            "warmup_latency_rounds": _execution_rounds("warmup", request),
            "latency_rounds": _execution_rounds("evaluation", request),
            "evaluated_frame_ids": evaluated_ids,
            "latency_rows": latency_rows,
            "eval_manifest_hash": str(manifest.get("manifest_hash", "")),
            "eval_manifest_path": str(request["eval_manifest_path"]),
            "fixed_manifest_enforced": True,
            "reset_after_warmup": True,
            "ap_iou_backend": "gpu",
            "strict_gpu_ap_iou": True,
            "gpu_ap_iou_smoke": gpu_ap_iou_smoke,
            "dataloader_num_workers": 8,
            "fixed_k": int(request["fixed_k"]),
            "engine_path": str(request["engine_path"]),
            "physical_device": str(request.get("physical_device", "")),
            "runner_allocation_report": runner.allocation_report(),
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "evaluation_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result.get("status"), "output": str(output)}))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_request"]
