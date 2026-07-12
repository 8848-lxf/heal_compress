"""One fixed real-validation PyTorch evaluator for every experiment candidate."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


THRESHOLDS = (0.03, 0.30, 0.50, 0.70)


def frame_list_hash(frame_ids: Sequence[str]) -> str:
    raw = json.dumps(
        [str(value) for value in frame_ids],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_manifest_binding(
    manifest: Mapping[str, Any],
    dataset_split_ids: Sequence[str],
) -> dict[str, Any]:
    expected = [str(value) for value in manifest.get("frame_ids", [])]
    declared_count = int(manifest.get("frame_count", 0))
    if declared_count != len(expected) or declared_count <= 0:
        raise ValueError(
            f"manifest frame count mismatch: declared={declared_count}, ids={len(expected)}"
        )
    observed = [str(value) for value in dataset_split_ids[:declared_count]]
    if observed != expected:
        raise ValueError("dataset split frame order differs from the fixed manifest")
    observed_hash = frame_list_hash(expected)
    if str(manifest.get("frame_list_hash", "")) != observed_hash:
        raise ValueError("manifest frame_list_hash does not match its exact frame IDs")
    return {
        "frame_ids": expected,
        "frame_count": declared_count,
        "frame_list_hash": observed_hash,
        "identity_source": "dataset.split_info",
        "selection_policy": manifest.get("selection_policy", ""),
    }


def _sample_index(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"expected one dataset-local sample index, got {value!r}")
        value = value[0]
    return int(value)


def evaluate_pytorch_model(
    model: Any,
    dataset: Any,
    manifest: Mapping[str, Any],
    *,
    device: str,
    heartbeat: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import torch
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    binding = validate_manifest_binding(manifest, getattr(dataset, "split_info", []))
    frame_ids = binding["frame_ids"]
    result_stat = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in THRESHOLDS
    }
    observed_indices: list[int] = []
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for index, _frame_id in enumerate(frame_ids):
            sample = dataset[index]
            if sample is None:
                raise RuntimeError(f"validation dataset returned None at fixed index {index}")
            batch = dataset.collate_batch_test([sample])
            if batch is None:
                raise RuntimeError(f"validation collate returned None at fixed index {index}")
            batch = train_utils.to_device(batch, device)
            if "sample_idx" not in batch["ego"]:
                raise RuntimeError("HEAL validation batch is missing dataset-local sample_idx")
            local_index = _sample_index(batch["ego"]["sample_idx"])
            if local_index != index:
                raise RuntimeError(
                    f"dataset-local sample index changed at position {index}: {local_index}"
                )
            observed_indices.append(local_index)
            inference = inference_utils.inference_intermediate_fusion(batch, model, dataset)
            for threshold in THRESHOLDS:
                eval_utils.caluclate_tp_fp(
                    inference["pred_box_tensor"],
                    inference["pred_score"],
                    inference["gt_box_tensor"],
                    result_stat,
                    threshold,
                )
            evaluated = index + 1
            if heartbeat is not None and evaluated % 25 == 0:
                heartbeat(
                    {
                        "evaluated_frames": evaluated,
                        "frame_count": len(frame_ids),
                        "last_frame_id": frame_ids[index],
                        "wall_seconds": time.time() - started,
                        "frame_list_hash": binding["frame_list_hash"],
                    }
                )
    metrics = {
        f"ap_{threshold:.2f}": float(eval_utils.calculate_ap(result_stat, threshold)[0])
        for threshold in THRESHOLDS
    }
    metrics["mAP"] = sum(metrics.values()) / len(metrics)
    return {
        **metrics,
        "frame_count": len(frame_ids),
        "evaluated_frames": len(observed_indices),
        "frame_list_hash": binding["frame_list_hash"],
        "frame_identity_source": binding["identity_source"],
        "dataset_local_sample_index_min": min(observed_indices),
        "dataset_local_sample_index_max": max(observed_indices),
        "validation_source_used": True,
        "synthetic_used": False,
        "evaluation_wall_seconds": time.time() - started,
    }


def apply_baseline_metrics(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(candidate)
    for key in ("ap_0.03", "ap_0.30", "ap_0.50", "ap_0.70", "mAP"):
        result[f"{key}_absolute_drop"] = float(baseline[key]) - float(candidate[key])
    result["mAP_retention"] = (
        float(candidate["mAP"]) / float(baseline["mAP"])
        if float(baseline["mAP"]) != 0.0
        else None
    )
    original_count = int(
        candidate.get("original_parameter_count", baseline.get("candidate_parameter_count", 0))
    )
    candidate_count = int(candidate.get("candidate_parameter_count", original_count))
    pruned = original_count - candidate_count
    result["actual_parameter_reduction"] = pruned
    result["actual_parameter_reduction_ratio"] = (
        float(pruned) / float(original_count) if original_count > 0 else None
    )
    result["delta_map_per_million_pruned_parameters"] = (
        float(result["mAP_absolute_drop"]) / (float(pruned) / 1_000_000.0)
        if pruned > 0
        else None
    )
    return result


__all__ = [
    "THRESHOLDS",
    "apply_baseline_metrics",
    "evaluate_pytorch_model",
    "frame_list_hash",
    "validate_manifest_binding",
]
