"""Parity-only TensorRT worker for CoBEVT Attention intermediate tensors."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from search.model_families.lidar_cobevt.attention_tensor_parity import (
    qk_metrics,
    residual_metrics,
    select_failure_frames,
    softmax_metrics,
    tensor_error_metrics,
)

from .evaluation_worker import _loader_worker_init, _move
from .lidar_cobevt_evaluation_worker import _load_manifest


def validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    validated = dict(request)
    if str(validated.get("model_family", "")) != "lidar_cobevt":
        raise RuntimeError("cobevt_parity_model_family_required")
    if not str(validated.get("device", "")).startswith("cuda:"):
        raise RuntimeError("cobevt_parity_cuda_device_required")
    if int(validated.get("num_workers", -1)) != 8:
        raise RuntimeError("cobevt_parity_num_workers_must_equal_8")
    if int(validated.get("fixed_k", 0)) != 29696:
        raise RuntimeError("cobevt_parity_fixed_k_29696_required")
    if int(validated.get("num_frames", 0)) != 10:
        raise RuntimeError("cobevt_parity_smoke10_required")
    if not bool(validated.get("diagnostic_latency_invalid", False)):
        raise RuntimeError("cobevt_parity_diagnostic_only_required")
    reference = str(validated.get("reference_engine_path", ""))
    candidate = str(validated.get("candidate_engine_path", ""))
    if not reference or reference == candidate:
        raise RuntimeError("cobevt_parity_engines_must_differ")
    specs = list(validated.get("output_specs", []))
    if not specs:
        raise RuntimeError("cobevt_parity_output_specs_required")
    names = [str(row.get("tensor_name", "")) for row in specs]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise RuntimeError("cobevt_parity_output_specs_invalid")
    validated["output_specs"] = specs
    return validated


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("block_id", "")),
                str(row.get("role", "")),
                str(row.get("tensor_name", "")),
            )
        ].append(row)
    result = []
    for (block_id, role, tensor_name), items in sorted(grouped.items()):
        numeric: dict[str, list[float]] = defaultdict(list)
        for item in items:
            for key, value in item.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric[str(key)].append(float(value))
        summary: dict[str, Any] = {
            "block_id": block_id,
            "frame_count": len(items),
            "role": role,
            "tensor_name": tensor_name,
        }
        for key, values in sorted(numeric.items()):
            finite = [value for value in values if value == value]
            if finite:
                summary[f"{key}_mean"] = sum(finite) / len(finite)
                summary[f"{key}_max"] = max(finite)
        result.append(summary)
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    request_path = Path(args.request).expanduser().resolve()
    raw = json.loads(request_path.read_text(encoding="utf-8"))
    output_path = Path(
        str(raw.get("output_path", request_path.with_suffix(".result.json")))
    )
    try:
        request = validate_request(raw)
        heal_root = Path(str(request["heal_root"])).resolve()
        for entry in (heal_root.parent, heal_root, Path.cwd(), Path.cwd() / "tests"):
            value = str(entry)
            if value not in sys.path:
                sys.path.insert(0, value)
        plugin_path = Path(str(request["plugin_path"])).resolve()
        if not plugin_path.is_file():
            raise RuntimeError(f"cobevt_scatter_plugin_missing:{plugin_path}")
        ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)

        from adapters.heal_lidar_adapter import HEALLiDARAdapter
        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils
        from quantization.export.heal_lidar_cobevt import prepare_cobevt_maxk_inputs
        from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

        device = torch.device(str(request["device"]))
        if not torch.cuda.is_available():
            raise RuntimeError("cobevt_parity_cuda_unavailable")
        torch.cuda.set_device(device)
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
        manifest, split_ids, _warmup_ids, evaluation_ids = _load_manifest(
            request,
            validation_split=Path(str(hypes["validate_dir"])),
        )
        reference_runner = TensorRTEngineRunner(
            str(request["reference_engine_path"]), device
        )
        candidate_runner = TensorRTEngineRunner(
            str(request["candidate_engine_path"]), device
        )
        specs = list(request["output_specs"])
        selected_ids = set(evaluation_ids)
        detailed: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        evaluated: list[str] = []
        skip_reasons: dict[str, int] = defaultdict(int)
        for index, batch in enumerate(loader):
            if index >= len(split_ids):
                break
            frame_id = split_ids[index]
            if frame_id not in selected_ids:
                continue
            try:
                batch = _move(batch, device)
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                inputs = prepare_cobevt_maxk_inputs(
                    ego, fixed_k=29696, max_cav=int(request.get("max_cav", 2))
                )
                reference = reference_runner.run(inputs)
                candidate = candidate_runner.run(inputs)
                frame_metrics: list[dict[str, Any]] = []
                by_block: dict[str, dict[str, str]] = defaultdict(dict)
                for spec in specs:
                    block_id = str(spec["block_id"])
                    role = str(spec["role"])
                    name = str(spec["tensor_name"])
                    if name not in reference or name not in candidate:
                        raise RuntimeError(f"cobevt_parity_output_missing:{name}")
                    metrics = tensor_error_metrics(reference[name], candidate[name])
                    if role in {"qk_matmul", "scaled_qk_logits"}:
                        metrics.update(qk_metrics(reference[name], candidate[name]))
                    if role == "softmax":
                        metrics.update(softmax_metrics(reference[name], candidate[name]))
                    row = {
                        "block_id": block_id,
                        "frame_id": frame_id,
                        "role": role,
                        "tensor_name": name,
                        **metrics,
                    }
                    frame_metrics.append(row)
                    by_block[block_id][role] = name
                for block_id, roles in by_block.items():
                    required = {
                        "residual_input",
                        "residual_attention_update",
                        "residual_add",
                    }
                    if not required.issubset(roles):
                        continue
                    metrics = residual_metrics(
                        reference[roles["residual_input"]],
                        reference[roles["residual_attention_update"]],
                        reference[roles["residual_add"]],
                        candidate[roles["residual_add"]],
                    )
                    for row in frame_metrics:
                        if row["block_id"] == block_id and row["role"] == "residual_add":
                            row.update(metrics)
                            break
                detailed.extend(frame_metrics)
                frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "maximum_absolute_error": max(
                            float(row["maximum_absolute_error"])
                            for row in frame_metrics
                        ),
                    }
                )
                evaluated.append(frame_id)
            except Exception as exc:  # noqa: BLE001
                skip_reasons[f"{type(exc).__name__}:{exc}"] += 1
                break
        complete = len(evaluated) == 10 and not skip_reasons
        result = {
            "attention_failure_frames": select_failure_frames(frame_rows, count=3),
            "candidate_engine_path": str(request["candidate_engine_path"]),
            "dataloader_num_workers": 8,
            "detailed_rows": detailed,
            "diagnostic_latency_invalid": True,
            "evaluated_frame_ids": evaluated,
            "eval_manifest_hash": str(manifest.get("manifest_hash", "")),
            "fixed_k": 29696,
            "num_evaluated_frames": len(evaluated),
            "num_skipped_frames": int(sum(skip_reasons.values())),
            "profile_name": str(request.get("profile_name", "")),
            "reference_engine_path": str(request["reference_engine_path"]),
            "skip_reason_counts": dict(skip_reasons),
            "status": "ok" if complete else "parity_failed",
            "summary_rows": _aggregate(detailed),
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "diagnostic_latency_invalid": True,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "status": "parity_failed",
            "traceback": traceback.format_exc(),
        }
    _write_json(output_path, result)
    print(json.dumps({"output": str(output_path), "status": result["status"]}))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_request"]
