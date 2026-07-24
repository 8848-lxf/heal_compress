#!/usr/bin/env python3
"""Evaluate frozen V2X-ViT diagnostic controls with ONNX Runtime.

The TensorRT-only PointPillarScatter node is replaced by the audited ScatterND
reference graph.  This is a diagnostic backend bridge, never a deployment
fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.int8_equivalence_tensor_parity import replace_scatter_plugin_for_ort


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def evaluate(model_path: Path, request: dict[str, Any], device: torch.device) -> dict[str, Any]:
    import onnxruntime as ort
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.utils import eval_utils
    from tests.test_baseline_eval import calculate_tp_fp_for_threshold
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from search.integration.evaluation_worker import IOU_THRESHOLDS, _dataloader_worker_init, _move, _verify_cuda_postprocess_backend
    from search.model_family.export import HealV2XViTExportPolicy, prepare_v2xvit_fixed_k_inputs

    cuda_postprocess = _verify_cuda_postprocess_backend(device)
    adapter = HEALLiDARAdapter(heal_repo=request["heal_root"], config={"model": {"hypes_yaml": request["model_config"]}})
    hypes = adapter._absolutize_dataset_paths(yaml_utils.load_yaml(adapter._resolve_heal_path(request["model_config"])))
    dataset = build_dataset(hypes, visualize=True, train=False)
    workers = int(request.get("dataloader_num_workers", 8))
    kwargs: dict[str, Any] = {"batch_size": 1, "shuffle": False, "num_workers": workers, "collate_fn": dataset.collate_batch_test, "pin_memory": workers > 0}
    if workers:
        kwargs.update({"prefetch_factor": 2, "persistent_workers": True, "worker_init_fn": _dataloader_worker_init})
    loader = DataLoader(dataset, **kwargs)
    policy = HealV2XViTExportPolicy(fixed_k=int(request["fixed_k"]), max_agents=int(request.get("max_agents", 2)))
    session = ort.InferenceSession(
        str(model_path),
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    session_inputs = {row.name for row in session.get_inputs()}
    session_outputs = {row.name for row in session.get_outputs()}
    missing_outputs = sorted(set(policy.output_names) - session_outputs)
    if missing_outputs:
        raise RuntimeError(f"ort_missing_policy_outputs:{missing_outputs}")

    manifest_path = Path(request["eval_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warmup_ids = [str(x) for x in manifest["warmup_frame_ids"][: int(request["warmup_frames"])]]
    evaluation_ids = [str(x) for x in manifest["evaluation_frame_ids"][: int(request["num_frames"])]]
    split_ids = [str(x) for x in json.loads(Path(str(hypes["validate_dir"])).read_text(encoding="utf-8"))]
    stats = {t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in IOU_THRESHOLDS}
    warmed: list[str] = []
    evaluated: list[str] = []
    skipped: list[str] = []
    reasons: Counter[str] = Counter()

    def phase(role: str, wanted: list[str]) -> None:
        wanted_set = set(wanted)
        collected = warmed if role == "warmup" else evaluated
        for index, batch in enumerate(loader):
            if len(collected) == len(wanted):
                break
            frame_id = split_ids[index]
            if frame_id not in wanted_set:
                continue
            try:
                if batch is None:
                    raise RuntimeError("empty_batch")
                batch = _move(batch, device)
                prepared = prepare_v2xvit_fixed_k_inputs(batch["ego"], policy=policy)
                feed = {name: tensor.detach().cpu().numpy() for name, tensor in prepared.items() if name in session_inputs}
                if set(feed) != session_inputs:
                    raise RuntimeError(f"ort_input_contract_mismatch:missing={sorted(session_inputs-set(feed))}:extra={sorted(set(feed)-session_inputs)}")
                values = session.run(list(policy.output_names), feed)
                outputs = {name: torch.from_numpy(np.asarray(value)).to(device=device, dtype=torch.float32) for name, value in zip(policy.output_names, values)}
                rows = OrderedDict(); rows["ego"] = outputs
                pred_box, pred_score, gt_box = dataset.post_process(batch, rows)
                if role == "warmup":
                    warmed.append(frame_id)
                    continue
                for threshold in IOU_THRESHOLDS:
                    calculate_tp_fp_for_threshold(pred_box, pred_score, gt_box, stats, threshold, "gpu", device)
                evaluated.append(frame_id)
            except Exception as exc:  # noqa: BLE001
                skipped.append(frame_id); reasons[f"{type(exc).__name__}:{exc}"] += 1
                raise

    phase("warmup", warmup_ids)
    phase("evaluation", evaluation_ids)
    if warmed != warmup_ids or evaluated != evaluation_ids or skipped:
        raise RuntimeError(f"ort_manifest_incomplete:warmup={len(warmed)}:eval={len(evaluated)}:skip={len(skipped)}")
    ap: dict[str, float] = {}
    for threshold in IOU_THRESHOLDS:
        value = 0.0
        if stats[threshold]["gt"] > 0 and stats[threshold]["score"]:
            value, _, _ = eval_utils.calculate_ap(stats, threshold)
        ap[f"AP@{threshold:.1f}"] = float(value)
    return {
        "status": "ok", **ap, "mAP": float(sum(ap.values()) / len(ap)),
        "backend": "onnxruntime", "providers": session.get_providers(),
        "scatter_semantics": "PointPillarScatterTRT_replaced_by_ScatterND_reference",
        "diagnostic_backend_bridge": True, "eval_manifest_path": str(manifest_path),
        "eval_manifest_hash": manifest.get("manifest_hash", ""), "num_warmup_frames": len(warmed),
        "num_evaluated_frames": len(evaluated), "num_skipped_frames": len(skipped),
        "evaluated_frame_ids": evaluated, "skip_reason_counts": dict(reasons),
        "cuda_postprocess_audit": cuda_postprocess,
    }


def run(args: argparse.Namespace) -> int:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected_exactly_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    source = args.source_root.resolve(); out = args.output_root.resolve()
    inherited = json.loads((source / "evaluation/v2xvit_greedy005_final_fixed50/evaluation_request.json").read_text(encoding="utf-8"))
    request = {key: inherited[key] for key in ("heal_root", "model_config", "fixed_k", "max_agents", "eval_manifest_path")}
    request.update({"warmup_frames": 5, "num_frames": 50, "dataloader_num_workers": 8})
    results: dict[str, Any] = {}
    for name in args.controls:
        control_dir = out / "ort" / name
        control_dir.mkdir(parents=True, exist_ok=True)
        backend_name = name.replace("-", "_")
        source_onnx = out / "tensorrt" / backend_name / "physical_mixed_qdq.onnx"
        reference = control_dir / "reference_scatter.onnx"
        try:
            selected = replace_scatter_plugin_for_ort(source_onnx, reference) if not reference.is_file() else {}
            result = evaluate(reference, request, device)
            result["selected_intermediate_outputs"] = selected
        except Exception as exc:  # noqa: BLE001
            result = {"status": "evaluation_failed", "failure_reason": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()}
        write(control_dir / "evaluation.json", result)
        results[name] = result
    write(out / "reports/phase2_ort_evaluations.json", {"schema_version": "v2xvit-greedy005-ort-alignment-v1", "controls": results})
    print(json.dumps({name: row.get("status") for name, row in results.items()}, sort_keys=True))
    return 0 if all(row.get("status") == "ok" for row in results.values()) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controls", nargs="+", default=["S32", "S16", "JMIX-FRESH"])
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
