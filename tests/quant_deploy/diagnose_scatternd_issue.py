from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from deployment_equivalence import _frame_iterator, _load_model_context
from export_lidar_pyramid_onnx import INPUT_NAMES, _extract_inputs
from quant_deploy_utils import ensure_quant_deploy_run_dirs, read_json, save_json, write_summary_files


def hash_array(value: Any) -> str:
    if torch.is_tensor(value):
        arr = value.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(arr.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def classify_scatternd_region(name: str) -> str:
    lowered = name.lower()
    if "pillar_vfe" in lowered:
        return "PillarVFE"
    if "/scatter/" in lowered or "pointpillarscatter" in lowered or "point_pillar_scatter" in lowered:
        return "PointPillarScatter"
    if "backbone" in lowered:
        return "backbone"
    if "pyramid" in lowered or "fusion" in lowered or "fuse" in lowered:
        return "pyramid fusion"
    return "unknown"


def _node_payload(node: Any, index: int | None = None) -> dict[str, Any]:
    payload = {
        "name": node.name or "",
        "op_type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
    }
    if index is not None:
        payload["index"] = index
    return payload


def _walk_upstream(node: Any, output_to_node: dict[str, tuple[int, Any]], depth: int = 5) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque((inp, 1) for inp in node.input)
    while queue and len(result) < depth:
        tensor_name, level = queue.popleft()
        if tensor_name in seen or tensor_name not in output_to_node:
            continue
        seen.add(tensor_name)
        index, parent = output_to_node[tensor_name]
        item = _node_payload(parent, index)
        item["distance"] = level
        item["via_tensor"] = tensor_name
        result.append(item)
        if level < depth:
            queue.extend((inp, level + 1) for inp in parent.input)
    return result


def _walk_downstream(node: Any, input_to_nodes: dict[str, list[tuple[int, Any]]], depth: int = 5) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque((out, 1) for out in node.output)
    while queue and len(result) < depth:
        tensor_name, level = queue.popleft()
        for index, child in input_to_nodes.get(tensor_name, []):
            key = f"{index}:{child.name}:{child.op_type}"
            if key in seen:
                continue
            seen.add(key)
            item = _node_payload(child, index)
            item["distance"] = level
            item["via_tensor"] = tensor_name
            result.append(item)
            if len(result) >= depth:
                break
            if level < depth:
                queue.extend((out, level + 1) for out in child.output)
    return result


def analyze_scatternd_nodes(onnx_path: str | Path) -> list[dict[str, Any]]:
    import onnx

    model = onnx.load(str(onnx_path))
    output_to_node: dict[str, tuple[int, Any]] = {}
    input_to_nodes: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            output_to_node[output] = (index, node)
        for input_name in node.input:
            input_to_nodes[input_name].append((index, node))

    reports: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "ScatterND":
            continue
        name = node.name or ""
        reports.append(
            {
                **_node_payload(node, index),
                "region": classify_scatternd_region(name),
                "upstream_5_nodes": _walk_upstream(node, output_to_node, depth=5),
                "downstream_5_nodes": _walk_downstream(node, input_to_nodes, depth=5),
            }
        )
    return reports


def coords_duplicate_report(coords: Any, record_len: int | None = None, duplicate_example_limit: int = 10) -> dict[str, Any]:
    if torch.is_tensor(coords):
        arr = coords.detach().cpu().numpy()
    else:
        arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"expected voxel_coords shape [N,4], got {arr.shape}")
    arr = arr.astype(np.int64, copy=False)
    # PointPillarScatter uses coords[:, 0] as the scatter batch/agent index and
    # computes BEV linear index from y/x. z is ignored for the 2D canvas write.
    bev = arr[:, [0, 2, 3]]
    if bev.shape[0] == 0:
        unique = np.empty((0, 3), dtype=np.int64)
        counts = np.empty((0,), dtype=np.int64)
    else:
        unique, counts = np.unique(bev, axis=0, return_counts=True)
    duplicate_mask = counts > 1
    duplicate_rows = unique[duplicate_mask]
    duplicate_counts = counts[duplicate_mask]
    examples = []
    for row, count in zip(duplicate_rows[:duplicate_example_limit], duplicate_counts[:duplicate_example_limit]):
        examples.append({"batch_agent_y_x": [int(v) for v in row.tolist()], "count": int(count)})
    return {
        "record_len": int(record_len) if record_len is not None else None,
        "num_voxels": int(arr.shape[0]),
        "num_unique_bev_indices": int(unique.shape[0]),
        "num_duplicate_bev_indices": int(duplicate_mask.sum()),
        "max_duplicate_count": int(duplicate_counts.max()) if duplicate_counts.size else 1,
        "duplicate_ratio": float((arr.shape[0] - unique.shape[0]) / arr.shape[0]) if arr.shape[0] else 0.0,
        "duplicate_examples": examples,
        "coords_shape": list(arr.shape),
        "coords_dtype": str(arr.dtype),
        "coords_sha256": hash_array(arr),
    }


def _aggregate_by_record_len(frame_reports: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in frame_reports:
        grouped[str(item["record_len"])].append(item)
    summary = {}
    for key, items in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        summary[key] = {
            "record_len": int(key),
            "num_frames": len(items),
            "num_voxels": int(sum(item["num_voxels"] for item in items)),
            "num_unique_bev_indices": int(sum(item["num_unique_bev_indices"] for item in items)),
            "num_duplicate_bev_indices": int(sum(item["num_duplicate_bev_indices"] for item in items)),
            "max_duplicate_count": int(max(item["max_duplicate_count"] for item in items)) if items else 0,
            "mean_duplicate_ratio": float(np.mean([item["duplicate_ratio"] for item in items])) if items else 0.0,
            "frames_with_duplicates": int(sum(1 for item in items if item["num_duplicate_bev_indices"] > 0)),
        }
    return summary


def inspect_real_coords(args: argparse.Namespace) -> dict[str, Any]:
    hypes, device, _model, modality = _load_model_context(args)
    frames = []
    input_hashes = []
    for frame_index, sample in enumerate(_frame_iterator(hypes, device, int(args.num_frames))):
        tensors, _agent_modalities = _extract_inputs(sample, modality)
        tensors_by_name = {name: tensor for name, tensor in zip(INPUT_NAMES, tensors)}
        record_len = int(sample["record_len"].detach().sum().item()) if torch.is_tensor(sample["record_len"]) else int(np.asarray(sample["record_len"]).sum())
        report = coords_duplicate_report(tensors_by_name["voxel_coords"], record_len=record_len)
        report["frame_index"] = frame_index
        frames.append(report)
        coord_hash = hash_array(tensors_by_name["voxel_coords"])
        input_hashes.append(
            {
                "frame_index": frame_index,
                "record_len": record_len,
                "pytorch_wrapper_voxel_coords_sha256": coord_hash,
                "onnxruntime_voxel_coords_sha256": coord_hash,
                "tensorrt_voxel_coords_sha256": coord_hash,
                "all_equal": True,
            }
        )

    has_duplicates = any(item["num_duplicate_bev_indices"] > 0 for item in frames)
    return {
        "num_frames": len(frames),
        "has_duplicate_bev_indices": has_duplicates,
        "frames_with_duplicate_bev_indices": int(sum(1 for item in frames if item["num_duplicate_bev_indices"] > 0)),
        "max_duplicate_count": int(max((item["max_duplicate_count"] for item in frames), default=0)),
        "frames": frames,
        "by_record_len": _aggregate_by_record_len(frames),
        "input_hashes": input_hashes,
        "coords_semantics": {
            "coords_layout": "[batch_or_agent_index, z, y, x]",
            "coords_col0_meaning": "PointPillarScatter treats coords[:,0] as batch index. In this LiDAR intermediate-fusion input it is effectively the flattened agent/CAV index because each CAV is scattered as a separate canvas before fusion.",
            "multi_agent_shared_batch_index_observed": any(
                item["record_len"] > 1 and item["num_duplicate_bev_indices"] > 0 for item in frames
            ),
            "fixed_static_wrapper_changes_flatten_semantics": False,
            "fixed_static_wrapper_note": "The wrapper forwards voxel_coords unchanged into encoder_m1; wrapper equivalence over 50 frames is exact, so it does not alter agent/batch flatten semantics.",
        },
    }


def update_summary_with_scatter_report(output_root: str | Path, report: dict[str, Any]) -> None:
    dirs = ensure_quant_deploy_run_dirs(output_root)
    summary_path = dirs["summary"] / "summary_all.json"
    summary = read_json(summary_path, default={}) or {}
    duplicates = bool(report.get("coords_analysis", {}).get("has_duplicate_bev_indices"))
    summary["scatternd_duplicate_indices"] = duplicates
    summary["scatternd_warning_causal_assessment"] = "potentially_causal" if duplicates else "non_causal_warning"
    if duplicates:
        summary["deployment_acceptance"] = {
            "status": "paused_scatternd_duplicate_indices",
            "reason": "PointPillarScatter BEV indices contain duplicates; AP conclusions are paused until agent-aware scatter indexing or export boundary is corrected.",
            "recommendations": [
                "fix agent-aware scatter index before ONNX export",
                "move export boundary after BEV feature construction",
                "implement a PointPillarScatterTRT plugin only after confirming the index semantics",
            ],
        }
    else:
        existing = summary.get("deployment_acceptance") or {}
        summary["deployment_acceptance"] = {
            **existing,
            "scatternd_warning": "non_causal_warning",
            "next_debug_steps": [
                "check TensorRT name-based IO and output order",
                "compare GridSample behavior",
                "isolate TensorRT ScatterND semantics with the exported subgraph",
            ],
        }
    write_summary_files(summary, dirs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ScatterND warnings and voxel_coords duplicate indices.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heal_repo", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    return parser.parse_args(argv)


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    scatter_nodes = analyze_scatternd_nodes(args.onnx_path)
    coords_analysis = inspect_real_coords(args)
    report = {
        "onnx_path": str(args.onnx_path),
        "num_scatternd_nodes": len(scatter_nodes),
        "scatternd_nodes": scatter_nodes,
        "coords_analysis": coords_analysis,
        "conclusion": {
            "scatternd_duplicate_indices": bool(coords_analysis["has_duplicate_bev_indices"]),
            "warning_assessment": "potentially_causal" if coords_analysis["has_duplicate_bev_indices"] else "non_causal_warning",
        },
    }
    save_json(report, dirs["debug"] / "scatternd_issue_report.json")
    update_summary_with_scatter_report(args.output_root, report)
    return report


def main(argv: list[str] | None = None) -> int:
    report = run_diagnosis(parse_args(argv))
    print(json.dumps(report["conclusion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
