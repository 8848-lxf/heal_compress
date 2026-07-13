#!/usr/bin/env python3
"""Collect entropy-style activation scales at the signal-maxK wrapper boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO.parent, Path("/home/lixingfeng/UniAD_examine/HEAL")):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

CHECKPOINT = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
CONFIG = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_SOURCE = REPO / "outputs/int8_baseline_equivalence_audit_20260713_021108"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def entropy_threshold(histogram: np.ndarray, absolute_maximum: float, quantized_bins: int = 128) -> dict[str, Any]:
    """Use NVIDIA Model Optimizer's installed entropy threshold implementation."""

    from modelopt.torch.quantization.calib.histogram import _compute_amax_entropy

    hist = np.asarray(histogram, dtype=np.int64)
    if hist.ndim != 1 or hist.size < quantized_bins or hist.sum() <= 0:
        raise RuntimeError("invalid_entropy_histogram")
    edges = np.linspace(0.0, float(absolute_maximum), hist.size + 1, dtype=np.float64)
    threshold_value = float(
        _compute_amax_entropy(
            hist.copy(),
            edges,
            num_bits=8,
            unsigned=False,
            stride=1,
            start_bin=quantized_bins,
        ).item()
    )
    best_threshold = int(round(threshold_value / float(absolute_maximum) * hist.size))
    best_threshold = min(max(best_threshold, quantized_bins), hist.size)
    return {
        "histogram_bins": int(hist.size),
        "quantized_bins": int(quantized_bins),
        "selected_bin": int(best_threshold),
        "absolute_maximum": float(absolute_maximum),
        "clipping_threshold": threshold_value,
        "clipped_fraction": float(hist[best_threshold:].sum() / max(hist.sum(), 1.0)),
        "implementation": "modelopt.torch.quantization.calib.histogram._compute_amax_entropy",
        "scale": threshold_value / 127.0,
    }


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            result = first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, (list, tuple)):
        for item in value:
            result = first_tensor(item)
            if result is not None:
                return result
    return None


def per_channel_weight_scale(weight: np.ndarray, op_type: str, node: Any) -> tuple[list[float], int]:
    if op_type == "Conv":
        axis = 0
    elif op_type == "ConvTranspose":
        axis = 1
    elif op_type == "MatMul":
        axis = 1
    elif op_type == "Gemm":
        trans_b = next((int(attr.i) for attr in node.attribute if str(attr.name) == "transB"), 0)
        axis = 0 if trans_b else 1
    else:
        raise RuntimeError(f"unsupported_weight_op:{op_type}")
    reduce_axes = tuple(index for index in range(weight.ndim) if index != axis)
    amax = np.max(np.abs(weight), axis=reduce_axes)
    if np.any(amax <= 0) or not np.all(np.isfinite(amax)):
        raise RuntimeError("invalid_per_channel_weight_amax")
    return (amax / 127.0).astype(np.float32).tolist(), axis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--histogram-bins", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--coverage", choices=("current22", "legacy67"), default="current22")
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV", "") != "univ2x-opt":
        raise RuntimeError("entropy_calibration_requires_univ2x-opt")
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"force_rebuild_destination_exists:{output}")
    output.mkdir(parents=True)

    import onnx
    from onnx import numpy_helper
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from torch.utils.data import DataLoader
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from quantization.config import OnnxExportConfig
    from quantization.export.heal_lidar_pyramid import prepare_signal_maxk_inputs
    from search.integration.calibration_provider import _paired_batchnorm_path
    from search.integration.trt_compatible_export import build_search_trt_compatible_export_module

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    adapter = HEALLiDARAdapter(heal_repo=HEAL_ROOT, config={"model": {"hypes_yaml": str(CONFIG)}})
    model = adapter.build_model(CONFIG, CHECKPOINT).to(device).eval()
    qcfg = OnnxExportConfig(fixed_k=29696, min_agents=1, opt_agents=2, max_agents=2)
    wrapper = build_search_trt_compatible_export_module(
        model,
        output_names=qcfg.output_names,
        fixed_k=qcfg.fixed_k,
        modality="m1",
    ).to(device).eval()
    wrapper_modules = dict(wrapper.named_modules())

    source = args.source_audit_root.resolve()
    artifacts = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    mapping = read_json(artifacts / "canonical_layer_map.json")
    entries = {str(row["module_path"]): row for row in mapping["entries"]}
    if args.coverage == "legacy67":
        protected = {
            "encoder_m1.pillar_vfe.pfn_layers.0.linear",
            "pyramid_backbone.single_head_2",
        }
        module_paths = [
            str(row["module_path"])
            for row in mapping["entries"]
            if str(row["module_path"]) not in protected
        ]
    else:
        manifest = read_json(artifacts / "calibration_manifest.json")
        module_paths = [str(value) for value in manifest["module_paths"]]
    base_onnx = onnx.load(str(artifacts / "pruned_fp32.onnx"))
    nodes = {str(node.name): node for node in base_onnx.graph.node}
    initializers = {str(row.name): numpy_helper.to_array(row) for row in base_onnx.graph.initializer}
    output_paths = {name: (_paired_batchnorm_path(model, name) or name) for name in module_paths}
    missing = [
        value
        for name in module_paths
        for value in (f"model.{name}", f"model.{output_paths[name]}")
        if value not in wrapper_modules
    ]
    if missing:
        raise RuntimeError(f"entropy_observer_modules_missing:{sorted(set(missing))}")

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(CONFIG))
    hypes = adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        pin_memory=False,
        drop_last=False,
    )

    state = {
        name: {
            "input_amax": None,
            "output_amax": None,
            "input_hist": None,
            "output_hist": None,
            "input_count": 0,
            "output_count": 0,
        }
        for name in module_paths
    }
    phase = {"name": "amax"}
    handles = []

    def observe(name: str, role: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().float().abs()
        row = state[name]
        if phase["name"] == "amax":
            maximum = value.amax()
            key = f"{role}_amax"
            row[key] = maximum if row[key] is None else torch.maximum(row[key], maximum)
            row[f"{role}_count"] += 1
        else:
            maximum = float(row[f"{role}_amax"].item())
            histogram = torch.histc(value, bins=int(args.histogram_bins), min=0.0, max=maximum)
            key = f"{role}_hist"
            row[key] = histogram if row[key] is None else row[key] + histogram

    def input_hook(name: str):
        def hook(_module: Any, inputs: Any) -> None:
            tensor = first_tensor(inputs)
            if tensor is None:
                raise RuntimeError(f"entropy_input_tensor_missing:{name}")
            observe(name, "input", tensor)

        return hook

    def output_hook(name: str):
        def hook(_module: Any, _inputs: Any, output_value: Any) -> None:
            tensor = first_tensor(output_value)
            if tensor is None:
                raise RuntimeError(f"entropy_output_tensor_missing:{name}")
            observe(name, "output", tensor)

        return hook

    for name in module_paths:
        handles.append(wrapper_modules[f"model.{name}"].register_forward_pre_hook(input_hook(name)))
        handles.append(wrapper_modules[f"model.{output_paths[name]}"].register_forward_hook(output_hook(name)))

    def run_pass(pass_name: str) -> int:
        phase["name"] = pass_name
        reset_seed(args.seed)
        count = 0
        with torch.inference_mode():
            for batch in loader:
                if count >= int(args.frames):
                    break
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                prepared = prepare_signal_maxk_inputs(ego, config=qcfg, modality="m1")
                inputs = {name: tensor.to(device) for name, tensor in prepared.items()}
                wrapper(**inputs)
                count += 1
        return count

    try:
        first_count = run_pass("amax")
        second_count = run_pass("histogram")
    finally:
        for handle in handles:
            handle.remove()
    if first_count != int(args.frames) or second_count != int(args.frames):
        raise RuntimeError(f"entropy_frame_count_mismatch:{first_count}:{second_count}:{args.frames}")

    histogram_payload = {}
    for name, row in state.items():
        histogram_payload[f"{name}::input"] = row["input_hist"].cpu().numpy().astype(np.int64)
        histogram_payload[f"{name}::output"] = row["output_hist"].cpu().numpy().astype(np.int64)
    np.savez_compressed(output / "entropy_histograms.npz", **histogram_payload)

    scales: dict[str, dict[str, Any]] = {}
    lineage: dict[str, Any] = {}
    for name in module_paths:
        row = state[name]
        if row["input_count"] != first_count or row["output_count"] != first_count:
            raise RuntimeError(f"entropy_observation_count_mismatch:{name}")
        input_result = entropy_threshold(row["input_hist"].cpu().numpy(), float(row["input_amax"].item()))
        output_result = entropy_threshold(row["output_hist"].cpu().numpy(), float(row["output_amax"].item()))
        entry = entries[name]
        node = nodes[str(entry["canonical_node_name"])]
        weight = initializers[str(entry["weight_initializer"])]
        weight_scales, axis = per_channel_weight_scale(weight, str(node.op_type), node)
        scales[name] = {
            "activation_input_scale": input_result["scale"],
            "weight_scale": weight_scales,
            "weight_axis": axis,
            "weight_granularity": "per_channel",
            "coverage": args.coverage,
            "weight_scale_shape": [len(weight_scales)],
            "activation_output_scale": output_result["scale"],
            "activation_input_tensor": str(node.input[0]),
            "activation_output_tensor": str(node.output[0]),
            "activation_scale_source": "signal_maxK_wrapper_entropy_KL_2048_to_128",
            "weight_scale_source": "final_folded_onnx_initializer",
        }
        lineage[name] = {
            "input_observer_module": f"model.{name}:forward_pre_hook",
            "output_observer_module": f"model.{output_paths[name]}:forward_hook",
            "q_input_tensor": str(node.input[0]),
            "q_output_tensor": str(node.output[0]),
            "input": input_result,
            "output": output_result,
            "calls_per_pass": first_count,
            "fixed_k": qcfg.fixed_k,
            "valid_voxel_mask_applied_by_wrapper": True,
        }

    result = {
        "metadata": {
            "algorithm": "NVIDIA_ModelOpt_symmetric_entropy_KL_absolute_histogram_2048_to_128",
            "implementation": "modelopt.torch.quantization.calib.histogram._compute_amax_entropy",
            "scale_formula": "selected_clipping_threshold / 127",
            "frames": first_count,
            "passes": 2,
            "seed": args.seed,
            "split": "train",
            "fixed_k": qcfg.fixed_k,
            "observer_graph": "SearchTensorRTCompatibleLidarPyramid",
            "observer_tensor_identity": "module/canonical mapping exact; numerical ORT identity audited separately",
            "weight_granularity": "per_channel",
            "histogram_artifact": str(output / "entropy_histograms.npz"),
            "histogram_sha256": sha256_file(output / "entropy_histograms.npz"),
            "base_onnx": str(artifacts / "pruned_fp32.onnx"),
            "base_onnx_sha256": sha256_file(artifacts / "pruned_fp32.onnx"),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "config": str(CONFIG),
            "config_sha256": sha256_file(CONFIG),
        },
        "scales": scales,
        "lineage": lineage,
    }
    write_json(output / "entropy_calibration_scales.json", result)
    write_json(output / "entropy_calibration_manifest.json", result["metadata"])
    print(json.dumps({"output": str(output), "frames": first_count, "module_count": len(scales)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
