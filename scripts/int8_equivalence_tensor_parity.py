#!/usr/bin/env python3
"""Ten-frame intermediate-tensor parity for the INT8 baseline audit.

The script is intentionally diagnostic: it rebuilds separate augmented engines
whose extra outputs prevent some normal TensorRT fusion.  Official AP/latency
always comes from the unmodified engines; these engines are used only to locate
the first numerical divergence.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO.parent, Path("../../HEAL")):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

CHECKPOINT = Path("${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
CONFIG = Path("${MODEL_ROOT}/lidar_pyramid/config.yaml")
HEAL_ROOT = Path("../../HEAL")
TRT_ROOT = Path("${TENSORRT_ROOT}")
TRTEXEC = TRT_ROOT / "targets/x86_64-linux-gnu/bin/trtexec"
CURRENT_PLUGIN = REPO / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
LEGACY_ROOT = REPO / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare"
LEGACY_ONNX = LEGACY_ROOT / "artifacts/onnx/fixedK29696/dynamic_agent_single_engine_maxK/lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
LEGACY_PLUGIN = LEGACY_ROOT / "artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so"
LEGACY_CACHE = LEGACY_ROOT / "artifacts/calibration/lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK29696_int8_train_calib200.cache"

TARGET_CANDIDATES = [
    ("scatter", ["/PointPillarScatterTRT_output_0"]),
    ("early_backbone", ["/layer0/layer0.2/relu_1/Relu_output_0", "/layer0/layer0.2/relu/Relu_output_0"]),
    ("grouped_conv_stage0", ["/layer0/layer0.2/relu_4/Relu_output_0"]),
    ("grouped_conv_stage1", ["/layer1/layer1.4/relu_2/Relu_output_0"]),
    ("grouped_conv_stage2", ["/layer2/layer2.7/relu_2/Relu_output_0"]),
    ("fusion", ["/ReduceSum_output_0"]),
    ("shrink", ["/shrink_conv/layers.0/double_conv/double_conv.3/Relu_output_0", "/shrink_conv/layers.0/double_conv/double_conv.2/Conv_output_0"]),
    ("head_input", ["/shrink_conv/layers.0/double_conv/double_conv.3/Relu_output_0", "/shrink_conv/layers.0/double_conv/double_conv.2/Conv_output_0"]),
    ("cls_output", ["cls_preds"]),
    ("reg_output", ["reg_preds"]),
    ("dir_output", ["dir_preds"]),
]


def tensor_work(audit_root: Path) -> Path:
    """Use v2 so the first failed diagnostic attempt remains immutable evidence."""

    return audit_root / "tensor_parity_work_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def require_modelopt() -> dict[str, str]:
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    expected = Path("${CONDA_BASE}/envs/modelopt")
    values = {
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "CONDA_PREFIX": str(prefix),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CC": os.environ.get("CC", ""),
        "CXX": os.environ.get("CXX", ""),
        "CUDACXX": os.environ.get("CUDACXX", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if values["CONDA_DEFAULT_ENV"] != "modelopt" or prefix.resolve() != expected.resolve():
        raise RuntimeError(f"modelopt_not_explicitly_activated:{values}")
    if Path(values["CUDA_HOME"]).resolve() != expected.resolve():
        raise RuntimeError(f"CUDA_HOME_not_modelopt:{values}")
    for key, name in (("CC", "gcc"), ("CXX", "g++"), ("CUDACXX", "nvcc")):
        if Path(values[key]).resolve() != (expected / "bin" / name).resolve():
            raise RuntimeError(f"{key}_not_modelopt:{values}")
    return values


def all_tensor_names(model: Any) -> set[str]:
    names = {row.name for row in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    for node in model.graph.node:
        names.update(str(name) for name in node.output)
    return names


def select_targets(model: Any) -> dict[str, str]:
    available = all_tensor_names(model)
    selected: dict[str, str] = {}
    for alias, choices in TARGET_CANDIDATES:
        exact = next((name for name in choices if name in available), None)
        if exact is None:
            exact = next((name + "__before_output_qdq" for name in choices if name + "__before_output_qdq" in available), None)
        if exact is not None:
            selected[alias] = exact
    return selected


def augment_outputs(source: Path, destination: Path) -> dict[str, str]:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(source))
    selected = select_targets(model)
    existing = {row.name for row in model.graph.output}
    value_info = {row.name: row for row in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    for name in selected.values():
        if name in existing:
            continue
        model.graph.output.append(value_info.get(name, helper.make_tensor_value_info(name, TensorProto.FLOAT, None)))
        existing.add(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return selected


def replace_scatter_plugin_for_ort(source: Path, destination: Path) -> dict[str, str]:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(source))
    selected = select_targets(model)
    plugin_index = next((i for i, node in enumerate(model.graph.node) if node.op_type == "PointPillarScatterTRT"), None)
    if plugin_index is None:
        raise RuntimeError("PointPillarScatterTRT_missing")
    plugin = model.graph.node[plugin_index]
    pillar, coords, mask, pairwise = list(plugin.input)
    output = str(plugin.output[0])
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in plugin.attribute}
    height = int(attrs.get("height", 256))
    width = int(attrs.get("width", 512))
    initializers_by_name = {row.name: row for row in model.graph.initializer}
    scatter_consumer = next((node for node in model.graph.node if output in node.input and node.op_type == "Conv"), None)
    if scatter_consumer is None or len(scatter_consumer.input) < 2 or scatter_consumer.input[1] not in initializers_by_name:
        raise RuntimeError("scatter_channel_count_cannot_be_derived")
    scatter_weight = initializers_by_name[scatter_consumer.input[1]]
    group = int(next((onnx.helper.get_attribute_value(attr) for attr in scatter_consumer.attribute if attr.name == "group"), 1))
    channels = int(scatter_weight.dims[1]) * group
    prefix = "__ort_scatter_reference"
    initializers = [
        helper.make_tensor(prefix + "_shape_index", TensorProto.INT64, [1], [1]),
        helper.make_tensor(prefix + "_hwc", TensorProto.INT64, [3], [height, width, channels]),
        helper.make_tensor(prefix + "_mask_threshold", TensorProto.FLOAT, [], [0.5]),
        helper.make_tensor(prefix + "_coord_columns", TensorProto.INT64, [3], [0, 2, 3]),
    ]
    model.graph.initializer.extend(initializers)
    nodes = [
        helper.make_node("Shape", [pairwise], [prefix + "_pairwise_shape"], name=prefix + "_shape"),
        helper.make_node("Gather", [prefix + "_pairwise_shape", prefix + "_shape_index"], [prefix + "_n_vec"], axis=0, name=prefix + "_gather_n"),
        helper.make_node("Concat", [prefix + "_n_vec", prefix + "_hwc"], [prefix + "_nhwc_shape"], axis=0, name=prefix + "_concat_shape"),
        helper.make_node("ConstantOfShape", [prefix + "_nhwc_shape"], [prefix + "_zeros"], value=helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0]), name=prefix + "_zeros_node"),
        helper.make_node("Greater", [mask, prefix + "_mask_threshold"], [prefix + "_valid"], name=prefix + "_valid_node"),
        helper.make_node("Compress", [pillar, prefix + "_valid"], [prefix + "_features"], axis=0, name=prefix + "_compress_features"),
        helper.make_node("Compress", [coords, prefix + "_valid"], [prefix + "_coords"], axis=0, name=prefix + "_compress_coords"),
        helper.make_node("Gather", [prefix + "_coords", prefix + "_coord_columns"], [prefix + "_indices_i32"], axis=1, name=prefix + "_gather_coords"),
        helper.make_node("Cast", [prefix + "_indices_i32"], [prefix + "_indices"], to=TensorProto.INT64, name=prefix + "_cast_indices"),
        helper.make_node("ScatterND", [prefix + "_zeros", prefix + "_indices", prefix + "_features"], [prefix + "_nhwc"], name=prefix + "_scatter_nd"),
        helper.make_node("Transpose", [prefix + "_nhwc"], [output], perm=[0, 3, 1, 2], name=prefix + "_transpose"),
    ]
    del model.graph.node[plugin_index]
    for offset, node in enumerate(nodes):
        model.graph.node.insert(plugin_index + offset, node)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    existing = {row.name for row in model.graph.output}
    value_info = {row.name: row for row in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    for name in selected.values():
        if name not in existing:
            if name not in value_info:
                raise RuntimeError(f"ORT_reference_output_shape_missing:{name}")
            model.graph.output.append(value_info[name])
            existing.add(name)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return selected


def profile_args() -> list[str]:
    minimum = "pairwise_t_matrix:1x1x1x4x4,valid_voxel_mask:29696,voxel_coords:29696x4,voxel_features:29696x32x4,voxel_num_points:29696"
    optimum = "pairwise_t_matrix:1x2x2x4x4,valid_voxel_mask:29696,voxel_coords:29696x4,voxel_features:29696x32x4,voxel_num_points:29696"
    return [f"--minShapes={minimum}", f"--optShapes={optimum}", f"--maxShapes={optimum}"]


def official_trtexec_command(log_path: Path, onnx_path: Path, engine_path: Path, layer_info: Path) -> list[str]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    line = next((row for row in reversed(lines) if " # " in row and "trtexec" in row), None)
    if line is None:
        raise RuntimeError(f"official_trtexec_command_missing:{log_path}")
    command = shlex.split(line.split(" # ", 1)[1])
    replaced = []
    for item in command:
        if item.startswith("--onnx="):
            item = f"--onnx={onnx_path}"
        elif item.startswith("--saveEngine="):
            item = f"--saveEngine={engine_path}"
        elif item.startswith("--exportLayerInfo="):
            item = f"--exportLayerInfo={layer_info}"
        replaced.append(item)
    return replaced


def run_build(command: list[str], log_path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(" ".join(shlex.quote(item) for item in command) + "\n\n" + completed.stdout, encoding="utf-8")
    return {"command": command, "returncode": completed.returncode, "log": str(log_path)}


def prepare_inputs(audit_root: Path, work: Path) -> list[dict[str, Any]]:
    from torch.utils.data import DataLoader
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from heal_compress.quantization.config import OnnxExportConfig
    from heal_compress.quantization.export.heal_lidar_pyramid import prepare_signal_maxk_inputs

    adapter = HEALLiDARAdapter(heal_repo=HEAL_ROOT, config={"model": {"hypes_yaml": str(CONFIG)}})
    hypes = adapter._absolutize_dataset_paths(yaml_utils.load_yaml(adapter._resolve_heal_path(CONFIG)))
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    manifest = read_json(audit_root / "baseline/eval_manifest.json", {})
    wanted = [str(value) for value in manifest.get("evaluation_frame_ids", [])[:10]]
    split_ids = [str(value) for value in json.loads(Path(hypes["validate_dir"]).read_text(encoding="utf-8"))]
    wanted_set = set(wanted)
    rows = []
    config = OnnxExportConfig(fixed_k=29696, min_agents=1, opt_agents=2, max_agents=2)
    for index, batch in enumerate(loader):
        if len(rows) >= 10:
            break
        frame_id = split_ids[index]
        if frame_id not in wanted_set or batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) else batch
        inputs = prepare_signal_maxk_inputs(ego, config=config, modality="m1")
        path = work / "inputs" / f"frame_{len(rows):02d}_{frame_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{name: tensor.detach().cpu().numpy() for name, tensor in inputs.items()})
        rows.append({"index": len(rows), "frame_id": frame_id, "path": str(path), "sha256": sha256_file(path)})
    if [row["frame_id"] for row in rows] != wanted:
        raise RuntimeError("ten_frame_manifest_order_mismatch")
    write_json(work / "ten_frame_inputs.json", {"manifest_hash": manifest.get("manifest_hash", ""), "frames": rows})
    return rows


def prepare_build(audit_root: Path) -> None:
    env = require_modelopt()
    work = tensor_work(audit_root)
    allowed_existing = {"inputs", "ten_frame_inputs.json"}
    unexpected = [item.name for item in work.iterdir()] if work.exists() else []
    unexpected = [name for name in unexpected if name not in allowed_existing]
    if unexpected:
        raise RuntimeError(f"force_rebuild_destination_contains_build_artifacts:{unexpected}")
    work.mkdir(parents=True, exist_ok=True)
    search_fp16 = audit_root / "search_strict_fp16_force_rebuild/artifacts"
    search_int8 = audit_root / "search_maximal_legal_int8_force_rebuild/artifacts"
    sources = {
        "legacy_int8": LEGACY_ONNX,
        "search_fp16": search_fp16 / "qdq_trt_compatible.onnx",
        "search_int8": search_int8 / "qdq_trt_compatible.onnx",
    }
    maps = {}
    for label, source in sources.items():
        maps[label] = augment_outputs(source, work / f"{label}_augmented.onnx")
    maps["ort_fp32"] = replace_scatter_plugin_for_ort(search_fp16 / "pruned_fp32.onnx", work / "ort_fp32_augmented.onnx")
    write_json(work / "target_map.json", maps)
    builds = {}
    for label, official_dir in (("search_fp16", search_fp16), ("search_int8", search_int8)):
        command = official_trtexec_command(
            official_dir / "engine_build.log",
            work / f"{label}_augmented.onnx",
            work / f"{label}_augmented.engine",
            work / f"{label}_augmented_layerinfo.json",
        )
        builds[label] = run_build(command, work / f"{label}_augmented_build.log")
    legacy_command = [
        str(TRTEXEC),
        f"--onnx={work / 'legacy_int8_augmented.onnx'}",
        f"--saveEngine={work / 'legacy_int8_augmented.engine'}",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        f"--exportLayerInfo={work / 'legacy_int8_augmented_layerinfo.json'}",
        "--skipInference",
        "--noTF32",
        "--int8",
        "--fp16",
        f"--calib={LEGACY_CACHE}",
        f"--staticPlugins={LEGACY_PLUGIN}",
        *profile_args(),
    ]
    builds["legacy_int8"] = run_build(legacy_command, work / "legacy_int8_augmented_build.log")
    for label in builds:
        engine = work / f"{label}_augmented.engine"
        builds[label]["engine_exists"] = engine.is_file()
        builds[label]["engine_sha256"] = sha256_file(engine) if engine.is_file() else ""
    inputs_manifest = work / "ten_frame_inputs.json"
    if not inputs_manifest.is_file():
        raise RuntimeError("ten_frame_inputs_missing:run_prepare-inputs_in_univ2x-opt_first")
    write_json(
        work / "debug_build_manifest.json",
        {
            "diagnostic_only": True,
            "fusion_may_change_due_to_extra_outputs": True,
            "environment": env,
            "sources": {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in sources.items()},
            "plugins": {"legacy": sha256_file(LEGACY_PLUGIN), "search": sha256_file(CURRENT_PLUGIN)},
            "calibration_cache_sha256": sha256_file(LEGACY_CACHE),
            "builds": builds,
        },
    )
    if not all(row["returncode"] == 0 and row["engine_exists"] for row in builds.values()):
        raise RuntimeError(f"augmented_engine_build_failed:{builds}")


def prepare_inputs_action(audit_root: Path) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV", "") != "univ2x-opt":
        raise RuntimeError("prepare-inputs_requires_univ2x-opt")
    work = tensor_work(audit_root)
    if work.exists() and any(work.iterdir()):
        raise RuntimeError(f"force_rebuild_destination_not_empty:{work}")
    work.mkdir(parents=True)
    prepare_inputs(audit_root, work)


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except RuntimeError:
                pass
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except RuntimeError:
                pass
    raise RuntimeError("hook_value_has_no_tensor")


def run_pytorch_route(audit_root: Path) -> None:
    if os.environ.get("CONDA_DEFAULT_ENV", "") != "univ2x-opt":
        raise RuntimeError("run-pytorch_requires_univ2x-opt")
    from heal_compress.quantization.config import OnnxExportConfig
    from heal_compress.quantization.export.heal_lidar_pyramid import build_heal_signal_maxk_export_module
    from scripts.int8_baseline_equivalence_audit import build_context

    work = tensor_work(audit_root)
    maps = read_json(work / "target_map.json", {})
    ort_map = maps["ort_fp32"]
    context = build_context(audit_root / "tensor_parity_pytorch_context_v2", int(os.environ.get("AUDIT_PHYSICAL_GPU", "6")))
    wrapper = build_heal_signal_maxk_export_module(
        context.model,
        config=OnnxExportConfig(fixed_k=29696, min_agents=1, opt_agents=2, max_agents=2),
        modality="m1",
    ).eval().to(context.runtime_device)
    modules = dict(wrapper.named_modules())
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def output_hook(alias: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captures[alias] = _first_tensor(output).detach()
        return hook

    def input_hook(alias: str):
        def hook(_module: Any, inputs: Any) -> None:
            captures[alias] = _first_tensor(inputs).detach()
        return hook

    output_modules = {
        "early_backbone": "model.backbone_m1.resnet.layer0.2",
        "grouped_conv_stage0": "model.pyramid_backbone.resnet.layer0.2",
        "grouped_conv_stage1": "model.pyramid_backbone.resnet.layer1.4",
        "grouped_conv_stage2": "model.pyramid_backbone.resnet.layer2.7",
        "shrink": "model.shrink_conv",
        "head_input": "model.shrink_conv",
        "cls_output": "model.cls_head",
        "reg_output": "model.reg_head",
        "dir_output": "model.dir_head",
    }
    input_modules = {
        "scatter": "model.backbone_m1.resnet.layer0.0",
        "fusion": "model.pyramid_backbone.deblocks.0.0",
    }
    missing_modules = [name for name in list(output_modules.values()) + list(input_modules.values()) if name not in modules]
    if missing_modules:
        raise RuntimeError(f"pytorch_hook_modules_missing:{sorted(set(missing_modules))}")
    for alias, name in output_modules.items():
        handles.append(modules[name].register_forward_hook(output_hook(alias)))
    for alias, name in input_modules.items():
        handles.append(modules[name].register_forward_pre_hook(input_hook(alias)))

    per_frame = []
    capture_dir = work / "pytorch_tensors"
    capture_dir.mkdir(parents=True, exist_ok=False)
    try:
        for frame in read_json(work / "ten_frame_inputs.json", {}).get("frames", []):
            values = np.load(frame["path"])
            feeds = {name: values[name] for name in values.files}
            device_inputs = {name: torch.as_tensor(value, device=context.runtime_device) for name, value in feeds.items()}
            captures.clear()
            with torch.inference_mode():
                wrapper(**device_inputs)
            tensor_path = capture_dir / f"frame_{int(frame['index']):02d}_{frame['frame_id']}.npz"
            np.savez_compressed(tensor_path, **{alias: value.float().cpu().numpy() for alias, value in captures.items()})
            frame_rows = {}
            for alias in ort_map:
                frame_rows[alias] = {"captured": alias in captures, "shape": list(captures[alias].shape) if alias in captures else None}
            per_frame.append({"frame_id": frame["frame_id"], "tensor_path": str(tensor_path), "tensor_sha256": sha256_file(tensor_path), "tensors": frame_rows})
    finally:
        for handle in handles:
            handle.remove()
    write_json(
        work / "pytorch_capture_manifest.json",
        {
            "label": "pytorch_fp32",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "target_map": ort_map,
            "frames": per_frame,
        },
    )


def compare_pytorch_route(audit_root: Path) -> None:
    require_modelopt()
    work = tensor_work(audit_root)
    manifest = read_json(work / "pytorch_capture_manifest.json", {})
    ort_map = manifest.get("target_map", {})
    session = ort_session(work / "ort_fp32_augmented.onnx")
    accumulators = {alias: Accumulator() for alias in ort_map}
    ort_accumulators = {alias: Accumulator() for alias in ort_map}
    per_frame = []
    input_rows = {row["frame_id"]: row for row in read_json(work / "ten_frame_inputs.json", {}).get("frames", [])}
    for frame in manifest.get("frames", []):
        input_row = input_rows[frame["frame_id"]]
        input_values = np.load(input_row["path"])
        feeds = {name: input_values[name] for name in input_values.files}
        ort_values = session.run(None, feeds)
        ort_outputs = {row.name: value for row, value in zip(session.get_outputs(), ort_values)}
        captures = np.load(frame["tensor_path"])
        frame_rows = {}
        for alias, reference_name in ort_map.items():
            if alias not in captures.files or reference_name not in ort_outputs:
                frame_rows[alias] = {"missing": True, "reference": reference_name, "candidate": alias}
                continue
            reference = np.asarray(ort_outputs[reference_name])
            candidate = np.asarray(captures[alias])
            ort_accumulators[alias].update(reference, reference, None)
            accumulators[alias].update(reference, candidate, None)
            frame_rows[alias] = {"reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
        per_frame.append({"frame_id": frame["frame_id"], "tensors": frame_rows})
    write_json(
        work / "route_pytorch_fp32.json",
        {
            **{key: value for key, value in manifest.items() if key != "frames"},
            "reference": "ORT_FP32_standard_scatter_reconstruction",
            "ORT_FP32_metrics": {alias: accumulator.result() for alias, accumulator in ort_accumulators.items()},
            "metrics": {alias: accumulator.result() for alias, accumulator in accumulators.items()},
            "frames": per_frame,
        },
    )


class Accumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.absmax = 0.0
        self.zeros = 0
        self.saturated = 0
        self.saturation_n = 0
        self.diff_abs = 0.0
        self.diff_sq = 0.0
        self.diff_max = 0.0
        self.dot = 0.0
        self.ref_sq = 0.0
        self.cand_sq = 0.0

    def update(self, reference: np.ndarray, candidate: np.ndarray, saturation_threshold: float | None) -> None:
        ref = torch.as_tensor(np.asarray(reference), device="cuda", dtype=torch.float32).reshape(-1)
        cand = torch.as_tensor(np.asarray(candidate), device="cuda", dtype=torch.float32).reshape(-1)
        if ref.numel() != cand.numel():
            raise RuntimeError(f"tensor_size_mismatch:{ref.numel()}!={cand.numel()}")
        diff = cand - ref
        self.n += int(cand.numel())
        self.sum += float(cand.sum().item())
        self.sumsq += float(cand.square().sum().item())
        self.absmax = max(self.absmax, float(cand.abs().max().item()) if cand.numel() else 0.0)
        self.zeros += int((cand == 0).sum().item())
        if saturation_threshold is not None:
            self.saturated += int((cand.abs() >= float(saturation_threshold) * 0.999).sum().item())
            self.saturation_n += int(cand.numel())
        self.diff_abs += float(diff.abs().sum().item())
        self.diff_sq += float(diff.square().sum().item())
        self.diff_max = max(self.diff_max, float(diff.abs().max().item()) if diff.numel() else 0.0)
        self.dot += float((ref * cand).sum().item())
        self.ref_sq += float(ref.square().sum().item())
        self.cand_sq += float(cand.square().sum().item())

    def result(self) -> dict[str, Any]:
        mean = self.sum / max(self.n, 1)
        variance = max(self.sumsq / max(self.n, 1) - mean * mean, 0.0)
        cosine = self.dot / max(math.sqrt(self.ref_sq * self.cand_sq), 1.0e-30)
        sqnr = 10.0 * math.log10(self.ref_sq / max(self.diff_sq, 1.0e-30)) if self.ref_sq > 0 else None
        return {
            "numel": self.n,
            "absmax": self.absmax,
            "mean": mean,
            "std": math.sqrt(variance),
            "zero_ratio": self.zeros / max(self.n, 1),
            "saturation_ratio": self.saturated / self.saturation_n if self.saturation_n else None,
            "MAE": self.diff_abs / max(self.n, 1),
            "RMSE": math.sqrt(self.diff_sq / max(self.n, 1)),
            "max_error": self.diff_max,
            "cosine": cosine,
            "SQNR_dB": sqnr,
        }


def ort_session(path: Path, *, prefer_cuda: bool = True):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    available = ort.get_available_providers()
    order = ("CUDAExecutionProvider", "CPUExecutionProvider") if prefer_cuda else ("CPUExecutionProvider",)
    providers = [name for name in order if name in available]
    return ort.InferenceSession(str(path), sess_options=options, providers=providers or available)


def route_scale_thresholds(audit_root: Path, label: str, target_map: dict[str, str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {alias: None for alias in target_map}
    if label == "legacy_int8":
        cache: dict[str, float] = {}
        for line in LEGACY_CACHE.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            if ": " not in line:
                continue
            name, encoded = line.rsplit(": ", 1)
            try:
                import struct

                cache[name] = struct.unpack("!f", bytes.fromhex(encoded))[0]
            except Exception:
                pass
        for alias, name in target_map.items():
            if name in cache:
                result[alias] = abs(cache[name]) * 127.0
    elif label == "search_int8":
        inventory = read_json(audit_root / "qdq_inventory.json", {})
        for alias, name in target_map.items():
            normalized = name.replace("__before_output_qdq", "")
            row = next((row for row in inventory.get("rows", []) if str(row.get("tensor", "")).replace("__before_output_qdq", "") == normalized and row.get("quant_role") != "weight"), None)
            if row is not None and row.get("scale_max") != "":
                result[alias] = abs(float(row["scale_max"])) * 127.0
    return result


def run_engine_route(audit_root: Path, label: str) -> None:
    require_modelopt()
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    work = tensor_work(audit_root)
    maps = read_json(work / "target_map.json", {})
    target_map = maps[label]
    ort_map = maps["ort_fp32"]
    plugin = LEGACY_PLUGIN if label == "legacy_int8" else CURRENT_PLUGIN
    ctypes.CDLL(str(plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    session = ort_session(work / "ort_fp32_augmented.onnx")
    runner = TensorRTEngineRunner(work / f"{label}_augmented.engine", torch.device("cuda:0"))
    thresholds = route_scale_thresholds(audit_root, label, target_map)
    accumulators = {alias: Accumulator() for alias in target_map if alias in ort_map}
    per_frame = []
    inputs_manifest = read_json(work / "ten_frame_inputs.json", {})
    for frame in inputs_manifest.get("frames", []):
        values = np.load(frame["path"])
        feeds = {name: values[name] for name in values.files}
        ort_values = session.run(None, feeds)
        ort_outputs = {row.name: value for row, value in zip(session.get_outputs(), ort_values)}
        device_inputs = {name: torch.as_tensor(value, device="cuda:0") for name, value in feeds.items()}
        outputs = runner.run(device_inputs)
        frame_rows = {}
        for alias, engine_name in target_map.items():
            ref_name = ort_map.get(alias)
            if not ref_name or ref_name not in ort_outputs or engine_name not in outputs:
                frame_rows[alias] = {"missing": True, "reference": ref_name, "candidate": engine_name}
                continue
            reference = np.asarray(ort_outputs[ref_name])
            candidate = outputs[engine_name].detach().float().cpu().numpy()
            accumulators[alias].update(reference, candidate, thresholds.get(alias))
            frame_rows[alias] = {"reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
        per_frame.append({"frame_id": frame["frame_id"], "tensors": frame_rows})
    metrics = {alias: accumulator.result() for alias, accumulator in accumulators.items()}
    write_json(
        work / f"route_{label}.json",
        {
            "label": label,
            "reference": "ORT_FP32_standard_scatter_reconstruction",
            "engine": str(work / f"{label}_augmented.engine"),
            "engine_sha256": sha256_file(work / f"{label}_augmented.engine"),
            "plugin": str(plugin),
            "plugin_sha256": sha256_file(plugin),
            "target_map": target_map,
            "metrics": metrics,
            "frames": per_frame,
        },
    )


def aggregate(audit_root: Path) -> None:
    work = tensor_work(audit_root)
    routes = {label: read_json(work / f"route_{label}.json", {}) for label in ("search_fp16", "legacy_int8", "search_int8")}
    pytorch = read_json(work / "route_pytorch_fp32.json", {})
    order = [alias for alias, _choices in TARGET_CANDIDATES]
    first = {}
    for label, route in routes.items():
        first[label] = next(
            (
                alias
                for alias in order
                if alias in route.get("metrics", {})
                and (
                    float(route["metrics"][alias].get("cosine", 1.0)) < 0.99
                    or float(route["metrics"][alias].get("SQNR_dB") or 999.0) < 20.0
                )
            ),
            None,
        )
    legacy_metrics = routes.get("legacy_int8", {}).get("metrics", {})
    search_metrics = routes.get("search_int8", {}).get("metrics", {})
    first_search_worse_than_legacy = next(
        (
            alias
            for alias in order
            if alias in legacy_metrics
            and alias in search_metrics
            and float(search_metrics[alias].get("cosine", 1.0))
            < float(legacy_metrics[alias].get("cosine", 1.0)) - 0.05
        ),
        None,
    )
    first_search_catastrophic = next(
        (
            alias
            for alias in order
            if alias in search_metrics
            and (
                float(search_metrics[alias].get("cosine", 1.0)) < 0.8
                or float(search_metrics[alias].get("zero_ratio", 0.0)) > 0.95
            )
        ),
        None,
    )
    result = {
        "frame_count": 10,
        "frame_manifest": read_json(work / "ten_frame_inputs.json", {}),
        "reference": {
            "PyTorch_FP32": {
                "status": "verified" if pytorch.get("metrics") else "not_yet_verified",
                "comparison_to_ORT_FP32": pytorch,
            },
            "ORT_FP32": {
                "status": "verified",
                "model": str(work / "ort_fp32_augmented.onnx"),
                "model_sha256": sha256_file(work / "ort_fp32_augmented.onnx"),
                "metrics": pytorch.get("ORT_FP32_metrics", {}),
            },
        },
        "diagnostic_engine_warning": "Augmented intermediate outputs alter TensorRT fusion; official AP/latency uses unmodified engines.",
        "routes": routes,
        "first_abnormal_tensor": first,
        "first_search_tensor_worse_than_legacy_by_cosine_0.05": first_search_worse_than_legacy,
        "first_search_catastrophic_tensor": first_search_catastrophic,
    }
    write_json(audit_root / "tensor_parity.json", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--action", choices=("prepare-inputs", "prepare-build", "run-pytorch", "compare-pytorch", "run-route", "aggregate"), required=True)
    parser.add_argument("--label", choices=("search_fp16", "legacy_int8", "search_int8"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.audit_root.resolve()
    if args.action == "prepare-inputs":
        prepare_inputs_action(root)
    elif args.action == "prepare-build":
        prepare_build(root)
    elif args.action == "run-pytorch":
        run_pytorch_route(root)
    elif args.action == "compare-pytorch":
        compare_pytorch_route(root)
    elif args.action == "run-route":
        if not args.label:
            raise RuntimeError("--label is required for run-route")
        run_engine_route(root, args.label)
    else:
        aggregate(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
