from __future__ import annotations

import json
import hashlib
import re
import shutil
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch


REQUIRED_HEAL_INPUTS = {
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
}
ADAPTER_AVAILABLE_INPUTS = {
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "record_len",
    "pairwise_t_matrix",
    "valid_voxel_mask",
}
EXPECTED_HEAL_OUTPUTS = {"cls_preds", "reg_preds", "dir_preds"}
ONNX_ORIGIN_MAP_UNIQUE_NAME_POLICY_VERSION = "canonical_v2_trt_safe_max68"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def escape_canonical_module_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip(".")).strip("_") or "module"


def _shape_signature(value: Any) -> Any:
    if torch.is_tensor(value):
        return [int(dim) if isinstance(dim, int) or str(dim).isdigit() else str(dim) for dim in value.shape]
    if isinstance(value, (list, tuple)):
        return [_shape_signature(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _shape_signature(item) for key, item in value.items()}
    return str(type(value).__name__)


def _module_mapped_onnx_op_type(module: torch.nn.Module) -> str:
    if isinstance(module, torch.nn.ConvTranspose2d):
        return "ConvTranspose"
    if isinstance(module, torch.nn.Conv2d):
        return "Conv"
    if isinstance(module, torch.nn.Linear):
        return "MatMul"
    return ""


def _module_trace_record(
    *,
    canonical_module_name: str,
    module_call_index: int,
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: Any,
) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    return {
        "canonical_module_name": canonical_module_name,
        "module_call_index": int(module_call_index),
        "module_type": type(module).__name__,
        "mapped_onnx_op_type": _module_mapped_onnx_op_type(module),
        "weight_shape": list(weight.shape) if torch.is_tensor(weight) else [],
        "bias_shape": list(bias.shape) if torch.is_tensor(bias) else [],
        "groups": int(getattr(module, "groups", 1) or 1),
        "kernel_size": list(getattr(module, "kernel_size", []) or []),
        "stride": list(getattr(module, "stride", []) or []),
        "padding": list(getattr(module, "padding", []) or []),
        "dilation": list(getattr(module, "dilation", []) or []),
        "input_shape_signature": _shape_signature(inputs),
        "output_shape_signature": _shape_signature(output),
    }


def _canonical_name_from_wrapper_module_name(name: str) -> str:
    name = str(name).strip(".")
    if name.startswith("model."):
        name = name[len("model.") :]
    return name


@contextmanager
def capture_weighted_module_call_trace(model: torch.nn.Module):
    trace: list[dict[str, Any]] = []
    handles = []
    counter = {"value": 0}

    def hook(canonical_module_name: str, module: torch.nn.Module):
        def _inner(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            trace.append(
                _module_trace_record(
                    canonical_module_name=canonical_module_name,
                    module_call_index=counter["value"],
                    module=module,
                    inputs=inputs,
                    output=output,
                )
            )
            counter["value"] += 1

        return _inner

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            canonical = _canonical_name_from_wrapper_module_name(name)
            if canonical:
                handles.append(module.register_forward_hook(hook(canonical, module)))
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def _patch_torch_onnx_export_for_origin_trace(trace_sink: list[dict[str, Any]]):
    original_export = torch.onnx.export

    def _wrapped_export(model: Any, args: Any = (), f: Any = None, *pos: Any, **kwargs: Any):
        if isinstance(model, torch.nn.Module):
            with capture_weighted_module_call_trace(model) as trace:
                result = original_export(model, args, f, *pos, **kwargs)
            trace_sink.clear()
            trace_sink.extend(trace)
            return result
        return original_export(model, args, f, *pos, **kwargs)

    return original_export, _wrapped_export


def _onnx_attribute_ints(node: Any, name: str, default: Iterable[int] | None = None) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            if attr.ints:
                return [int(value) for value in attr.ints]
            if attr.i:
                return [int(attr.i)]
    return [int(value) for value in (default or [])]


def _onnx_attribute_int(node: Any, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return int(default)


def _initializer_shapes(model: Any) -> dict[str, list[int]]:
    return {initializer.name: [int(dim) for dim in initializer.dims] for initializer in model.graph.initializer}


def _node_spec(node: Any, initializer_shapes: Mapping[str, list[int]], graph_index: int) -> dict[str, Any]:
    weight_name = str(node.input[1]) if len(node.input) > 1 else ""
    bias_name = str(node.input[2]) if len(node.input) > 2 else ""
    kernel = _onnx_attribute_ints(node, "kernel_shape", [])
    strides = _onnx_attribute_ints(node, "strides", [1, 1])
    pads = _onnx_attribute_ints(node, "pads", [0, 0, 0, 0])
    dilations = _onnx_attribute_ints(node, "dilations", [1, 1])
    return {
        "onnx_node_name_original": str(node.name),
        "onnx_op_type": str(node.op_type),
        "graph_index": int(graph_index),
        "onnx_weight_initializer": weight_name,
        "onnx_bias_initializer": bias_name,
        "weight_shape": list(initializer_shapes.get(weight_name, [])),
        "bias_shape": list(initializer_shapes.get(bias_name, [])),
        "groups": _onnx_attribute_int(node, "group", 1),
        "kernel_size": kernel,
        "stride": strides,
        "padding": pads,
        "dilation": dilations,
        "inputs": list(node.input),
        "outputs": list(node.output),
    }


def _trace_matches_node(trace: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    trace_op = str(trace.get("mapped_onnx_op_type", ""))
    node_op = str(node.get("onnx_op_type", ""))
    if trace_op == "MatMul":
        if node_op not in {"MatMul", "Gemm"}:
            return False
    elif trace_op != node_op:
        return False
    trace_weight = [int(value) for value in trace.get("weight_shape", [])]
    node_weight = [int(value) for value in node.get("weight_shape", [])]
    if trace_weight and node_weight:
        if trace_op == "MatMul":
            if node_weight not in (trace_weight, list(reversed(trace_weight))):
                return False
        elif node_weight != trace_weight:
            return False
    if trace.get("bias_shape") and node.get("bias_shape") and list(trace.get("bias_shape") or []) != list(node.get("bias_shape") or []):
        return False
    if trace_op in {"Conv", "ConvTranspose"}:
        if int(trace.get("groups", 1) or 1) != int(node.get("groups", 1) or 1):
            return False
        for trace_key, node_key in (("kernel_size", "kernel_size"), ("stride", "stride"), ("dilation", "dilation")):
            trace_value = [int(value) for value in trace.get(trace_key, [])]
            node_value = [int(value) for value in node.get(node_key, [])]
            if trace_value and node_value and trace_value != node_value:
                return False
        trace_padding = [int(value) for value in trace.get("padding", [])]
        node_padding = [int(value) for value in node.get("padding", [])]
        if trace_padding and node_padding:
            if len(node_padding) == 2:
                node_padding = node_padding + node_padding
            if len(trace_padding) == 2:
                trace_padding = trace_padding + trace_padding
            if trace_padding != node_padding:
                return False
    return True


def _unique_node_name(canonical_module_name: str, op_type: str, module_call_index: int) -> str:
    escaped = escape_canonical_module_name(canonical_module_name)
    full = f"__canonical__{escaped}__{op_type}__call{int(module_call_index):05d}"
    if len(full) <= 68:
        return full
    digest = hashlib.sha1(str(canonical_module_name).encode("utf-8")).hexdigest()[:10]
    suffix = f"__{op_type}__call{int(module_call_index):05d}"
    max_escaped = max(8, 68 - len("__canonical__") - len(suffix) - len(digest) - 1)
    return f"__canonical__{escaped[:max_escaped]}_{digest}{suffix}"


def build_onnx_export_origin_map(onnx_path: str | Path, call_trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    import onnx

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    init_shapes = _initializer_shapes(model)
    weighted_ops = {"Conv", "Gemm", "MatMul", "ConvTranspose"}
    nodes = [
        _node_spec(node, init_shapes, idx)
        for idx, node in enumerate(model.graph.node)
        if node.op_type in weighted_ops and len(node.input) > 1 and str(node.input[1]) in init_shapes
    ]
    traces = [
        dict(row)
        for row in call_trace
        if str(row.get("mapped_onnx_op_type", "")) in {"Conv", "MatMul", "ConvTranspose"} and row.get("weight_shape")
    ]
    _write_json(onnx_path.parent / "onnx_export_module_call_trace.json", traces)
    failures: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    used_nodes: set[int] = set()
    trace_indices = list(range(len(traces)))
    while trace_indices:
        trace_idx = trace_indices[0]
        trace = traces[trace_idx]
        candidate_node_indices = [
            node_idx
            for node_idx, node in enumerate(nodes)
            if node_idx not in used_nodes and _trace_matches_node(trace, node)
        ]
        if not candidate_node_indices:
            failures.append(
                {
                    "failure_reason": "no_onnx_node_matches_module_call",
                    "trace_index": trace_idx,
                    "trace": trace,
                }
            )
            trace_indices.pop(0)
            continue
        similar_trace_indices = [
            idx
            for idx in trace_indices
            if set(candidate_node_indices)
            == {node_idx for node_idx, node in enumerate(nodes) if node_idx not in used_nodes and _trace_matches_node(traces[idx], node)}
        ]
        if len(candidate_node_indices) < len(similar_trace_indices):
            failures.append(
                {
                    "failure_reason": "not_enough_onnx_nodes_for_equivalent_module_calls",
                    "trace_indices": similar_trace_indices,
                    "candidate_node_indices": candidate_node_indices,
                }
            )
            for idx in similar_trace_indices:
                trace_indices.remove(idx)
            continue
        ordered_traces = sorted(similar_trace_indices, key=lambda idx: int(traces[idx].get("module_call_index", idx)))
        ordered_nodes = sorted(candidate_node_indices, key=lambda idx: int(nodes[idx].get("graph_index", idx)))[: len(ordered_traces)]
        for idx, node_idx in zip(ordered_traces, ordered_nodes):
            trace_row = traces[idx]
            node_row = nodes[node_idx]
            unique_name = _unique_node_name(
                str(trace_row.get("canonical_module_name", "")),
                str(node_row.get("onnx_op_type", trace_row.get("mapped_onnx_op_type", ""))),
                int(trace_row.get("module_call_index", idx)),
            )
            entries.append(
                {
                    **trace_row,
                    **node_row,
                    "onnx_node_name_unique": unique_name,
                    "trt_metadata_match_key": unique_name,
                }
            )
            used_nodes.add(node_idx)
        for idx in ordered_traces:
            trace_indices.remove(idx)
    unmapped_nodes = [node for idx, node in enumerate(nodes) if idx not in used_nodes]
    if failures:
        report = {
            "success": False,
            "onnx_path": str(onnx_path),
            "failure_reason": "onnx_export_origin_map_not_unique",
            "failures": failures,
            "mapped_entry_count": len(entries),
            "weighted_onnx_node_count": len(nodes),
            "module_call_trace_count": len(traces),
            "unmapped_weighted_onnx_nodes": unmapped_nodes[:50],
        }
        _write_json(onnx_path.parent / "onnx_export_origin_map_failure_report.json", report)
        raise ValueError(f"onnx_export_origin_map_not_unique:{len(failures)}")
    origin_map = {
        "success": True,
        "onnx_path": str(onnx_path),
        "unique_name_policy_version": ONNX_ORIGIN_MAP_UNIQUE_NAME_POLICY_VERSION,
        "entries": entries,
        "entry_count": len(entries),
        "module_call_trace_count": len(traces),
        "weighted_onnx_node_count": len(nodes),
        "unmapped_weighted_onnx_nodes": unmapped_nodes,
    }
    _write_json(onnx_path.parent / "onnx_export_origin_map.json", origin_map)
    return origin_map


def rename_onnx_compute_nodes_with_origin_map(onnx_path: str | Path, origin_map: Mapping[str, Any], output_path: str | Path | None = None) -> dict[str, Any]:
    import onnx

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path is not None else onnx_path
    model = onnx.load(str(onnx_path))
    by_original: dict[str, list[dict[str, Any]]] = {}
    for row in origin_map.get("entries", []):
        if isinstance(row, Mapping):
            by_original.setdefault(str(row.get("onnx_node_name_original", "")), []).append(dict(row))
    renamed = []
    for node in model.graph.node:
        rows = by_original.get(str(node.name), [])
        if not rows:
            continue
        if len(rows) > 1:
            rows = [row for row in rows if str(row.get("onnx_op_type", "")) == str(node.op_type)]
        if len(rows) != 1:
            raise ValueError(f"onnx_node_rename_origin_not_unique:{node.name}")
        original = str(node.name)
        unique = str(rows[0].get("onnx_node_name_unique", ""))
        if not unique:
            raise ValueError(f"onnx_node_rename_missing_unique_name:{original}")
        node.name = unique
        renamed.append({"onnx_node_name_original": original, "onnx_node_name_unique": unique, "op_type": str(node.op_type)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    updated_map = dict(origin_map)
    updated_map["onnx_path"] = str(output_path)
    updated_map["renamed_model_path"] = str(output_path)
    _write_json(output_path.parent / "onnx_export_origin_map.json", updated_map)
    report = {
        "success": True,
        "onnx_path": str(output_path),
        "renamed_node_count": len(renamed),
        "renamed_nodes": renamed,
    }
    _write_json(output_path.parent / "onnx_node_rename_report.json", report)
    return report


def inspect_signal_maxk_onnx(path: Path, *, validate_onnx: bool = True) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    checker_passed = None
    checker_error = ""
    if validate_onnx:
        try:
            onnx.checker.check_model(model)
            checker_passed = True
        except Exception as exc:  # noqa: BLE001
            checker_passed = False
            checker_error = f"{type(exc).__name__}: {exc}"
    return {
        "onnx_path": str(path),
        "input_names": input_names,
        "output_names": output_names,
        "has_required_heal_inputs": REQUIRED_HEAL_INPUTS <= set(input_names),
        "has_toy_input_only": input_names == ["input.1"] or input_names == ["input"],
        "has_expected_outputs": bool(EXPECTED_HEAL_OUTPUTS & set(output_names)),
        "onnx_checker_passed": checker_passed,
        "onnx_checker_error": checker_error,
    }


def _run_formal_signal_maxk_export(
    *,
    pruned_model_path: Path | None = None,
    output_onnx_path: Path,
    model_config: Path,
    checkpoint: Path,
    heal_root: Path,
    fixed_k: int,
    dynamic_axes: bool,
    trt_root: Path | None = None,
) -> dict[str, Any]:
    from quantization.export.export_single_engine_maxk_onnx import export_single_engine_maxk_onnx

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_onnx_path.parent / "_formal_signal_maxk_export"
    args = SimpleNamespace(
        config=str(model_config),
        checkpoint=str(checkpoint),
        fixed_k=int(fixed_k),
        precision="fp16",
        output_dir=str(output_dir),
        heal_repo=str(heal_root),
        device="cuda:0",
        max_cav=2,
        opset=17,
        trt_root=str(trt_root) if trt_root else "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118",
        trtexec_path=None,
        export_sample_split="train",
        max_scan_samples=128,
        overwrite=True,
        dynamic_axes=bool(dynamic_axes),
    )
    module_call_trace: list[dict[str, Any]] = []
    original_export, wrapped_export = _patch_torch_onnx_export_for_origin_trace(module_call_trace)
    torch.onnx.export = wrapped_export
    try:
        report = _export_with_pruned_model_context(args, pruned_model_path=pruned_model_path, heal_root=heal_root) if pruned_model_path else export_single_engine_maxk_onnx(args)
    finally:
        torch.onnx.export = original_export
    formal_path = Path(str(report.get("formal_onnx_path") or ""))
    if not formal_path.is_file():
        legacy_path = Path(str(report.get("onnx_path") or ""))
        formal_path = legacy_path if legacy_path.is_file() else formal_path
    if formal_path.is_file() and formal_path.resolve() != output_onnx_path.resolve():
        output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(formal_path, output_onnx_path)
    return {
        "success": bool(report.get("success")) and output_onnx_path.is_file(),
        "source_exporter": "quantization.export.export_single_engine_maxk_onnx",
        "formal_report": report,
        "onnx_export_module_call_trace": module_call_trace,
        "onnx_export_module_call_trace_count": len(module_call_trace),
    }


def _load_pruned_model_object(path: Path, *, heal_root: Path, device: torch.device) -> torch.nn.Module:
    root = str(Path(heal_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model = payload.get("model_object") if isinstance(payload, dict) else payload
    if not isinstance(model, torch.nn.Module):
        raise TypeError("pruned artifact does not contain torch.nn.Module model_object")
    return model.to(device).eval()


def _export_with_pruned_model_context(args: SimpleNamespace, *, pruned_model_path: Path | None, heal_root: Path) -> dict[str, Any]:
    if pruned_model_path is None or not Path(pruned_model_path).is_file():
        from quantization.export.export_single_engine_maxk_onnx import export_single_engine_maxk_onnx

        return export_single_engine_maxk_onnx(args)
    from quantization.utils.paths import load_quant_deploy_module

    legacy = load_quant_deploy_module("export_dynamic_single_engine_maxk_onnx")
    deployment = load_quant_deploy_module("deployment_equivalence")
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    deployment._add_paths(str(heal_root))
    hypes = deployment._load_hypes(str(args.config), str(heal_root))
    model = _load_pruned_model_object(Path(pruned_model_path), heal_root=Path(heal_root), device=device)
    modality = deployment._infer_modality(model)
    old_context = legacy._load_model_context

    def _context(_legacy_args: Any):
        return hypes, device, model, modality

    legacy._load_model_context = _context
    try:
        from quantization.export.export_single_engine_maxk_onnx import export_single_engine_maxk_onnx

        report = export_single_engine_maxk_onnx(args)
    finally:
        legacy._load_model_context = old_context
    report["pruned_model_object_export"] = True
    report["pruned_model_path"] = str(pruned_model_path)
    return report


def export_pruned_lidar_pyramid_signal_maxk_onnx(
    *,
    pruned_model_path: Path,
    pruning_manifest_path: Path,
    output_onnx_path: Path,
    model_config: Path,
    checkpoint: Path,
    heal_root: Path,
    fixed_k: int,
    calibration_or_dummy_batch_source: str,
    dynamic_axes: bool,
    validate_onnx: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "export_success": False,
        "onnx_path": str(output_onnx_path),
        "input_names": [],
        "output_names": [],
        "fixed_k": int(fixed_k),
        "dynamic_axes": bool(dynamic_axes),
        "source_exporter": "quantization.export.export_single_engine_maxk_onnx",
        "pruned_model_path": str(pruned_model_path),
        "pruning_manifest_path": str(pruning_manifest_path),
        "model_config": str(model_config),
        "checkpoint": str(checkpoint),
        "heal_root": str(heal_root),
        "calibration_or_dummy_batch_source": str(calibration_or_dummy_batch_source),
        "shape_check_passed": False,
        "onnx_checker_passed": None,
        "failure_reason": "",
    }
    try:
        export_result = _run_formal_signal_maxk_export(
            pruned_model_path=Path(pruned_model_path),
            output_onnx_path=Path(output_onnx_path),
            model_config=Path(model_config),
            checkpoint=Path(checkpoint),
            heal_root=Path(heal_root),
            fixed_k=int(fixed_k),
            dynamic_axes=bool(dynamic_axes),
        )
        report["source_exporter"] = str(export_result.get("source_exporter", report["source_exporter"]))
        if not export_result.get("success"):
            report["failure_reason"] = str(export_result.get("failure_reason") or "formal_signal_maxk_export_failed")
            return report
        module_call_trace = list(export_result.get("onnx_export_module_call_trace") or [])
        report["onnx_export_module_call_trace_count"] = len(module_call_trace)
        if module_call_trace:
            try:
                origin_map = build_onnx_export_origin_map(Path(output_onnx_path), module_call_trace)
                rename_report = rename_onnx_compute_nodes_with_origin_map(Path(output_onnx_path), origin_map)
                report["origin_map_success"] = True
                report["origin_map_entry_count"] = int(origin_map.get("entry_count", 0))
                report["onnx_node_rename_report"] = rename_report
            except Exception as exc:  # noqa: BLE001
                report["origin_map_success"] = False
                report["failure_reason"] = f"onnx_export_origin_map_failed:{type(exc).__name__}: {exc}"
                report["traceback"] = traceback.format_exc()
                return report
        else:
            report["origin_map_success"] = False
            report["origin_map_entry_count"] = 0
        info = inspect_signal_maxk_onnx(Path(output_onnx_path), validate_onnx=validate_onnx)
        report.update(info)
        report["shape_check_passed"] = bool(info.get("has_required_heal_inputs")) and not bool(info.get("has_toy_input_only"))
        if not report["shape_check_passed"]:
            report["failure_reason"] = "exported_onnx_missing_required_heal_bindings"
            return report
        if validate_onnx and info.get("onnx_checker_passed") is False:
            report["failure_reason"] = str(info.get("onnx_checker_error") or "onnx_checker_failed")
            return report
        report["export_success"] = True
        report["failure_reason"] = ""
        return report
    except Exception as exc:  # noqa: BLE001
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        return report
