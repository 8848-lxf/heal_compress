from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.tools.compression.latency_lut.precision_constraint_graph import (
    build_precision_constraint_graph_from_onnx,
)
from opencood.tools.compression.latency_lut.precision_resolver import resolve_precision_constraints


FLOAT_OPS = {"Conv", "Gemm", "MatMul"}
MERGE_OPS = {"Add", "Sub", "Mul", "Div", "Concat", "GridSample"}


def _load_candidate(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _precision_config(candidate: dict[str, Any]) -> tuple[str, dict[str, str]]:
    config = dict(candidate.get("precision_config") or {})
    default = str(config.get("default", "FP16")).upper()
    overrides = {}
    nested = config.get("overrides")
    if isinstance(nested, dict):
        overrides.update({str(k): str(v).upper() for k, v in nested.items()})
    for key, value in config.items():
        if key not in {"default", "overrides"}:
            overrides[str(key)] = str(value).upper()
    return default, overrides


def _node_matches(node: Any, pattern: str) -> bool:
    # Match the operator itself and its owned parameters, not arbitrary data
    # inputs/outputs. Downstream heads can consume shrink tensors, so matching
    # all tensor names would over-apply a `shrink` override to cls/reg/dir heads.
    owned_inputs = [name for name in node.input if "weight" in name.lower() or "bias" in name.lower()]
    text = " ".join([node.name, node.op_type, *owned_inputs]).lower()
    pattern_text = pattern.lower()
    normalized_text = re.sub(r"[^a-z0-9]+", "", text)
    normalized_pattern = re.sub(r"[^a-z0-9]+", "", pattern_text)
    return (
        pattern_text in text
        or pattern_text.replace(".", "_") in text
        or pattern_text.replace(".", "/") in text
        or (normalized_pattern and normalized_pattern in normalized_text)
    )


def _load_mapping(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _patterns_for_unit(unit_name: str, mapping: dict[str, Any]) -> list[str]:
    entry = mapping.get(unit_name) or {}
    patterns = entry.get("onnx_node_patterns") if isinstance(entry, dict) else None
    if patterns:
        return [str(v) for v in patterns]
    return [unit_name]


def _matches_unit(node: Any, unit_name: str, mapping: dict[str, Any]) -> bool:
    return any(_node_matches(node, pattern) for pattern in _patterns_for_unit(unit_name, mapping))


def _compute_nodes(model: Any) -> list[Any]:
    return [node for node in model.graph.node if node.op_type in FLOAT_OPS]


def _node_key(node: Any) -> str:
    return str(node.name or (node.output[0] if node.output else ""))


def _tensor_dtype_from_elem_type(elem_type: int) -> str | None:
    from onnx import TensorProto

    if elem_type == TensorProto.FLOAT16:
        return "FP16"
    if elem_type == TensorProto.FLOAT:
        return "FP32"
    return None


def _tensor_proto_for_precision(precision: str) -> int:
    from onnx import TensorProto

    return TensorProto.FLOAT16 if precision == "FP16" else TensorProto.FLOAT


def _dtype_map(model: Any) -> dict[str, str]:
    from onnx import numpy_helper

    dtypes: dict[str, str] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value.type.HasField("tensor_type"):
            dtype = _tensor_dtype_from_elem_type(value.type.tensor_type.elem_type)
            if dtype:
                dtypes[value.name] = dtype
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        if array.dtype == np.float16:
            dtypes[init.name] = "FP16"
        elif array.dtype == np.float32:
            dtypes[init.name] = "FP32"
    return dtypes


def _graph_output_dtype(model: Any, name: str) -> str | None:
    for value in model.graph.output:
        if value.name == name and value.type.HasField("tensor_type"):
            return _tensor_dtype_from_elem_type(value.type.tensor_type.elem_type)
    return None


def _set_value_dtype(model: Any, name: str, precision: str) -> None:
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value.name == name and value.type.HasField("tensor_type"):
            value.type.tensor_type.elem_type = _tensor_proto_for_precision(precision)


def _target_from_inputs(input_dtypes: list[str]) -> str:
    return "FP32" if "FP32" in input_dtypes else "FP16"


def _safe_name(text: str) -> str:
    return text.replace(".", "_").replace("/", "_").replace(":", "_")


def _constant_output_dtype(node: Any) -> str | None:
    from onnx import TensorProto

    for attr in node.attribute:
        if attr.name == "value" and attr.HasField("t"):
            return _tensor_dtype_from_elem_type(attr.t.data_type)
        if attr.name in {"value_float", "value_floats"}:
            return "FP32"
        if attr.name == "dtype":
            return _tensor_dtype_from_elem_type(int(attr.i))
        if attr.name == "value" and attr.type == TensorProto.FLOAT:
            return "FP32"
    return None


def rewrite_onnx_precision(
    input_onnx: str | Path,
    output_onnx: str | Path,
    candidate: dict[str, Any],
    report_path: str | Path,
    layer_mapping: str | Path | None = None,
    resolved_precision_profile: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    default, overrides = _precision_config(candidate)
    mapping = _load_mapping(layer_mapping)
    model = onnx.load(str(input_onnx))
    constraint_graph = build_precision_constraint_graph_from_onnx(model)
    if resolved_precision_profile is None:
        resolved_report = resolve_precision_constraints(
            constraint_graph,
            candidate,
            output_path=Path(report_path).with_name("resolved_precision_profile.json"),
        )
    elif isinstance(resolved_precision_profile, (str, Path)):
        resolved_report = json.loads(Path(resolved_precision_profile).read_text(encoding="utf-8"))
    else:
        resolved_report = dict(resolved_precision_profile)
    resolved_precision = dict(resolved_report.get("resolved_precision_config") or {})
    fp16_nodes: list[str] = []
    fp32_nodes: list[str] = []
    inserted_cast_nodes: list[str] = []
    converted_weight_initializers: list[str] = []
    unmatched_overrides: list[str] = []
    checker_warnings: list[str] = []
    initializer_by_name = {init.name: init for init in model.graph.initializer}
    node_precision: dict[str, tuple[str, str]] = {}

    override_nodes: dict[str, tuple[str, str]] = {}
    for unit_name, precision in overrides.items():
        if precision == "INT8":
            continue
        matches = [node for node in _compute_nodes(model) if _matches_unit(node, unit_name, mapping)]
        if not matches:
            unmatched_overrides.append(unit_name)
            continue
        for node in matches:
            override_nodes[_node_key(node)] = (unit_name, precision)

    if unmatched_overrides:
        status = "layer_name_mapping_failed"
        error = f"unmatched precision overrides: {unmatched_overrides}"
    else:
        status = "success"
        error = None

    for node in _compute_nodes(model):
        key = _node_key(node)
        if key in override_nodes:
            unit_name, precision = override_nodes[key]
        else:
            unit_name, precision = "__default__", default
        precision = str(resolved_precision.get(unit_name, resolved_precision.get(key, precision))).upper()
        if precision == "INT8":
            continue
        if precision not in {"FP16", "FP32", "TRT_FP16", "TRT_FP32"}:
            raise ValueError(f"unsupported FP16/FP32 rewrite precision: {precision}")
        precision = "FP16" if precision in {"FP16", "TRT_FP16"} else "FP32"
        if key in override_nodes or precision == "FP16":
            if precision == "FP32":
                fp32_nodes.append(node.name or node.output[0])
            elif precision == "FP16":
                fp16_nodes.append(node.name or node.output[0])
            node_precision[key] = (unit_name, precision)
            for input_name in node.input:
                init = initializer_by_name.get(input_name)
                if init is None:
                    continue
                array = numpy_helper.to_array(init)
                target = np.float16 if precision == "FP16" else np.float32
                if array.dtype != target:
                    converted = numpy_helper.from_array(array.astype(target), name=init.name)
                    init.CopyFrom(converted)
                    converted_weight_initializers.append(init.name)

    new_nodes = []
    cast_counter = 0
    dtypes = _dtype_map(model)
    producer: dict[str, Any] = {}
    graph_output_names = {output.name for output in model.graph.output}
    graph_output_expected = {output.name: _graph_output_dtype(model, output.name) for output in model.graph.output}
    num_add_fixed = 0
    num_concat_fixed = 0
    num_grid_sample_fixed = 0
    num_plugin_boundary_fixed = 0

    for node in model.graph.node:
        unit_precision = node_precision.get(_node_key(node))
        if unit_precision:
            unit_name, precision = unit_precision
            for idx, input_name in enumerate(list(node.input)):
                if input_name in initializer_by_name:
                    continue
                if dtypes.get(input_name) == precision and unit_name == "__default__":
                    continue
                safe_unit = _safe_name(unit_name)
                cast_output = f"{input_name}_{safe_unit}_{precision.lower()}_cast_{cast_counter}"
                cast_name = f"Cast_{safe_unit}_{precision}_{idx}_{cast_counter}"
                cast_counter += 1
                cast = helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_output],
                    name=cast_name,
                    to=TensorProto.FLOAT16 if precision == "FP16" else TensorProto.FLOAT,
                )
                new_nodes.append(cast)
                node.input[idx] = cast_output
                dtypes[cast_output] = precision
                inserted_cast_nodes.append(cast_name)
        if node.op_type in MERGE_OPS:
            known_input_dtypes = [dtypes[name] for name in node.input if dtypes.get(name) in {"FP16", "FP32"}]
            if len(set(known_input_dtypes)) > 1:
                target = _target_from_inputs(known_input_dtypes)
                for idx, input_name in enumerate(list(node.input)):
                    if dtypes.get(input_name) in {"FP16", "FP32"} and dtypes[input_name] != target:
                        cast_output = f"{input_name}_{node.op_type.lower()}_{target.lower()}_cast_{cast_counter}"
                        cast_name = f"Cast_{_safe_name(_node_key(node))}_{target}_{idx}_{cast_counter}"
                        cast_counter += 1
                        cast = helper.make_node(
                            "Cast",
                            inputs=[input_name],
                            outputs=[cast_output],
                            name=cast_name,
                            to=_tensor_proto_for_precision(target),
                        )
                        new_nodes.append(cast)
                        node.input[idx] = cast_output
                        dtypes[cast_output] = target
                        inserted_cast_nodes.append(cast_name)
                if node.op_type == "Concat":
                    num_concat_fixed += 1
                elif node.op_type == "GridSample":
                    num_grid_sample_fixed += 1
                else:
                    num_add_fixed += 1
        new_nodes.append(node)
        if unit_precision:
            _unit_name, precision = unit_precision
            for output_name in node.output:
                dtypes[output_name] = precision
                producer[output_name] = node
        elif node.op_type == "Cast":
            attr = next((attr for attr in node.attribute if attr.name == "to"), None)
            if attr is not None:
                dtype = _tensor_dtype_from_elem_type(int(attr.i))
                if dtype:
                    for output_name in node.output:
                        dtypes[output_name] = dtype
                        producer[output_name] = node
        elif node.op_type in MERGE_OPS:
            known_input_dtypes = [dtypes[name] for name in node.input if dtypes.get(name) in {"FP16", "FP32"}]
            if known_input_dtypes:
                target = _target_from_inputs(known_input_dtypes)
                for output_name in node.output:
                    dtypes[output_name] = target
                    producer[output_name] = node
        elif node.op_type in {"Constant", "ConstantOfShape"}:
            dtype = _constant_output_dtype(node)
            if dtype:
                for output_name in node.output:
                    dtypes[output_name] = dtype
                    producer[output_name] = node
        else:
            first_dtype = next((dtypes[name] for name in node.input if dtypes.get(name) in {"FP16", "FP32"}), None)
            if first_dtype:
                for output_name in node.output:
                    dtypes[output_name] = first_dtype
                    producer[output_name] = node

    # Keep runner-visible graph outputs at their declared dtype by inserting
    # a final Cast only when needed. This keeps internal strongly-typed regions
    # free to use FP16/FP32 while preserving the exported interface.
    for output_name in list(graph_output_names):
        expected = graph_output_expected.get(output_name)
        actual = dtypes.get(output_name)
        if expected in {"FP16", "FP32"} and actual in {"FP16", "FP32"} and expected != actual:
            producing_node = producer.get(output_name)
            if producing_node is None:
                continue
            internal_name = f"{output_name}_pre_output_{actual.lower()}"
            for idx, name in enumerate(producing_node.output):
                if name == output_name:
                    producing_node.output[idx] = internal_name
            cast_name = f"Cast_graph_output_{_safe_name(output_name)}_{expected}_{cast_counter}"
            cast_counter += 1
            cast = helper.make_node(
                "Cast",
                inputs=[internal_name],
                outputs=[output_name],
                name=cast_name,
                to=_tensor_proto_for_precision(expected),
            )
            new_nodes.append(cast)
            inserted_cast_nodes.append(cast_name)
            dtypes[internal_name] = actual
            dtypes[output_name] = expected

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    remaining_dtype_mismatches: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type not in MERGE_OPS:
            continue
        input_dtypes = [dtypes.get(name) for name in node.input]
        float_input_dtypes = [dtype for dtype in input_dtypes if dtype in {"FP16", "FP32"}]
        if len(set(float_input_dtypes)) > 1:
            remaining_dtype_mismatches.append(
                {
                    "node": _node_key(node),
                    "op_type": node.op_type,
                    "inputs": list(node.input),
                    "input_dtypes": input_dtypes,
                }
            )

    output_onnx = Path(output_onnx)
    if remaining_dtype_mismatches:
        status = "dtype_rewrite_remaining_mismatch"
        error = "remaining Add/Concat/elementwise dtype mismatches after rewrite"
    if status == "success":
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        try:
            onnx.checker.check_model(model)
        except Exception as exc:
            # The full deployment ONNX contains the custom PointPillarScatterTRT
            # op. ONNX's generic checker does not know that op, so record the
            # warning instead of blocking a TensorRT-targeted graph.
            checker_warnings.append(str(exc))
        onnx.save(model, str(output_onnx))
    report = {
        "status": status,
        "success": status == "success",
        "error": error,
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_onnx),
        "requested_default": default,
        "requested_overrides": overrides,
        "layer_mapping": str(layer_mapping) if layer_mapping else None,
        "fp16_nodes": fp16_nodes,
        "fp32_nodes": fp32_nodes,
        "inserted_cast_nodes": inserted_cast_nodes,
        "converted_weight_initializers": converted_weight_initializers,
        "unmatched_overrides": unmatched_overrides,
        "checker_warnings": checker_warnings,
        "resolved_precision_profile": resolved_report,
        "precision_constraint_graph": constraint_graph.to_dict(),
        "num_cast_inserted": len(inserted_cast_nodes),
        "num_add_fixed": num_add_fixed,
        "num_concat_fixed": num_concat_fixed,
        "num_grid_sample_fixed": num_grid_sample_fixed,
        "num_plugin_boundary_fixed": num_plugin_boundary_fixed,
        "remaining_dtype_mismatches": remaining_dtype_mismatches,
        "uses_qdq": False,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-onnx", required=True)
    parser.add_argument("--output-onnx", required=True)
    parser.add_argument("--candidate", "--precision-config", dest="candidate", required=True)
    parser.add_argument("--layer-mapping", default=None)
    parser.add_argument("--resolved-precision-profile", default=None)
    parser.add_argument("--report", default="outputs/latency_lut/full_onnx_precision_rewrite_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = rewrite_onnx_precision(
        args.input_onnx,
        args.output_onnx,
        _load_candidate(args.candidate),
        args.report,
        layer_mapping=args.layer_mapping,
        resolved_precision_profile=args.resolved_precision_profile,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
