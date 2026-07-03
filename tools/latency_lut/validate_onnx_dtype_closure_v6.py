from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


FLOAT_OPS = {"Conv", "Gemm", "MatMul"}
MERGE_OPS = {"Add", "Concat", "Sub", "Mul", "Div", "Sum", "Where", "GridSample"}


def _load_candidate(path_or_dict: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return json.loads(Path(path_or_dict).read_text(encoding="utf-8"))


def _elem_dtype(elem_type: int) -> str:
    if elem_type == TensorProto.FLOAT:
        return "FLOAT"
    if elem_type == TensorProto.FLOAT16:
        return "FLOAT16"
    if elem_type == TensorProto.INT8:
        return "INT8"
    if elem_type == TensorProto.UINT8:
        return "UINT8"
    if elem_type == TensorProto.INT64:
        return "INT64"
    if elem_type == TensorProto.INT32:
        return "INT32"
    if elem_type == TensorProto.BOOL:
        return "BOOL"
    return "UNKNOWN"


def _dtype_from_array(array: np.ndarray) -> str:
    if array.dtype == np.float32:
        return "FLOAT"
    if array.dtype == np.float16:
        return "FLOAT16"
    if array.dtype == np.int8:
        return "INT8"
    if array.dtype == np.uint8:
        return "UINT8"
    if array.dtype == np.int64:
        return "INT64"
    if array.dtype == np.int32:
        return "INT32"
    if array.dtype == np.bool_:
        return "BOOL"
    return "UNKNOWN"


def _constant_dtype(node: Any) -> str:
    for attr in node.attribute:
        if attr.name == "value":
            return _dtype_from_array(numpy_helper.to_array(attr.t))
        if attr.name == "value_float" or attr.name == "value_floats":
            return "FLOAT"
        if attr.name in {"value_int", "value_ints"}:
            return "INT64"
    return "UNKNOWN"


def _first_known(dtypes: dict[str, str], names: list[str], allowed: set[str] | None = None) -> str:
    for name in names:
        dtype = dtypes.get(name)
        if dtype and dtype != "UNKNOWN" and (allowed is None or dtype in allowed):
            return dtype
    return "UNKNOWN"


def infer_tensor_dtypes(model: Any) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value.type.HasField("tensor_type"):
            dtypes[value.name] = _elem_dtype(value.type.tensor_type.elem_type)
    for init in model.graph.initializer:
        dtypes[init.name] = _dtype_from_array(numpy_helper.to_array(init))
    for node in model.graph.node:
        if node.op_type == "Constant":
            dtype = _constant_dtype(node)
        elif node.op_type == "ConstantOfShape":
            dtype = _constant_dtype(node)
            if dtype == "UNKNOWN":
                dtype = "FLOAT"
        elif node.op_type == "Cast":
            to_attr = next((attr for attr in node.attribute if attr.name == "to"), None)
            dtype = _elem_dtype(int(to_attr.i)) if to_attr is not None else "UNKNOWN"
        elif node.op_type == "QuantizeLinear":
            dtype = dtypes.get(node.input[2], "INT8") if len(node.input) >= 3 else "INT8"
        elif node.op_type == "DequantizeLinear":
            dtype = dtypes.get(node.input[1], "FLOAT") if len(node.input) >= 2 else "FLOAT"
        elif node.op_type in {"Shape", "NonZero", "ArgMax", "ArgMin"}:
            dtype = "INT64"
        elif node.op_type in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And", "Or", "Not"}:
            dtype = "BOOL"
        elif node.op_type in {"Unsqueeze", "Squeeze", "Slice", "Gather", "GatherElements", "Transpose", "Reshape", "Expand", "Tile", "Flatten", "Identity"}:
            dtype = _first_known(dtypes, list(node.input[:1]))
        elif node.op_type == "Where":
            dtype = _first_known(dtypes, list(node.input[1:]))
        elif node.op_type in FLOAT_OPS:
            dtype = next((dtypes.get(name) for name in node.input[:2] if dtypes.get(name) in {"FLOAT", "FLOAT16"}), None) or "UNKNOWN"
        elif node.op_type in MERGE_OPS:
            dtype = next((dtypes.get(name) for name in node.input if dtypes.get(name) in {"FLOAT", "FLOAT16"}), None) or "UNKNOWN"
        else:
            dtype = next((dtypes.get(name) for name in node.input if dtypes.get(name) != "UNKNOWN"), None) or "UNKNOWN"
        for out in node.output:
            dtypes[out] = dtype
    return dtypes


def _producer_map(model: Any) -> dict[str, Any]:
    return {out: node for node in model.graph.node for out in node.output}


def _is_qdq_tensor(tensor: str, producers: dict[str, Any]) -> bool:
    prod = producers.get(tensor)
    return bool(prod and prod.op_type == "DequantizeLinear")


def _check_qdq_pair_not_broken(model: Any) -> list[dict[str, Any]]:
    producers = _producer_map(model)
    errors: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        q = producers.get(node.input[0]) if node.input else None
        if q is None or q.op_type != "QuantizeLinear":
            errors.append({"node": node.name, "op_type": node.op_type, "reason": "dequantize_input_is_not_quantize_output"})
    return errors


def validate_onnx_dtype_closure(
    onnx_path: str | Path,
    candidate: str | Path | dict[str, Any],
    report_path: str | Path,
) -> dict[str, Any]:
    model = onnx.load(str(onnx_path))
    _candidate = _load_candidate(candidate)
    dtypes = infer_tensor_dtypes(model)
    producers = _producer_map(model)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    conv_checked = 0
    merge_checked = 0
    qdq_checked = 0
    for node in model.graph.node:
        if node.op_type in FLOAT_OPS:
            conv_checked += 1
            input_dtype = dtypes.get(node.input[0], "UNKNOWN") if node.input else "UNKNOWN"
            weight_dtype = dtypes.get(node.input[1], "UNKNOWN") if len(node.input) > 1 else input_dtype
            bias_dtype = dtypes.get(node.input[2], input_dtype) if len(node.input) > 2 else input_dtype
            if "UNKNOWN" in {input_dtype, weight_dtype, bias_dtype}:
                errors.append({"node": node.name, "op_type": node.op_type, "reason": "unknown_dtype", "input_dtype": input_dtype, "weight_dtype": weight_dtype, "bias_dtype": bias_dtype})
            if input_dtype != weight_dtype:
                errors.append({"node": node.name, "op_type": node.op_type, "reason": "input_weight_dtype_mismatch", "input_dtype": input_dtype, "weight_dtype": weight_dtype})
            if bias_dtype not in {input_dtype, weight_dtype}:
                errors.append({"node": node.name, "op_type": node.op_type, "reason": "bias_dtype_mismatch", "input_dtype": input_dtype, "weight_dtype": weight_dtype, "bias_dtype": bias_dtype})
        if node.op_type in MERGE_OPS:
            merge_checked += 1
            merge_inputs = node.input[1:] if node.op_type == "Where" else node.input
            input_dtypes = {name: dtypes.get(name, "UNKNOWN") for name in merge_inputs}
            float_dtypes = {dtype for dtype in input_dtypes.values() if dtype in {"FLOAT", "FLOAT16"}}
            if "UNKNOWN" in input_dtypes.values():
                warnings.append({"node": node.name, "op_type": node.op_type, "reason": "unknown_merge_input_dtype", "input_dtypes": input_dtypes})
            if len(float_dtypes) > 1:
                errors.append({"node": node.name, "op_type": node.op_type, "reason": "merge_dtype_mismatch", "input_dtypes": input_dtypes})
        if node.op_type == "DequantizeLinear":
            qdq_checked += 1
    errors.extend(_check_qdq_pair_not_broken(model))
    payload = {
        "candidate_id": _candidate.get("candidate_id", ""),
        "precision_profile_id": _candidate.get("precision_profile_id", ""),
        "onnx_path": str(onnx_path),
        "valid": len(errors) == 0,
        "num_conv_gemm_checked": conv_checked,
        "num_merge_checked": merge_checked,
        "num_qdq_patterns_checked": qdq_checked,
        "num_errors": len(errors),
        "errors": errors,
        "warnings": warnings,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_onnx_dtype_closure(args.onnx, args.candidate, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
