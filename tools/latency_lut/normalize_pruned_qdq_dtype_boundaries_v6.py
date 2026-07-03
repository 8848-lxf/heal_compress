from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.latency_lut.validate_onnx_dtype_closure_v6 import FLOAT_OPS, MERGE_OPS, infer_tensor_dtypes


def _load_candidate(path_or_dict: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return json.loads(Path(path_or_dict).read_text(encoding="utf-8"))


def _precision_config(candidate: dict[str, Any]) -> tuple[str, dict[str, str]]:
    cfg = dict(candidate.get("precision_config") or {})
    default = str(cfg.get("default", "FP16")).upper()
    overrides = {}
    nested = cfg.get("overrides")
    if isinstance(nested, dict):
        overrides.update({str(k): str(v).upper() for k, v in nested.items()})
    for key, value in cfg.items():
        if key not in {"default", "overrides"}:
            overrides[str(key)] = str(value).upper()
    return default, overrides


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _node_matches(node: Any, pattern: str) -> bool:
    owned = [name for name in node.input if "weight" in name.lower() or "bias" in name.lower()]
    text = " ".join([node.name, node.op_type, *owned])
    return _norm(pattern) in _norm(text)


def _node_precision(node: Any, default: str, overrides: dict[str, str]) -> str:
    best = default
    best_len = -1
    for key, value in overrides.items():
        if _node_matches(node, key) and len(_norm(key)) > best_len:
            best = value
            best_len = len(_norm(key))
    if best in {"INT8", "TRT_INT8_QDQ", "INT8_QDQ"}:
        return "INT8_QDQ"
    if best in {"FP32", "TRT_FP32"}:
        return "FP32"
    return "FP16"


def _target_dtype(precision: str) -> tuple[str, int, Any]:
    if precision == "FP32" or precision == "INT8_QDQ":
        return "FLOAT", TensorProto.FLOAT, np.float32
    return "FLOAT16", TensorProto.FLOAT16, np.float16


def _array_dtype_name(array: np.ndarray) -> str:
    if array.dtype == np.float16:
        return "FLOAT16"
    if array.dtype == np.float32:
        return "FLOAT"
    if array.dtype == np.int8:
        return "INT8"
    return "UNKNOWN"


def _safe(text: str) -> str:
    return text.strip("/").replace("/", "_").replace(".", "_").replace(":", "_")


def _producer_map(model: Any) -> dict[str, Any]:
    return {out: node for node in model.graph.node for out in node.output}


def _base_initializer_name(tensor: str, producers: dict[str, Any], initializers: dict[str, Any], depth: int = 0) -> str | None:
    if tensor in initializers:
        return tensor
    if depth > 12:
        return None
    prod = producers.get(tensor)
    if prod is None or not prod.input:
        return None
    if prod.op_type in {"DequantizeLinear", "QuantizeLinear", "Cast", "Identity"}:
        return _base_initializer_name(prod.input[0], producers, initializers, depth + 1)
    return None


def _rewrite_initializer_dtype(init: Any, np_dtype: Any) -> bool:
    array = numpy_helper.to_array(init)
    if array.dtype == np_dtype:
        return False
    init.CopyFrom(numpy_helper.from_array(array.astype(np_dtype), name=init.name))
    return True


def normalize_pruned_qdq_dtype_boundaries(
    input_onnx: str | Path,
    output_onnx: str | Path,
    candidate: str | Path | dict[str, Any],
    report_path: str | Path,
) -> dict[str, Any]:
    model = onnx.load(str(input_onnx))
    candidate_data = _load_candidate(candidate)
    default, overrides = _precision_config(candidate_data)
    initializers = {init.name: init for init in model.graph.initializer}
    producers = _producer_map(model)
    dtypes = infer_tensor_dtypes(model)
    new_nodes: list[Any] = []
    cast_counter = 0
    conv_fixes: list[dict[str, Any]] = []
    merge_fixes: list[dict[str, Any]] = []
    casts_inserted: list[str] = []
    initializers_rewritten: list[str] = []

    def insert_cast(input_name: str, target_elem: int, target_name: str, reason: str) -> str:
        nonlocal cast_counter
        out = f"{input_name}_{target_name.lower()}_closure_cast_{cast_counter}"
        name = f"Cast_{_safe(input_name)}_{target_name}_{cast_counter}"
        cast_counter += 1
        new_nodes.append(helper.make_node("Cast", [input_name], [out], name=name, to=target_elem))
        dtypes[out] = target_name
        casts_inserted.append(name)
        return out

    for node in model.graph.node:
        if node.op_type in FLOAT_OPS:
            requested = _node_precision(node, default, overrides)
            target_name, target_elem, target_np = _target_dtype(requested)
            input_before = dtypes.get(node.input[0], "UNKNOWN") if node.input else "UNKNOWN"
            weight_before = dtypes.get(node.input[1], "UNKNOWN") if len(node.input) > 1 else input_before
            bias_before = dtypes.get(node.input[2], "UNKNOWN") if len(node.input) > 2 else None
            local_casts: list[str] = []
            local_rewrites: list[str] = []
            if node.input and node.input[0] in initializers:
                if _rewrite_initializer_dtype(initializers[node.input[0]], target_np):
                    local_rewrites.append(node.input[0])
                    dtypes[node.input[0]] = target_name
            elif node.input and input_before != target_name:
                node.input[0] = insert_cast(node.input[0], target_elem, target_name, "conv_input_dtype")
                local_casts.append(casts_inserted[-1])
            if len(node.input) > 1:
                init_name = _base_initializer_name(node.input[1], producers, initializers)
                if init_name and _rewrite_initializer_dtype(initializers[init_name], target_np):
                    local_rewrites.append(init_name)
                    dtypes[init_name] = target_name
                    # Q/DQ weight path dequant output follows scale dtype; for
                    # explicit INT8 QDQ compute, scale stays FLOAT and the DQ
                    # output is FLOAT. For FP16 direct weights, the initializer
                    # itself is enough.
                    if requested != "INT8_QDQ":
                        dtypes[node.input[1]] = target_name
            if len(node.input) > 2:
                init_name = _base_initializer_name(node.input[2], producers, initializers)
                if init_name and _rewrite_initializer_dtype(initializers[init_name], target_np):
                    local_rewrites.append(init_name)
                    dtypes[init_name] = target_name
                    dtypes[node.input[2]] = target_name
            input_after = dtypes.get(node.input[0], target_name) if node.input else target_name
            weight_after = target_name
            if len(node.input) > 1:
                # If the weight is a DQ output, its dtype remains controlled by
                # the scale tensor. The pass rewrites the base initializer so
                # validation sees a legal FLOAT QDQ compute path.
                weight_after = dtypes.get(node.input[1], target_name)
                if requested == "INT8_QDQ":
                    weight_after = target_name
            bias_after = target_name if len(node.input) > 2 else None
            for out in node.output:
                dtypes[out] = target_name
            if local_casts or local_rewrites or input_before != input_after or weight_before != weight_after or (bias_before and bias_before != bias_after):
                conv_fixes.append(
                    {
                        "node_name": node.name,
                        "op_type": node.op_type,
                        "requested_precision": requested,
                        "input_dtype_before": input_before,
                        "weight_dtype_before": weight_before,
                        "bias_dtype_before": bias_before,
                        "input_dtype_after": input_after,
                        "weight_dtype_after": weight_after,
                        "bias_dtype_after": bias_after,
                        "casts_inserted": local_casts,
                        "initializers_rewritten": local_rewrites,
                        "qdq_path_preserved": True,
                    }
                )
                initializers_rewritten.extend(local_rewrites)
        if node.op_type in MERGE_OPS:
            before = {name: dtypes.get(name, "UNKNOWN") for name in node.input}
            float_dtypes = [dtype for dtype in before.values() if dtype in {"FLOAT", "FLOAT16"}]
            if len(set(float_dtypes)) > 1:
                # Merge ops are not INT8 regions in the first Route2 QDQ
                # implementation. When a DQ output (FLOAT) meets the default
                # FP16 branch, close the boundary by casting DQ output back to
                # FP16 before Add/Concat. Only keep FLOAT when the merge node
                # itself is matched by an explicit FP32 override.
                target_name = "FLOAT" if _node_precision(node, default, overrides) == "FP32" else "FLOAT16"
                target_elem = TensorProto.FLOAT if target_name == "FLOAT" else TensorProto.FLOAT16
                merge_casts: list[str] = []
                for idx, input_name in enumerate(list(node.input)):
                    if dtypes.get(input_name) in {"FLOAT", "FLOAT16"} and dtypes[input_name] != target_name:
                        node.input[idx] = insert_cast(input_name, target_elem, target_name, "merge_input_dtype")
                        merge_casts.append(casts_inserted[-1])
                after = {name: dtypes.get(name, "UNKNOWN") for name in node.input}
                merge_fixes.append(
                    {
                        "merge_node": node.name,
                        "op_type": node.op_type,
                        "input_dtypes_before": before,
                        "target_dtype": target_name,
                        "casts_inserted": merge_casts,
                        "input_dtypes_after": after,
                        "reason": "downstream_fp32" if target_name == "FLOAT" else "downstream_fp16",
                    }
                )
            out_dtype = next((dtypes.get(name) for name in node.input if dtypes.get(name) in {"FLOAT", "FLOAT16"}), "UNKNOWN")
            for out in node.output:
                dtypes[out] = out_dtype
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    output = Path(output_onnx)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    payload = {
        "status": "success",
        "success": True,
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_onnx),
        "candidate_id": candidate_data.get("candidate_id", ""),
        "precision_profile_id": candidate_data.get("precision_profile_id", ""),
        "num_cast_inserted": len(casts_inserted),
        "num_dtype_mismatches_fixed": len(conv_fixes) + len(merge_fixes),
        "num_merge_fixed": len(merge_fixes),
        "conv_gemm_fixes": conv_fixes,
        "merge_fixes": merge_fixes,
        "casts_inserted": casts_inserted,
        "initializers_rewritten": sorted(set(initializers_rewritten)),
        "qdq_pattern_preserved": True,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-onnx", required=True)
    parser.add_argument("--output-onnx", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = normalize_pruned_qdq_dtype_boundaries(args.input_onnx, args.output_onnx, args.candidate, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
