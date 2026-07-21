"""Replace the six CoBEVT F3 Attention GEMMs with explicit plugin oracles."""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path
from typing import Any, Iterable


_QK_EQUATION = "bhid,bhjd->bhij"
_AV_EQUATION = "bhij,bhjd->bhid"


def _equation(node: Any) -> str:
    for attribute in node.attribute:
        if attribute.name == "equation":
            return attribute.s.decode("utf-8").replace(" ", "")
    return ""


def _tensor_types(model: Any) -> dict[str, int]:
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    return {
        str(value.name): int(value.type.tensor_type.elem_type)
        for value in values
        if value.type.HasField("tensor_type")
    }


def _constant_scalar(name: str, producers: dict[str, Any], initializers: dict[str, Any]) -> float:
    import numpy as np
    from onnx import numpy_helper

    current = str(name)
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        if current in initializers:
            values = numpy_helper.to_array(initializers[current]).reshape(-1)
            if values.size != 1:
                break
            return float(values[0])
        node = producers.get(current)
        if node is None:
            break
        if node.op_type in {"Cast", "Identity"} and node.input:
            current = str(node.input[0])
            continue
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    values = numpy_helper.to_array(attribute.t).reshape(-1)
                    if values.size == 1:
                        return float(values[0])
                if attribute.name == "value_float":
                    return float(attribute.f)
                if attribute.name == "value_floats" and len(attribute.floats) == 1:
                    return float(attribute.floats[0])
        break
    raise RuntimeError(f"attention_plugin_scale_unresolved:{name}")


def _fp16_layout_operand(
    name: str, producers: dict[str, Any], types: dict[str, int]
) -> tuple[str, list[Any]]:
    from onnx import TensorProto

    current = str(name)
    reverse_layout: list[Any] = []
    while True:
        node = producers.get(current)
        if node is None:
            raise RuntimeError(f"attention_plugin_fp16_cast_source_unresolved:{name}")
        if node.op_type == "Cast":
            break
        if node.op_type not in {"Identity", "Reshape", "Transpose"} or not node.input:
            raise RuntimeError(
                f"attention_plugin_non_layout_operand_path:{name}:{node.name}:{node.op_type}"
            )
        reverse_layout.append(node)
        current = str(node.input[0])
    if len(node.input) != 1:
        raise RuntimeError(f"attention_plugin_fp16_cast_arity:{node.name}:{len(node.input)}")
    source = str(node.input[0])
    if int(types.get(source, 0)) != int(TensorProto.FLOAT16):
        raise RuntimeError(f"attention_plugin_operand_not_fp16:{source}:{types.get(source)}")
    cloned: list[Any] = []
    replacement = source
    for original in reversed(reverse_layout):
        if len(original.output) != 1:
            raise RuntimeError(
                f"attention_plugin_layout_output_arity:{original.name}:{len(original.output)}"
            )
        clone = copy.deepcopy(original)
        clone.name = f"{original.name}__F16Operand"
        clone.input[0] = replacement
        clone.output[0] = f"{original.output[0]}__F16Operand"
        replacement = str(clone.output[0])
        cloned.append(clone)
    return replacement, cloned


def _qk_inputs(
    node: Any,
    producers: dict[str, Any],
    types: dict[str, int],
    initializers: dict[str, Any],
) -> tuple[str, str, float, list[Any], str]:
    if len(node.input) != 2:
        raise RuntimeError(f"attention_plugin_qk_arity:{node.name}:{len(node.input)}")
    scaled_q = producers.get(str(node.input[0]))
    if scaled_q is None or scaled_q.op_type != "Mul" or len(scaled_q.input) != 2:
        raise RuntimeError(f"attention_plugin_qk_scale_node_unresolved:{node.name}")
    scale_candidates: list[tuple[str, float]] = []
    for value in scaled_q.input:
        try:
            scale_candidates.append((str(value), _constant_scalar(str(value), producers, initializers)))
        except RuntimeError:
            pass
    if len(scale_candidates) != 1:
        raise RuntimeError(f"attention_plugin_qk_scaled_operand_ambiguous:{node.name}")
    scale_name, scale = scale_candidates[0]
    data_inputs = [str(value) for value in scaled_q.input if str(value) != scale_name]
    if len(data_inputs) != 1:
        raise RuntimeError(f"attention_plugin_qk_data_operand_ambiguous:{node.name}")
    q_source, q_layout = _fp16_layout_operand(data_inputs[0], producers, types)
    k_source, k_layout = _fp16_layout_operand(str(node.input[1]), producers, types)
    return q_source, k_source, scale, [*q_layout, *k_layout], str(scaled_q.name)


def rewrite_f3_attention_einsums(
    input_onnx: str | Path,
    output_onnx: str | Path,
    *,
    families: Iterable[str],
) -> dict[str, Any]:
    """Rewrite exactly six selected QK/AV nodes and preserve all tensor names."""

    import onnx
    from onnx import TensorProto, helper

    selected = {str(value).upper() for value in families}
    if not selected or not selected <= {"QK", "AV"}:
        raise ValueError(f"attention_plugin_families_invalid:{sorted(selected)}")
    source = Path(input_onnx).expanduser().resolve()
    destination = Path(output_onnx).expanduser().resolve()
    model = onnx.load(str(source))
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"attention_plugin_pre_rewrite_type_inference_failed:{exc}") from exc
    types = _tensor_types(inferred)
    producers = {str(output): node for node in model.graph.node for output in node.output}
    initializers = {str(value.name): value for value in model.graph.initializer}
    rewritten = []
    records: list[dict[str, Any]] = []
    ignored_node_names: list[str] = []
    counts = {"QK": 0, "AV": 0}
    for node in model.graph.node:
        family = "QK" if node.op_type == "Einsum" and _equation(node) == _QK_EQUATION else (
            "AV" if node.op_type == "Einsum" and _equation(node) == _AV_EQUATION else ""
        )
        if family not in selected:
            rewritten.append(node)
            continue
        if family == "QK":
            left, right, scale, layout_nodes, scale_node_name = _qk_inputs(
                node, producers, types, initializers
            )
            rewritten.extend(layout_nodes)
            ignored_node_names.append(scale_node_name)
            output_type = 0
            expected_output = int(TensorProto.FLOAT)
        else:
            if len(node.input) != 2:
                raise RuntimeError(f"attention_plugin_av_arity:{node.name}:{len(node.input)}")
            left, right = map(str, node.input)
            if any(int(types.get(value, 0)) != int(TensorProto.FLOAT16) for value in (left, right)):
                raise RuntimeError(f"attention_plugin_av_operand_not_fp16:{node.name}")
            scale = 1.0
            output_type = 1
            expected_output = int(TensorProto.FLOAT16)
        output = str(node.output[0])
        if int(types.get(output, 0)) != expected_output:
            raise RuntimeError(f"attention_plugin_output_dtype_mismatch:{node.name}:{types.get(output)}")
        plugin = helper.make_node(
            "QKMixedAccumPlugin",
            [left, right],
            list(node.output),
            name=f"{node.name}__F16A32PluginOracle",
            family=0 if family == "QK" else 1,
            output_type=output_type,
            scale=float(scale),
            plugin_version="1",
            plugin_namespace="",
        )
        rewritten.append(plugin)
        counts[family] += 1
        records.append(
            {
                "family": family,
                "original_node": str(node.name),
                "plugin_node": str(plugin.name),
                "left_operand": left,
                "right_operand": right,
                "operand_dtype": "FP16",
                "accumulator_dtype": "FP32",
                "output_dtype": "FP32" if output_type == 0 else "FP16",
                "scale": float(scale),
            }
        )
    expected = {family: 6 for family in selected}
    actual = {family: counts[family] for family in selected}
    if actual != expected:
        raise RuntimeError(f"attention_plugin_rewrite_count_mismatch:{actual}:{expected}")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    aliases = {
        row["original_node"]: row["plugin_node"] for row in records
    }
    overrides = {}
    for row in records:
        if row["family"] != "QK":
            continue
        block = row["original_node"].lstrip("/").split("/fn/", 1)[0].replace("/", ".")
        overrides[f"fusion_net.{block}.fn::qk_matmul"] = "fp16"
    return {
        "input_onnx": str(source),
        "output_onnx": str(destination),
        "output_onnx_sha256": digest,
        "qk_replaced_count": counts["QK"],
        "av_replaced_count": counts["AV"],
        "replacement_records": records,
        "ignored_node_names": sorted(set(ignored_node_names)),
        "attention_inventory_node_aliases": aliases,
        "implementation": "plugin_oracle",
        "native_tensorrt": False,
        "precision_realization_overrides": overrides,
    }


__all__ = ["rewrite_f3_attention_einsums"]
