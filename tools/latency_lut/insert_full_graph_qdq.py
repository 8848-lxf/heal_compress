from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _precision_overrides(candidate: dict[str, Any]) -> dict[str, str]:
    config = dict(candidate.get("precision_config") or {})
    overrides: dict[str, str] = {}
    nested = config.get("overrides")
    if isinstance(nested, dict):
        overrides.update({str(k): str(v).upper() for k, v in nested.items()})
    for key, value in config.items():
        if key not in {"default", "overrides"}:
            overrides[str(key)] = str(value).upper()
    return overrides


def _node_matches(node: Any, pattern: str) -> bool:
    aliases = {
        "shrink": ["shrink_conv"],
        "detection_head": ["cls_head", "reg_head", "dir_head"],
        "head": ["cls_head", "reg_head", "dir_head"],
        "pyramid_fusion": ["fusion", "pyramid"],
    }
    patterns = [pattern.lower(), pattern.lower().replace(".", "_"), *aliases.get(pattern.lower(), [])]
    # Deliberately avoid matching data tensor names here. In the full ONNX, head
    # layers consume shrink outputs, so input/output matching can accidentally
    # quantize cls/reg/dir heads when only `shrink` was requested.
    node_text = " ".join([node.name, *[x for x in node.input if "weight" in x or "bias" in x]]).lower()
    return any(p in node_text for p in patterns)


def _load_inventory(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("units") or data)


def _requested_int8_units(candidate: dict[str, Any]) -> list[str]:
    return [name for name, precision in _precision_overrides(candidate).items() if precision in {"INT8", "TRT_INT8_QDQ"}]


def _unit_matches_request(unit: dict[str, Any], requested: str) -> bool:
    return requested == unit.get("unit_id") or requested == unit.get("module_path") or str(unit.get("unit_id", "")).startswith(requested + ".")


def _scale_for_tensor(unit_scale: dict[str, Any], tensor_name: str) -> float | None:
    entry = (unit_scale.get("input_activation_tensors") or {}).get(tensor_name)
    if not entry:
        entry = (unit_scale.get("output_activation_tensors") or {}).get(tensor_name)
    if not entry:
        # FP16/FP32 typed rewrite can insert Cast tensors such as
        # "/Concat_9_output_0___default___fp16_cast_13". The calibration cache
        # is keyed by the original ONNX tensor name, so permit a strict prefix
        # alias only when it unambiguously maps to a cached tensor in this unit.
        candidates = {
            **(unit_scale.get("input_activation_tensors") or {}),
            **(unit_scale.get("output_activation_tensors") or {}),
        }
        matches = [(name, value) for name, value in candidates.items() if tensor_name.startswith(str(name) + "_")]
        if len(matches) == 1:
            entry = matches[0][1]
    if isinstance(entry, dict) and entry.get("scale") is not None:
        return float(entry["scale"])
    # Backward-compatible old schema.
    if unit_scale.get("input_scale") is not None:
        return float(unit_scale["input_scale"])
    return None


def _weight_scale(unit_scale: dict[str, Any], tensor_name: str, array: np.ndarray) -> tuple[np.ndarray, bool, int | None]:
    entry = (unit_scale.get("weight_tensors") or {}).get(tensor_name)
    if isinstance(entry, dict) and entry.get("scale") is not None:
        scale = np.asarray(entry["scale"], dtype=np.float32)
        if scale.ndim == 1 and scale.size > 1:
            return np.maximum(scale, 1.0e-8), True, int(entry.get("axis", 0))
        return np.asarray([max(float(scale.reshape(-1)[0]), 1.0e-8)], dtype=np.float32), False, None
    if array.ndim >= 1:
        flat = np.abs(array.reshape(array.shape[0], -1)).max(axis=1)
        return np.maximum(flat.astype(np.float32) / 127.0, 1.0e-8), True, 0
    return np.asarray([max(float(np.max(np.abs(array))) / 127.0, 1.0e-8)], dtype=np.float32), False, None


def insert_qdq(
    input_onnx: str | Path,
    output_onnx: str | Path,
    candidate: dict[str, Any],
    scale_cache: dict[str, Any],
    report_path: str | Path,
    inventory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(input_onnx))
    int8_units = _requested_int8_units(candidate)
    if int8_units and not scale_cache.get("success", False):
        status = "activation_scale_unavailable"
        report = {
            "status": status,
            "success": False,
            "error": "full-graph W8A8 Q/DQ insertion requires a successful activation scale cache; refusing synthetic/default scales",
            "input_onnx": str(input_onnx),
            "output_onnx": str(output_onnx),
            "int8_units": int8_units,
            "inserted_qdq_nodes": [],
            "quantized_weight_initializers": [],
            "unmatched_int8_units": [],
            "weight_scale_granularity": "per_tensor",
            "activation_scale_granularity": "per_tensor",
            "zero_point": 0,
            "scale_source": scale_cache.get("scale_source"),
            "scale_cache_status": scale_cache.get("status"),
            "uses_qdq": False,
            "num_qdq_nodes_inserted": 0,
            "int8_units_requested": int8_units,
            "int8_units_rewritten": [],
            "int8_units_skipped": [{"unit_id": unit, "reason": status} for unit in int8_units],
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    initializer_by_name = {init.name: init for init in model.graph.initializer}
    inserted_nodes: list[str] = []
    quantized_initializers: list[str] = []
    unmatched: list[str] = []
    rewritten_units: list[str] = []
    skipped_units: list[dict[str, Any]] = []
    inventory = inventory or []
    inventory_by_node: dict[str, dict[str, Any]] = {}
    for unit in inventory:
        for node_name in unit.get("covered_onnx_nodes") or []:
            inventory_by_node[str(node_name)] = unit
    units_by_id = {str(unit.get("unit_id")): unit for unit in inventory}
    qdq_by_node: dict[str, list[Any]] = {}
    int8_output_tensors: set[str] = set()
    used_per_channel_weight_scale = False

    for unit in int8_units:
        candidate_units = [u for u in inventory if _unit_matches_request(u, unit)] if inventory else []
        if inventory and not candidate_units:
            unmatched.append(unit)
            skipped_units.append({"unit_id": unit, "reason": "int8_unit_not_found_in_inventory"})
            continue
        if inventory:
            unsupported = [u for u in candidate_units if not u.get("int8_supported")]
            if unsupported:
                skipped_units.extend({"unit_id": u.get("unit_id"), "reason": "int8_unit_not_supported"} for u in unsupported)
                continue
            node_names = {node for u in candidate_units for node in (u.get("covered_onnx_nodes") or [])}
            matches = [node for node in model.graph.node if str(node.name) in node_names]
        else:
            matches = [node for node in model.graph.node if _node_matches(node, unit)]
        if not matches:
            unmatched.append(unit)
            continue
        for node in matches:
            if node.op_type not in {"Conv", "Gemm", "MatMul"}:
                continue
            node_unit = inventory_by_node.get(str(node.name)) or units_by_id.get(unit) or {"unit_id": unit}
            node_unit_id = str(node_unit.get("unit_id") or unit)
            unit_scales = (scale_cache.get("units") or {}).get(node_unit_id, {})
            if not unit_scales:
                skipped_units.append({"unit_id": node_unit_id, "reason": "int8_activation_scale_missing"})
                continue
            for idx, input_name in enumerate(list(node.input[:2])):
                scale_name = f"{input_name}_{unit.replace('.', '_')}_act_scale"
                zp_name = f"{input_name}_{unit.replace('.', '_')}_act_zero_point"
                q_name = f"{input_name}_{unit.replace('.', '_')}_QuantizeLinear"
                dq_name = f"{input_name}_{unit.replace('.', '_')}_DequantizeLinear"
                q_out = f"{input_name}_{unit.replace('.', '_')}_q"
                dq_out = f"{input_name}_{unit.replace('.', '_')}_dq"
                if input_name in initializer_by_name:
                    array = numpy_helper.to_array(initializer_by_name[input_name]).astype(np.float32)
                    scale, per_channel, axis = _weight_scale(unit_scales, input_name, array)
                    used_per_channel_weight_scale = used_per_channel_weight_scale or bool(per_channel)
                    zero_point = np.zeros(scale.shape, dtype=np.int8) if per_channel else np.array([0], dtype=np.int8)
                    model.graph.initializer.extend(
                        [
                            numpy_helper.from_array(scale, name=scale_name),
                            numpy_helper.from_array(zero_point, name=zp_name),
                        ]
                    )
                    quantized_initializers.append(input_name)
                else:
                    act_scale = _scale_for_tensor(unit_scales, input_name)
                    if act_scale is None:
                        skipped_units.append({"unit_id": node_unit_id, "tensor": input_name, "reason": "int8_activation_scale_missing"})
                        continue
                    per_channel = False
                    axis = None
                    model.graph.initializer.extend(
                        [
                            numpy_helper.from_array(np.array([act_scale], dtype=np.float32), name=scale_name),
                            numpy_helper.from_array(np.array([0], dtype=np.int8), name=zp_name),
                        ]
                    )
                attrs = {"axis": axis} if per_channel and axis is not None else {}
                q_node = helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name], [q_out], name=q_name, **attrs)
                dq_node = helper.make_node("DequantizeLinear", [q_out, scale_name, zp_name], [dq_out], name=dq_name, **attrs)
                qdq_by_node.setdefault(str(node.name), []).extend([q_node, dq_node])
                node.input[idx] = dq_out
                inserted_nodes.extend([q_name, dq_name])
            if len(node.input) >= 3 and node.input[2] in initializer_by_name:
                bias_init = initializer_by_name[node.input[2]]
                bias_array = numpy_helper.to_array(bias_init)
                if bias_array.dtype != np.float32:
                    bias_init.CopyFrom(numpy_helper.from_array(bias_array.astype(np.float32), name=bias_init.name))
            int8_output_tensors.update(str(name) for name in node.output)
            if inserted_nodes:
                rewritten_units.append(node_unit_id)

    status = "success"
    error = None
    if unmatched:
        status = "int8_qdq_layer_mapping_failed"
        error = f"unmatched INT8 overrides: {unmatched}"
    if skipped_units:
        status = "qdq_insertion_failed"
        error = f"INT8 Q/DQ insertion skipped required units/tensors: {skipped_units[:5]}"
    output_onnx = Path(output_onnx)
    if status == "success":
        ordered_nodes = []
        boundary_cast_outputs: dict[str, str] = {}
        for node in model.graph.node:
            if node.op_type in {"Relu", "Sigmoid"} and any(input_name in int8_output_tensors for input_name in node.input):
                int8_output_tensors.update(str(name) for name in node.output)
            if str(node.name) not in qdq_by_node and node.op_type in {"Conv", "Gemm", "MatMul"}:
                for idx, input_name in enumerate(list(node.input[:1])):
                    if input_name in int8_output_tensors:
                        cast_output = boundary_cast_outputs.get(input_name)
                        if cast_output is None:
                            safe_input = input_name.strip("/").replace("/", "_").replace(".", "_")
                            cast_name = f"Cast_{safe_input}_fp16_after_int8_region"
                            cast_output = f"{input_name}_fp16_after_int8_region"
                            boundary_cast_outputs[input_name] = cast_output
                            ordered_nodes.append(
                                helper.make_node(
                                    "Cast",
                                    inputs=[input_name],
                                    outputs=[cast_output],
                                    name=cast_name,
                                    to=TensorProto.FLOAT16,
                                )
                            )
                        node.input[idx] = cast_output
            ordered_nodes.extend(qdq_by_node.get(str(node.name), []))
            ordered_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(ordered_nodes)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_onnx))
    report = {
        "status": status,
        "success": status == "success",
        "error": error,
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_onnx),
        "int8_units": int8_units,
        "inserted_qdq_nodes": inserted_nodes,
        "quantized_weight_initializers": quantized_initializers,
        "unmatched_int8_units": unmatched,
        "weight_scale_granularity": "per_channel" if used_per_channel_weight_scale else "per_tensor",
        "activation_scale_granularity": "per_tensor",
        "zero_point": 0,
        "scale_source": scale_cache.get("scale_source", "missing_or_default"),
        "uses_qdq": status == "success" and bool(inserted_nodes),
        "num_cast_inserted": len(boundary_cast_outputs) if status == "success" else 0,
        "num_qdq_nodes_inserted": len(inserted_nodes),
        "num_int8_activation_tensors": sum(1 for name in inserted_nodes if "QuantizeLinear" in name) - len(quantized_initializers),
        "num_int8_weight_tensors": len(quantized_initializers),
        "num_add_fixed": 0,
        "num_concat_fixed": 0,
        "num_grid_sample_fixed": 0,
        "remaining_dtype_mismatches": [],
        "int8_units_requested": int8_units,
        "int8_units_rewritten": sorted(set(rewritten_units)),
        "int8_units_skipped": skipped_units,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-onnx", required=True)
    parser.add_argument("--output-onnx", required=True)
    parser.add_argument("--precision-config", required=True)
    parser.add_argument("--scale-cache", required=True)
    parser.add_argument("--inventory", default=None)
    parser.add_argument("--layer-mapping", default=None)
    parser.add_argument("--report", default="outputs/latency_lut/full_graph_qdq_insertion_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = insert_qdq(
        args.input_onnx,
        args.output_onnx,
        _load_json(args.precision_config),
        _load_json(args.scale_cache),
        args.report,
        inventory=_load_inventory(args.inventory),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
