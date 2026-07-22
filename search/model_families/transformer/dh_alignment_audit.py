"""Shape, tactic, fusion, padding and fallback evidence for d_h engines."""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


_PADDING_TOKENS = ("pad", "padding", "zero-fill", "zerofill")
# A TensorRT-selected cuBLAS GEMV/small-N GEMM tactic is still a realized
# tactic, not proof of precision or shape fallback.  Its performance is kept
# in the tactic/latency inventory and judged empirically.  Only explicit
# fallback/scalar evidence is classified as FALLBACK here.
_FALLBACK_TOKENS = ("fallback", "scalar")
_TENSOR_CORE_TOKENS = ("tensor core", "tc_", "hmma", "imma", "mma")


def audit_onnx_head_dimension(
    onnx_path: str | Path,
    *,
    selected_module_paths: Iterable[str],
    origin_map: Mapping[str, Any],
    heads: int,
    target_d_h: int,
) -> dict[str, Any]:
    import onnx
    from onnx import numpy_helper, shape_inference

    path = Path(onnx_path)
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)
    inferred = shape_inference.infer_shapes(graph)
    initializers = {str(value.name): numpy_helper.to_array(value) for value in inferred.graph.initializer}
    entries = {str(row["module_path"]): row for row in origin_map.get("entries", ())}
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    width = int(heads) * int(target_d_h)
    for module_path in selected_module_paths:
        matched = [row for path, row in entries.items() if path == module_path or path.startswith(f"{module_path}.")]
        if not matched:
            issues.append(f"origin_mapping_missing:{module_path}")
            continue
        for row in matched:
            initializer_name = str(row.get("weight_initializer", ""))
            weight = initializers.get(initializer_name)
            shape = list(weight.shape) if weight is not None else []
            role_path = str(row.get("module_path", ""))
            is_output = role_path.endswith((".out_proj", ".to_out.0")) or ".a_linears." in role_path
            expected_axis = 0 if is_output else 1
            physical = bool(shape and width in shape)
            if not physical:
                issues.append(f"onnx_initializer_dh_missing:{role_path}:{shape}:{width}")
            rows.append(
                {
                    "module_path": role_path,
                    "onnx_node": str(row.get("canonical_node_name", "")),
                    "initializer": initializer_name,
                    "initializer_shape": shape,
                    "expected_projection_width": width,
                    "projection_width_present": physical,
                    "projection_axis_semantics": "W_O_input" if is_output else "QKV_output",
                }
            )
    return {
        "checker_passed": True,
        "shape_inference_passed": True,
        "heads": int(heads),
        "target_d_h": int(target_d_h),
        "logical_projection_width": width,
        "rows": rows,
        "issues": issues,
        "passed": not issues,
    }


def _walk_layer_info(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for value in payload:
            rows.extend(_walk_layer_info(value))
    elif isinstance(payload, dict):
        if any(key in payload for key in ("Name", "LayerType", "TacticName", "name", "type")):
            rows.append(payload)
        for value in payload.values():
            if isinstance(value, (dict, list)):
                rows.extend(_walk_layer_info(value))
    return rows


def audit_engine_alignment(
    layer_info_path: str | Path,
    *,
    logical_d_h: int,
    projection_width: int,
) -> dict[str, Any]:
    path = Path(layer_info_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = _walk_layer_info(payload)
    evidence: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    materialized_padding = False
    internal_padding = False
    fallback = False
    tensor_core = False
    for row in layers:
        name = str(row.get("Name", row.get("name", "")))
        layer_type = str(row.get("LayerType", row.get("type", "")))
        tactic = str(row.get("TacticName", row.get("Tactic", row.get("tactic", ""))))
        inputs = row.get("Inputs", row.get("inputs", []))
        outputs = row.get("Outputs", row.get("outputs", []))
        text = json.dumps(row, sort_keys=True).lower()
        # Bind padding/reformat evidence to the physical projection width or a
        # projection/QK layer name.  Generic CNN Pad/Reformat layers elsewhere
        # in the full engine are not evidence that TensorRT padded d_h.
        relevant = any(
            token in name.lower() for token in (
                "q_proj", "k_proj", "v_proj", "out_proj", "einsum",
                "q_linears", "k_linears", "v_linears", "a_linears",
                "to_qkv", "to_out", "qk_matmul", "av_matmul",
            )
        )
        if not relevant:
            continue
        explicit_pad = layer_type.lower() in {"padding", "pad"} or any(token in name.lower() for token in _PADDING_TOKENS)
        reformat = "reformat" in layer_type.lower() or "reformat" in name.lower()
        shuffle = "shuffle" in layer_type.lower() or "shuffle" in name.lower()
        tactic_pad = any(token in tactic.lower() for token in _PADDING_TOKENS)
        tactic_fallback = any(token in tactic.lower() for token in _FALLBACK_TOKENS)
        uses_tc = any(token in tactic.lower() for token in _TENSOR_CORE_TOKENS)
        materialized_padding |= explicit_pad
        internal_padding |= tactic_pad
        fallback |= tactic_fallback
        tensor_core |= uses_tc
        counters["reformat"] += int(reformat)
        counters["shuffle"] += int(shuffle)
        counters["cast"] += int("cast" in layer_type.lower() or "cast" in name.lower())
        counters["padding"] += int(explicit_pad)
        evidence.append(
            {
                "name": name,
                "layer_type": layer_type,
                "tactic": tactic,
                "inputs": inputs,
                "outputs": outputs,
                "explicit_padding": explicit_pad,
                "internal_padding_hint": tactic_pad,
                "fallback_hint": tactic_fallback,
                "tensor_core_hint": uses_tc,
            }
        )
    if materialized_padding:
        status = "MATERIALIZED_PADDED"
    elif fallback:
        status = "FALLBACK"
    elif internal_padding:
        status = "INTERNAL_PADDED"
    elif evidence:
        status = "EXACT_NONALIGNED" if int(logical_d_h) % 8 else "EXACT_ALIGNED"
    else:
        status = "UNKNOWN"
    return {
        "logical_d_h": int(logical_d_h),
        "projection_width": int(projection_width),
        "padding_status": status,
        "materialized_padding": materialized_padding,
        "internal_padding_hint": internal_padding,
        "fallback_hint": fallback,
        "tensor_core_hint": tensor_core,
        "cast_count": counters["cast"],
        "reformat_count": counters["reformat"],
        "shuffle_count": counters["shuffle"],
        "padding_layer_count": counters["padding"],
        "evidence": evidence,
        "evidence_level": "B" if evidence else "C",
    }


def latency_beneficial(
    *,
    baseline_p50_ms: float,
    candidate_p50_ms: float,
    baseline_repeat_cv: float,
    baseline_replay_drift: float,
) -> dict[str, Any]:
    reduction = (float(baseline_p50_ms) - float(candidate_p50_ms)) / float(baseline_p50_ms)
    threshold = max(0.01, 3.0 * float(baseline_repeat_cv), float(baseline_replay_drift))
    return {
        "latency_reduction": reduction,
        "required_reduction": threshold,
        "latency_beneficial": reduction > threshold,
    }


def parse_engine_memory_audit(build_log: str | Path) -> dict[str, int | None]:
    text = Path(build_log).read_text(encoding="utf-8", errors="replace")

    def value(label: str) -> int | None:
        match = re.search(rf"{re.escape(label)}:\s*([0-9]+)\s*bytes", text)
        return int(match.group(1)) if match else None

    return {
        "host_persistent_bytes": value("Total Host Persistent Memory"),
        "device_persistent_bytes": value("Total Device Persistent Memory"),
        "max_scratch_bytes": value("Max Scratch Memory"),
        "activation_bytes": value("Total Activation Memory"),
        "weights_bytes": value("Total Weights Memory"),
    }


__all__ = [
    "audit_engine_alignment",
    "audit_onnx_head_dimension",
    "latency_beneficial",
    "parse_engine_memory_audit",
]
