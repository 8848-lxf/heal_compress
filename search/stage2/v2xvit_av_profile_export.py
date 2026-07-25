"""Explicit AV operand boundaries and strict TensorRT realization audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from quantization.tensorrt.layer_info import (
    has_canonical_identity,
    layer_metadata,
    layer_name,
    load_layer_info,
)

from ..quantization_space.v2xvit_av_merge import av_contract


def rewrite_onnx_av_profile(
    onnx_path: str | Path,
    attention_audit: Mapping[str, Any],
    *,
    profile: str,
    operand_scales: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Insert explicit AV P/V boundaries without changing Softmax compute."""

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    targets = {str(row["node_name"]) for row in attention_audit.get("av_nodes", ())}
    return rewrite_onnx_av_profiles(
        onnx_path,
        attention_audit,
        profiles={name: str(profile).upper() for name in targets},
        operand_scales=operand_scales,
    )


def rewrite_onnx_av_profiles(
    onnx_path: str | Path,
    attention_audit: Mapping[str, Any],
    *,
    profiles: Mapping[str, str],
    operand_scales: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Rewrite independently requested AV profiles at all 12 instances."""

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    targets = {str(row["node_name"]) for row in attention_audit.get("av_nodes", ())}
    normalized = {str(name): av_contract(value).profile for name, value in profiles.items()}
    if set(normalized) != targets:
        raise RuntimeError(
            "v2xvit_av_profile_node_schema_mismatch:"
            f"missing={sorted(targets-set(normalized))}:extra={sorted(set(normalized)-targets)}"
        )
    if all(value == "AV32" for value in normalized.values()):
        return {
            "profile": "AV32",
            "profile_counts": {"AV32": len(targets), "AV16": 0, "AV8": 0},
            "rewritten_node_count": 0,
            "qdq_pair_count": 0,
            "cast_count": 0,
            "softmax_compute_precision": "FP32",
        }
    if len(targets) != 12:
        raise RuntimeError(f"v2xvit_av_node_count_not_12:{len(targets)}")
    model = onnx.load(str(onnx_path), load_external_data=False)
    scales = dict(operand_scales or {})
    rebuilt = []
    qdq_pairs = 0
    casts = 0
    rewritten = []
    for node in model.graph.node:
        name = str(node.name)
        if name not in targets:
            rebuilt.append(node)
            continue
        contract = av_contract(normalized[name])
        if contract.profile == "AV32":
            rebuilt.append(node)
            continue
        prefix = name.replace("/", "_").replace(".", "_")
        insertions = []
        for input_index, source in enumerate(list(node.input)):
            if contract.profile == "AV16":
                boundary = f"{source}__av16_operand_{input_index}"
                insertions.append(helper.make_node(
                    "Cast", [source], [boundary],
                    name=f"{name}__av16_input_cast_{input_index}",
                    to=TensorProto.FLOAT16,
                ))
                casts += 1
            else:
                values = scales.get(name)
                if values is None or len(values) != 2:
                    raise RuntimeError(f"v2xvit_av8_operand_scale_missing:{name}")
                scale = float(values[input_index])
                if not np.isfinite(scale) or scale <= 0.0:
                    raise RuntimeError(f"v2xvit_av8_operand_scale_invalid:{name}:{scale}")
                scale_name = f"{prefix}__av8_scale_{input_index}"
                zero_name = f"{prefix}__av8_zero_{input_index}"
                quantized = f"{source}__av8_q_{input_index}"
                boundary = f"{source}__av8_dq_{input_index}"
                model.graph.initializer.extend([
                    numpy_helper.from_array(np.asarray(scale, dtype=np.float32), scale_name),
                    numpy_helper.from_array(np.asarray(0, dtype=np.int8), zero_name),
                ])
                insertions.extend([
                    helper.make_node(
                        "QuantizeLinear", [source, scale_name, zero_name], [quantized],
                        name=f"{name}__av8_quantize_{input_index}",
                    ),
                    helper.make_node(
                        "DequantizeLinear", [quantized, scale_name, zero_name], [boundary],
                        name=f"{name}__av8_dequantize_{input_index}",
                    ),
                ])
                qdq_pairs += 1
            node.input[input_index] = boundary
        rebuilt.extend(insertions)
        original_outputs = list(node.output)
        for output_index, output in enumerate(original_outputs):
            raw = f"{output}__{contract.profile.lower()}_raw"
            node.output[output_index] = raw
        rebuilt.append(node)
        for output_index, output in enumerate(original_outputs):
            rebuilt.append(helper.make_node(
                "Cast", [node.output[output_index]], [output],
                name=f"{name}__{contract.profile.lower()}_output_cast_{output_index}",
                to=TensorProto.FLOAT16,
            ))
            casts += 1
        rewritten.append(name)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    return {
        "profile": "MIXED" if len(set(normalized.values())) > 1 else next(iter(normalized.values())),
        "profile_counts": {
            value: sum(profile == value for profile in normalized.values())
            for value in ("AV32", "AV16", "AV8")
        },
        "rewritten_node_count": len(rewritten),
        "rewritten_nodes": sorted(rewritten),
        "qdq_pair_count": qdq_pairs,
        "cast_count": casts,
        "softmax_compute_precision": "FP32",
        "node_profiles": dict(sorted(normalized.items())),
    }


def _formats(row: Mapping[str, Any], field: str) -> list[str]:
    items = row.get(field) or row.get(field.lower()) or ()
    return [
        str(item.get("Format/Datatype") or item.get("format") or "").lower()
        for item in items
        if isinstance(item, Mapping)
    ]


def audit_trt_av_profile(
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    attention_audit: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """Require one real AV compute layer with both requested operands."""

    nodes = {str(row["node_name"]) for row in attention_audit.get("av_nodes", ())}
    return audit_trt_av_profiles(
        layer_info,
        attention_audit,
        profiles={name: str(profile).upper() for name in nodes},
    )


def audit_trt_av_profiles(
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    attention_audit: Mapping[str, Any],
    *,
    profiles: Mapping[str, str],
) -> dict[str, Any]:
    """Audit independently requested profiles without allowing fallback."""

    layers = load_layer_info(layer_info)
    normalized = {str(name): av_contract(value).profile for name, value in profiles.items()}
    rows = []
    for source in attention_audit.get("av_nodes", ()):
        canonical = str(source["node_name"])
        if canonical not in normalized:
            raise RuntimeError(f"v2xvit_av_requested_profile_missing:{canonical}")
        contract = av_contract(normalized[canonical])
        expected_input = {"AV32": "float", "AV16": "half", "AV8": "int8"}[
            contract.profile
        ]
        expected_output = {"AV32": "float", "AV16": "half", "AV8": "half"}[
            contract.profile
        ]
        matches = [
            row for row in layers
            if has_canonical_identity(row, canonical) or canonical in layer_metadata(row)
        ]
        compute_matches = []
        for row in matches:
            inputs = _formats(row, "Inputs")
            outputs = _formats(row, "Outputs")
            exact = (
                len(inputs) >= 2
                and all(expected_input in value for value in inputs[:2])
                and bool(outputs)
                and expected_output in outputs[0]
            )
            if contract.profile == "AV8":
                tactic = str(row.get("TacticName") or row.get("tactic") or "").lower()
                exact = exact and any(token in tactic for token in ("int8", "imma", "s8"))
            if exact:
                compute_matches.append(row)
        rows.append({
            "canonical_node": canonical,
            "profile": contract.profile,
            "requested_probability_precision": contract.probability_precision,
            "requested_value_precision": contract.value_precision,
            "requested_compute_precision": contract.compute_precision,
            "requested_output_precision": contract.output_precision,
            "matched_layer_names": [layer_name(row) for row in matches],
            "exact_compute_layer_names": [layer_name(row) for row in compute_matches],
            "requested_realized_exact": len(compute_matches) == 1,
            "unmapped": not matches,
            "fallback": bool(matches) and not compute_matches,
            "conflict": len(compute_matches) != 1,
        })
    return {
        "schema_version": "v2xvit-av-profile-trt-audit-v1",
        "profile": "MIXED" if len(set(normalized.values())) > 1 else next(iter(normalized.values())),
        "profile_counts": {
            value: sum(profile == value for profile in normalized.values())
            for value in ("AV32", "AV16", "AV8")
        },
        "rows": rows,
        "passed": len(rows) == 12 and all(row["requested_realized_exact"] for row in rows),
        "conflict_count": sum(bool(row["conflict"]) for row in rows),
        "unmapped_count": sum(bool(row["unmapped"]) for row in rows),
        "fallback_count": sum(bool(row["fallback"]) for row in rows),
    }


__all__ = [
    "audit_trt_av_profile",
    "audit_trt_av_profiles",
    "rewrite_onnx_av_profile",
    "rewrite_onnx_av_profiles",
]
