#!/usr/bin/env python3
"""Build the minimal typed PointPillarScatterTRT graph and command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _boundary_tensor_type(boundary_dtype: str) -> int:
    from onnx import TensorProto

    normalized = str(boundary_dtype).lower()
    if normalized == "fp16":
        return int(TensorProto.FLOAT16)
    if normalized == "fp32":
        return int(TensorProto.FLOAT)
    raise ValueError("scatter_boundary_must_be_fp16_or_fp32")


def build_typed_scatter_onnx(
    destination: str | Path,
    *,
    boundary_dtype: str,
    fixed_k: int,
    channels: int,
    num_agents: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Emit a production-signature scatter graph with an explicit feature dtype."""

    import onnx
    from onnx import TensorProto, helper

    tensor_type = _boundary_tensor_type(boundary_dtype)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    cast_output = "pillar_features_typed"
    scatter_output = "spatial_features"
    nodes = [
        helper.make_node(
            "Cast",
            ["pillar_features"],
            [cast_output],
            name="ScatterBoundaryFeatureCast",
            to=tensor_type,
        ),
        helper.make_node(
            "PointPillarScatterTRT",
            [cast_output, "voxel_coords", "valid_voxel_mask", "pairwise_t_matrix"],
            [scatter_output],
            name="PointPillarScatterTRT",
            num_agents=0,
            height=int(height),
            width=int(width),
            plugin_version="1",
            plugin_namespace="",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        f"strongly_typed_scatter_{str(boundary_dtype).lower()}",
        [
            helper.make_tensor_value_info(
                "pillar_features", TensorProto.FLOAT, [int(fixed_k), int(channels)]
            ),
            helper.make_tensor_value_info(
                "voxel_coords", TensorProto.INT32, [int(fixed_k), 4]
            ),
            helper.make_tensor_value_info(
                "valid_voxel_mask", TensorProto.FLOAT, [int(fixed_k)]
            ),
            helper.make_tensor_value_info(
                "pairwise_t_matrix",
                TensorProto.FLOAT,
                [1, int(num_agents), int(num_agents), 4, 4],
            ),
        ],
        [
            helper.make_tensor_value_info(
                scatter_output,
                tensor_type,
                [int(num_agents), int(channels), int(height), int(width)],
            )
        ],
        value_info=[
            helper.make_tensor_value_info(
                cast_output, tensor_type, [int(fixed_k), int(channels)]
            )
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="heal_compress_strongly_typed_scatter_probe",
    )
    onnx.save(model, str(output))
    return {
        "onnx_path": str(output),
        "plugin_boundary_dtype": str(boundary_dtype).upper(),
        "plugin_int8_boundary": False,
        "plugin_input_tensor": cast_output,
        "plugin_output_tensor": scatter_output,
    }


def strongly_typed_trtexec_command(
    *,
    trtexec_path: str | Path,
    onnx_path: str | Path,
    engine_path: str | Path,
    layer_info_path: str | Path,
    plugin_path: str | Path,
) -> list[str]:
    """Return a strongly typed command with no weak precision controls."""

    return [
        str(trtexec_path),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--staticPlugins={plugin_path}",
        "--stronglyTyped",
        "--noTF32",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_path}",
        "--skipInference",
        "--verbose",
    ]


def audit_scatter_layer_info(
    layer_info_path: str | Path,
    *,
    boundary_dtype: str,
) -> dict[str, Any]:
    """Require exactly one floating PointPillarScatterTRT inspector row."""

    _boundary_tensor_type(boundary_dtype)
    normalized = str(boundary_dtype).lower()
    expected = "Half" if normalized == "fp16" else "Float"
    payload = json.loads(Path(layer_info_path).read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("Layers", [])
        if str(row.get("PluginType", "")) == "PointPillarScatterTRT"
        or (
            str(row.get("LayerType", "")) == "PluginV2"
            and "PointPillarScatterTRT" in str(row.get("Name", ""))
        )
    ]
    issues: list[str] = []
    if len(rows) != 1:
        issues.append(f"pointpillar_plugin_row_count:{len(rows)}")
    row = rows[0] if len(rows) == 1 else {}
    inputs = list(row.get("Inputs", []))
    outputs = list(row.get("Outputs", []))
    input_dtype = str(inputs[0].get("Format/Datatype", "")) if inputs else ""
    output_dtype = str(outputs[0].get("Format/Datatype", "")) if outputs else ""
    if input_dtype != expected:
        issues.append(f"plugin_input_dtype:{input_dtype}!={expected}")
    if output_dtype != expected:
        issues.append(f"plugin_output_dtype:{output_dtype}!={expected}")
    if "Int8" in {input_dtype, output_dtype}:
        issues.append("plugin_int8_boundary_forbidden")
    return {
        "passed": not issues,
        "boundary_dtype": normalized.upper(),
        "plugin_row_count": len(rows),
        "plugin_input_dtype": input_dtype,
        "plugin_output_dtype": output_dtype,
        "plugin_int8_boundary": "Int8" in {input_dtype, output_dtype},
        "issues": issues,
    }


__all__ = [
    "audit_scatter_layer_info",
    "build_typed_scatter_onnx",
    "strongly_typed_trtexec_command",
]
