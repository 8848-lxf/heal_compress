from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto


def _tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        result[str(value.name)] = int(value.type.tensor_type.elem_type)
    return result


@pytest.mark.parametrize(
    ("boundary_dtype", "expected_type"),
    (("fp16", TensorProto.FLOAT16), ("fp32", TensorProto.FLOAT)),
)
def test_typed_scatter_graph_declares_floating_plugin_boundary(
    tmp_path: Path,
    boundary_dtype: str,
    expected_type: int,
) -> None:
    from scripts.run_strongly_typed_scatter_probe import build_typed_scatter_onnx

    output = tmp_path / f"scatter_{boundary_dtype}.onnx"
    report = build_typed_scatter_onnx(
        output,
        boundary_dtype=boundary_dtype,
        fixed_k=8,
        channels=4,
        num_agents=2,
        height=3,
        width=5,
    )

    model = onnx.load(str(output))
    types = _tensor_types(model)
    plugin = next(node for node in model.graph.node if node.op_type == "PointPillarScatterTRT")
    assert plugin.domain == ""
    assert types[plugin.input[0]] == expected_type
    assert types[plugin.output[0]] == expected_type
    assert types["voxel_coords"] == TensorProto.INT32
    assert types["valid_voxel_mask"] == TensorProto.FLOAT
    assert types["pairwise_t_matrix"] == TensorProto.FLOAT
    assert report["plugin_boundary_dtype"] == boundary_dtype.upper()
    assert report["plugin_int8_boundary"] is False


def test_typed_scatter_graph_rejects_int8_boundary(tmp_path: Path) -> None:
    from scripts.run_strongly_typed_scatter_probe import build_typed_scatter_onnx

    with pytest.raises(ValueError, match="scatter_boundary_must_be_fp16_or_fp32"):
        build_typed_scatter_onnx(
            tmp_path / "scatter_int8.onnx",
            boundary_dtype="int8",
            fixed_k=8,
            channels=4,
            num_agents=2,
            height=3,
            width=5,
        )


def test_strongly_typed_scatter_command_forbids_weak_precision_options(
    tmp_path: Path,
) -> None:
    from scripts.run_strongly_typed_scatter_probe import strongly_typed_trtexec_command

    command = strongly_typed_trtexec_command(
        trtexec_path=Path("/opt/tensorrt/bin/trtexec"),
        onnx_path=tmp_path / "scatter.onnx",
        engine_path=tmp_path / "scatter.plan",
        layer_info_path=tmp_path / "layer_info.json",
        plugin_path=tmp_path / "scatter.so",
    )

    assert "--stronglyTyped" in command
    forbidden = (
        "--fp16",
        "--int8",
        "--precisionConstraints",
        "--layerPrecisions",
        "--layerOutputTypes",
    )
    assert not any(part.startswith(forbidden) for part in command)
    assert any(part.startswith("--staticPlugins=") for part in command)
    assert any(part.startswith("--exportLayerInfo=") for part in command)


@pytest.mark.parametrize(
    ("boundary_dtype", "inspector_dtype"),
    (("fp16", "Half"), ("fp32", "Float")),
)
def test_scatter_inspector_audit_requires_exact_floating_boundary(
    tmp_path: Path,
    boundary_dtype: str,
    inspector_dtype: str,
) -> None:
    from scripts.run_strongly_typed_scatter_probe import audit_scatter_layer_info

    layer_info = tmp_path / "layer_info.json"
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "PointPillarScatterTRT",
                        "LayerType": "PluginV2",
                        "PluginType": "PointPillarScatterTRT",
                        "Inputs": [
                            {"Name": "pillar", "Format/Datatype": inspector_dtype},
                            {"Name": "coords", "Format/Datatype": "Int32"},
                            {"Name": "mask", "Format/Datatype": "Float"},
                            {"Name": "pairwise", "Format/Datatype": "Float"},
                        ],
                        "Outputs": [
                            {"Name": "spatial", "Format/Datatype": inspector_dtype}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_scatter_layer_info(layer_info, boundary_dtype=boundary_dtype)
    assert report["passed"] is True
    assert report["plugin_input_dtype"] == inspector_dtype
    assert report["plugin_output_dtype"] == inspector_dtype
    assert report["plugin_int8_boundary"] is False

    mismatched = audit_scatter_layer_info(
        layer_info,
        boundary_dtype="fp32" if boundary_dtype == "fp16" else "fp16",
    )
    assert mismatched["passed"] is False
    assert mismatched["issues"]
