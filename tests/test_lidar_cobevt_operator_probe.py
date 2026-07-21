from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _write_operator_graph(path: Path, *, unknown_custom: bool) -> None:
    nodes = [
        helper.make_node(
            "PointPillarScatterTRT",
            ["pillar", "coords", "mask", "pairwise"],
            ["spatial"],
            name="scatter",
            domain="trt",
        ),
        helper.make_node("Relu", ["spatial"], ["output"], name="relu"),
    ]
    opsets = [helper.make_opsetid("", 17), helper.make_opsetid("trt", 1)]
    if unknown_custom:
        nodes.insert(
            1,
            helper.make_node(
                "UnknownCoBEVTOp",
                ["spatial"],
                ["unknown"],
                name="unknown",
                domain="cobevt",
            ),
        )
        nodes[-1].input[0] = "unknown"
        opsets.append(helper.make_opsetid("cobevt", 1))
    graph = helper.make_graph(
        nodes,
        "operator_probe",
        [
            helper.make_tensor_value_info("pillar", TensorProto.FLOAT, [8, 4]),
            helper.make_tensor_value_info("coords", TensorProto.INT32, [8, 4]),
            helper.make_tensor_value_info("mask", TensorProto.FLOAT, [8]),
            helper.make_tensor_value_info("pairwise", TensorProto.FLOAT, [1, 2, 2, 4, 4]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4, 3, 5])],
    )
    onnx.save(helper.make_model(graph, opset_imports=opsets), str(path))


def test_operator_audit_accepts_only_registered_scatter(tmp_path):
    from search.model_families.lidar_cobevt.operator_probe import audit_onnx_operators

    path = tmp_path / "registered.onnx"
    _write_operator_graph(path, unknown_custom=False)

    report = audit_onnx_operators(path)

    assert report.registered_custom_ops == ("trt::PointPillarScatterTRT",)
    assert report.unregistered_custom_ops == ()
    assert report.weak_precision_fallback_requested is False


def test_operator_audit_reports_unknown_custom_op(tmp_path):
    from search.model_families.lidar_cobevt.operator_probe import audit_onnx_operators

    path = tmp_path / "unknown.onnx"
    _write_operator_graph(path, unknown_custom=True)

    report = audit_onnx_operators(path)

    assert report.unregistered_custom_ops == ("cobevt::UnknownCoBEVTOp",)
    assert report.passed is False


def test_strongly_typed_probe_command_forbids_weak_precision_flags(tmp_path):
    from search.model_families.lidar_cobevt.operator_probe import (
        strongly_typed_probe_command,
    )

    command = strongly_typed_probe_command(
        trtexec_path=Path("/opt/TensorRT/bin/trtexec"),
        onnx_path=tmp_path / "model.onnx",
        engine_path=tmp_path / "model.plan",
        layer_info_path=tmp_path / "layers.json",
        plugin_path=tmp_path / "scatter.so",
        shapes={"record_len": "1", "voxel_features": "8x32x4"},
    )

    assert "--stronglyTyped" in command
    assert "--noTF32" in command
    assert "--skipInference" in command
    forbidden = (
        "--fp16",
        "--int8",
        "--precisionConstraints",
        "--layerPrecisions",
        "--layerOutputTypes",
    )
    assert not any(token.startswith(forbidden) for token in command)


def test_static_probe_command_omits_shape_profile_flags(tmp_path):
    from search.model_families.lidar_cobevt.operator_probe import (
        strongly_typed_probe_command,
    )

    command = strongly_typed_probe_command(
        trtexec_path=Path("/opt/TensorRT/bin/trtexec"),
        onnx_path=tmp_path / "static.onnx",
        engine_path=tmp_path / "static.plan",
        layer_info_path=tmp_path / "layers.json",
        plugin_path=tmp_path / "scatter.so",
        shapes={},
    )

    assert not any("Shapes=" in token for token in command)


def test_scatter_parser_copy_removes_only_trt_domain(tmp_path):
    from search.model_families.lidar_cobevt.operator_probe import (
        make_scatter_parser_compatible,
    )

    source = tmp_path / "source.onnx"
    destination = tmp_path / "parser.onnx"
    _write_operator_graph(source, unknown_custom=False)

    report = make_scatter_parser_compatible(source, destination)
    model = onnx.load(str(destination))
    scatter = next(node for node in model.graph.node if node.op_type == "PointPillarScatterTRT")

    assert scatter.domain == ""
    assert report["changed_nodes"] == ["scatter"]
    assert all(opset.domain != "trt" for opset in model.opset_import)
