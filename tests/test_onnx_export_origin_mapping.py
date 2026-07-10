from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _write_ambiguous_layer0_onnx(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "w_backbone", "b_backbone"],
                ["backbone_out"],
                name="/layer0/layer0.0/conv1/Conv",
                kernel_shape=[1, 1],
                strides=[1, 1],
                pads=[0, 0, 0, 0],
                dilations=[1, 1],
                group=1,
            ),
            helper.make_node(
                "Conv",
                ["backbone_out", "w_pyramid", "b_pyramid"],
                ["pyramid_out"],
                name="/layer0/layer0.0/conv1_1/Conv",
                kernel_shape=[1, 1],
                strides=[1, 1],
                pads=[0, 0, 0, 0],
                dilations=[1, 1],
                group=1,
            ),
        ],
        "ambiguous_layer0",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 4, 4])],
        [helper.make_tensor_value_info("pyramid_out", TensorProto.FLOAT, [1, 8, 4, 4])],
        [
            numpy_helper.from_array(np.ones((8, 8, 1, 1), dtype=np.float32), "w_backbone"),
            numpy_helper.from_array(np.zeros((8,), dtype=np.float32), "b_backbone"),
            numpy_helper.from_array(np.ones((8, 8, 1, 1), dtype=np.float32), "w_pyramid"),
            numpy_helper.from_array(np.zeros((8,), dtype=np.float32), "b_pyramid"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _call_trace() -> list[dict[str, object]]:
    base = {
        "module_type": "Conv2d",
        "mapped_onnx_op_type": "Conv",
        "weight_shape": [8, 8, 1, 1],
        "bias_shape": [8],
        "groups": 1,
        "kernel_size": [1, 1],
        "stride": [1, 1],
        "padding": [0, 0],
        "dilation": [1, 1],
        "input_shape_signature": [[1, 8, 4, 4]],
        "output_shape_signature": [1, 8, 4, 4],
    }
    return [
        {
            **base,
            "canonical_module_name": "backbone_m1.resnet.layer0.0.conv1",
            "module_call_index": 17,
        },
        {
            **base,
            "canonical_module_name": "pyramid_backbone.resnet.layer0.0.conv1",
            "module_call_index": 42,
        },
    ]


def _profile() -> dict:
    return {
        "precision_group_assignments": {
            "pg_backbone": {
                "member_modules": ["backbone_m1.resnet.layer0.0.conv1"],
                "requested_precision": "int8",
                "final_precision": "int8",
            },
            "pg_pyramid": {
                "member_modules": ["pyramid_backbone.resnet.layer0.0.conv1"],
                "requested_precision": "int8",
                "final_precision": "int8",
            },
        },
        "layer_precision_assignment": {
            "backbone_m1.resnet.layer0.0.conv1": "int8",
            "pyramid_backbone.resnet.layer0.0.conv1": "int8",
        },
    }


def test_export_origin_map_renames_ambiguous_backbone_and_pyramid_nodes(tmp_path: Path) -> None:
    import onnx
    import quant_deploy.pruned_signal_maxk_exporter as exporter

    onnx_path = tmp_path / "model_signal_maxk.onnx"
    _write_ambiguous_layer0_onnx(onnx_path)

    origin_map = exporter.build_onnx_export_origin_map(onnx_path, _call_trace())
    rename_report = exporter.rename_onnx_compute_nodes_with_origin_map(onnx_path, origin_map)
    model = onnx.load(str(onnx_path))
    node_names = [node.name for node in model.graph.node if node.op_type == "Conv"]

    assert origin_map["success"] is True
    assert rename_report["renamed_node_count"] == 2
    assert "__canonical__backbone_m1_resnet_layer0_0_conv1__Conv__call00017" in node_names
    assert "__canonical__pyramid_backbone_resnet_layer0_0_conv1__Conv__call00042" in node_names
    assert (tmp_path / "onnx_export_origin_map.json").is_file()
    assert (tmp_path / "onnx_export_module_call_trace.json").is_file()
    assert (tmp_path / "onnx_node_rename_report.json").is_file()


def test_no_origin_map_keeps_ambiguous_mapping_as_gate_failure(tmp_path: Path) -> None:
    import pytest
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model_signal_maxk.onnx"
    _write_ambiguous_layer0_onnx(onnx_path)

    with pytest.raises(ValueError, match="onnx_node_matches_multiple_canonical_modules"):
        builder.build_canonical_precision_mapping(onnx_path, _profile())


def test_origin_map_makes_canonical_mapping_specs_and_qdq_exact(tmp_path: Path) -> None:
    import quant_deploy.pruned_signal_maxk_exporter as exporter
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model_signal_maxk.onnx"
    output_onnx = tmp_path / "model_mixed_qdq.onnx"
    _write_ambiguous_layer0_onnx(onnx_path)
    origin_map = exporter.build_onnx_export_origin_map(onnx_path, _call_trace())
    exporter.rename_onnx_compute_nodes_with_origin_map(onnx_path, origin_map)

    profile = _profile()
    mapping = builder.build_canonical_precision_mapping(onnx_path, profile)
    profile["canonical_precision_mapping"] = mapping
    specs = builder.precision_constraint_specs_from_canonical_mapping(mapping)
    qdq = builder.insert_mixed_precision_qdq(
        input_onnx=onnx_path,
        output_onnx=output_onnx,
        profile=profile,
        scale_table={"pg_backbone": {"scale": 0.1}, "pg_pyramid": {"scale": 0.1}},
    )

    unique_names = {row["onnx_node_name_unique"] for row in mapping["entries"]}
    assert mapping["success"] is True
    assert mapping["origin_map_used"] is True
    assert mapping["ambiguous_mapping_count"] == 0
    assert unique_names == {
        "__canonical__backbone_m1_resnet_layer0_0_conv1__Conv__call00017",
        "__canonical__pyramid_backbone_resnet_layer0_0_conv1__Conv__call00042",
    }
    assert {spec.split(":")[0] for spec in specs} == unique_names
    assert qdq["unmatched_int8_precision_groups"] == []
    assert {row["onnx_node_name_unique"] for row in qdq["inserted_qdq_nodes"]} == unique_names
    assert {row["onnx_node_name_original"] for row in qdq["inserted_qdq_nodes"]} == {
        "/layer0/layer0.0/conv1/Conv",
        "/layer0/layer0.0/conv1_1/Conv",
    }
    assert all(row["canonical_module_name"] for row in qdq["inserted_qdq_nodes"])
    persisted = json.loads((tmp_path / "onnx_export_origin_map.json").read_text(encoding="utf-8"))
    assert persisted["entry_count"] == 2


def test_trt_metadata_prefers_unique_name_and_rejects_ambiguous_original(tmp_path: Path) -> None:
    import quant_deploy.pruned_signal_maxk_exporter as exporter
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model_signal_maxk.onnx"
    _write_ambiguous_layer0_onnx(onnx_path)
    origin_map = exporter.build_onnx_export_origin_map(onnx_path, _call_trace())
    exporter.rename_onnx_compute_nodes_with_origin_map(onnx_path, origin_map)
    mapping = builder.build_canonical_precision_mapping(onnx_path, _profile())

    assert (
        builder.canonical_module_from_trt_metadata(
            "[ONNX Layer: __canonical__pyramid_backbone_resnet_layer0_0_conv1__Conv__call00042]",
            mapping,
        )
        == "pyramid_backbone.resnet.layer0.0.conv1"
    )
    assert (
        builder.canonical_module_from_trt_metadata(
            "[ONNX Layer: /layer0/layer0.0/conv1/Conv]",
            mapping,
        )
        == "__ambiguous__"
    )
