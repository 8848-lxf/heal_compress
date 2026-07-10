from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _write_conv_onnx(path: Path, node_names: list[str]) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    initializers = []
    input_name = "input"
    in_channels = 3
    for idx, node_name in enumerate(node_names):
        module_key = node_name.strip("/").replace("/", ".").replace(".Conv", "")
        weight = f"{module_key}.weight"
        bias = f"{module_key}.bias"
        output_name = f"out_{idx}"
        nodes.append(helper.make_node("Conv", [input_name, weight, bias], [output_name], name=node_name))
        initializers.extend(
            [
                numpy_helper.from_array(np.ones((4, in_channels, 1, 1), dtype=np.float32), weight),
                numpy_helper.from_array(np.zeros((4,), dtype=np.float32), bias),
            ]
        )
        input_name = output_name
        in_channels = 4
    graph = helper.make_graph(
        nodes,
        "canonical_mapping_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 4, 8, 8])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def test_canonical_precision_mapping_is_unique_and_generates_precision_specs(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model.onnx"
    _write_conv_onnx(onnx_path, ["/stem/Conv", "/head/Conv"])
    profile = {
        "precision_group_assignments": {
            "pg_stem": {
                "member_modules": ["stem"],
                "requested_precision": "int8",
                "final_precision": "int8",
            },
            "pg_head": {
                "member_modules": ["head"],
                "requested_precision": "int8",
                "final_precision": "fp16",
                "fallback_precision": "fp16",
                "fallback_reason": "head_constraint",
            },
        },
        "layer_precision_assignment": {"stem": "int8", "head": "fp16"},
    }

    mapping = builder.build_canonical_precision_mapping(onnx_path, profile)
    specs = builder.precision_constraint_specs_from_canonical_mapping(mapping)

    assert mapping["success"] is True
    assert {row["canonical_module_name"] for row in mapping["entries"]} == {"stem", "head"}
    stem = next(row for row in mapping["entries"] if row["canonical_module_name"] == "stem")
    assert stem["onnx_node_name"] == "/stem/Conv"
    assert stem["onnx_weight_initializer"] == "stem.weight"
    assert stem["trt_metadata_match_key"] == "/stem/Conv"
    assert stem["requested_precision"] == "int8"
    assert stem["fallback_precision"] == ""
    assert "/stem/Conv:int8" in specs
    assert "/head/Conv:fp16" in specs
    assert all(spec.split(":")[0] in {"/stem/Conv", "/head/Conv"} for spec in specs)


def test_canonical_mapping_rejects_ambiguous_module_to_multiple_onnx_nodes(tmp_path: Path) -> None:
    import pytest
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model.onnx"
    _write_conv_onnx(onnx_path, ["/stem/Conv", "/stem/block/Conv"])
    profile = {
        "precision_group_assignments": {
            "pg_stem": {"member_modules": ["stem"], "requested_precision": "int8", "final_precision": "int8"},
        },
        "layer_precision_assignment": {"stem": "int8"},
    }

    with pytest.raises(ValueError, match="matches_multiple_onnx_nodes"):
        builder.build_canonical_precision_mapping(onnx_path, profile)


def test_canonical_mapping_rejects_one_onnx_node_to_multiple_modules(tmp_path: Path) -> None:
    import pytest
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model.onnx"
    _write_conv_onnx(onnx_path, ["/stem/0/Conv"])
    profile = {
        "precision_group_assignments": {
            "pg_stem": {"member_modules": ["stem"], "requested_precision": "int8", "final_precision": "int8"},
            "pg_stem0": {"member_modules": ["stem.0"], "requested_precision": "int8", "final_precision": "int8"},
        },
        "layer_precision_assignment": {"stem": "int8", "stem.0": "int8"},
    }

    with pytest.raises(ValueError, match="matches_multiple_canonical_modules"):
        builder.build_canonical_precision_mapping(onnx_path, profile)


def test_canonical_mapping_does_not_alias_deblocks_to_single_head(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model.onnx"
    _write_conv_onnx(onnx_path, ["/single_head_0/Conv"])
    profile = {
        "precision_group_assignments": {
            "pg_deblock": {
                "member_modules": ["pyramid_backbone.deblocks.0.0"],
                "requested_precision": "int8",
                "final_precision": "int8",
            },
            "pg_head": {
                "member_modules": ["pyramid_backbone.single_head_0"],
                "requested_precision": "fp16",
                "final_precision": "fp16",
            },
        },
        "layer_precision_assignment": {
            "pyramid_backbone.deblocks.0.0": "int8",
            "pyramid_backbone.single_head_0": "fp16",
        },
    }

    mapping = builder.build_canonical_precision_mapping(onnx_path, profile)

    assert [row["canonical_module_name"] for row in mapping["entries"]] == ["pyramid_backbone.single_head_0"]


def test_qdq_insert_report_can_reverse_map_qdq_nodes_to_canonical_module(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model.onnx"
    output_onnx = tmp_path / "model_qdq.onnx"
    _write_conv_onnx(onnx_path, ["/stem/Conv"])
    profile = {
        "precision_group_assignments": {
            "pg_stem": {"member_modules": ["stem"], "requested_precision": "int8", "final_precision": "int8"},
        },
        "layer_precision_assignment": {"stem": "int8"},
    }
    mapping = builder.build_canonical_precision_mapping(onnx_path, profile)
    profile["canonical_precision_mapping"] = mapping

    report = builder.insert_mixed_precision_qdq(
        input_onnx=onnx_path,
        output_onnx=output_onnx,
        profile=profile,
        scale_table={"pg_stem": {"scale": 0.1}},
    )

    assert report["inserted_qdq_nodes"]
    assert {row["canonical_module_name"] for row in report["inserted_qdq_nodes"]} == {"stem"}
    assert {row["onnx_node_name"] for row in report["inserted_qdq_nodes"]} == {"/stem/Conv"}


def test_trt_metadata_reverse_maps_to_canonical_module(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model.onnx"
    _write_conv_onnx(onnx_path, ["/stem/Conv"])
    profile = {
        "precision_group_assignments": {
            "pg_stem": {"member_modules": ["stem"], "requested_precision": "int8", "final_precision": "int8"},
        },
        "layer_precision_assignment": {"stem": "int8"},
    }
    mapping = builder.build_canonical_precision_mapping(onnx_path, profile)

    assert (
        builder.canonical_module_from_trt_metadata(
            "[ONNX Layer: stem_weight_QuantizeLinear]\x1e[ONNX Layer: /stem/Conv]\x1e[ONNX Layer: /stem/Relu]",
            mapping,
        )
        == "stem"
    )


def test_write_and_load_canonical_precision_mapping(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    payload = {"success": True, "entries": [{"canonical_module_name": "stem", "onnx_node_name": "/stem/Conv"}]}

    builder.write_canonical_precision_mapping(tmp_path, payload)
    loaded = builder.load_canonical_precision_mapping(tmp_path / "canonical_precision_mapping.json")

    assert loaded == payload
    assert json.loads((tmp_path / "canonical_precision_mapping.json").read_text(encoding="utf-8")) == payload
