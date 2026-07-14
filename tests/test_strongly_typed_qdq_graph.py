from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult


def _synthetic_qdq_graph(path: Path) -> CanonicalPrecisionMappingResult:
    scale = numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="zero")
    int8_weight = numpy_helper.from_array(
        np.ones((4, 4, 1, 1), dtype=np.float32), name="int8_weight"
    )
    fp16_weight = numpy_helper.from_array(
        np.ones((4, 4, 1, 1), dtype=np.float32), name="fp16_weight"
    )
    nodes = [
        helper.make_node(
            "PointPillarScatterTRT",
            ["pillar_features", "voxel_coords", "valid_voxel_mask", "pairwise_t_matrix"],
            ["scatter_output"],
            name="PointPillarScatterTRT",
            num_agents=0,
            height=3,
            width=5,
            plugin_version="1",
            plugin_namespace="",
        ),
        helper.make_node(
            "QuantizeLinear", ["scatter_output", "scale", "zero"], ["activation_q"], name="activation_q"
        ),
        helper.make_node(
            "DequantizeLinear", ["activation_q", "scale", "zero"], ["activation_dq"], name="activation_dq"
        ),
        helper.make_node(
            "QuantizeLinear", ["int8_weight", "scale", "zero"], ["weight_q"], name="weight_q"
        ),
        helper.make_node(
            "DequantizeLinear", ["weight_q", "scale", "zero"], ["weight_dq"], name="weight_dq"
        ),
        helper.make_node(
            "Conv", ["activation_dq", "weight_dq"], ["int8_conv_output"], name="__canonical__int8_conv"
        ),
        helper.make_node("Relu", ["int8_conv_output"], ["int8_relu"], name="int8_relu"),
        helper.make_node(
            "Conv", ["scatter_output", "fp16_weight"], ["fp16_conv_output"], name="__canonical__fp16_conv"
        ),
        helper.make_node("Relu", ["fp16_conv_output"], ["fp16_relu"], name="fp16_relu"),
        helper.make_node("Add", ["int8_relu", "fp16_relu"], ["merge_output"], name="merge_add"),
        helper.make_node("Relu", ["merge_output"], ["output"], name="merge_relu"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "synthetic_typed_qdq",
            [
                helper.make_tensor_value_info("pillar_features", TensorProto.FLOAT, [8, 4]),
                helper.make_tensor_value_info("voxel_coords", TensorProto.INT32, [8, 4]),
                helper.make_tensor_value_info("valid_voxel_mask", TensorProto.FLOAT, [8]),
                helper.make_tensor_value_info("pairwise_t_matrix", TensorProto.FLOAT, [1, 2, 2, 4, 4]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4, 3, 5])],
            [scale, zero, int8_weight, fp16_weight],
            value_info=[
                helper.make_tensor_value_info("scatter_output", TensorProto.FLOAT, [2, 4, 3, 5])
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, str(path))
    return CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="int8_conv",
                canonical_node_name="__canonical__int8_conv",
                precision_group="pg_int8",
                requested_precision="int8",
                realized_request_precision="int8",
                realized_output_precision="fp16",
                weight_initializer="int8_weight",
                onnx_op_type="Conv",
            ),
            CanonicalPrecisionEntry(
                module_path="fp16_conv",
                canonical_node_name="__canonical__fp16_conv",
                precision_group="pg_fp16",
                requested_precision="fp16",
                realized_request_precision="fp16",
                weight_initializer="fp16_weight",
                onnx_op_type="Conv",
            ),
        ],
        auxiliary_layer_precisions={"merge_add": "fp16", "merge_relu": "fp16"},
        auxiliary_layer_output_types={"merge_add": "fp16", "merge_relu": "fp16"},
    )


def _producer_by_tensor(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {str(output): node for node in model.graph.node for output in node.output}


@pytest.mark.parametrize(
    ("plugin_boundary", "expected_type"),
    (("fp16", TensorProto.FLOAT16), ("fp32", TensorProto.FLOAT)),
)
def test_typed_graph_closes_qdq_fp16_merge_and_plugin_boundary(
    tmp_path: Path,
    plugin_boundary: str,
    expected_type: int,
) -> None:
    from quantization.precision.typed_graph import apply_strongly_typed_precision_contract

    source = tmp_path / "source.onnx"
    destination = tmp_path / f"typed_{plugin_boundary}.onnx"
    mapping = _synthetic_qdq_graph(source)

    report = apply_strongly_typed_precision_contract(
        source,
        destination,
        mapping,
        plugin_boundary=plugin_boundary,
    )

    model = onnx.load(str(destination))
    nodes = {str(node.name): node for node in model.graph.node}
    producers = _producer_by_tensor(model)
    plugin = nodes["PointPillarScatterTRT"]
    plugin_input_cast = producers[str(plugin.input[0])]
    assert plugin_input_cast.op_type == "Cast"
    assert next(attribute.i for attribute in plugin_input_cast.attribute if attribute.name == "to") == expected_type

    plugin_output_types = {
        str(value.name): int(value.type.tensor_type.elem_type)
        for value in [*model.graph.value_info, *model.graph.output]
    }
    assert plugin_output_types[str(plugin.output[0])] == expected_type

    activation_q = nodes["activation_q"]
    q_input_cast = producers[str(activation_q.input[0])]
    assert q_input_cast.op_type == "Cast"
    assert next(attribute.i for attribute in q_input_cast.attribute if attribute.name == "to") == TensorProto.FLOAT

    int8_relu = nodes["int8_relu"]
    int8_output_cast = producers[str(int8_relu.input[0])]
    assert int8_output_cast.op_type == "Cast"
    assert next(attribute.i for attribute in int8_output_cast.attribute if attribute.name == "to") == TensorProto.FLOAT16

    fp16_conv = nodes["__canonical__fp16_conv"]
    assert all(producers[str(value)].op_type == "Cast" for value in fp16_conv.input[:2])
    assert all(
        next(attribute.i for attribute in producers[str(value)].attribute if attribute.name == "to")
        == TensorProto.FLOAT16
        for value in fp16_conv.input[:2]
    )

    merge = nodes["merge_add"]
    assert all(producers[str(value)].op_type == "Cast" for value in merge.input)
    assert all(
        next(attribute.i for attribute in producers[str(value)].attribute if attribute.name == "to")
        == TensorProto.FLOAT16
        for value in merge.input
    )
    assert report["canonical_precision_counts"] == {"INT8": 1, "FP16": 1, "FP32": 0}
    assert report["plugin_boundary_dtype"] == plugin_boundary.upper()
    assert report["plugin_qdq_count"] == 0
    assert report["unresolved_tensor_dtype_count"] == 0


def test_typed_graph_rejects_int8_plugin_boundary(tmp_path: Path) -> None:
    from quantization.precision.typed_graph import apply_strongly_typed_precision_contract

    source = tmp_path / "source.onnx"
    mapping = _synthetic_qdq_graph(source)
    with pytest.raises(ValueError, match="scatter_boundary_must_be_fp16_or_fp32"):
        apply_strongly_typed_precision_contract(
            source,
            tmp_path / "typed.onnx",
            mapping,
            plugin_boundary="int8",
        )


def test_typed_graph_expands_functional_mapping_and_aligns_grid_sample_inputs(
    tmp_path: Path,
) -> None:
    from quantization.precision.typed_graph import apply_strongly_typed_precision_contract

    source = tmp_path / "functional.onnx"
    destination = tmp_path / "functional_typed.onnx"
    left = numpy_helper.from_array(
        np.ones((1, 4, 3), dtype=np.float32), name="grid_left"
    )
    right = numpy_helper.from_array(
        np.ones((1, 3, 2), dtype=np.float32), name="grid_right"
    )
    add_constant = numpy_helper.from_array(
        np.asarray(1.0e-4, dtype=np.float32), name="add_constant"
    )
    compare_constant = numpy_helper.from_array(
        np.asarray(0.0, dtype=np.float32), name="compare_constant"
    )
    nodes = [
        helper.make_node(
            "PointPillarScatterTRT",
            ["pillar_features", "voxel_coords", "valid_voxel_mask", "pairwise_t_matrix"],
            ["scatter_output"],
            name="PointPillarScatterTRT",
            num_agents=0,
            height=2,
            width=2,
            plugin_version="1",
            plugin_namespace="",
        ),
        helper.make_node(
            "Cast", ["feature"], ["feature_half"], name="feature_half_cast", to=TensorProto.FLOAT16
        ),
        helper.make_node(
            "Add", ["feature_half", "add_constant"], ["feature_plus"], name="feature_add"
        ),
        helper.make_node(
            "MatMul", ["grid_left", "grid_right"], ["grid_half"], name="functional_matmul"
        ),
        helper.make_node(
            "Cast", ["grid_half"], ["grid_float"], name="legacy_grid_float_cast", to=TensorProto.FLOAT
        ),
        helper.make_node(
            "GridSample", ["feature_plus", "grid_float"], ["warped"], name="functional_grid_sample"
        ),
        helper.make_node(
            "Equal", ["warped", "compare_constant"], ["is_zero"], name="feature_equal"
        ),
        helper.make_node(
            "Where", ["is_zero", "compare_constant", "warped"], ["selected"], name="feature_where"
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "functional_grid",
            [
                helper.make_tensor_value_info("pillar_features", TensorProto.FLOAT, [8, 4]),
                helper.make_tensor_value_info("voxel_coords", TensorProto.INT32, [8, 4]),
                helper.make_tensor_value_info("valid_voxel_mask", TensorProto.FLOAT, [8]),
                helper.make_tensor_value_info("pairwise_t_matrix", TensorProto.FLOAT, [1, 2, 2, 4, 4]),
                helper.make_tensor_value_info("feature", TensorProto.FLOAT, [1, 1, 2, 2]),
            ],
            [helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1, 1, 2, 2])],
            [left, right, add_constant, compare_constant],
            value_info=[
                helper.make_tensor_value_info("scatter_output", TensorProto.FLOAT, [2, 4, 2, 2]),
                helper.make_tensor_value_info("feature_half", TensorProto.FLOAT16, [1, 1, 2, 2]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, str(source))
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="functional_affine_grid_matmul",
                canonical_node_name="__canonical__functional_matmul_group",
                precision_group="pg_functional",
                requested_precision="fp16",
                realized_request_precision="fp16",
                onnx_op_type="MatMulGroup",
                constraint_node_names=("functional_matmul",),
            )
        ]
    )

    apply_strongly_typed_precision_contract(
        source,
        destination,
        mapping,
        plugin_boundary="fp16",
    )

    typed = onnx.load(str(destination))
    nodes_by_name = {str(node.name): node for node in typed.graph.node}
    producers = _producer_by_tensor(typed)
    matmul = nodes_by_name["functional_matmul"]
    assert all(
        next(attribute.i for attribute in producers[str(value)].attribute if attribute.name == "to")
        == TensorProto.FLOAT16
        for value in matmul.input
    )
    grid_sample = nodes_by_name["functional_grid_sample"]
    feature_add = nodes_by_name["feature_add"]
    add_cast = producers[str(feature_add.input[1])]
    assert add_cast.op_type == "Cast"
    assert next(attribute.i for attribute in add_cast.attribute if attribute.name == "to") == TensorProto.FLOAT16
    grid_cast = producers[str(grid_sample.input[1])]
    assert grid_cast.op_type == "Cast"
    assert next(attribute.i for attribute in grid_cast.attribute if attribute.name == "to") == TensorProto.FLOAT16
    equal = nodes_by_name["feature_equal"]
    equal_cast = producers[str(equal.input[1])]
    assert equal_cast.op_type == "Cast"
    assert next(attribute.i for attribute in equal_cast.attribute if attribute.name == "to") == TensorProto.FLOAT16
    value_types = {
        str(value.name): int(value.type.tensor_type.elem_type)
        for value in [*typed.graph.value_info, *typed.graph.output]
    }
    assert value_types["is_zero"] == TensorProto.BOOL
    where = nodes_by_name["feature_where"]
    where_then_cast = producers[str(where.input[1])]
    assert where_then_cast.op_type == "Cast"
    assert next(attribute.i for attribute in where_then_cast.attribute if attribute.name == "to") == TensorProto.FLOAT16
    assert value_types["selected"] == TensorProto.FLOAT16
