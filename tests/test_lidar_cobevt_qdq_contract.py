from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def test_layernorm_scale_and_bias_are_cast_to_typed_activation(tmp_path):
    from search.model_families.lidar_cobevt.quantization_recipe import (
        apply_cobevt_auxiliary_typed_contract,
    )

    source = tmp_path / "source.onnx"
    output = tmp_path / "typed.onnx"
    scale = numpy_helper.from_array(np.ones(4, dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="bias")
    nodes = [
        helper.make_node(
            "Cast", ["input"], ["half_input"], name="input_half", to=TensorProto.FLOAT16
        ),
        helper.make_node(
            "LayerNormalization",
            ["half_input", "scale", "bias"],
            ["output"],
            name="fusion_norm",
            axis=-1,
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "cobevt_norm",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4])],
            [scale, bias],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, str(source))

    report = apply_cobevt_auxiliary_typed_contract(source, output)
    typed = onnx.load(str(output))
    by_name = {node.name: node for node in typed.graph.node}
    norm = by_name["fusion_norm"]

    assert report["layernorm_count"] == 1
    assert report["layernorm_parameter_cast_count"] == 2
    assert by_name[norm.input[1]].op_type == "Cast"
    assert by_name[norm.input[2]].op_type == "Cast"
    assert all(
        next(attribute.i for attribute in by_name[name].attribute if attribute.name == "to")
        == TensorProto.FLOAT16
        for name in norm.input[1:]
    )


def test_auxiliary_contract_does_not_insert_scatter_qdq(tmp_path):
    from search.model_families.lidar_cobevt.quantization_recipe import (
        apply_cobevt_auxiliary_typed_contract,
    )

    source = tmp_path / "scatter.onnx"
    output = tmp_path / "scatter_typed.onnx"
    node = helper.make_node(
        "PointPillarScatterTRT",
        ["pillar", "coords", "mask", "pairwise"],
        ["spatial"],
        name="scatter",
        domain="trt",
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "scatter",
            [
                helper.make_tensor_value_info("pillar", TensorProto.FLOAT, [8, 4]),
                helper.make_tensor_value_info("coords", TensorProto.INT32, [8, 4]),
                helper.make_tensor_value_info("mask", TensorProto.FLOAT, [8]),
                helper.make_tensor_value_info("pairwise", TensorProto.FLOAT, [1, 2, 2, 4, 4]),
            ],
            [helper.make_tensor_value_info("spatial", TensorProto.FLOAT, [2, 4, 3, 5])],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("trt", 1)],
    )
    onnx.save(model, str(source))

    report = apply_cobevt_auxiliary_typed_contract(source, output)
    typed = onnx.load(str(output))

    assert report["scatter_qdq_count"] == 0
    assert not any(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in typed.graph.node
    )


def test_attention_elementwise_bias_is_cast_to_activation_dtype(tmp_path):
    from search.model_families.lidar_cobevt.quantization_recipe import (
        apply_cobevt_auxiliary_typed_contract,
    )

    source = tmp_path / "attention_add.onnx"
    output = tmp_path / "attention_add_typed.onnx"
    nodes = [
        helper.make_node(
            "Cast", ["activation"], ["activation_half"], name="activation_half", to=TensorProto.FLOAT16
        ),
        helper.make_node(
            "Reshape", ["bias", "shape"], ["bias_reshaped"], name="bias_reshape"
        ),
        helper.make_node(
            "Add", ["activation_half", "bias_reshaped"], ["output"], name="attention_bias_add"
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "attention_add",
            [
                helper.make_tensor_value_info("activation", TensorProto.FLOAT, [1, 4]),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, str(source))

    report = apply_cobevt_auxiliary_typed_contract(source, output)
    typed = onnx.load(str(output))
    by_name = {node.name: node for node in typed.graph.node}
    add = by_name["attention_bias_add"]

    assert report["elementwise_input_cast_count"] == 1
    assert by_name[add.input[1]].op_type == "Cast"
    assert next(
        attribute.i
        for attribute in by_name[add.input[1]].attribute
        if attribute.name == "to"
    ) == TensorProto.FLOAT16


def test_attention_where_branches_are_cast_to_same_dtype(tmp_path):
    from search.model_families.lidar_cobevt.quantization_recipe import (
        apply_cobevt_auxiliary_typed_contract,
    )

    source = tmp_path / "attention_where.onnx"
    output = tmp_path / "attention_where_typed.onnx"
    nodes = [
        helper.make_node(
            "Cast", ["activation"], ["activation_half"], name="activation_half", to=TensorProto.FLOAT16
        ),
        helper.make_node(
            "Where", ["condition", "fallback", "activation_half"], ["output"], name="attention_where"
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "attention_where",
            [
                helper.make_tensor_value_info("condition", TensorProto.BOOL, [1, 4]),
                helper.make_tensor_value_info("fallback", TensorProto.FLOAT, [1, 4]),
                helper.make_tensor_value_info("activation", TensorProto.FLOAT, [1, 4]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, str(source))

    report = apply_cobevt_auxiliary_typed_contract(source, output)
    typed = onnx.load(str(output))
    by_name = {node.name: node for node in typed.graph.node}
    where = by_name["attention_where"]

    assert report["where_input_cast_count"] == 1
    assert by_name[where.input[1]].op_type == "Cast"
    assert where.input[0] == "condition"
    assert next(
        attribute.i
        for attribute in by_name[where.input[1]].attribute
        if attribute.name == "to"
    ) == TensorProto.FLOAT16


def test_backbone_concat_mixed_float_inputs_are_closed_to_fp16(tmp_path):
    from search.model_families.lidar_cobevt.quantization_recipe import (
        apply_cobevt_auxiliary_typed_contract,
    )

    source = tmp_path / "backbone_concat.onnx"
    output = tmp_path / "backbone_concat_typed.onnx"
    nodes = [
        helper.make_node(
            "Cast",
            ["branch_half_source"],
            ["branch_half"],
            name="branch_half",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "Concat",
            ["branch_float", "branch_half"],
            ["merged"],
            name="/backbone_m1/Concat",
            axis=1,
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "backbone_concat",
            [
                helper.make_tensor_value_info(
                    "branch_float", TensorProto.FLOAT, [1, 2, 3, 3]
                ),
                helper.make_tensor_value_info(
                    "branch_half_source", TensorProto.FLOAT, [1, 2, 3, 3]
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "merged", TensorProto.FLOAT16, [1, 4, 3, 3]
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, str(source))

    report = apply_cobevt_auxiliary_typed_contract(source, output)
    typed = onnx.load(str(output))
    by_name = {node.name: node for node in typed.graph.node}
    concat = by_name["/backbone_m1/Concat"]

    assert report["concat_input_cast_count"] == 1
    assert report["concat_contract_dtype"] == "FP16"
    assert by_name[concat.input[0]].op_type == "Cast"
    assert next(
        attribute.i
        for attribute in by_name[concat.input[0]].attribute
        if attribute.name == "to"
    ) == TensorProto.FLOAT16
