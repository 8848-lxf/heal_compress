from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _functional_onnx(path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    fp32 = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    fp16 = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)
    inputs = [
        fp32("x", [1, 2, 4]),
        fp32("ln_scale", [4]),
        fp32("ln_bias", [4]),
        fp16("half_a", [1, 2, 4]),
        fp16("half_b", [1, 2, 4]),
    ]
    weights = [
        numpy_helper.from_array(np.eye(4, dtype=np.float32), name=name)
        for name in ("q.weight", "k.weight", "v.weight", "ffn2.weight")
    ]
    nodes = [
        helper.make_node("MatMul", ["x", "q.weight"], ["q"], name="q_canonical"),
        helper.make_node("MatMul", ["x", "k.weight"], ["k"], name="k_canonical"),
        helper.make_node("MatMul", ["x", "v.weight"], ["v"], name="v_canonical"),
        helper.make_node("Transpose", ["k"], ["kt"], name="kt", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["score"], name="qk"),
        helper.make_node("Softmax", ["score"], ["prob"], name="softmax", axis=-1),
        helper.make_node("MatMul", ["prob", "v"], ["av"], name="av"),
        helper.make_node(
            "LayerNormalization",
            ["x", "ln_scale", "ln_bias"],
            ["ln"],
            name="/layers.0.0/layers.0.0/norm/LayerNormalization",
            axis=-1,
        ),
        helper.make_node("Add", ["half_a", "half_b"], ["res_att"], name="/layers.0.0/Add"),
        helper.make_node(
            "Add", ["half_a", "half_b"], ["merge"],
            name="/layers.0.0/layers.0.1/fn/split_attn/Add_3",
        ),
        helper.make_node("QuantizeLinear", ["x", "qscale", "qzero"], ["xq"], name="ffn_act_q"),
        helper.make_node("DequantizeLinear", ["xq", "qscale", "qzero"], ["xdq"], name="ffn_act_dq"),
        helper.make_node("MatMul", ["xdq", "ffn2.weight"], ["ffn"], name="ffn2_canonical"),
        helper.make_node("Cast", ["ffn"], ["ffn16"], name="ffn_to_half", to=TensorProto.FLOAT16),
        helper.make_node("Add", ["ffn16", "half_a"], ["output"], name="/Add_2"),
    ]
    graph = helper.make_graph(
        nodes,
        "functional_closure",
        inputs,
        [fp16("output", [1, 2, 4])],
        initializer=weights
        + [
            numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), name="qscale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="qzero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _unit(unit_id, role, state, owner, **metadata):
    return SimpleNamespace(
        unit_id=unit_id,
        role=role,
        default_state=state,
        activation_only=role not in {"ffn2"},
        metadata={"functional_owner": owner, **metadata},
    )


def test_v2xvit_functional_precision_mapping_is_complete_and_inspector_exact(tmp_path) -> None:
    from search.stage2.v2xvit_functional_precision import (
        audit_trt_v2xvit_functional_precision,
        build_v2xvit_functional_onnx_mapping,
    )

    path = tmp_path / "functional.onnx"
    _functional_onnx(path)
    attention_path = "fusion_net.encoder.layers.0.0.layers.0.0.fn"
    ffn_path = "fusion_net.encoder.layers.0.1.fn"
    units = (
        _unit(f"transformer_precision::{attention_path}::qk_matmul", "qk_matmul", "A32", attention_path),
        _unit(f"transformer_precision::{attention_path}::softmax", "softmax", "A32", attention_path),
        _unit(f"transformer_precision::{attention_path}::av", "av_matmul", "A32", attention_path),
        _unit(
            "transformer_precision::attention_residual", "residual_add", "A16", attention_path,
            boundary_kind="attention_residual_add", attention_adapter="v2xvit_hgt",
        ),
        _unit(
            "transformer_precision::window_merge", "attention_merge", "A16",
            "fusion_net.encoder.layers.0.0.layers.0.1.fn",
            boundary_kind="window_family_merge", attention_adapter="v2xvit_window",
        ),
        _unit(
            "transformer_precision::fusion_net.encoder.layers.0.0.layers.0.0.norm::layernorm",
            "layernorm", "A32", "fusion_net.encoder.layers.0.0.layers.0.0.norm",
            boundary_kind="layernorm",
        ),
        _unit(f"transformer_precision::{ffn_path}::ffn2", "ffn2", "W8A8", ffn_path),
        _unit(
            "transformer_precision::ffn_residual", "residual_add", "A16", ffn_path,
            boundary_kind="ffn_residual_add",
        ),
    )
    attention = SimpleNamespace(
        module_path=attention_path,
        q_projection_paths=("attn.q",),
        k_projection_paths=("attn.k",),
        v_projection_paths=("attn.v",),
    )
    ffn = SimpleNamespace(
        module_path=ffn_path,
        ffn_type="standard",
        second_projection_path="ffn.fc2",
        down_projection_path="",
    )
    origin = SimpleNamespace(
        entries=(
            SimpleNamespace(module_path="attn.q", canonical_node_name="q_canonical"),
            SimpleNamespace(module_path="attn.k", canonical_node_name="k_canonical"),
            SimpleNamespace(module_path="attn.v", canonical_node_name="v_canonical"),
            SimpleNamespace(module_path="ffn.fc2", canonical_node_name="ffn2_canonical"),
        )
    )
    requested = {unit.unit_id: unit.default_state for unit in units}
    onnx_mapping = build_v2xvit_functional_onnx_mapping(
        path,
        origin_map=origin,
        precision_units=units,
        attention_instances=(attention,),
        ffn_instances=(ffn,),
        requested_states=requested,
    )
    assert onnx_mapping["passed"]
    assert not onnx_mapping["missing_unit_ids"]
    assert onnx_mapping["ffn_activation_binding_count"] == 1

    layer_rows = []
    for row in onnx_mapping["rows"]:
        requested_precision = row["requested_compute_precision"]
        token = {"FP32": "Float", "FP16": "Half", "INT8": "Int8"}[requested_precision]
        layer_rows.append(
            {
                "Name": row["onnx_node"],
                "Metadata": f"[ONNX Layer: {row['onnx_node']}]",
                "LayerType": "MatrixMultiply" if row["role"] in {"qk_matmul", "av_matmul", "ffn_activation_input"} else row["onnx_op_type"],
                "Precision": requested_precision,
                "Inputs": [{"Format/Datatype": token}],
                "Outputs": [{"Format/Datatype": token}],
            }
        )
    trt = audit_trt_v2xvit_functional_precision(layer_rows, onnx_mapping)
    assert trt["passed"]
    assert trt["unmapped_count"] == trt["conflict_count"] == trt["fallback_count"] == 0


def test_v2xvit_functional_precision_mapping_fails_closed_on_unmapped_trt_layer(tmp_path) -> None:
    from search.stage2.v2xvit_functional_precision import (
        audit_trt_v2xvit_functional_precision,
    )

    mapping = {
        "rows": [
            {
                "unit_id": "softmax",
                "role": "softmax",
                "onnx_node": "softmax",
                "onnx_op_type": "Softmax",
                "requested_compute_precision": "FP32",
                "requested_output_precision": "FP32",
                "input_types": ["FP32"],
                "output_types": ["FP32"],
                "onnx_contract_exact": True,
            }
        ]
    }
    report = audit_trt_v2xvit_functional_precision([], mapping)
    assert not report["passed"]
    assert report["unmapped_count"] == report["conflict_count"] == 1
