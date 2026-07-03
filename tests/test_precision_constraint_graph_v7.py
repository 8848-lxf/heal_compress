from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from tools.latency_lut.audit_precision_constraint_graph_v7 import build_precision_constraint_components


def test_residual_add_and_concat_create_precision_components(tmp_path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [4, 4], [0.0] * 16)
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [4, 4], [0.0] * 16)
    gemm1 = helper.make_node("Gemm", ["x", "w1"], ["a"], name="backbone.branch1.Gemm")
    gemm2 = helper.make_node("Gemm", ["y", "w2"], ["b"], name="backbone.branch2.Gemm")
    add = helper.make_node("Add", ["a", "b"], ["sum"], name="backbone.residual.Add")
    concat = helper.make_node("Concat", ["a", "b"], ["cat"], name="backbone.concat", axis=1)
    out = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 4])
    model = helper.make_model(helper.make_graph([gemm1, gemm2, add, concat], "g", [x, y], [out], [w1, w2]))
    path = tmp_path / "m.onnx"
    onnx.save(model, path)

    comps = build_precision_constraint_components(path)
    types = {c["component_type"] for c in comps}
    assert "residual_add" in types
    assert "concat" in types
    assert all(c["must_share_precision"] for c in comps)
