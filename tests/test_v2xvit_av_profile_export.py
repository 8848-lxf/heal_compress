from __future__ import annotations

import json


def _model(path):
    import onnx
    from onnx import TensorProto, helper

    nodes = []
    inputs = []
    outputs = []
    audit = {"av_nodes": []}
    for index in range(12):
        left = f"p{index}"
        right = f"v{index}"
        output = f"o{index}"
        name = f"/attention.{index}/Einsum_1"
        inputs.extend([
            helper.make_tensor_value_info(left, TensorProto.FLOAT, [1, 2, 2]),
            helper.make_tensor_value_info(right, TensorProto.FLOAT, [1, 2, 2]),
        ])
        outputs.append(helper.make_tensor_value_info(output, TensorProto.FLOAT16, [1, 2, 2]))
        nodes.append(helper.make_node("MatMul", [left, right], [output], name=name))
        audit["av_nodes"].append({"node_name": name})
    graph = helper.make_graph(nodes, "av", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.save(model, path)
    return audit


def test_av16_rewrite_inserts_explicit_operand_casts(tmp_path) -> None:
    import onnx

    from search.stage2.v2xvit_av_profile_export import rewrite_onnx_av_profile

    path = tmp_path / "av16.onnx"
    audit = _model(path)
    report = rewrite_onnx_av_profile(path, audit, profile="AV16")
    graph = onnx.load(path)
    assert report["rewritten_node_count"] == 12
    assert report["qdq_pair_count"] == 0
    assert sum(node.op_type == "Cast" for node in graph.graph.node) == 36


def test_av8_rewrite_has_two_qdq_pairs_per_av(tmp_path) -> None:
    import onnx

    from search.stage2.v2xvit_av_profile_export import rewrite_onnx_av_profile

    path = tmp_path / "av8.onnx"
    audit = _model(path)
    scales = {row["node_name"]: (1.0 / 127.0, 0.05) for row in audit["av_nodes"]}
    report = rewrite_onnx_av_profile(path, audit, profile="AV8", operand_scales=scales)
    graph = onnx.load(path)
    assert report["qdq_pair_count"] == 24
    assert sum(node.op_type == "QuantizeLinear" for node in graph.graph.node) == 24
    assert sum(node.op_type == "DequantizeLinear" for node in graph.graph.node) == 24


def test_av8_trt_audit_rejects_fp16_tactic() -> None:
    from search.stage2.v2xvit_av_profile_export import audit_trt_av_profile

    audit = {"av_nodes": [{"node_name": f"/attention.{index}/Einsum_1"} for index in range(12)]}
    layers = [
        {
            "Name": f"av{index}",
            "LayerType": "gemm",
            "Metadata": f"[ONNX Layer: /attention.{index}/Einsum_1]",
            "Inputs": [{"Format/Datatype": "Half"}, {"Format/Datatype": "Half"}],
            "Outputs": [{"Format/Datatype": "Half"}],
            "TacticName": "fp16_gemm",
        }
        for index in range(12)
    ]
    result = audit_trt_av_profile(layers, audit, profile="AV8")
    assert result["passed"] is False
    assert result["fallback_count"] == 12
