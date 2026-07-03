from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.latency_lut.normalize_pruned_qdq_dtype_boundaries_v6 import normalize_pruned_qdq_dtype_boundaries
from tools.latency_lut.validate_onnx_dtype_closure_v6 import validate_onnx_dtype_closure


def _make_qdq_to_fp16_conv_mismatch(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 1, 4, 4])
    w_int8 = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="int8_weight")
    w_fp16 = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float16), name="fp16_weight")
    scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="scale")
    zp = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="zp")
    q_act = helper.make_node("QuantizeLinear", ["input", "scale", "zp"], ["input_q"], name="int8_act_Q")
    dq_act = helper.make_node("DequantizeLinear", ["input_q", "scale", "zp"], ["input_dq"], name="int8_act_DQ")
    q_w = helper.make_node("QuantizeLinear", ["int8_weight", "scale", "zp"], ["w_q"], name="int8_w_Q")
    dq_w = helper.make_node("DequantizeLinear", ["w_q", "scale", "zp"], ["w_dq"], name="int8_w_DQ")
    int8_conv = helper.make_node("Conv", ["input_dq", "w_dq"], ["int8_out"], name="int8_conv")
    fp16_conv = helper.make_node("Conv", ["int8_out", "fp16_weight"], ["output"], name="default_fp16_conv")
    graph = helper.make_graph([q_act, dq_act, q_w, dq_w, int8_conv, fp16_conv], "qdq_mismatch", [x], [y], [w_int8, w_fp16, scale, zp])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(path))


def _make_add_mismatch(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 1, 4, 4])
    w_fp16 = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float16), name="fp16_weight")
    w_fp32 = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="fp32_weight")
    conv16 = helper.make_node("Conv", ["input", "fp16_weight"], ["half_branch"], name="half_conv")
    conv32 = helper.make_node("Conv", ["input", "fp32_weight"], ["float_branch"], name="float_conv")
    add = helper.make_node("Add", ["half_branch", "float_branch"], ["output"], name="residual_add")
    graph = helper.make_graph([conv16, conv32, add], "add_mismatch", [x], [y], [w_fp16, w_fp32])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(path))


def test_pruned_qdq_closure_casts_float_dq_output_before_default_fp16_conv(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "closed.onnx"
    report = tmp_path / "closure.json"
    validation = tmp_path / "validation.json"
    _make_qdq_to_fp16_conv_mismatch(src)

    payload = normalize_pruned_qdq_dtype_boundaries(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"int8_conv": "INT8_QDQ"}}},
        report,
    )
    valid = validate_onnx_dtype_closure(
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"int8_conv": "INT8_QDQ"}}},
        validation,
    )

    model = onnx.load(str(dst))
    fp16_conv = next(node for node in model.graph.node if node.name == "default_fp16_conv")
    assert payload["success"] is True
    assert payload["num_cast_inserted"] >= 1
    assert any(item["node_name"] == "default_fp16_conv" for item in payload["conv_gemm_fixes"])
    assert fp16_conv.input[0] != "int8_out"
    assert valid["valid"] is True
    assert valid["num_errors"] == 0


def test_pruned_qdq_closure_aligns_add_inputs_without_breaking_qdq_pair(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "closed.onnx"
    report = tmp_path / "closure.json"
    validation = tmp_path / "validation.json"
    _make_add_mismatch(src)

    payload = normalize_pruned_qdq_dtype_boundaries(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"float_conv": "FP32"}}},
        report,
    )
    valid = validate_onnx_dtype_closure(
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"float_conv": "FP32"}}},
        validation,
    )

    assert payload["success"] is True
    assert payload["num_merge_fixed"] == 1
    assert valid["valid"] is True
    assert valid["num_errors"] == 0
