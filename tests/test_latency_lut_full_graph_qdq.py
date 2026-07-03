from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.latency_lut.insert_full_graph_qdq import insert_qdq


def _toy_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="shrink_weight")
    conv = helper.make_node("Conv", ["input", "shrink_weight"], ["output"], name="shrink_conv")
    graph = helper.make_graph([conv], "toy", [x], [y], [w])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(path))


def test_int8_qdq_insertion_adds_quantize_and_dequantize(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    report = tmp_path / "report.json"
    _toy_onnx(src)

    payload = insert_qdq(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"shrink": "INT8"}}},
        {"success": True, "scale_source": "unit_test", "units": {"shrink": {"input_scale": 0.1}}},
        report,
    )

    model = onnx.load(str(dst))
    op_types = [node.op_type for node in model.graph.node]
    assert payload["success"] is True
    assert "QuantizeLinear" in op_types
    assert "DequantizeLinear" in op_types
    assert payload["weight_scale_granularity"] == "per_channel"
    assert payload["remaining_dtype_mismatches"] == []


def test_missing_int8_mapping_fails_without_output(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    report = tmp_path / "report.json"
    _toy_onnx(src)

    payload = insert_qdq(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"missing": "INT8"}}},
        {"success": True, "scale_source": "unit_test", "units": {}},
        report,
    )

    assert payload["success"] is False
    assert payload["status"] == "int8_qdq_layer_mapping_failed"
    assert not dst.exists()
