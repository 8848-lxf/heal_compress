from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.latency_lut.rewrite_full_onnx_precision import rewrite_onnx_precision


def _toy_onnx(path: Path, node_name: str = "detection_head_cls") -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="head_weight")
    conv = helper.make_node("Conv", ["input", "head_weight"], ["output"], name=node_name)
    graph = helper.make_graph([conv], "toy", [x], [y], [w])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(path))


def _toy_add_concat_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])
    w_backbone = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="backbone_weight")
    w_head = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="head_weight")
    conv_backbone = helper.make_node("Conv", ["input", "backbone_weight"], ["backbone_y"], name="backbone_conv")
    conv_head = helper.make_node("Conv", ["input", "head_weight"], ["head_y"], name="detection_head_cls")
    add = helper.make_node("Add", ["backbone_y", "head_y"], ["add_y"], name="residual_add")
    concat = helper.make_node("Concat", ["add_y", "backbone_y"], ["output"], name="fusion_concat", axis=1)
    graph = helper.make_graph([conv_backbone, conv_head, add, concat], "toy_add", [x], [out], [w_backbone, w_head])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(path))


def _toy_grid_sample_onnx(path: Path) -> None:
    feature = helper.make_tensor_value_info("feature", TensorProto.FLOAT, [1, 1, 4, 4])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 4, 4, 2])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="backbone_weight")
    conv = helper.make_node("Conv", ["feature", "backbone_weight"], ["feature_fp16"], name="backbone_conv")
    sample = helper.make_node(
        "GridSample",
        ["feature_fp16", "grid"],
        ["output"],
        name="fusion_grid_sample",
        align_corners=0,
        mode=b"bilinear",
        padding_mode=b"zeros",
    )
    graph = helper.make_graph([conv, sample], "toy_grid", [feature, grid], [out], [w])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(path))


def test_fp16_fp32_rewrite_does_not_insert_qdq(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    report = tmp_path / "report.json"
    _toy_onnx(src)

    payload = rewrite_onnx_precision(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"detection_head": "FP32"}}},
        report,
    )

    model = onnx.load(str(dst))
    op_types = [node.op_type for node in model.graph.node]
    assert payload["success"] is True
    assert "Cast" in op_types
    assert "QuantizeLinear" not in op_types
    assert "DequantizeLinear" not in op_types
    assert json.loads(report.read_text())["uses_qdq"] is False


def test_missing_precision_override_mapping_fails(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    report = tmp_path / "report.json"
    _toy_onnx(src)

    payload = rewrite_onnx_precision(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"not_present": "FP32"}}},
        report,
    )

    assert payload["success"] is False
    assert payload["status"] == "layer_name_mapping_failed"
    assert not dst.exists()


def test_rewrite_uses_layer_mapping_and_applies_default_fp16(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    report = tmp_path / "report.json"
    mapping = tmp_path / "mapping.yaml"
    _toy_onnx(src, node_name="/cls_head/Conv")
    mapping.write_text(
        """
detection_head:
  onnx_node_patterns:
    - cls_head
""",
        encoding="utf-8",
    )

    payload = rewrite_onnx_precision(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"detection_head": "FP32"}}},
        report,
        layer_mapping=mapping,
    )

    model = onnx.load(str(dst))
    assert payload["success"] is True
    assert payload["fp32_nodes"] == ["/cls_head/Conv"]
    assert payload["unmatched_overrides"] == []
    assert "QuantizeLinear" not in [node.op_type for node in model.graph.node]


def test_rewrite_repairs_add_and_concat_dtype_mismatches(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    report = tmp_path / "report.json"
    _toy_add_concat_onnx(src)

    payload = rewrite_onnx_precision(
        src,
        dst,
        {"precision_config": {"default": "FP16", "overrides": {"detection_head": "FP32"}}},
        report,
    )

    model = onnx.load(str(dst))
    op_types = [node.op_type for node in model.graph.node]
    saved = json.loads(report.read_text())
    assert payload["success"] is True
    assert "Cast" in op_types
    assert saved["num_add_fixed"] == 1
    assert saved["num_concat_fixed"] == 1
    assert saved["remaining_dtype_mismatches"] == []
    assert saved["uses_qdq"] is False


def test_rewrite_repairs_grid_sample_dtype_mismatch(tmp_path: Path):
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    report = tmp_path / "report.json"
    _toy_grid_sample_onnx(src)

    payload = rewrite_onnx_precision(
        src,
        dst,
        {"precision_config": {"default": "FP16"}},
        report,
    )

    saved = json.loads(report.read_text())
    assert payload["success"] is True
    assert saved["num_grid_sample_fixed"] == 1
    assert saved["remaining_dtype_mismatches"] == []
