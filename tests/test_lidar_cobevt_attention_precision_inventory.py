from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _typed_projection(path: Path) -> None:
    weight = numpy_helper.from_array(np.eye(4, dtype=np.float16), name="weight")
    nodes = [
        helper.make_node(
            "Cast", ["input"], ["half_input"], name="input_to_half", to=TensorProto.FLOAT16
        ),
        helper.make_node(
            "MatMul", ["half_input", "weight"], ["q_raw"], name="q_canonical"
        ),
        helper.make_node(
            "Cast", ["q_raw"], ["q"], name="q_to_float", to=TensorProto.FLOAT
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "projection",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 4])],
            [weight],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, str(path))


def _boundary_report() -> dict:
    return {
        "profile": {
            "profile_name": "A1_qkv_projection_fp16_core_fp32",
            "profile_hash": "profile-hash",
        },
        "node_records": [
            {
                "attention_kind": "window",
                "block_id": "layers.0.window_attention",
                "compute_dtype": "FP16",
                "input_cast_nodes": ["input_to_half"],
                "input_tensors_after": ["half_input", "weight"],
                "input_tensors_before": ["input", "weight"],
                "node_name": "q_canonical",
                "op_type": "MatMul",
                "output_cast_nodes": ["q_to_float"],
                "output_dtype": "FP32",
                "output_tensors_after": ["q_raw"],
                "output_tensors_before": ["q"],
                "role": "q_projection",
            }
        ],
    }


def test_inventory_joins_explicit_casts_with_realized_fused_gemm(tmp_path: Path):
    from search.reporting.cobevt_attention_precision_inventory import (
        build_attention_precision_inventory,
    )

    typed = tmp_path / "typed.onnx"
    layer_info = tmp_path / "layers.json"
    _typed_projection(typed)
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "fused_qkv_gemm",
                        "LayerType": "gemm",
                        "Metadata": (
                            "[ONNX Layer: q_canonical]"
                            "[ONNX Layer: k_canonical]"
                            "[ONNX Layer: v_canonical]"
                        ),
                        "Inputs": [{"Format/Datatype": "Half"}],
                        "Outputs": [{"Format/Datatype": "Float"}],
                        "TacticName": "sm80_xmma_gemm_f16f16_f16f16_f16_nn",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = build_attention_precision_inventory(
        typed, _boundary_report(), layer_info
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["module_path"] == "fusion_net.layers.0.window_attention.fn.q_proj"
    assert row["requested_dtype"] == "FP16"
    assert row["onnx_input_dtypes"] == ["FP16", "FP16"]
    assert row["onnx_output_dtypes"] == ["FP16"]
    assert row["explicit_input_cast"] is True
    assert row["explicit_output_cast"] is True
    assert row["realized_input_dtypes"] == ["FP16"]
    assert row["realized_output_dtypes"] == ["FP32"]
    assert row["fused"] is True
    assert row["realization_status"] == "matched"


def test_inventory_fails_closed_when_engine_layer_is_unmapped(tmp_path: Path):
    from search.reporting.cobevt_attention_precision_inventory import (
        build_attention_precision_inventory,
    )

    typed = tmp_path / "typed.onnx"
    layer_info = tmp_path / "layers.json"
    _typed_projection(typed)
    layer_info.write_text(json.dumps({"Layers": []}), encoding="utf-8")

    rows = build_attention_precision_inventory(
        typed, _boundary_report(), layer_info
    )

    assert rows[0]["realization_status"] == "unresolved"
    assert rows[0]["realization_failure_reason"] == "engine_layer_not_found"


def test_inventory_marks_fused_downstream_output_as_unexposed_not_fallback(
    tmp_path: Path,
):
    from search.reporting.cobevt_attention_precision_inventory import (
        build_attention_precision_inventory,
    )

    typed = tmp_path / "typed.onnx"
    layer_info = tmp_path / "layers.json"
    _typed_projection(typed)
    report = _boundary_report()
    report["node_records"][0]["output_cast_nodes"] = []
    report["node_records"][0]["output_dtype"] = "FP16"
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "fused_projection_and_fp32_consumer",
                        "LayerType": "gemm",
                        "Metadata": (
                            "[ONNX Layer: q_canonical]"
                            "[ONNX Layer: downstream_fp32]"
                        ),
                        "Inputs": [{"Format/Datatype": "Half"}],
                        "Outputs": [{"Format/Datatype": "Float"}],
                        "TacticName": "sm80_xmma_gemm_f16f16_f16f16_f16_nn",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = build_attention_precision_inventory(typed, report, layer_info)[0]

    assert row["realization_status"] == "matched_fused_output_unexposed"
    assert row["realization_failure_reason"] == ""


def test_inventory_accepts_fp32_softmax_hidden_behind_fused_upstream_cast(
    tmp_path: Path,
):
    from search.reporting.cobevt_attention_precision_inventory import (
        build_attention_precision_inventory,
    )

    typed = tmp_path / "typed.onnx"
    layer_info = tmp_path / "layers.json"
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Cast", ["half_logits"], ["float_logits"], name="qk_to_float", to=TensorProto.FLOAT
                ),
                helper.make_node(
                    "Softmax", ["float_logits"], ["probability"], name="softmax", axis=-1
                ),
            ],
            "softmax_island",
            [helper.make_tensor_value_info("half_logits", TensorProto.FLOAT16, [1, 4])],
            [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, str(typed))
    report = {
        "node_records": [
            {
                "block_id": "layers.0.window_attention",
                "compute_dtype": "FP32",
                "input_cast_nodes": [],
                "input_tensors_after": ["float_logits"],
                "input_tensors_before": ["float_logits"],
                "node_name": "softmax",
                "op_type": "Softmax",
                "output_cast_nodes": [],
                "output_dtype": "FP32",
                "output_tensors_after": ["probability"],
                "output_tensors_before": ["probability"],
                "role": "softmax",
            }
        ]
    }
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "fused_qk_cast_softmax",
                        "LayerType": "kgen",
                        "Metadata": "[ONNX Layer: qk_to_float][ONNX Layer: softmax]",
                        "Inputs": [{"Format/Datatype": "Half"}],
                        "Outputs": [{"Format/Datatype": "Float"}],
                        "TacticName": "fused_softmax",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = build_attention_precision_inventory(typed, report, layer_info)[0]

    assert row["onnx_input_dtypes"] == ["FP32"]
    assert row["realized_input_dtypes"] == ["FP16"]
    assert row["realized_output_dtypes"] == ["FP32"]
    assert row["realization_status"] == "matched_fused_input_unexposed"
    assert row["realization_failure_reason"] == ""


def test_inventory_unions_split_fused_softmax_realization(tmp_path: Path):
    from search.reporting.cobevt_attention_precision_inventory import (
        build_attention_precision_inventory,
    )

    typed = tmp_path / "typed.onnx"
    layer_info = tmp_path / "layers.json"
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Cast", ["logits"], ["half_logits"], name="to_half", to=TensorProto.FLOAT16
                ),
                helper.make_node(
                    "Softmax", ["half_logits"], ["half_probability"], name="softmax", axis=-1
                ),
                helper.make_node(
                    "Cast", ["half_probability"], ["probability"], name="to_float", to=TensorProto.FLOAT
                ),
            ],
            "split_softmax",
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, str(typed))
    report = {
        "node_records": [
            {
                "block_id": "layers.0.window_attention",
                "compute_dtype": "FP16",
                "input_cast_nodes": ["to_half"],
                "input_tensors_after": ["half_logits"],
                "input_tensors_before": ["logits"],
                "node_name": "softmax",
                "op_type": "Softmax",
                "output_cast_nodes": ["to_float"],
                "output_dtype": "FP32",
                "output_tensors_after": ["half_probability"],
                "output_tensors_before": ["probability"],
                "role": "softmax",
            }
        ]
    }
    metadata = "[ONNX Layer: softmax]"
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "softmax_front",
                        "LayerType": "kgen",
                        "Metadata": "[ONNX Layer: to_half]" + metadata,
                        "Inputs": [{"Format/Datatype": "Float"}],
                        "Outputs": [{"Format/Datatype": "Half"}],
                        "TacticName": "softmax_front",
                    },
                    {
                        "Name": "softmax_back",
                        "LayerType": "kgen",
                        "Metadata": metadata + "[ONNX Layer: to_float]",
                        "Inputs": [{"Format/Datatype": "Half"}],
                        "Outputs": [{"Format/Datatype": "Float"}],
                        "TacticName": "softmax_back",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    row = build_attention_precision_inventory(typed, report, layer_info)[0]

    assert row["realized_input_dtypes"] == ["FP16", "FP32"]
    assert row["realized_output_dtypes"] == ["FP16", "FP32"]
    assert row["realization_status"] == "matched"


def test_inventory_separates_compute_output_from_explicit_public_output_cast(
    tmp_path: Path,
):
    from search.reporting.cobevt_attention_precision_inventory import (
        build_attention_precision_inventory,
    )

    typed = tmp_path / "typed.onnx"
    layer_info = tmp_path / "layers.json"
    _typed_projection(typed)
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "q_gemm",
                        "LayerType": "gemm",
                        "Metadata": "[ONNX Layer: q_canonical]",
                        "Inputs": [{"Format/Datatype": "Half"}],
                        "Outputs": [{"Format/Datatype": "Half"}],
                        "TacticName": "sm80_xmma_gemm_f16f16_f16f16_f16_nn",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = build_attention_precision_inventory(
        typed, _boundary_report(), layer_info
    )[0]

    assert row["requested_dtype"] == "FP16"
    assert row["realized_output_dtypes"] == ["FP16"]
    assert row["explicit_output_cast"] is True
    assert row["realization_status"] == "matched_separate_output_cast"
    assert row["realization_failure_reason"] == ""


def test_inventory_writer_emits_machine_and_block_markdown(tmp_path: Path):
    from search.reporting.cobevt_attention_precision_inventory import (
        write_attention_precision_inventory,
    )

    rows = [
        {
            "block_id": "layers.0.window_attention",
            "role": "softmax",
            "node_name": "/layers.0/window_attention/fn/attend/Softmax",
            "op_type": "Softmax",
            "requested_dtype": "FP16",
            "onnx_input_dtypes": ["FP16"],
            "onnx_output_dtypes": ["FP16"],
            "realized_input_dtypes": ["FP16"],
            "realized_output_dtypes": ["FP32"],
            "fused": True,
            "realization_status": "matched",
        }
    ]

    result = write_attention_precision_inventory(
        rows,
        tmp_path / "attention_precision_inventory.json",
        tmp_path / "attention_precision_inventory.md",
    )

    assert result["row_count"] == 1
    assert json.loads((tmp_path / "attention_precision_inventory.json").read_text()) == rows
    markdown = (tmp_path / "attention_precision_inventory.md").read_text()
    assert "layers.0.window_attention" in markdown
    assert "softmax" in markdown
    assert "FP16" in markdown
