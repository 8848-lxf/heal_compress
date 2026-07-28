from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np


def test_stage2_fixed_k_requires_exact_audited_contract(tmp_path) -> None:
    import pytest

    from scripts.smoke_transformer_stage2_engine import _resolve_fixed_k

    contract = tmp_path / "fixed_k_contract.json"
    contract.write_text(
        json.dumps(
            {
                "fixed_k": 128,
                "contract_hash": "fixed-k-hash",
                "overflow_count": 0,
            }
        ),
        encoding="utf-8",
    )
    fixed_k, provenance = _resolve_fixed_k(96, 128, contract)
    assert fixed_k == 128
    assert provenance["source"] == "audited_fixed_k_contract"
    assert provenance["contract_hash"] == "fixed-k-hash"
    assert provenance["overflow_count"] == 0
    with pytest.raises(RuntimeError, match="stage2_fixed_k_contract_mismatch"):
        _resolve_fixed_k(96, 127, contract)
    with pytest.raises(RuntimeError, match="stage2_fixed_k_below_observed"):
        _resolve_fixed_k(129, 128, contract)


def _attention_onnx(path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4])
    weight = numpy_helper.from_array(
        np.arange(48, dtype=np.float32).reshape(4, 12), "qkv.weight"
    )
    nodes = [
        helper.make_node("MatMul", ["x", "qkv.weight"], ["qkv"], name="qkv_canonical"),
        helper.make_node(
            "Split",
            ["qkv"],
            ["q", "k", "v"],
            name="split",
            axis=2,
            split=[4, 4, 4],
        ),
        helper.make_node("Transpose", ["k"], ["kt"], name="kt", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["score"], name="qk"),
        helper.make_node("Softmax", ["score"], ["prob"], name="softmax", axis=-1),
        helper.make_node("MatMul", ["prob", "v"], ["context"], name="av"),
        # A later attention-like sequence is reachable through the residual
        # stream but must not be attributed to the selected QKV instance.
        helper.make_node(
            "Transpose", ["context"], ["context_t"], name="later_kt", perm=[0, 2, 1]
        ),
        helper.make_node(
            "MatMul", ["context", "context_t"], ["later_score"], name="later_qk"
        ),
        helper.make_node(
            "Softmax", ["later_score"], ["later_prob"], name="later_softmax", axis=-1
        ),
        helper.make_node("MatMul", ["later_prob", "context"], ["y"], name="later_av"),
    ]
    graph = helper.make_graph(nodes, "attention", [x], [y], [weight])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 11)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_transformer_mapping_is_strict_and_onnx_qk_softmax_audit_is_fp32(tmp_path) -> None:
    from search.stage2.transformer_precision_export import (
        audit_onnx_attention_fp32_contract,
        build_transformer_precision_mapping,
    )

    origin = SimpleNamespace(
        entries=(
            SimpleNamespace(
                module_path="attn.qkv",
                canonical_node_name="qkv_canonical",
                original_node_name="qkv",
                weight_initializer="qkv.weight",
                onnx_op_type="MatMul",
                call_index=0,
                graph_index=0,
            ),
        ),
        origin_map_hash="origin-hash",
    )
    mapping = build_transformer_precision_mapping(
        origin, {"attn.qkv": "fp32"}, profile_id="toy"
    )
    assert mapping.entries[0].requested_precision == "fp32"
    model_path = tmp_path / "attention.onnx"
    _attention_onnx(model_path)
    audit = audit_onnx_attention_fp32_contract(
        model_path, qkv_canonical_node_names=("qkv_canonical",)
    )
    assert audit["passed"]
    assert audit["qk_nodes"][0]["node_name"] == "qk"
    assert audit["softmax_nodes"][0]["node_name"] == "softmax"
    assert audit["av_nodes"][0]["node_name"] == "av"
    assert len(audit["softmax_nodes"]) == 1
    assert len(audit["av_nodes"]) == 1


def test_trt_attention_contract_requires_exact_fp32_inspector_evidence() -> None:
    from search.stage2.transformer_precision_export import (
        audit_trt_attention_fp32_contract,
    )

    contract = {
        "qk_nodes": [{"node_name": "qk", "op_type": "MatMul"}],
        "softmax_nodes": [{"node_name": "softmax", "op_type": "Softmax"}],
    }
    passed = audit_trt_attention_fp32_contract(
        [
            {"Name": "qk", "Precision": "FP32", "LayerType": "MatrixMultiply"},
            {"Name": "softmax", "Precision": "FP32", "LayerType": "Softmax"},
        ],
        contract,
    )
    assert passed["passed"]
    failed = audit_trt_attention_fp32_contract(
        [
            {"Name": "qk", "Precision": "FP16", "LayerType": "MatrixMultiply"},
            {"Name": "softmax", "Precision": "FP32", "LayerType": "Softmax"},
        ],
        contract,
    )
    assert not failed["passed"]
    assert not failed["qk_fp32_protected"]


def test_trt_softmax_contract_ignores_bool_mask_but_rejects_half_logits() -> None:
    from search.stage2.transformer_precision_export import (
        audit_trt_attention_fp32_contract,
    )

    contract = {
        "qk_nodes": [],
        "softmax_nodes": [{"node_name": "softmax", "op_type": "Softmax"}],
    }
    fused = {
        "Name": "fused_mask_softmax",
        "Precision": "FP16",
        "LayerType": "kgen",
        "Metadata": "[ONNX Layer: softmax]",
        "Inputs": [
            {"Format/Datatype": "Bool"},
            {"Format/Datatype": "Float"},
        ],
        "Outputs": [{"Format/Datatype": "Float"}],
    }
    passed = audit_trt_attention_fp32_contract([fused], contract)
    assert passed["passed"]
    assert passed["softmax_compute_fp32"]
    assert passed["evidence"][0]["numeric_input_formats"] == ["float"]
    assert passed["evidence"][0]["ignored_control_input_formats"] == ["bool"]

    fused["Inputs"][1]["Format/Datatype"] = "Half"
    failed = audit_trt_attention_fp32_contract([fused], contract)
    assert not failed["passed"]
    assert not failed["softmax_compute_fp32"]


def test_fused_int8_gemm_is_weighted_compute_but_pointwise_fusion_is_not() -> None:
    from quantization.config import TensorRTValidationConfig
    from quantization.tensorrt.layer_info import is_weighted_compute_layer
    from quantization.tensorrt.precision_checker import validate_precision_realization
    from quantization.tensorrt.structure_checker import validate_engine_structure
    from quantization.types import (
        CanonicalPrecisionEntry,
        CanonicalPrecisionMappingResult,
    )

    canonical = "__canonical__ffn__MatMul__call00000"
    weighted = {
        "Name": "__myl_FcCastAddErf",
        "LayerType": "fusion",
        "TacticName": "sm80_xmma_gemm_i8i8_i8i32_f32_tn",
        "Metadata": f"[ONNX Layer: {canonical}]",
        "Inputs": [{"Format/Datatype": "Int8"}],
        "Outputs": [{"Format/Datatype": "Half"}],
    }
    pointwise = {
        "Name": "__myl_AddRelu",
        "LayerType": "fusion",
        "TacticName": "pointwise_vectorized",
        "Metadata": "[ONNX Layer: activation]",
    }
    assert is_weighted_compute_layer(weighted)
    assert not is_weighted_compute_layer(pointwise)

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                "ffn",
                canonical,
                "ffn_group",
                "int8",
                "int8",
                weight_initializer="ffn.weight",
                onnx_op_type="MatMul",
            )
        ]
    )
    precision = validate_precision_realization([weighted, pointwise], mapping)
    structure = validate_engine_structure(
        [weighted, pointwise],
        mapping,
        config=TensorRTValidationConfig(require_physical_snapshot_v2=False),
    )
    assert precision.passed and precision.realized_int8_count == 1
    assert structure.passed and structure.matched_canonical_count == 1


def test_attention_rewrite_keeps_softmax_and_av_fp32(tmp_path) -> None:
    from scripts.run_v2xvit_greedy005_stage2 import _force_attention_fp32_contract
    from search.stage2.transformer_precision_export import (
        audit_onnx_attention_fp32_contract,
    )

    path = tmp_path / "attention.onnx"
    _attention_onnx(path)
    before = audit_onnx_attention_fp32_contract(
        path, qkv_canonical_node_names=("qkv_canonical",)
    )
    report = _force_attention_fp32_contract(path, before)
    after = audit_onnx_attention_fp32_contract(
        path, qkv_canonical_node_names=("qkv_canonical",)
    )
    assert report["roles"]["av"] == 1
    softmax = next(row for row in after["softmax_nodes"] if row["node_name"] == "softmax")
    av = next(row for row in after["av_nodes"] if row["node_name"] == "av")
    assert softmax["output_dtypes"] == ["FLOAT"]
    assert av["input_dtypes"] == ["FLOAT", "FLOAT"]
    assert av["output_dtypes"] == ["FLOAT"]


def test_attention_rewrite_promotes_fixed_layernorm_boundary_to_fp32(tmp_path) -> None:
    import onnx
    from onnx import TensorProto, helper

    from scripts.run_v2xvit_greedy005_stage2 import _force_attention_fp32_contract

    path = tmp_path / "layernorm_half.onnx"
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4]),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [4]),
        helper.make_tensor_value_info("bias", TensorProto.FLOAT16, [4]),
    ]
    node = helper.make_node(
        "LayerNormalization",
        ["x", "scale", "bias"],
        ["y"],
        name="/layers.0.1/norm/LayerNormalization",
        axis=-1,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "layernorm_fp32_contract",
            inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save(model, path)
    report = _force_attention_fp32_contract(
        path, {"qk_nodes": [], "softmax_nodes": [], "av_nodes": []}
    )
    rewritten = onnx.shape_inference.infer_shapes(onnx.load(path))
    layernorm = next(
        row for row in rewritten.graph.node if row.op_type == "LayerNormalization"
    )
    assert report["roles"]["layernorm"] == 1
    assert all("layernorm_fp32_input" in value for value in layernorm.input)
    assert all("layernorm_raw" in value for value in layernorm.output)
