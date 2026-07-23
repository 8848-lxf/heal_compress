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
