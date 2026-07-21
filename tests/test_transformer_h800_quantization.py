from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from search.model_families.transformer.accumulator_contract import (
    build_accumulator_evidence,
    infer_accumulator_from_layer,
)
from search.orchestration.lidar_transformer_h800_accumulator import (
    EXPERIMENTS,
    build_matrix,
)
from search.orchestration.lidar_transformer_h800_smoothquant import _qdq_inventory
from search.model_families.transformer.canonical_roles import classify_weighted_module
from search.model_families.transformer.fp8_profiles import (
    capability_from_probe,
    require_deployable,
)
from search.model_families.transformer.precision_contract import (
    precision_profiles,
    qk_accumulator_variant,
)
from search.model_families.transformer.search_space import (
    build_profile_library,
    build_search_space,
)
from search.model_families.transformer.realized_precision import (
    audit_realized_precision,
    normalize_trt_dtype,
)
from search.model_families.transformer.qdq_adjacency import (
    restore_projection_qdq_adjacency,
    validate_projection_qdq_adjacency,
)
from quantization.tensorrt.layer_info import precision_name
from search.model_families.transformer.smoothquant_profiles import (
    choose_alpha,
    selective_smoothquant_config,
    smoothquant_profiles,
)
from search.integration.runtime_environment import (
    configure_modelopt_inprocess,
    runtime_cuda_index_for_physical,
)
from search.reporting.transformer_h800_quantization import _profile_semantics


def test_physical_gpu_is_not_silently_replaced_by_cuda_zero(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert runtime_cuda_index_for_physical(6) == 6
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")
    assert runtime_cuda_index_for_physical(6) == 1
    with pytest.raises(RuntimeError, match="physical_gpu_not_visible"):
        runtime_cuda_index_for_physical(4)


def test_modelopt_inprocess_toolchain_rejects_system_compilers(
    tmp_path, monkeypatch
) -> None:
    toolchain = configure_modelopt_inprocess(
        output_root=tmp_path, cache_namespace="unit"
    )
    prefix = toolchain["conda_prefix"]
    assert all(toolchain[name].startswith(prefix) for name in ("nvcc", "gcc", "g++", "ninja"))
    assert toolchain["nvcc"] != "/usr/local/cuda/bin/nvcc"
    assert toolchain["torch_cuda_arch_list"] == "9.0"
    assert toolchain["cpu_fallback"] == "false"


def test_model_family_role_mapping_is_separate() -> None:
    cobevt = classify_weighted_module(
        "lidar_cobevt", "fusion_net.layers.0.window_ffd.fn.net.0"
    )
    v2xvit = classify_weighted_module(
        "lidar_v2xvit", "fusion_net.fusion_net.encoder.layers.0.1.fn.net.0"
    )
    assert cobevt.canonical_role == v2xvit.canonical_role == "ffn1"
    wrong_family = classify_weighted_module(
        "lidar_cobevt", "fusion_net.fusion_net.encoder.layers.0.1.fn.net.0"
    )
    assert wrong_family.canonical_role != "ffn1"


def test_f3_q_and_k_boundaries_are_symmetric() -> None:
    profile = precision_profiles()["B3_F3"]
    for role in ("q_projection", "k_projection", "fused_qkv_projection"):
        assert profile[role].operand_precision == "FP16"
        assert profile[role].output_precision == "FP32"
    assert profile["qk_matmul"].operand_precision == "FP32"
    assert profile["qk_matmul"].accumulator_precision == "FP32"


def test_onnx_float16_dtype_is_not_misparsed_as_fp32() -> None:
    assert normalize_trt_dtype("FLOAT16") == "FP16"
    assert normalize_trt_dtype("TensorRT Half") == "FP16"
    assert normalize_trt_dtype("FLOAT") == "FP32"
    assert precision_name({"Precision": "BF16"}) == "bf16"
    assert precision_name({"Precision": "FP8"}) == "fp8"


def test_single_role_profiles_only_lower_named_roles() -> None:
    fp32 = precision_profiles()["B1_TRT_ATTN_FP32"]
    layernorm = precision_profiles()["P6_LAYERNORM_FP16"]
    changed = {
        role
        for role in fp32
        if fp32[role].operand_precision != layernorm[role].operand_precision
    }
    assert changed == {"layernorm"}


def test_bf16_sensitivity_is_role_isolated_from_attention_fp32() -> None:
    profiles = precision_profiles()
    fp32 = profiles["B1_TRT_ATTN_FP32"]
    qkv = profiles["P13_QKV_BF16"]
    changed = {
        role for role in fp32
        if fp32[role].operand_precision != qkv[role].operand_precision
    }
    assert changed == {
        "q_projection", "k_projection", "v_projection", "fused_qkv_projection"
    }
    assert qkv["qk_matmul"].operand_precision == "FP32"


def test_h800_bf16_joint_profiles_preserve_declared_qk_contract() -> None:
    profiles = precision_profiles()
    assert profiles["H1_QKV_BF16_QK_FP32"]["qk_matmul"].operand_precision == "FP32"
    h2 = profiles["H2_QKV_BF16_QK_BF16A32"]["qk_matmul"]
    assert h2.operand_precision == "BF16"
    assert h2.accumulator_precision == "FP32"
    assert h2.evidence_required == "level_a_compute_contract"
    p13 = _profile_semantics("P13_QKV_BF16")
    assert p13["qk_projection_precision"] == "BF16"
    assert p13["layernorm_precision"] == "FP32"
    h1 = _profile_semantics("H1_QKV_BF16_QK_FP32")
    assert h1["qk_projection_precision"] == "BF16_OUTPUT_FP32"


def test_native_int8_qk_requires_int32_accumulator() -> None:
    contract = qk_accumulator_variant(
        "Q4_I8A32I", operand="INT8", accumulator="INT32"
    )
    assert contract.quantization == "native_int8_operands"
    with pytest.raises(ValueError, match="native_int8_qk_requires_int32_accumulator"):
        qk_accumulator_variant("bad", operand="INT8", accumulator="FP32")


def test_output_dtype_does_not_prove_accumulator() -> None:
    accumulator, level, _ = infer_accumulator_from_layer(
        {"Outputs": [{"Format/Datatype": "Half"}], "TacticName": "opaque"}
    )
    assert accumulator == "unknown"
    assert level == "B"
    evidence = build_accumulator_evidence(
        operator="QK",
        requested_operand="FP16",
        requested_accumulator="FP32",
        realized_operand="FP16",
        layer={"Outputs": [{"Format/Datatype": "Half"}], "TacticName": "opaque"},
    )
    assert not evidence.searchable


def test_requested_accumulator_unknown_realization_is_a_conflict(tmp_path) -> None:
    layer_info = tmp_path / "layers.json"
    layer_info.write_text(
        '{"Layers":[{"Name":"__canonical__qk","LayerType":"Einsum",'
        '"Inputs":[{"Format/Datatype":"BF16"}],'
        '"Outputs":[{"Format/Datatype":"BF16"}],"TacticName":"opaque"}]}',
        encoding="utf-8",
    )
    records = audit_realized_precision(
        model="lidar_cobevt",
        profile="H2",
        requested_rows=[{
            "onnx_node": "__canonical__qk",
            "role": "qk_matmul",
            "requested_precision": "BF16",
            "requested_accumulator": "FP32",
        }],
        layer_info_path=layer_info,
    )
    assert records[0].realized_precision == "BF16"
    assert records[0].realized_accumulator == "unknown"
    assert records[0].conflict == "requested_accumulator_FP32_realized_unknown"


def test_int8_accumulator_tactic_is_level_a() -> None:
    evidence = build_accumulator_evidence(
        operator="QK",
        requested_operand="INT8",
        requested_accumulator="INT32",
        realized_operand="INT8",
        layer={"TacticName": "sm90_i8i8_i32_tensorcore"},
    )
    assert evidence.realized_accumulator == "INT32"
    assert evidence.evidence_level == "A"
    assert evidence.searchable


def test_accumulator_phase_covers_q0_q5_and_a0_a5(tmp_path) -> None:
    names = {row[0] for row in EXPERIMENTS}
    assert names == {
        "Q0_F32A32", "Q1_F16_DEFAULT", "Q2_F16A32", "Q3_BF16A32",
        "Q4_I8A32I", "Q5_FP8", "A0_F32A32", "A1_F16_DEFAULT",
        "A2_F16A32", "A3_BF16A32", "A4_I8A32I", "A5_FP8",
    }
    rows = build_matrix(tmp_path)
    assert len(rows) == 24
    native = [row for row in rows if row["experiment"] in {"Q4_I8A32I", "Q5_FP8", "A4_I8A32I", "A5_FP8"}]
    assert native
    assert all(row["status"] == "unsupported" for row in native)
    assert all(row["projection_only_substitute_forbidden"] for row in native)


def test_qdq_inventory_resolves_canonical_role_and_granularity(tmp_path) -> None:
    scale = numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), "zero")
    weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), "weight")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"], name="q_proj/input_quantizer/QuantizeLinear"),
        helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["dq"], name="q_proj/input_quantizer/DequantizeLinear"),
        helper.make_node("MatMul", ["dq", "weight"], ["y"], name="canonical_q_proj"),
    ]
    graph = helper.make_graph(
        nodes, "qdq", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], [scale, zero, weight]
    )
    path = tmp_path / "qdq.onnx"
    onnx.save(helper.make_model(graph), path)
    rows = _qdq_inventory(
        path,
        [{
            "onnx_node": "canonical_q_proj", "module_path": "fusion.q_proj",
            "role": "q_projection", "requested_precision": "INT8",
        }],
    )
    assert len(rows) == 2
    assert all(row["canonical_layer"] == "canonical_q_proj" for row in rows)
    assert all(row["canonical_role"] == "q_projection" for row in rows)
    assert all(row["granularity"] == "per_tensor" for row in rows)
    assert all(row["symmetric"] for row in rows)


def _projection_qdq_graph(*, with_casts: bool, existing_output_cast: bool = False):
    nodes = [
        helper.make_node("DequantizeLinear", ["qa", "s", "z"], ["a_dq"], name="a_dq"),
        helper.make_node("DequantizeLinear", ["qw", "s", "z"], ["w_dq"], name="w_dq"),
        helper.make_node("Transpose", ["w_dq"], ["w_t"], name="w_t"),
    ]
    activation, weight = "a_dq", "w_t"
    if with_casts:
        nodes.extend(
            [
                helper.make_node("Cast", [activation], ["a_half"], name="a_half", to=TensorProto.FLOAT16),
                helper.make_node("Cast", [weight], ["w_half"], name="w_half", to=TensorProto.FLOAT16),
            ]
        )
        activation, weight = "a_half", "w_half"
    nodes.append(helper.make_node("MatMul", [activation, weight], ["y"], name="projection"))
    output = "y"
    if existing_output_cast:
        nodes.append(helper.make_node("Cast", ["y"], ["typed_y"], name="typed_output", to=TensorProto.FLOAT))
        output = "typed_y"
    return helper.make_model(
        helper.make_graph(
            nodes,
            "qdq",
            [
                helper.make_tensor_value_info("qa", TensorProto.INT8, [1, 4]),
                helper.make_tensor_value_info("qw", TensorProto.INT8, [4, 4]),
                helper.make_tensor_value_info("s", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("z", TensorProto.INT8, []),
            ],
            [helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, 4])],
        )
    )


def test_projection_qdq_adjacency_fails_closed_on_intervening_cast() -> None:
    validate_projection_qdq_adjacency(
        _projection_qdq_graph(with_casts=False), ("projection",)
    )
    with pytest.raises(ValueError, match="projection_qdq_not_adjacent"):
        validate_projection_qdq_adjacency(
            _projection_qdq_graph(with_casts=True), ("projection",)
        )


def test_projection_qdq_adjacency_repair_preserves_typed_output_boundary() -> None:
    model = _projection_qdq_graph(with_casts=True)
    rows = restore_projection_qdq_adjacency(
        model, ("projection",), output_cast_precisions={"projection": "FP16"}
    )
    validate_projection_qdq_adjacency(model, ("projection",))
    projection = next(node for node in model.graph.node if node.name == "projection")
    output_cast = next(
        node for node in model.graph.node if node.name == "projection__output_fp16"
    )
    assert list(projection.input) == ["a_dq", "w_t"]
    assert list(output_cast.input) == ["y__before_quantized_projection_output_cast"]
    assert list(output_cast.output) == ["y"]
    assert rows[0]["removed_cast_nodes"] == ["a_half", "w_half"]


def test_projection_qdq_adjacency_repair_rejects_duplicate_output_cast() -> None:
    model = _projection_qdq_graph(with_casts=True, existing_output_cast=True)
    with pytest.raises(
        ValueError,
        match="projection_output_cast_conflicts_with_existing_typed_boundary",
    ):
        restore_projection_qdq_adjacency(
            model, ("projection",), output_cast_precisions={"projection": "FP16"}
        )


def test_bf16_and_fp8_capabilities_fail_closed() -> None:
    capability = capability_from_probe(
        "FP8",
        {
            "hardware_supported": True,
            "tensorrt_api_supported": True,
            "modelopt_supported": True,
            "explicit_graph_supported": False,
            "realized_auditable": False,
            "reasons": ["no_realized_engine_evidence"],
        },
    )
    assert not capability.deployable
    with pytest.raises(RuntimeError, match="fp8_unsupported_fail_closed"):
        require_deployable(capability)


def test_smoothquant_profiles_protect_attention_core() -> None:
    profiles = {row.profile_id: row for row in smoothquant_profiles()}
    assert profiles["SQ4_FFN"].int8_roles == ("ffn1", "ffn2")
    assert profiles["SQ5_QK_PLUS_FFN"].qk_contract == "DQ_FP32_QK"
    assert "qk_matmul" in profiles["SQ6_ALL_TRANSFORMER_LINEAR"].protected_roles


def test_selective_smoothquant_config_is_per_channel_weight() -> None:
    path = "fusion.layers.0.ffn.fn.net.0"
    config = selective_smoothquant_config((path,), alpha=0.7)
    assert config["quant_cfg"][f"*{path}*weight_quantizer"]["axis"] == 0
    assert config["quant_cfg"][f"*{path}*input_quantizer"]["axis"] is None
    assert config["quant_cfg"]["default"] == {"enable": False}


def test_alpha_selection_uses_all_required_metrics() -> None:
    rows = []
    for alpha, value in ((0.5, 2.0), (0.7, 1.0), (0.8, 3.0)):
        rows.append(
            {
                "alpha": alpha,
                "projection_relative_l2": value,
                "qk_relative_l2": value,
                "softmax_js": value,
                "ffn_output_relative_l2": value,
                "residual_update_relative_l2": value,
                "saturation_ratio": value,
                "scale_stability_delta": value,
            }
        )
    assert choose_alpha(rows)["alpha"] == 0.7


def test_profile_library_requires_all_acceptance_evidence() -> None:
    accepted = {
        "profile_id": "F3",
        "physical_graph_legal": True,
        "onnx_success": True,
        "engine_success": True,
        "requested_realized_match": True,
        "zero_skip": True,
        "fixed500_safe": True,
        "latency_benefit": True,
        "no_precision_conflict": True,
        "qk_projection_precision": "FP16",
    }
    incomplete = {**accepted, "profile_id": "fixed50_only", "fixed500_safe": False}
    wrong_alpha = {
        **accepted,
        "profile_id": "wrong_alpha",
        "alpha_contract_match": False,
    }
    library = build_profile_library(
        {"cobevt": [accepted, incomplete, wrong_alpha], "v2xvit": [accepted]}
    )
    assert [row["profile_id"] for row in library["cobevt"]["allowed"]] == ["F3"]
    assert library["cross_model_profiles"]["portable"] == ["F3"]
    space = build_search_space(library)
    assert not space["cartesian_product_automatically_opened"]
    assert space["subgraph_lut_additive"] is False
