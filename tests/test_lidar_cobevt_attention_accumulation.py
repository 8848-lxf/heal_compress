from __future__ import annotations

import json

import pytest
import torch


def test_fp16_accumulation_profiles_keep_operands_and_accumulator_distinct():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        accumulation_profile,
    )

    r1 = accumulation_profile("R1_fp16_operands_default_accum")
    r2 = accumulation_profile("R2_fp16_operands_forced_fp32_accum")
    r3 = accumulation_profile("R3_fp16_projection_cast_fp32_operands_fp32_accum")

    assert r1.projection_dtype == "FP16"
    assert r1.qk_operand_dtype == "FP16"
    assert r1.requested_accumulator_dtype == "DEFAULT"
    assert r2.qk_operand_dtype == "FP16"
    assert r2.requested_accumulator_dtype == "FP32"
    assert r3.projection_dtype == "FP16"
    assert r3.qk_operand_dtype == "FP32"
    assert r3.requested_accumulator_dtype == "FP32"


def test_trt_10_9_strongly_typed_cannot_claim_separate_fp16_accumulator_control():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        trt_accumulator_realization,
    )

    result = trt_accumulator_realization(
        profile_name="R2_fp16_operands_forced_fp32_accum",
        strongly_typed=True,
        layer_info={"Inputs": [{"Format/Datatype": "Half"}]},
        trt_version="10.9.0.34",
    )

    assert result["status"] == "unsupported_exact_semantics"
    assert result["accumulator_precision"] == "unknown"
    assert result["evidence_sufficient"] is False


def test_tactic_evidence_identifies_f3_as_fp32_operands_and_compute():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        classify_qk_tactic,
    )

    result = classify_qk_tactic(
        "sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_stage3"
    )

    assert result == {
        "left_operand_precision": "FP32",
        "right_operand_precision": "FP32",
        "output_precision": "FP32",
        "compute_precision": "FP32",
        "accumulator_precision": "FP32",
        "evidence_sufficient": True,
    }


def test_unknown_tactic_does_not_guess_accumulator_precision():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        classify_qk_tactic,
    )

    result = classify_qk_tactic("__myl_opaque_attention_kernel")

    assert result["accumulator_precision"] == "unknown"
    assert result["evidence_sufficient"] is False


def test_int8_dq_fp32_matmul_is_not_native_int8_qk():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        accumulation_profile,
        classify_int8_realization,
    )

    profile = accumulation_profile("I2_int8_qk_dq_fp32_matmul")
    result = classify_int8_realization(
        profile,
        qk_input_dtype="FP32",
        qk_tactic="sm80_xmma_gemm_f32f32_f32f32_f32_nn",
    )

    assert result["native_int8_qk"] is False
    assert result["realized_multiplication_precision"] == "FP32"
    assert result["realized_accumulator_precision"] == "FP32"


def test_dynamic_quantize_api_capability_is_fp4_not_int8():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        dynamic_quantize_api_verdict,
    )

    verdict = dynamic_quantize_api_verdict(
        python_api_present=True,
        cpp_api_present=True,
        allowed_output_types=("FP4",),
        allowed_scale_types=("FP8",),
        block_sizes=(16,),
    )

    assert verdict["api_present"] is True
    assert verdict["int8_dynamic_quantization_supported"] is False
    assert verdict["sageattention_dynamic_int8_expressible"] is False


def test_decomposition_keeps_input_compute_accumulation_and_interaction_terms():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        decompose_qk_error,
    )

    values = decompose_qk_error(
        reference_error=0.0,
        fp16_projection_fp32_qk_error=0.1,
        fp32_projection_fp16_default_error=0.4,
        fp32_projection_fp16_fp32_accum_error=0.25,
        fp16_projection_fp16_default_error=0.6,
    )

    assert values["input_rounding_contribution"] == pytest.approx(0.1)
    assert values["multiplication_contribution"] == pytest.approx(0.25)
    assert values["accumulation_contribution"] == pytest.approx(0.15)
    assert values["interaction_term"] == pytest.approx(0.1)


def test_numerical_matrix_uses_real_capture_metadata_and_is_finite(tmp_path):
    from search.model_families.lidar_cobevt.attention_accumulation import (
        run_fp16_numerical_matrix,
    )

    capture = tmp_path / "capture.pt"
    torch.save(
        {
            "attention_type": "layers.0.window_attention",
            "frame_id": "frame-001",
            "q_fp32": torch.randn(2, 8, 32, 32),
            "k_fp32": torch.randn(2, 8, 32, 32),
            "q_fp16_projection": torch.randn(2, 8, 32, 32).half(),
            "k_fp16_projection": torch.randn(2, 8, 32, 32).half(),
            "scale": 32**-0.5,
        },
        capture,
    )

    rows = run_fp16_numerical_matrix((capture,), device=torch.device("cpu"))

    assert {row["profile"] for row in rows} == {
        "R0_fp32_operands_fp32_accum",
        "R1_fp16_operands_default_accum",
        "R2_fp16_operands_forced_fp32_accum",
        "R3_fp16_projection_cast_fp32_operands_fp32_accum",
        "R4_fp32_projection_cast_fp16_default_accum",
        "R5_fp32_projection_fp16_operands_fp32_accum",
    }
    assert all(row["frame_id"] == "frame-001" for row in rows)
    assert all(row["finite"] for row in rows)


def test_tp_contract_reports_shared_indices_without_per_head_balance(tmp_path):
    from search.model_families.lidar_cobevt.attention_accumulation import (
        write_tp_attention_contract,
    )

    destination = tmp_path / "tp.md"
    report = write_tp_attention_contract(destination)

    assert report["qkv_same_indices"] is True
    assert report["out_proj_input_and_output_pruned"] is True
    assert report["per_original_head_balance_guaranteed"] is False
    assert "TransformerSemanticResolver" in destination.read_text(encoding="utf-8")


def test_accumulation_manifest_is_stable_and_complete():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        ACCUMULATION_PROFILE_NAMES,
        accumulation_profile_manifest,
    )

    first = accumulation_profile_manifest()
    second = accumulation_profile_manifest()

    assert first == second
    assert set(first) == set(ACCUMULATION_PROFILE_NAMES)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_precision_parser_uses_role_contract_for_fused_fp16_projection():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        classify_attention_precision_row,
    )

    row = classify_attention_precision_row(
        profile_contract_id="F3_rest_fp16_qk_fp32_minimal_island",
        role="q_projection",
        layer_info={
            "Name": "q+k+v_projection",
            "LayerType": "gemm",
            "Inputs": [{"Format/Datatype": "Half"}],
            "Outputs": [{"Format/Datatype": "Half"}],
            "TacticName": "ampere_h16816gemm_256x128_ldg8_nn_v1",
        },
    )

    assert row["requested_precision"] == "FP16"
    assert row["requested_precision_source"] == "profile_contract.roles.q_projection"
    assert row["realized_precision"] == "FP16"
    assert row["realized_precision_source"] == "engine_inspector+tactic"
    assert row["requested_realized_match"] is True
    assert row["classification_conflict"] is False


def test_precision_parser_classifies_f3_qk_and_av_from_engine_evidence():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        classify_attention_precision_row,
    )

    qk = classify_attention_precision_row(
        profile_contract_id="F3_rest_fp16_qk_fp32_minimal_island",
        role="qk_matmul",
        layer_info={
            "Name": "qk",
            "LayerType": "gemm",
            "Inputs": [
                {"Format/Datatype": "Float"},
                {"Format/Datatype": "Float"},
            ],
            "Outputs": [{"Format/Datatype": "Float"}],
            "TacticName": "sm80_xmma_gemm_f32f32_f32f32_f32_nn",
        },
    )
    av = classify_attention_precision_row(
        profile_contract_id="F3_rest_fp16_qk_fp32_minimal_island",
        role="av_matmul",
        layer_info={
            "Name": "av",
            "LayerType": "gemm",
            "Inputs": [
                {"Format/Datatype": "Half"},
                {"Format/Datatype": "Half"},
            ],
            "Outputs": [{"Format/Datatype": "Half"}],
            "TacticName": "ampere_h16816gemm_64x64_ldg8_nn_v1",
        },
    )

    assert qk["requested_precision"] == qk["realized_precision"] == "FP32"
    assert qk["accumulator_precision"] == "FP32"
    assert av["requested_precision"] == av["realized_precision"] == "FP16"


def test_precision_parser_fails_closed_on_dtype_tactic_conflict():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        classify_attention_precision_row,
    )

    row = classify_attention_precision_row(
        profile_contract_id="F3_rest_fp16_qk_fp32_minimal_island",
        role="output_projection",
        layer_info={
            "Name": "out",
            "LayerType": "gemm",
            "Inputs": [{"Format/Datatype": "Float"}],
            "Outputs": [{"Format/Datatype": "Float"}],
            "TacticName": "ampere_h16816gemm_256x128_ldg8_nn_v1",
        },
    )

    assert row["realized_precision"] == "unknown"
    assert row["classification_conflict"] is True
    assert row["requested_realized_match"] is False


def test_precision_parser_uses_explicit_onnx_boundary_only_without_engine_evidence():
    from search.model_families.lidar_cobevt.attention_accumulation import (
        classify_attention_precision_row,
    )

    row = classify_attention_precision_row(
        profile_contract_id="F3_rest_fp16_qk_fp32_minimal_island",
        role="softmax",
        layer_info={},
        onnx_compute_precision="FP16",
        explicit_onnx_boundary=True,
    )

    assert row["requested_precision"] == "FP16"
    assert row["realized_precision"] == "FP16"
    assert row["realized_precision_source"] == "onnx_explicit_boundary"
    assert row["classification_confidence"] == "medium"
    assert row["requested_realized_match"] is True
