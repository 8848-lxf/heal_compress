from __future__ import annotations

import pytest


def test_compute_contract_keeps_storage_operand_accumulator_and_output_distinct():
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        ComputeContract,
    )

    contract = ComputeContract(
        storage_precision="FP16",
        left_operand_precision="FP16",
        right_operand_precision="FP16",
        multiplication_precision="FP16",
        accumulator_precision="FP32",
        output_precision="FP32",
    )

    assert contract.phenotype == "F16A32"
    assert contract.storage_precision == "FP16"
    assert contract.accumulator_precision == "FP32"


def test_f3_is_f32a32_not_f16a32():
    from search.model_families.lidar_cobevt.attention_accumulation_profiles import (
        accumulation_profile,
    )

    profile = accumulation_profile("A0_F3_REFERENCE")

    assert profile.qk_contract.storage_precision == "FP16"
    assert profile.qk_contract.left_operand_precision == "FP32"
    assert profile.qk_contract.accumulator_precision == "FP32"
    assert profile.qk_contract.phenotype == "F32A32"


def test_r1_accumulator_is_unknown_and_not_searchable():
    from search.model_families.lidar_cobevt.attention_accumulation_profiles import (
        accumulation_profile,
    )

    profile = accumulation_profile("A1_R1_LOW_ATTENTION")

    assert profile.qk_contract.phenotype == "UNKNOWN_ACCUM"
    assert profile.search_eligible is False


def test_int8_accumulator_is_int32_not_fp32():
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        ComputeContract,
    )

    contract = ComputeContract(
        storage_precision="INT8",
        left_operand_precision="INT8",
        right_operand_precision="INT8",
        multiplication_precision="INT8",
        accumulator_precision="INT32",
        output_precision="INT32",
    )

    assert contract.phenotype == "I8A32I"
    assert contract.accumulator_precision != "FP32"


def test_materialized_fp16_to_fp32_cast_is_f32a32_after_cast():
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        classify_cast_materialization,
    )

    result = classify_cast_materialization(
        source_precision="FP16",
        gemm_operand_precision="FP32",
        cast_execution_layer_present=True,
        cast_tensor_written_to_memory=True,
        cast_absorbed_by_gemm=False,
    )

    assert result.phenotype == "F32A32_after_materialized_cast"
    assert result.materialized is True
    assert result.is_f16a32 is False


def test_absorbed_cast_requires_direct_kernel_evidence_for_f16a32():
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        classify_cast_materialization,
    )

    inferred = classify_cast_materialization(
        source_precision="FP16",
        gemm_operand_precision="FP32",
        cast_execution_layer_present=False,
        cast_tensor_written_to_memory=False,
        cast_absorbed_by_gemm=True,
        kernel_accumulator_precision="FP32",
        kernel_contract_evidence_level="C",
    )
    direct = classify_cast_materialization(
        source_precision="FP16",
        gemm_operand_precision="FP32",
        cast_execution_layer_present=False,
        cast_tensor_written_to_memory=False,
        cast_absorbed_by_gemm=True,
        kernel_accumulator_precision="FP32",
        kernel_contract_evidence_level="A",
    )

    assert inferred.phenotype == "UNKNOWN_ACCUM"
    assert direct.phenotype == "F16A32"


def test_dq_to_fp32_qk_cannot_be_native_int8():
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        classify_int8_qk,
    )

    result = classify_int8_qk(
        storage_precision="INT8",
        qk_input_precision="FP32",
        dequantize_before_qk=True,
        kernel_operand_precision="FP32",
        kernel_accumulator_precision="FP32",
        evidence_level="A",
    )

    assert result.native_int8 is False
    assert result.phenotype == "F32A32_after_dequantize"


def test_unknown_accumulator_cannot_enter_search():
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        SearchEvidence,
        search_eligibility,
    )

    evidence = SearchEvidence(
        evidence_level="C",
        accumulator_precision="unknown",
        requested_realized_match=True,
        fixed500_delta_map=0.0,
        formal_latency_gain=True,
        precision_conflicts=0,
        skipped_frames=0,
    )

    verdict = search_eligibility(evidence)
    assert verdict.eligible is False
    assert "accumulator_evidence_below_level_a" in verdict.reasons


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (0.001, "SAFE_FIXED500"),
        (-0.003, "SAFE_FIXED500"),
        (-0.004, "BORDERLINE_FIXED500"),
        (-0.010, "BORDERLINE_FIXED500"),
        (-0.010001, "UNSAFE_FIXED500"),
    ],
)
def test_fixed500_threshold_classification(delta: float, expected: str):
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        classify_fixed500,
    )

    assert classify_fixed500(delta, finite=True, skipped_frames=0) == expected


def test_invalid_compute_contract_fails_closed():
    from search.model_families.lidar_cobevt.attention_compute_contract import (
        ComputeContract,
    )

    with pytest.raises(ValueError, match="int8_accumulator_must_be_int32"):
        ComputeContract(
            storage_precision="INT8",
            left_operand_precision="INT8",
            right_operand_precision="INT8",
            multiplication_precision="INT8",
            accumulator_precision="FP32",
            output_precision="FP32",
        )


def test_trt_header_inventory_does_not_invent_accumulator_api():
    from search.orchestration.lidar_cobevt_attention_accumulation_capability import (
        parse_installed_header_capability,
    )

    header = """
    class IMatrixMultiplyLayer : public ILayer {};
    Strongly-typed networks reject calls to method setPrecision.
    IMatrixMultiplyLayer* addMatrixMultiply(ITensor& a, MatrixOperation op0,
                                             ITensor& b, MatrixOperation op1);
    """

    result = parse_installed_header_capability(header)
    assert result["matrix_multiply_layer_present"] is True
    assert result["strongly_typed_rejects_set_precision"] is True
    assert result["separate_accumulator_precision_api"] is False


def test_precision_capability_matrix_marks_native_f16a32_unproven():
    from search.orchestration.lidar_cobevt_attention_accumulation_capability import (
        build_precision_capability_rows,
    )

    rows = build_precision_capability_rows(
        trt_version="10.9.0.34",
        matrix_multiply_present=True,
        separate_accumulator_api=False,
        inspector_accumulator_field=False,
        bf16_type_present=True,
        quantize_layer_present=True,
    )
    by_name = {row["phenotype"]: row for row in rows}

    assert by_name["F32A32"]["native_request_status"] == "supported"
    assert by_name["F16A32"]["native_request_status"] == "not_separately_expressible"
    assert by_name["F16A32"]["maximum_native_evidence"] == "C"
    assert by_name["I8A32I"]["accumulator_type"] == "INT32"


def test_tensorrt_python_probe_requires_matching_library_path(tmp_path):
    from search.orchestration.lidar_cobevt_attention_accumulation_capability import (
        validate_trt_runtime_paths,
    )

    root = tmp_path / "TensorRT"
    (root / "lib").mkdir(parents=True)
    (root / "lib/libnvinfer.so.10").write_bytes(b"fake")

    with pytest.raises(ValueError, match="tensorrt_library_path_not_active"):
        validate_trt_runtime_paths(root, ld_library_path="/somewhere/else")

    result = validate_trt_runtime_paths(
        root, ld_library_path=f"{root / 'lib'}:/somewhere/else"
    )
    assert result["libnvinfer"] == str(root / "lib/libnvinfer.so.10")


def test_installed_tensorrt_10_9_uses_engine_inspector_python_name():
    from search.orchestration.lidar_cobevt_attention_accumulation_capability import (
        resolve_engine_inspector_api_name,
    )

    assert resolve_engine_inspector_api_name({"EngineInspector"}) == "EngineInspector"
    assert resolve_engine_inspector_api_name({"IEngineInspector"}) == "IEngineInspector"
    with pytest.raises(ValueError, match="engine_inspector_api_missing"):
        resolve_engine_inspector_api_name(set())
