from __future__ import annotations

from pathlib import Path

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


def test_derive_attention_tensors_reconstructs_qk_softmax_and_av():
    import torch

    from search.model_families.lidar_cobevt.attention_numerical_boundary import (
        derive_attention_tensors,
    )

    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    qkv = torch.cat((q, k, v), dim=-1)
    mask = torch.tensor([[[[True, False]]]])

    tensors = derive_attention_tensors(
        qkv=qkv,
        heads=1,
        scale=0.5,
        relative_bias=torch.zeros(1, 2, 2),
        attention_mask=mask,
    )

    assert tuple(tensors["q"].shape) == (1, 1, 2, 2)
    assert torch.equal(tensors["scaled_q"], tensors["q"] * 0.5)
    assert torch.isneginf(tensors["masked_logits"][..., 1]).all()
    assert torch.equal(tensors["probability"][..., 1], torch.zeros(1, 1, 2))
    assert torch.allclose(tensors["av"], torch.tensor([[[[2.0, 4.0], [2.0, 4.0]]]]))


def test_tensor_statistics_reports_subnormal_zero_and_finite_ratios():
    import torch

    from search.model_families.lidar_cobevt.attention_numerical_boundary import (
        tensor_statistics,
    )

    values = torch.tensor([0.0, torch.finfo(torch.float32).tiny / 2, 2.0, float("inf")])
    stats = tensor_statistics(values)

    assert stats["zero_ratio"] == pytest.approx(0.25)
    assert stats["subnormal_ratio"] == pytest.approx(0.25)
    assert stats["finite_ratio"] == pytest.approx(0.75)
    assert stats["max"] == pytest.approx(2.0)


def test_capture_inventory_requires_all_six_attention_blocks_and_frames():
    from search.model_families.lidar_cobevt.attention_numerical_boundary import (
        validate_capture_inventory,
    )

    modules = tuple(
        f"fusion_net.layers.{layer}.{kind}_attention.fn"
        for layer in range(3)
        for kind in ("window", "grid")
    )
    rows = [
        {"frame_id": f"f{frame}", "module_name": module}
        for frame in range(10)
        for module in modules
    ]
    result = validate_capture_inventory(rows, expected_frames=10)
    assert result["module_count"] == 6
    assert result["frame_count"] == 10

    with pytest.raises(ValueError, match="capture_attention_block_incomplete"):
        validate_capture_inventory(rows[:-1], expected_frames=10)


def test_tensor_content_hash_changes_with_tensor_value():
    import torch

    from search.model_families.lidar_cobevt.attention_numerical_boundary import (
        tensor_content_hash,
    )

    first = tensor_content_hash(torch.tensor([1.0, 2.0]))
    second = tensor_content_hash(torch.tensor([1.0, 3.0]))
    assert first != second
    assert first == tensor_content_hash(torch.tensor([1.0, 2.0]))


def test_cublaslt_oracle_contracts_encode_exact_accumulator_types():
    from search.model_families.lidar_cobevt.cuda_oracle_contract import (
        oracle_contracts,
    )

    qk = {row.profile_id: row for row in oracle_contracts("QK")}

    assert qk["O1_F16A32"].operand_cuda_type == "CUDA_R_16F"
    assert qk["O1_F16A32"].compute_type == "CUBLAS_COMPUTE_32F"
    assert qk["O1_F16A32"].accumulator_precision == "FP32"
    assert qk["O2_F16A16"].compute_type == "CUBLAS_COMPUTE_16F"
    assert qk["O4_I8A32I"].compute_type == "CUBLAS_COMPUTE_32I"
    assert qk["O4_I8A32I"].accumulator_precision == "INT32"


def test_oracle_contract_serialization_preserves_level_a_evidence():
    from search.model_families.lidar_cobevt.cuda_oracle_contract import (
        oracle_contract,
    )

    payload = oracle_contract("O1_F16A32").to_manifest()

    assert payload["evidence_level"] == "A"
    assert payload["implementation"] == "cuBLASLt_or_CUTLASS_oracle"
    assert payload["phenotype"] == "F16A32"


def test_oracle_build_environment_rejects_system_nvcc():
    from search.model_families.lidar_cobevt.cuda_oracle_contract import (
        validate_oracle_toolchain,
    )

    with pytest.raises(ValueError, match="invalid_non_conda_cuda_toolchain"):
        validate_oracle_toolchain(
            conda_prefix="/home/lixingfeng/anaconda3/envs/modelopt",
            nvcc="/usr/bin/nvcc",
            cxx="/home/lixingfeng/anaconda3/envs/modelopt/bin/x86_64-conda-linux-gnu-g++",
        )


def test_accumulator_discriminative_probes_cover_required_boundaries():
    from search.model_families.lidar_cobevt.cuda_oracle_contract import (
        accumulator_discriminative_probes,
    )

    probes = accumulator_discriminative_probes(reduction_length=128)

    assert set(probes) == {
        "alternating_cancellation",
        "large_small_mixture",
        "long_reduction",
        "near_fp16_subnormal",
        "product_underflow",
        "softmax_small_margin",
        "zero_uniform_logits",
    }
    assert all(tuple(value[0].shape) == (1, 1, 1, 128) for value in probes.values())
    assert all(tuple(value[1].shape) == (1, 1, 1, 128) for value in probes.values())


def test_oracle_command_binds_exact_shape_profile_and_files(tmp_path):
    from search.orchestration.lidar_cobevt_attention_gemm_oracle import (
        oracle_command,
    )

    command = oracle_command(
        executable=tmp_path / "oracle",
        a_path=tmp_path / "a.bin",
        b_path=tmp_path / "b.bin",
        output_path=tmp_path / "out.bin",
        m=32,
        n=32,
        k=32,
        batch=64,
        phenotype="F16A32",
        warmup=20,
        iterations=100,
    )

    assert command[0] == str(tmp_path / "oracle")
    assert command[command.index("--profile") + 1] == "F16A32"
    assert command[command.index("--batch") + 1] == "64"
    assert command[command.index("--a") + 1] == str(tmp_path / "a.bin")


def test_native_micro_matrix_contains_qk_and_av_requested_profiles():
    from search.model_families.lidar_cobevt.attention_trt_realization import (
        native_micro_specs,
    )

    specs = {row.spec_id: row for row in native_micro_specs()}
    assert set(specs) == {
        "M0_QK_F32A32",
        "M1_QK_F16_DEFAULT",
        "M2_QK_F16A32_REQUEST",
        "M3_QK_F16A16_REQUEST",
        "M4_QK_I8A32I",
        "M5_QK_BF16A32",
        "N0_AV_F32A32",
        "N1_AV_F16_DEFAULT",
        "N2_AV_F16A32_REQUEST",
        "N3_AV_F16A16_REQUEST",
        "N4_AV_I8A32I",
        "N5_AV_BF16A32",
    }
    assert specs["M2_QK_F16A32_REQUEST"].explicit_cast_to_fp32 is True


def test_native_build_success_does_not_prove_f16a32_accumulator():
    from search.model_families.lidar_cobevt.attention_trt_realization import (
        classify_native_realization,
    )

    result = classify_native_realization(
        requested_phenotype="F16A32",
        build_success=True,
        input_precisions=("FP16", "FP16"),
        matmul_input_precisions=("FP16", "FP16"),
        output_precision="FP32",
        execution_layers=[{"name": "qk", "tactic": "opaque_xmma"}],
        direct_accumulator_metadata=None,
    )

    assert result["realized_accumulator_precision"] == "unknown"
    assert result["evidence_level"] == "C"
    assert result["requested_realized_match"] is False


def test_native_realization_detects_materialized_cast_not_f16a32():
    from search.model_families.lidar_cobevt.attention_trt_realization import (
        classify_native_realization,
    )

    result = classify_native_realization(
        requested_phenotype="F16A32",
        build_success=True,
        input_precisions=("FP16", "FP16"),
        matmul_input_precisions=("FP32", "FP32"),
        output_precision="FP32",
        execution_layers=[
            {"name": "cast_q", "type": "Cast", "output": "FP32"},
            {"name": "cast_k", "type": "Cast", "output": "FP32"},
            {"name": "qk", "type": "MatrixMultiply", "input": "FP32|FP32"},
        ],
        direct_accumulator_metadata="FP32",
    )

    assert result["realized_phenotype"] == "F32A32_after_materialized_cast"
    assert result["cast_materialized"] is True
    assert result["requested_realized_match"] is False


def test_int8_dq_float_matmul_is_not_classified_native():
    from search.model_families.lidar_cobevt.attention_trt_realization import (
        classify_native_realization,
    )

    result = classify_native_realization(
        requested_phenotype="I8A32I",
        build_success=True,
        input_precisions=("INT8", "INT8"),
        matmul_input_precisions=("FP32", "FP32"),
        output_precision="FP32",
        execution_layers=[{"name": "dq_q"}, {"name": "qk"}],
        direct_accumulator_metadata="FP32",
    )

    assert result["realized_phenotype"] == "F32A32_after_dequantize"
    assert result["native_int8"] is False


def test_native_micro_candidate_matrix_covers_six_blocks_without_cache_aliasing():
    from search.orchestration.lidar_cobevt_attention_trt_microbench import (
        native_candidate_matrix,
    )

    modules = tuple(f"fusion.layers.{idx}.{kind}" for idx in range(3) for kind in ("window", "grid"))
    rows = native_candidate_matrix(modules)

    assert len(rows) == 6 * 12
    assert len({row.candidate_id for row in rows}) == len(rows)
    assert {row.module_name for row in rows} == set(modules)
    assert all(row.fresh_build for row in rows)


def test_native_micro_graph_plan_keeps_requests_distinct_from_realization():
    from search.orchestration.lidar_cobevt_attention_trt_microbench import (
        micro_graph_plan,
    )

    f16a32 = micro_graph_plan("M2_QK_F16A32_REQUEST")
    i8 = micro_graph_plan("M4_QK_I8A32I")

    assert f16a32.input_precision == "FP16"
    assert f16a32.cast_inputs_to_fp32 is True
    assert f16a32.claimed_realized_phenotype is None
    assert i8.input_precision == "INT8"
    assert i8.native_int8_operands is True
    assert i8.dequantize_before_matmul is False


def test_accumulator_tactic_parser_only_accepts_explicit_kernel_metadata():
    from search.model_families.lidar_cobevt.attention_trt_realization import (
        infer_accumulator_from_tactics,
    )

    f32 = [{"TacticName": "sm50_xmma_cublas_smallN_NN_f32f32_f32_f32_nn"}]
    opaque = [{"TacticName": "ampere_h16816gemm_64x64"}]
    conflict = [
        {"TacticName": "kernel_f16f16_f32_f32"},
        {"TacticName": "kernel_f16f16_f16_f16"},
    ]

    assert infer_accumulator_from_tactics(f32) == "FP32"
    assert infer_accumulator_from_tactics(opaque) is None
    assert infer_accumulator_from_tactics(conflict) is None


def test_mixed_accum_plugin_contract_is_level_a_but_not_native_tensorrt():
    from search.model_families.lidar_cobevt.attention_plugin_oracle import (
        mixed_accum_plugin_contract,
    )

    qk = mixed_accum_plugin_contract("QK", output_precision="FP32", scale=0.125)

    assert qk.input_precision == "FP16"
    assert qk.multiplication_precision == "FP16"
    assert qk.accumulator_precision == "FP32"
    assert qk.phenotype == "F16A32"
    assert qk.evidence_level == "A"
    assert qk.implementation == "plugin_oracle"
    assert qk.native_tensorrt is False


def test_mixed_accum_plugin_serialization_and_shapes_are_deterministic():
    from search.model_families.lidar_cobevt.attention_plugin_oracle import (
        MixedAccumPluginContract,
        mixed_accum_output_shape,
    )

    original = MixedAccumPluginContract("AV", "FP16", "FP16", "FP32", "FP16", 1.0)
    restored = MixedAccumPluginContract.from_bytes(original.to_bytes())

    assert restored == original
    assert mixed_accum_output_shape("QK", (8, 8, 32, 32), (8, 8, 32, 32)) == (8, 8, 32, 32)
    assert mixed_accum_output_shape("AV", (8, 8, 32, 32), (8, 8, 32, 24)) == (8, 8, 32, 24)
    with pytest.raises(ValueError, match="mixed_accum_batch_shape_mismatch"):
        mixed_accum_output_shape("QK", (8, 8, 32, 32), (7, 8, 32, 32))


def test_plugin_oracle_matrix_has_qk_and_av_for_each_real_block():
    from search.orchestration.lidar_cobevt_attention_plugin_oracle import (
        plugin_candidate_matrix,
    )

    modules = tuple(f"layers.{layer}.{kind}" for layer in range(3) for kind in ("window", "grid"))
    rows = plugin_candidate_matrix(modules)

    assert len(rows) == 12
    assert {row.family for row in rows} == {"QK", "AV"}
    assert all(row.requested_phenotype == "F16A32" for row in rows)
    assert all(row.implementation == "plugin_oracle" for row in rows)


def test_plugin_creator_selection_uses_exact_registry_identity():
    from types import SimpleNamespace

    from search.orchestration.lidar_cobevt_attention_plugin_oracle import (
        select_plugin_creator,
    )

    target = SimpleNamespace(
        name="QKMixedAccumPlugin", plugin_version="1", plugin_namespace=""
    )
    other = SimpleNamespace(name="QKMixedAccumPlugin", plugin_version="2", plugin_namespace="")
    assert select_plugin_creator([other, target]) is target
    with pytest.raises(ValueError, match="qk_mixed_accum_plugin_creator_ambiguous"):
        select_plugin_creator([target, target])


def test_mixed_accum_plugin_workspace_contract_is_consistent():
    root = Path(__file__).resolve().parents[1] / "plugins" / "qk_mixed_accum"
    common = (root / "plugin_common.h").read_text()
    plugin = (root / "qk_mixed_accum_plugin.cpp").read_text()
    kernel = (root / "qk_mixed_accum_kernel.cu").read_text()

    assert "kWorkspaceBytes" in common
    assert "return kWorkspaceBytes;" in plugin
    assert "outputs[0], workspace, kWorkspaceBytes" in plugin
    assert "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES" in kernel
    assert "workspace, workspaceBytes, stream" in kernel


def test_mixed_accum_plugin_context_lifecycle_initializes_cublaslt_handle():
    source = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "qk_mixed_accum"
        / "qk_mixed_accum_plugin.cpp"
    ).read_text()

    assert "ensureLtHandle" in source
    assert "void QKMixedAccumPlugin::attachToContext" in source
    assert "ensureLtHandle();" in source
    assert "void QKMixedAccumPlugin::detachFromContext" in source
    assert "releaseLtHandle();" in source


def _six_block_f3_einsum_model():
    import onnx
    from onnx import TensorProto, helper

    nodes = []
    inputs = []
    outputs = []
    initializers = []
    value_info = []
    for block in range(6):
        prefix = f"/layers.{block // 2}/{'window' if block % 2 == 0 else 'grid'}_attention/fn"
        q = f"q_half_{block}"
        k = f"k_half_{block}"
        p = f"p_half_{block}"
        v = f"v_half_{block}"
        inputs.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 8, 32, 32])
            for name in (q, k, p, v)
        )
        q_float = f"q_float_{block}"
        k_float = f"k_float_{block}"
        q_reshape = f"q_reshape_{block}"
        k_reshape = f"k_reshape_{block}"
        q_transpose = f"q_transpose_{block}"
        k_transpose = f"k_transpose_{block}"
        scale = f"scale_{block}"
        scale_half = f"scale_half_{block}"
        scaled_q = f"scaled_q_{block}"
        qk = f"qk_{block}"
        av = f"av_{block}"
        initializers.append(helper.make_tensor(scale_half, TensorProto.FLOAT16, [], [0.1768]))
        nodes.extend(
            (
                helper.make_node("Cast", [q], [q_float], name=f"{prefix}/QCast", to=TensorProto.FLOAT),
                helper.make_node("Cast", [k], [k_float], name=f"{prefix}/KCast", to=TensorProto.FLOAT),
                helper.make_node("Reshape", [q_float, "shape"], [q_reshape], name=f"{prefix}/QReshape"),
                helper.make_node("Reshape", [k_float, "shape"], [k_reshape], name=f"{prefix}/KReshape"),
                helper.make_node("Transpose", [q_reshape], [q_transpose], name=f"{prefix}/QTranspose"),
                helper.make_node("Transpose", [k_reshape], [k_transpose], name=f"{prefix}/KTranspose"),
                helper.make_node("Cast", [scale_half], [scale], name=f"{prefix}/ScaleCast", to=TensorProto.FLOAT),
                helper.make_node("Mul", [q_transpose, scale], [scaled_q], name=f"{prefix}/Mul_6"),
                helper.make_node(
                    "Einsum", [scaled_q, k_transpose], [qk], name=f"{prefix}/Einsum",
                    equation="b h i d, b h j d -> b h i j",
                ),
                helper.make_node(
                    "Einsum", [p, v], [av], name=f"{prefix}/Einsum_1",
                    equation="b h i j, b h j d -> b h i d",
                ),
            )
        )
        value_info.extend(
            (
                helper.make_tensor_value_info(q_float, TensorProto.FLOAT, [1, 8, 32, 32]),
                helper.make_tensor_value_info(k_float, TensorProto.FLOAT, [1, 8, 32, 32]),
                helper.make_tensor_value_info(q_reshape, TensorProto.FLOAT, [1, 8, 32, 32]),
                helper.make_tensor_value_info(k_reshape, TensorProto.FLOAT, [1, 8, 32, 32]),
                helper.make_tensor_value_info(q_transpose, TensorProto.FLOAT, [1, 8, 32, 32]),
                helper.make_tensor_value_info(k_transpose, TensorProto.FLOAT, [1, 8, 32, 32]),
                helper.make_tensor_value_info(scaled_q, TensorProto.FLOAT, [1, 8, 32, 32]),
                helper.make_tensor_value_info(qk, TensorProto.FLOAT, [1, 8, 32, 32]),
            )
        )
        outputs.append(helper.make_tensor_value_info(av, TensorProto.FLOAT16, [1, 8, 32, 32]))
    initializers.append(helper.make_tensor("shape", TensorProto.INT64, [4], [1, 8, 32, 32]))
    graph = helper.make_graph(nodes, "f3", inputs, outputs, initializers, value_info=value_info)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_full_model_plugin_rewrite_replaces_exact_six_qk_and_av_nodes(tmp_path):
    import onnx

    from search.model_families.lidar_cobevt.attention_plugin_rewrite import (
        rewrite_f3_attention_einsums,
    )

    source = tmp_path / "f3.onnx"
    output = tmp_path / "plugin.onnx"
    onnx.save(_six_block_f3_einsum_model(), source)

    report = rewrite_f3_attention_einsums(source, output, families=("QK", "AV"))
    rewritten = onnx.load(output)
    plugins = [node for node in rewritten.graph.node if node.op_type == "QKMixedAccumPlugin"]

    assert report["qk_replaced_count"] == 6
    assert report["av_replaced_count"] == 6
    assert len(report["ignored_node_names"]) == 6
    assert all(name.endswith("/Mul_6") for name in report["ignored_node_names"])
    assert len(plugins) == 12
    assert all(not value.startswith(("q_float", "k_float", "scaled_q")) for node in plugins[:6] for value in node.input)
    assert all(row["operand_dtype"] == "FP16" for row in report["replacement_records"])


def test_full_model_plugin_rewrite_fails_closed_on_incomplete_attention_graph(tmp_path):
    import onnx
    import pytest

    from search.model_families.lidar_cobevt.attention_plugin_rewrite import (
        rewrite_f3_attention_einsums,
    )

    model = _six_block_f3_einsum_model()
    del model.graph.node[-1]
    source = tmp_path / "incomplete.onnx"
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match="attention_plugin_rewrite_count_mismatch"):
        rewrite_f3_attention_einsums(source, tmp_path / "out.onnx", families=("QK", "AV"))


def test_cobevt_deployment_and_evaluation_accept_additional_plugin_paths(tmp_path):
    from search.integration.lidar_cobevt_evaluation_provider import (
        build_cobevt_evaluation_request,
    )
    from search.model_families.lidar_cobevt.deployment_recipe import (
        append_static_plugins,
    )

    command = append_static_plugins(
        ["trtexec", "--staticPlugins=scatter.so"], [tmp_path / "mixed.so"]
    )
    assert command[-1].endswith("mixed.so")
    request = build_cobevt_evaluation_request(
        engine_path="engine.plan", checkpoint="model.pth", model_config="config.yaml",
        heal_root="HEAL", device="cuda:7", output_path="evaluation.json",
        plugin_path="scatter.so", additional_plugin_paths=[tmp_path / "mixed.so"],
        fixed_k=29696, num_frames=10, warmup_frames=20,
        eval_manifest_path="smoke10.json", num_workers=8, ap_iou_backend="gpu",
    )
    assert request["additional_plugin_paths"] == [str(tmp_path / "mixed.so")]


def test_cobevt_formal_latency_separates_warmup_and_timed_rounds(tmp_path):
    from search.integration.lidar_cobevt_evaluation_provider import (
        build_cobevt_evaluation_request,
    )
    from search.integration.lidar_cobevt_evaluation_worker import _execution_rounds

    request = build_cobevt_evaluation_request(
        engine_path="engine.plan", checkpoint="model.pth", model_config="config.yaml",
        heal_root="HEAL", device="cuda:7", output_path="evaluation.json",
        plugin_path="scatter.so", fixed_k=29696, num_frames=500, warmup_frames=20,
        eval_manifest_path="fixed500.json", num_workers=8, ap_iou_backend="gpu",
        latency_rounds=5, warmup_latency_rounds=10,
    )
    assert request["latency_rounds"] == 5
    assert request["warmup_latency_rounds"] == 10
    assert _execution_rounds("warmup", request) == 10
    assert _execution_rounds("evaluation", request) == 5


def test_full_model_accumulation_profiles_are_minimal_and_joint_is_gated():
    from search.orchestration.lidar_cobevt_attention_accumulation_full_model import (
        full_model_profile_matrix,
        profile_evaluation_is_allowed,
    )

    profiles = {row.profile_id: row for row in full_model_profile_matrix()}
    assert set(profiles) == {
        "A0_F3_REFERENCE",
        "A2_QK_F16A32_PLUGIN",
        "B1_AV_F16A32_PLUGIN",
        "C1_QK_AV_F16A32_PLUGIN",
    }
    assert profiles["A2_QK_F16A32_PLUGIN"].plugin_families == ("QK",)
    assert profiles["B1_AV_F16A32_PLUGIN"].plugin_families == ("AV",)
    assert profiles["C1_QK_AV_F16A32_PLUGIN"].plugin_families == ("QK", "AV")
    assert not profile_evaluation_is_allowed(
        "C1_QK_AV_F16A32_PLUGIN", {"A2_QK_F16A32_PLUGIN": True}
    )
    assert profile_evaluation_is_allowed(
        "C1_QK_AV_F16A32_PLUGIN",
        {"A2_QK_F16A32_PLUGIN": True, "B1_AV_F16A32_PLUGIN": True},
    )


def test_fixed500_safety_classification_uses_fresh_f3_delta():
    from search.orchestration.lidar_cobevt_attention_accumulation_full_model import (
        classify_fixed500_delta,
    )

    assert classify_fixed500_delta(-0.0029) == "SAFE_FIXED500"
    assert classify_fixed500_delta(-0.0031) == "BORDERLINE_FIXED500"
    assert classify_fixed500_delta(-0.0101) == "UNSAFE_FIXED500"


def test_boundary_report_marks_plugin_oracle_experimental_even_with_level_a():
    from search.reporting.cobevt_attention_accumulation_boundary import (
        build_search_contract,
    )

    contract = build_search_contract(
        [
            {
                "profile": "C1_QK_AV_F16A32_PLUGIN",
                "implementation": "plugin_oracle",
                "evidence_level": "Level A",
                "fixed500_safety": "SAFE_FIXED500",
                "formal_latency_gain": True,
            },
            {
                "profile": "F3_REFERENCE",
                "implementation": "native_tensorrt",
                "evidence_level": "Level A",
                "fixed500_safety": "SAFE_FIXED500",
                "formal_latency_gain": True,
            },
        ]
    )

    assert "F3_REFERENCE" in contract["allowed"]
    assert "C1_QK_AV_F16A32_PLUGIN" in contract["experimental"]
    assert "C1_QK_AV_F16A32_PLUGIN" not in contract["allowed"]


def test_boundary_report_rejects_unknown_accumulator_from_search_contract():
    from search.reporting.cobevt_attention_accumulation_boundary import (
        build_search_contract,
    )

    contract = build_search_contract(
        [
            {
                "profile": "R1_NATIVE_FUSED_UNKNOWN",
                "implementation": "native_tensorrt",
                "evidence_level": "Level C",
                "fixed500_safety": "SAFE_FIXED500",
                "formal_latency_gain": True,
            }
        ]
    )
    assert contract["allowed"] == []
    assert "R1_NATIVE_FUSED_UNKNOWN" in contract["rejected"]


def test_boundary_root_conclusion_keeps_native_and_plugin_evidence_separate():
    from search.reporting.cobevt_attention_accumulation_boundary import (
        render_root_conclusion,
    )

    report = render_root_conclusion(
        [
            {
                "profile": "A0_F3_REFERENCE",
                "mAP": 0.6488,
                "delta_mAP_vs_fresh_F3": 0.0,
                "formal_forward_p50_ms": 5.08,
                "formal_speedup_vs_A0": 1.0,
                "fixed500_safety": "SAFE_FIXED500",
            },
            {
                "profile": "C1_QK_AV_F16A32_PLUGIN",
                "mAP": 0.6484,
                "delta_mAP_vs_fresh_F3": -0.0004,
                "formal_forward_p50_ms": 5.02,
                "formal_speedup_vs_A0": 1.012,
                "fixed500_safety": "SAFE_FIXED500",
            },
        ],
        {
            "allowed": ["A0_F3_REFERENCE"],
            "experimental": ["C1_QK_AV_F16A32_PLUGIN"],
            "rejected": [],
            "unsupported": [],
        },
    )

    assert "native TensorRT 10.9" in report
    assert "plugin oracle" in report
    assert "不等同于 native TensorRT" in report
    assert "R1" in report
    assert "INT8" in report


def test_boundary_report_prefers_only_valid_strict_formal_replay(tmp_path):
    import json

    from search.reporting.cobevt_attention_accumulation_boundary import (
        _select_formal_replay_root,
    )

    retry1 = tmp_path / "latency/formal_replay_retry1"
    retry2 = tmp_path / "latency/formal_replay_retry2"
    retry1.mkdir(parents=True)
    retry2.mkdir(parents=True)
    (retry2 / "formal_protocol.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "warmup_executions": 200,
                "timed_executions": 2500,
                "latency_rounds": 5,
            }
        )
    )
    assert _select_formal_replay_root(tmp_path) == retry2

    (retry2 / "formal_protocol.json").write_text(
        json.dumps({"status": "ok", "warmup_executions": 100})
    )
    assert _select_formal_replay_root(tmp_path) == retry1
