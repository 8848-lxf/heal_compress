from __future__ import annotations

import pytest


def test_phenotype_keeps_output_and_accumulator_precision_separate():
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        ComputePhenotype,
    )

    o16 = ComputePhenotype("FP16", "FP16", "FP32", "FP16")
    o32 = ComputePhenotype("FP16", "FP16", "FP32", "FP32")
    assert o16.name == "F16A32O16"
    assert o32.name == "F16A32O32"
    assert o16 != o32


def test_materialized_cast_is_not_f16a32():
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        classify_native_phenotype,
    )

    result = classify_native_phenotype(
        storage_precision="FP16",
        operand_precision="FP32",
        output_precision="FP32",
        accumulator_precision="FP32",
        materialized_cast=True,
        evidence_level="A",
    )
    assert result.name == "F32A32_AFTER_MATERIALIZED_CAST"
    assert result.is_native_f16a32 is False


def test_level_b_cannot_be_upgraded_by_numeric_fingerprint():
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        classify_evidence,
    )

    assert classify_evidence(
        direct_accumulator=None,
        tensor_core_fp16=True,
        no_materialized_cast=True,
        oracle_matches_f16a32=True,
        oracle_separates_f16a16=True,
    ) == "LEVEL_B_STRONG"
    assert classify_evidence(
        direct_accumulator=None,
        tensor_core_fp16=True,
        no_materialized_cast=True,
        oracle_matches_f16a32=False,
        oracle_separates_f16a16=False,
    ) == "LEVEL_C_UNKNOWN"


def test_cache_binding_rejects_shape_gpu_trt_or_graph_mismatch():
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        CacheBinding,
        validate_cache_binding,
    )

    expected = CacheBinding("g", "SM89", "10.9.0.34", "11.8", "shape-a", "cache")
    assert validate_cache_binding(expected, expected) is True
    for field, value in (
        ("graph_signature", "other"),
        ("gpu_arch", "SM90"),
        ("tensorrt_version", "10.10"),
        ("shape_key", "shape-b"),
        ("cache_sha256", "other"),
    ):
        data = expected.__dict__.copy()
        data[field] = value
        with pytest.raises(ValueError, match="timing_cache_binding_mismatch"):
            validate_cache_binding(expected, CacheBinding(**data))


def test_only_six_of_six_level_a_native_shapes_can_form_profile():
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        shape_profile_gate,
    )

    rows = [
        {"block": i, "evidence_level": "LEVEL_A_DIRECT", "pinning_stable": True}
        for i in range(6)
    ]
    assert shape_profile_gate(rows, role="QK").status == "NATIVE_F16A32_PINNABLE"
    rows[4]["evidence_level"] = "LEVEL_B_STRONG"
    assert shape_profile_gate(rows, role="QK").status == "NATIVE_PROBABLE_F16A32"
    rows[5]["evidence_level"] = "LEVEL_C_UNKNOWN"
    assert shape_profile_gate(rows, role="QK").status == "UNKNOWN"


def test_partial_native_profile_cannot_be_called_full_native():
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        shape_profile_gate,
    )

    rows = [
        {"block": i, "evidence_level": "LEVEL_A_DIRECT", "pinning_stable": True}
        for i in range(5)
    ]
    assert shape_profile_gate(rows, role="AV").status == "UNKNOWN"


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.0, "SAFE"), (-0.003, "SAFE"), (-0.0031, "BORDERLINE"), (-0.0101, "UNSAFE")],
)
def test_native_fixed500_safety(delta, expected):
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        classify_fixed500,
    )

    assert classify_fixed500(delta, evaluated=500, skipped=0, finite=True) == expected


def test_plugin_and_no_latency_profiles_are_not_allowed():
    from search.model_families.lidar_cobevt.native_tactic_contract import (
        profile_search_eligibility,
    )

    assert profile_search_eligibility(
        implementation="plugin_oracle",
        shape_status="NATIVE_F16A32_PINNABLE",
        full_engine_preserved=True,
        fixed500_safety="SAFE",
        formal_latency_gain=True,
    ).allowed is False
    assert profile_search_eligibility(
        implementation="native_tensorrt",
        shape_status="NATIVE_F16A32_PINNABLE",
        full_engine_preserved=True,
        fixed500_safety="SAFE",
        formal_latency_gain=False,
    ).allowed is False


def test_installed_capability_parser_distinguishes_editable_cache_and_accumulator_api():
    from search.orchestration.lidar_cobevt_native_tactic_capability import (
        parse_header_evidence,
    )

    result = parse_header_evidence(
        """
        enum class BuilderFlag { kEDITABLE_TIMING_CACHE = 27 };
        class ITimingCache { queryKeys; query; update; };
        class IMatrixMultiplyLayer { addMatrixMultiply; };
        class IAlgorithmSelector {}; // deprecated Deprecated in TensorRT 10.8
        """
    )
    assert result["editable_timing_cache_flag"] is True
    assert result["timing_cache_update_api"] is True
    assert result["matrix_multiply_layer_present"] is True
    assert result["separate_accumulator_precision_api"] is False
    assert result["algorithm_selector_deprecated"] is True


def test_capability_rows_make_f16a32_non_separately_expressible_without_api():
    from search.orchestration.lidar_cobevt_native_tactic_capability import (
        build_capability_rows,
    )

    rows = build_capability_rows(
        trt_version="10.9.0.34",
        editable_timing_cache=True,
        timing_cache_update=True,
        accumulator_api=False,
        inspector_accumulator=False,
        algorithm_selector_deprecated=True,
    )
    f16 = next(row for row in rows if row["phenotype"] == "F16A32")
    assert f16["native_request_status"] == "not_separately_expressible"
    assert f16["maximum_evidence_level"] == "LEVEL_C_UNKNOWN"


def test_conda_toolchain_guard_rejects_system_nvcc():
    from search.orchestration.lidar_cobevt_native_tactic_capability import (
        validate_conda_toolchain,
    )

    with pytest.raises(ValueError, match="system_nvcc_forbidden"):
        validate_conda_toolchain(
            conda_prefix="/envs/modelopt",
            nvcc_path="/usr/bin/nvcc",
            cuda_home="/usr/bin",
            arch_list=["sm_89"],
        )
    assert validate_conda_toolchain(
        conda_prefix="/envs/modelopt",
        nvcc_path="/envs/modelopt/bin/nvcc",
        cuda_home="/envs/modelopt",
        arch_list=["sm_89"],
    )["cpu_fallback"] is False


def test_conda_toolchain_guard_accepts_multiline_nvcc_arch_output():
    from search.orchestration.lidar_cobevt_native_tactic_capability import (
        validate_conda_toolchain,
    )

    result = validate_conda_toolchain(
        conda_prefix="/envs/modelopt",
        nvcc_path="/envs/modelopt/bin/nvcc",
        cuda_home="/envs/modelopt",
        arch_list=["sm_86\nsm_89\nsm_90"],
    )
    assert result["sm89_supported"] is True


def test_editable_log_parser_collects_key_all_tactics_and_selected_hash():
    from search.model_families.lidar_cobevt.native_tactic_evidence import (
        parse_editable_timing_log,
    )

    lines = [
        "Autotuning op qk_matrix_multiply(key: 0x0123456789abcdef0123456789abcdef):",
        "tactic_id, cost(in ms), cost/fastest_cost, prediction_correlation, kernel_name, tactic_hash, tunable_parameter",
        " 1, 0.0030, 1.00000, 1.0, sm80_xmma_gemm_f16f16_f16f32_f32_nn, 0x111,",
        " 2, 0.0040, 1.33333, 1.0, sm80_xmma_gemm_f16f16_f16f16_f16_nn, 0x222,",
        "The selected tactic is (tactic hash, cost(in ms)):0x111, 0.0030",
    ]
    records = parse_editable_timing_log(lines)
    assert records[0]["op"] == "qk_matrix_multiply"
    assert records[0]["key"] == "0x0123456789abcdef0123456789abcdef"
    assert len(records[0]["available_tactics"]) == 2
    assert records[0]["selected_tactic"] == "0x111"


def test_tactic_kernel_metadata_is_level_a_and_keeps_output_precision():
    from search.model_families.lidar_cobevt.native_tactic_evidence import (
        classify_tactic_kernel,
    )

    f32 = classify_tactic_kernel("sm80_xmma_gemm_f16f16_f16f32_f32_nn")
    f16 = classify_tactic_kernel("sm80_xmma_gemm_f16f16_f16f16_f16_nn")
    opaque = classify_tactic_kernel("ampere_h16816gemm_64x64_ldg8_stages_64x6_nn_v1")
    assert (f32["phenotype"], f32["evidence_level"]) == ("F16A32O32", "LEVEL_A_DIRECT")
    assert (f16["phenotype"], f16["evidence_level"]) == ("F16A16O16", "LEVEL_A_DIRECT")
    assert opaque["phenotype"] == "UNKNOWN_ACCUM"
    assert opaque["evidence_level"] == "LEVEL_C_UNKNOWN"


def test_layer_output_dtype_overrides_kernel_output_token():
    from search.model_families.lidar_cobevt.native_tactic_evidence import (
        classify_tactic_kernel,
        realize_output_phenotype,
    )

    kernel = classify_tactic_kernel("sm80_xmma_gemm_f16f16_f16f32_f32_nn")
    assert realize_output_phenotype(kernel, "FP16") == "F16A32O16"
    assert realize_output_phenotype(kernel, "FP32") == "F16A32O32"


def test_numerical_boundary_marks_nonzero_reference_zero_output_unsafe():
    from search.model_families.lidar_cobevt.native_tactic_evidence import (
        classify_numerical_boundary,
    )

    assert classify_numerical_boundary(
        finite=True, reference_norm=1.0e-6, output_zero_ratio=1.0
    ) == "NUMERICAL_UNSAFE_OUTPUT_UNDERFLOW"
    assert classify_numerical_boundary(
        finite=True, reference_norm=1.0, output_zero_ratio=0.0
    ) == "NUMERICAL_SAFE"


def test_tactic_selection_prefers_f16a32_o32_and_returns_none_when_absent():
    from search.model_families.lidar_cobevt.native_tactic_evidence import (
        parse_editable_timing_log,
        select_f16a32_tactic,
    )

    records = parse_editable_timing_log(
        [
            "Autotuning op x(key: 0x0123456789abcdef0123456789abcdef):",
            "tactic_id, cost(in ms), cost/fastest_cost, prediction_correlation, kernel_name, tactic_hash, tunable_parameter",
            " 1, 0.002, 1.0, 1.0, sm80_xmma_gemm_f16f16_f16f32_f16_nn, 0x1,",
            " 2, 0.003, 1.5, 1.0, sm80_xmma_gemm_f16f16_f16f32_f32_nn, 0x2,",
        ]
    )
    assert select_f16a32_tactic(records[0])["tactic_hash"] == "0x2"
    assert select_f16a32_tactic({"available_tactics": []}) is None


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("/layers_0/window_attention/fn/Einsum", ("QK", 0, "window")),
        ("/layers_0/grid_attention/fn/Einsum_1", ("AV", 0, "grid")),
        ("/layers.2/window_attention/fn/Einsum(key suffix)", ("QK", 2, "window")),
        ("unrelated/gemm", None),
    ],
)
def test_full_graph_attention_role_parser(op, expected):
    from search.model_families.lidar_cobevt.native_full_tactic_pinning import (
        parse_attention_role,
    )

    assert parse_attention_role(op) == expected


def test_full_graph_target_selection_requires_exact_six_layers():
    from search.model_families.lidar_cobevt.native_full_tactic_pinning import (
        choose_full_graph_targets,
    )

    def record(layer, kind, suffix=""):
        return {
            "op": f"/layers_{layer}/{kind}_attention/fn/Einsum{suffix}",
            "key": f"0x{layer + (10 if kind == 'grid' else 0):032x}",
            "available_tactics": [
                {
                    "tactic_hash": f"0x{layer + 1:x}",
                    "cost_ms": 0.1,
                    "kernel_name": "sm80_xmma_gemm_f16f16_f16f32_f32_nn",
                    "kernel_evidence": {
                        "phenotype": "F16A32O32",
                        "compute_phenotype": "F16A32",
                        "evidence_level": "LEVEL_A_DIRECT",
                    },
                }
            ],
        }

    records = [record(i, kind) for i in range(3) for kind in ("window", "grid")]
    assert len(choose_full_graph_targets(records, role="QK")) == 6
    with pytest.raises(ValueError, match="full_graph_expected_six_targets"):
        choose_full_graph_targets(records[:-1], role="QK")


def test_full_engine_preservation_does_not_accept_fused_or_plugin_layers():
    from search.model_families.lidar_cobevt.native_full_tactic_pinning import (
        verify_full_engine_preservation,
    )

    targets = [
        {
            "role": "QK",
            "block": i // 2,
            "attention_kind": kind,
            "requested_tactic_hash": f"0x{i + 1:x}",
            "requested_kernel_name": f"kernel_{i}",
        }
        for i, kind in enumerate(("window", "grid", "window", "grid", "window", "grid"))
    ]
    layers = [
        {
            "Name": f"/layers_{i // 2}/{kind}_attention/fn/Einsum",
            "LayerType": "gemm",
            "TacticName": f"kernel_{i}",
            "Outputs": [{"Format/Datatype": "Half"}],
        }
        for i, kind in enumerate(("window", "grid", "window", "grid", "window", "grid"))
    ]
    result = verify_full_engine_preservation(targets, {"Layers": layers})
    assert result["preserved"] is True
    layers[2]["LayerType"] = "plugin"
    assert verify_full_engine_preservation(targets, {"Layers": layers})["preserved"] is False
    layers[2]["LayerType"] = "gemm"
    layers[2]["TacticName"] = "_gemm_mha_v2"
    assert verify_full_engine_preservation(targets, {"Layers": layers})["preserved"] is False


def test_native_qk_profile_changes_only_qk_compute_from_f3_contract():
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        attention_boundary_profile,
    )

    f3 = attention_boundary_profile("F3_rest_fp16_qk_fp32_minimal_island")
    native = attention_boundary_profile("N1_rest_fp16_qk_native_f16a32")
    changed = {
        role
        for role in f3.role_dtypes
        if f3.role_dtypes[role] != native.role_dtypes[role]
    }
    assert changed == {"qk_matmul"}
    assert native.role_dtypes["qk_matmul"] == "FP16"
    assert native.role_dtypes["qk_scale"] == "FP32"
    assert native.role_dtypes["softmax"] == "FP16"
    assert native.role_dtypes["av_matmul"] == "FP16"
    assert native.external_weighted_dtype == "FP16"
    assert "qk_matmul" not in native.output_recovery_roles
