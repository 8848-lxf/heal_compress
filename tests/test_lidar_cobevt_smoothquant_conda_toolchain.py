from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _prefix() -> Path:
    return Path("/opt/conda/envs/modelopt")


def test_non_conda_nvcc_is_rejected():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_cuda_toolchain_paths,
    )

    with pytest.raises(ValueError, match="invalid_non_conda_cuda_toolchain"):
        validate_cuda_toolchain_paths(
            conda_prefix=_prefix(),
            python_path=_prefix() / "bin/python",
            nvcc_path=Path("/usr/bin/nvcc"),
            cuda_home=_prefix(),
            cudacxx=_prefix() / "bin/nvcc",
        )


def test_usr_local_cuda_nvcc_is_rejected():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        reject_non_conda_compiler_log,
    )

    with pytest.raises(ValueError, match="invalid_non_conda_cuda_toolchain"):
        reject_non_conda_compiler_log(
            "[1/2] /usr/local/cuda/bin/nvcc -gencode arch=compute_89,code=sm_89",
            conda_prefix=_prefix(),
        )


def test_cuda_home_must_be_inside_conda_prefix():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_cuda_toolchain_paths,
    )

    with pytest.raises(ValueError, match="cuda_home_outside_conda_prefix"):
        validate_cuda_toolchain_paths(
            conda_prefix=_prefix(),
            python_path=_prefix() / "bin/python",
            nvcc_path=_prefix() / "bin/nvcc",
            cuda_home=Path("/usr/local/cuda"),
            cudacxx=_prefix() / "bin/nvcc",
        )


def test_cudacxx_must_be_inside_conda_prefix():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_cuda_toolchain_paths,
    )

    with pytest.raises(ValueError, match="cudacxx_outside_conda_prefix"):
        validate_cuda_toolchain_paths(
            conda_prefix=_prefix(),
            python_path=_prefix() / "bin/python",
            nvcc_path=_prefix() / "bin/nvcc",
            cuda_home=_prefix(),
            cudacxx=Path("/usr/bin/nvcc"),
        )


def test_cxx_must_be_inside_conda_prefix():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_cuda_toolchain_paths,
    )

    with pytest.raises(ValueError, match="cxx_outside_conda_prefix"):
        validate_cuda_toolchain_paths(
            conda_prefix=_prefix(),
            python_path=_prefix() / "bin/python",
            nvcc_path=_prefix() / "bin/nvcc",
            cuda_home=_prefix(),
            cudacxx=_prefix() / "bin/nvcc",
            cxx_path=Path("/usr/bin/g++"),
        )


def test_sm89_compile_log_is_required():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_cuda_extension_build,
    )

    with pytest.raises(ValueError, match="sm89_compile_flag_missing"):
        validate_cuda_extension_build(
            compile_log=f"{_prefix()}/bin/nvcc -gencode arch=compute_86,code=sm_86",
            conda_prefix=_prefix(),
            runtime_backend="cuda",
            runtime_success=True,
        )


def test_cpu_fallback_is_rejected():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_cuda_extension_build,
    )

    with pytest.raises(ValueError, match="cuda_extension_cpu_fallback"):
        validate_cuda_extension_build(
            compile_log=f"{_prefix()}/bin/nvcc -gencode arch=compute_89,code=sm_89",
            conda_prefix=_prefix(),
            runtime_backend="cpu_fallback",
            runtime_success=True,
        )


def test_extension_cache_is_bound_to_environment():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        extension_cache_identity,
    )

    a = extension_cache_identity(_prefix(), "nvcc-a", "8.9")
    b = extension_cache_identity(Path("/different/modelopt"), "nvcc-a", "8.9")
    assert a != b


def test_toolchain_manifest_contains_compiler_hash():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        build_toolchain_manifest,
    )

    payload = build_toolchain_manifest(
        conda_prefix=_prefix(),
        python_path=_prefix() / "bin/python",
        nvcc_path=_prefix() / "bin/nvcc",
        nvcc_sha256="a" * 64,
        nvcc_version="11.8.89",
        cuda_home=_prefix(),
        cudacxx=_prefix() / "bin/nvcc",
        sm89_supported=True,
        extension_binary_sha256="b" * 64,
    )
    assert payload["nvcc_sha256"] == "a" * 64
    assert payload["extension_binary_sha256"] == "b" * 64


def test_tensorrt_builder_does_not_inherit_cuda_launch_blocking():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        sanitize_tensorrt_builder_environment,
    )

    cleaned = sanitize_tensorrt_builder_environment(
        {
            "CUDA_LAUNCH_BLOCKING": "1",
            "CUDA_VISIBLE_DEVICES": "7",
            "LD_LIBRARY_PATH": "/opt/conda/envs/modelopt/lib",
        }
    )
    assert "CUDA_LAUNCH_BLOCKING" not in cleaned
    assert cleaned["CUDA_VISIBLE_DEVICES"] == "7"
    assert cleaned["LD_LIBRARY_PATH"] == "/opt/conda/envs/modelopt/lib"


def test_realization_checker_accepts_explicit_smoothquant_int8_overrides():
    from quantization.tensorrt.precision_checker import validate_precision_realization
    from quantization.types import (
        CanonicalPrecisionEntry,
        CanonicalPrecisionMappingResult,
    )

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="fusion.q_proj",
                canonical_node_name="q_projection",
                precision_group="attention",
                requested_precision="fp16",
                realized_request_precision="fp16",
            ),
            CanonicalPrecisionEntry(
                module_path="fusion.v_proj",
                canonical_node_name="v_projection",
                precision_group="attention",
                requested_precision="fp16",
                realized_request_precision="fp16",
            ),
        ]
    )
    rows = [
        {"Name": "q_projection", "LayerType": "gemm", "Precision": "Int8"},
        {"Name": "v_projection", "LayerType": "gemm", "Precision": "Half"},
    ]
    report = validate_precision_realization(
        rows,
        mapping,
        expected_precision_overrides={"fusion.q_proj": "int8"},
    )
    assert report.passed is True
    assert report.requested_int8_count == 1
    assert report.realized_int8_count == 1


def test_smoothquant_int8_fusion_layer_is_weighted_compute_evidence():
    from quantization.tensorrt.precision_checker import validate_precision_realization
    from quantization.types import (
        CanonicalPrecisionEntry,
        CanonicalPrecisionMappingResult,
    )

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="fusion.v_proj",
                canonical_node_name="v_projection",
                precision_group="attention",
                requested_precision="fp16",
                realized_request_precision="fp16",
            )
        ]
    )
    report = validate_precision_realization(
        [
            {
                "Name": "v_projection_fused",
                "LayerType": "fusion",
                "Inputs": [{"Format/Datatype": "Int8"}],
                "Outputs": [{"Format/Datatype": "Half"}],
                "TacticName": "sm80_xmma_gemm_i8f32_i8i32_f32_by_fusion_tactic",
                "Metadata": "[ONNX Layer: v_projection]",
            }
        ],
        mapping,
        expected_precision_overrides={"fusion.v_proj": "int8"},
    )
    assert report.passed is True
    assert report.realized_int8_count == 1


def test_non_gemm_fusion_is_not_generic_weighted_compute_evidence():
    from quantization.tensorrt.layer_info import is_weighted_compute_layer

    assert not is_weighted_compute_layer(
        {
            "LayerType": "fusion",
            "TacticName": "pointwise_activation_fusion",
            "Metadata": "[ONNX Layer: activation]",
        }
    )


def test_alpha_selection_uses_normalized_projection_qk_and_softmax_metrics():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        choose_smoothquant_alpha,
    )

    rows = [
        {"alpha": 0.5, "relative_l2": 0.05, "qk_relative_l2": 0.20, "softmax_js": 0.02},
        {"alpha": 0.6, "relative_l2": 0.06, "qk_relative_l2": 0.06, "softmax_js": 0.01},
        {"alpha": 0.7, "relative_l2": 0.10, "qk_relative_l2": 0.05, "softmax_js": 0.005},
    ]
    selected = choose_smoothquant_alpha(rows)
    assert selected["alpha"] == pytest.approx(0.6)
    assert set(selected["normalized_metrics"]) == {
        "relative_l2",
        "qk_relative_l2",
        "softmax_js",
    }


def test_alpha_grid_contains_requested_values():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        SMOOTHQUANT_ALPHA_GRID,
    )

    assert SMOOTHQUANT_ALPHA_GRID == (0.5, 0.6, 0.7, 0.75, 0.8)


def test_smoothquant_reparameterization_is_equivalent_without_rounding():
    torch = pytest.importorskip("torch")
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        smoothquant_reparameterize,
    )

    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    w = torch.tensor([[2.0, -1.0, 4.0], [-3.0, 0.5, 2.0]], dtype=torch.float64)
    scale = torch.tensor([0.5, 2.0, 4.0], dtype=torch.float64)
    x_smooth, w_smooth = smoothquant_reparameterize(x, w, scale)
    assert torch.allclose(x @ w.T, x_smooth @ w_smooth.T, atol=1e-12, rtol=1e-12)


def test_larger_alpha_strengthens_activation_smoothing():
    torch = pytest.importorskip("torch")
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        smoothquant_scale,
    )

    activation_max = torch.tensor([100.0, 10.0], dtype=torch.float64)
    weight_max = torch.tensor([1.0, 1.0], dtype=torch.float64)
    low = smoothquant_scale(activation_max, weight_max, alpha=0.5)
    high = smoothquant_scale(activation_max, weight_max, alpha=0.8)
    assert high[0] / high[1] > low[0] / low[1]


def test_pre_quant_scale_must_cover_every_input_channel():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_pre_quant_scale,
    )

    with pytest.raises(ValueError, match="pre_quant_scale_incomplete"):
        validate_pre_quant_scale([1.0, 2.0], input_features=3)


def test_sq1_and_sq2_roles_remain_exact():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        smoothquant_profiles,
    )

    profiles = {row.profile_id: row for row in smoothquant_profiles()}
    assert profiles["SQ1"].int8_projection_roles == ("q_projection", "k_projection")
    assert profiles["SQ2"].int8_projection_roles == (
        "q_projection",
        "k_projection",
        "v_projection",
    )


def test_sq3_alpha_sweep_requires_successful_sq2_build():
    from search.orchestration.lidar_cobevt_smoothquant_projection import (
        resolve_alpha_sweep_profiles,
    )

    with pytest.raises(ValueError, match="smoothquant_alpha_prerequisite_failed:SQ3:SQ2"):
        resolve_alpha_sweep_profiles(("SQ3",), successful_profile_ids=())
    selected = resolve_alpha_sweep_profiles(
        ("SQ3",), successful_profile_ids=("SQ2",)
    )
    assert [row.profile_id for row in selected] == ["SQ3"]


def test_smoothquant_precision_inventory_summary_is_role_and_tactic_based():
    from search.reporting.cobevt_smoothquant_conda_toolchain import (
        build_profile_realization_row,
        summarize_precision_inventory,
    )

    rows = [
        {
            "block_id": "layers.0.window_attention",
            "role": "q_projection",
            "requested_dtype": "INT8",
            "tactic_precision": "INT8",
            "realized_input_dtypes": ["INT8"],
            "realized_output_dtypes": ["FP32"],
            "tactic_names": ["sm80_xmma_gemm_i8f32_i8i32_f32"],
        },
        {
            "block_id": "layers.0.window_attention",
            "role": "v_projection",
            "requested_dtype": "FP16",
            "tactic_precision": "UNKNOWN",
            "realized_input_dtypes": ["FP16"],
            "realized_output_dtypes": ["FP16"],
            "tactic_names": [],
        },
        {
            "block_id": "layers.0.window_attention",
            "role": "qk_matmul",
            "requested_dtype": "FP32",
            "tactic_precision": "FP32",
            "realized_input_dtypes": ["FP32", "FP32"],
            "realized_output_dtypes": ["FP32"],
            "tactic_names": ["sm80_xmma_gemm_f32f32_f32f32_f32"],
        },
    ]
    summary = summarize_precision_inventory("SQ1", rows)
    assert summary[0]["requested_realized_match"] is True
    assert summary[0]["realized_precision"] == "INT8"
    assert summary[2]["qk_accumulator_precision"] == "FP32"
    assert summary[2]["requested_realized_match"] is True
    profile = build_profile_realization_row(
        "SQ1", summary, status="ok", engine_sha256="engine"
    )
    assert profile["selected_projection_count"] == 1
    assert profile["realized_int8_projection_count"] == 1
    assert profile["requested_realized_match"] is True


def test_qk_requires_dq_to_fp32_and_native_int8_is_forbidden():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_smoothquant_realization,
    )

    with pytest.raises(ValueError, match="native_int8_qk_forbidden"):
        validate_smoothquant_realization(
            projection_rows=[{"role": "q_projection", "requested": "INT8", "realized": "INT8"}],
            required_projection_roles=("q_projection",),
            q_dtype="INT8",
            k_dtype="INT8",
            qk_input_dtypes=("INT8", "INT8"),
            qk_output_dtype="INT32",
            qk_accumulator_precision="INT32",
        )


def test_requested_realized_mismatch_fails_closed():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_smoothquant_realization,
    )

    with pytest.raises(ValueError, match="smoothquant_projection_precision_fallback"):
        validate_smoothquant_realization(
            projection_rows=[{"role": "q_projection", "requested": "INT8", "realized": "FP16"}],
            required_projection_roles=("q_projection",),
            q_dtype="FP32",
            k_dtype="FP32",
            qk_input_dtypes=("FP32", "FP32"),
            qk_output_dtype="FP32",
            qk_accumulator_precision="FP32",
        )


def test_s2_retained_indices_must_not_change():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_s2_retained_indices,
    )

    with pytest.raises(ValueError, match="s2_retained_indices_changed"):
        validate_s2_retained_indices(
            expected={"block": {"qk": [0, 1], "v": [0, 1]}},
            actual={"block": {"qk": [0, 2], "v": [0, 1]}},
        )


def test_additive_lut_cannot_be_restored_to_search_ready():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        build_latency_model_contract,
    )

    contract = build_latency_model_contract(
        unit_lut_status="not_additive",
        full_engine_anchor_hashes=(hashlib.sha256(b"S0").hexdigest(),),
    )
    assert contract == {
        "unit_lut": "not_additive",
        "formal_source": "full_engine_action_anchors",
        "full_engine_anchor_hashes": [hashlib.sha256(b"S0").hexdigest()],
    }


def test_full_engine_anchor_is_required():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        build_latency_model_contract,
    )

    with pytest.raises(ValueError, match="full_engine_anchor_required"):
        build_latency_model_contract(
            unit_lut_status="not_additive", full_engine_anchor_hashes=()
        )


def test_staged_exports_must_use_fresh_processes_and_stop_after_failure():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        summarize_staged_exports,
    )

    summary = summarize_staged_exports(
        [
            {"stage": "E0", "process_id": 101, "return_code": 0},
            {"stage": "E1", "process_id": 102, "return_code": 0},
            {"stage": "E2", "process_id": 103, "return_code": 139},
        ]
    )
    assert summary["first_failing_stage"] == "E2"
    assert summary["all_completed"] is False

    with pytest.raises(ValueError, match="staged_export_process_reused"):
        summarize_staged_exports(
            [
                {"stage": "E0", "process_id": 101, "return_code": 0},
                {"stage": "E1", "process_id": 101, "return_code": 0},
            ]
        )
    with pytest.raises(ValueError, match="stage_executed_after_failure"):
        summarize_staged_exports(
            [
                {"stage": "E0", "process_id": 101, "return_code": 1},
                {"stage": "E1", "process_id": 102, "return_code": 0},
            ]
        )


def test_final_contract_never_admits_incomplete_smoothquant_profile():
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        classify_smoothquant_candidate,
    )

    assert classify_smoothquant_candidate(
        toolchain_pass=True,
        extension_cuda=True,
        export_success=True,
        build_success=True,
        realized_precision_pass=False,
        smoke10_pass=True,
        fixed50_pass=True,
        formal_latency_pass=True,
    ) == "blocked"
    assert classify_smoothquant_candidate(
        toolchain_pass=True,
        extension_cuda=True,
        export_success=True,
        build_success=True,
        realized_precision_pass=True,
        smoke10_pass=True,
        fixed50_pass=True,
        formal_latency_pass=True,
    ) == "allowed"


def test_export_stage_specs_keep_sq3_gated_on_sq2():
    from search.orchestration.lidar_cobevt_smoothquant_conda_toolchain import (
        export_stage_specs,
    )

    specs = {row.stage: row for row in export_stage_specs()}
    assert tuple(specs) == ("E0", "E1", "E2", "E3", "E4", "E5")
    assert specs["E0"].scope == "toy_linear"
    assert specs["E1"].scope == "real_q_projection"
    assert specs["E2"].scope == "single_attention_qk"
    assert specs["E3"].profile_id == "SQ1"
    assert specs["E4"].profile_id == "SQ2"
    assert specs["E5"].profile_id == "SQ3"
    assert specs["E5"].requires_stage == "E4"


def test_smoothquant_projection_qdq_must_remain_adjacent():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        validate_projection_qdq_adjacency,
    )

    def graph(with_cast: bool):
        nodes = [
            helper.make_node("DequantizeLinear", ["qa", "s", "z"], ["a_dq"], name="a_dq"),
            helper.make_node("DequantizeLinear", ["qw", "s", "z"], ["w_dq"], name="w_dq"),
        ]
        activation = "a_dq"
        if with_cast:
            nodes.append(helper.make_node("Cast", [activation], ["a_half"], name="a_half", to=TensorProto.FLOAT16))
            activation = "a_half"
        nodes.append(helper.make_node("MatMul", [activation, "w_dq"], ["y"], name="projection"))
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
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            )
        )

    validate_projection_qdq_adjacency(graph(False), ("projection",))
    with pytest.raises(ValueError, match="projection_qdq_not_adjacent"):
        validate_projection_qdq_adjacency(graph(True), ("projection",))


def test_smoothquant_projection_rewrite_removes_only_intervening_fp16_casts():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from search.model_families.lidar_cobevt.conda_cuda_toolchain import (
        restore_projection_qdq_adjacency,
        validate_projection_qdq_adjacency,
    )

    nodes = [
        helper.make_node("DequantizeLinear", ["qa", "s", "z"], ["a_dq"], name="a_dq"),
        helper.make_node("DequantizeLinear", ["qw", "s", "z"], ["w_dq"], name="w_dq"),
        helper.make_node("Transpose", ["w_dq"], ["w_t"], name="w_t"),
        helper.make_node("Cast", ["a_dq"], ["a_half"], name="a_half", to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["w_t"], ["w_half"], name="w_half", to=TensorProto.FLOAT16),
        helper.make_node("MatMul", ["a_half", "w_half"], ["y"], name="projection"),
        helper.make_node("Identity", ["y"], ["outside"], name="outside"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "qdq",
            [
                helper.make_tensor_value_info("qa", TensorProto.INT8, [1, 4]),
                helper.make_tensor_value_info("qw", TensorProto.INT8, [4, 4]),
                helper.make_tensor_value_info("s", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("z", TensorProto.INT8, []),
            ],
            [helper.make_tensor_value_info("outside", TensorProto.FLOAT, [1, 4])],
        )
    )

    report = restore_projection_qdq_adjacency(
        model, ("projection",), output_cast_precisions={"projection": "FP16"}
    )
    validate_projection_qdq_adjacency(model, ("projection",))
    projection = next(node for node in model.graph.node if node.name == "projection")
    output_cast = next(node for node in model.graph.node if node.name == "projection__output_fp16")
    outside = next(node for node in model.graph.node if node.name == "outside")

    assert list(projection.input) == ["a_dq", "w_t"]
    assert list(projection.output) == ["y__before_smoothquant_output_cast"]
    assert list(output_cast.input) == ["y__before_smoothquant_output_cast"]
    assert list(output_cast.output) == ["y"]
    assert output_cast.attribute[0].i == TensorProto.FLOAT16
    assert list(outside.input) == ["y"]
    assert report[0]["removed_cast_nodes"] == ["a_half", "w_half"]
    assert report[0]["output_cast_precision"] == "FP16"
