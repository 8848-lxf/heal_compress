from __future__ import annotations

import math
import json
import inspect
from pathlib import Path

import pytest


def test_mandatory_structure_profiles_are_exact_and_include_v_only():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        mandatory_structure_profiles,
    )

    rows = mandatory_structure_profiles()
    assert [(row.profile_id, row.d_qk, row.d_v) for row in rows] == [
        ("S0", 32, 32),
        ("S1", 24, 24),
        ("S2", 16, 16),
        ("S3", 24, 32),
        ("S4", 32, 24),
    ]
    assert rows[3].family == "qk_only"
    assert rows[4].family == "v_only"
    assert all(row.embed_dim == 256 and row.heads == 8 for row in rows)


def test_cross_precision_contracts_only_change_attention_roles():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        cross_precision_contracts,
    )

    p0, f3 = cross_precision_contracts()
    assert p0.external_weighted_dtype == f3.external_weighted_dtype == "FP16"
    assert set(p0.role_dtypes.values()) == {"FP32"}
    assert f3.role_dtypes == {
        "layernorm": "FP32",
        "q_projection": "FP16",
        "k_projection": "FP16",
        "v_projection": "FP16",
        "qk_scale": "FP32",
        "qk_matmul": "FP32",
        "softmax": "FP16",
        "av_matmul": "FP16",
        "output_projection": "FP16",
        "residual_add": "FP16",
    }


def test_structure_precision_interaction_uses_same_s0_reference():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        structure_precision_interaction,
    )

    result = structure_precision_interaction(
        s0_p0_map=0.65,
        s0_f3_map=0.648,
        candidate_p0_map=0.64,
        candidate_f3_map=0.637,
    )
    assert result["delta_prune"] == pytest.approx(-0.01)
    assert result["delta_f3_base"] == pytest.approx(-0.002)
    assert result["delta_f3_candidate"] == pytest.approx(-0.003)
    assert result["interaction"] == pytest.approx(-0.001)


def test_smoothquant_profiles_are_ordered_and_sq3_depends_on_sq2():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        smoothquant_profiles,
    )

    rows = smoothquant_profiles()
    assert [row.profile_id for row in rows] == ["SQ0", "SQ1", "SQ2", "SQ3"]
    assert rows[0].int8_projection_roles == ()
    assert rows[1].int8_projection_roles == ("q_projection", "k_projection")
    assert rows[2].int8_projection_roles == (
        "q_projection",
        "k_projection",
        "v_projection",
    )
    assert rows[3].requires_successful_profile == "SQ2"
    assert "output_projection" in rows[3].int8_projection_roles
    assert all(row.qk_precision == "FP32" for row in rows)


def test_selective_smoothquant_config_disables_unselected_modules():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        selective_smoothquant_config,
    )

    config = selective_smoothquant_config(
        ("q_projection", "k_projection"), alpha=0.5
    )
    assert config["algorithm"] == {"method": "smoothquant", "alpha": 0.5}
    assert config["quant_cfg"]["default"] == {"enable": False}
    assert config["quant_cfg"]["*q_proj*weight_quantizer"]["axis"] == 0
    assert config["quant_cfg"]["*q_proj*input_quantizer"]["axis"] is None
    assert "*v_proj*weight_quantizer" not in config["quant_cfg"]


def test_alpha_selection_is_deterministic_and_finite():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        choose_smoothquant_alpha,
    )

    rows = [
        {"alpha": 0.7, "relative_l2": 0.1, "qk_relative_l2": 0.01, "softmax_js": 0.02},
        {"alpha": 0.3, "relative_l2": 0.1, "qk_relative_l2": 0.01, "softmax_js": 0.02},
        {"alpha": 0.5, "relative_l2": 0.08, "qk_relative_l2": 0.01, "softmax_js": 0.03},
    ]
    assert choose_smoothquant_alpha(rows)["alpha"] == 0.5
    with pytest.raises(ValueError, match="smoothquant_metric_nonfinite"):
        choose_smoothquant_alpha(
            [{"alpha": 0.5, "relative_l2": math.nan, "qk_relative_l2": 0.0, "softmax_js": 0.0}]
        )


def _blocks():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        AttentionDeploymentShape,
    )

    return tuple(
        AttentionDeploymentShape(
            block_id=f"layers.{layer}.{kind}",
            attention_kind=kind,
            batch=512,
            num_heads=8,
            sq=32,
            skv=32,
            external_embed_dim=256,
            input_layout="BHSD",
            mask_kind="cobevt_relation_mask",
        )
        for layer in range(3)
        for kind in ("window", "grid")
    )


def test_f3_lut_matrix_has_54_complete_unique_records():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        build_f3_lut_candidates,
    )

    rows = build_f3_lut_candidates(
        _blocks(),
        gpu_arch="SM89",
        gpu_uuid="GPU-test",
        tensorrt_version="10.9.0.34",
        cuda_version="11.8",
        driver="580.105.08",
    )
    assert len(rows) == 54
    assert len({row.key_hash for row in rows}) == 54
    assert {(row.d_qk, row.d_v) for row in rows} == {
        (qk, v) for qk in (16, 24, 32) for v in (16, 24, 32)
    }
    assert all(row.precision_profile_id == "F3_PRIMITIVE_V1" for row in rows)


@pytest.mark.parametrize(
    ("realized_match", "fusion_kind", "isolated", "error"),
    [
        (False, "primitive", True, "lut_requested_realized_mismatch"),
        (True, "complete_fused_mha", True, "lut_fused_phenotype_forbidden"),
        (True, "primitive", False, "lut_formal_requires_isolated_gpu"),
    ],
)
def test_lut_admission_fails_closed(
    realized_match: bool, fusion_kind: str, isolated: bool, error: str
):
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        validate_f3_lut_admission,
    )

    with pytest.raises(ValueError, match=error):
        validate_f3_lut_admission(
            requested_realized_match=realized_match,
            fusion_kind=fusion_kind,
            isolated_gpu=isolated,
            formal=True,
        )


def test_shared_gpu_uses_screening_filename_only():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        latency_lut_filename,
    )

    assert latency_lut_filename(formal=True) == "f3_primitive_latency_lut.csv"
    assert latency_lut_filename(formal=False) == (
        "f3_primitive_latency_lut_screening.csv"
    )


def test_latency_repetition_aggregation_and_full_engine_error():
    from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
        aggregate_latency_repetitions,
        validate_lut_full_engine_deltas,
    )

    summary = aggregate_latency_repetitions(
        [
            {"p50_ms": 1.0, "p90_ms": 1.2, "p95_ms": 1.3, "p99_ms": 1.4},
            {"p50_ms": 1.2, "p90_ms": 1.4, "p95_ms": 1.5, "p99_ms": 1.6},
            {"p50_ms": 1.1, "p90_ms": 1.3, "p95_ms": 1.4, "p99_ms": 1.5},
        ]
    )
    assert summary["p50_ms"] == pytest.approx(1.1)
    validation = validate_lut_full_engine_deltas(
        [
            {"candidate_id": "S1", "predicted_delta_ms": -0.20, "actual_delta_ms": -0.22},
            {"candidate_id": "S2", "predicted_delta_ms": -0.40, "actual_delta_ms": -0.50},
        ]
    )
    assert validation["mean_relative_error"] == pytest.approx((2 / 22 + 1 / 5) / 2)
    assert validation["status"] == "screening_only"


def test_tensorrt_root_is_resolved_from_accepted_manifest(tmp_path: Path):
    from search.orchestration.lidar_cobevt_minimal_structure_quant_latency import (
        resolve_tensorrt_root_from_history,
    )

    root = tmp_path / "TensorRT"
    trtexec = root / "targets/x86_64-linux-gnu/bin/trtexec"
    trtexec.parent.mkdir(parents=True)
    trtexec.write_bytes(b"accepted-trtexec")
    history = tmp_path / "history"
    history.mkdir()
    (history / "run_manifest.json").write_text(
        json.dumps(
            {
                "environment": {
                    "resolved_tensorrt_root": str(root),
                    "tensorrt_version": "10.9.0.34",
                }
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_tensorrt_root_from_history(history)
    assert resolved == root


def test_f3_deployment_unit_inspector_recognizes_trt_fused_projection_sequence():
    from search.model_families.lidar_cobevt.f3_attention_unit import (
        inspect_f3_deployment_unit_layers,
    )

    def tensor(name: str, dtype: str, dims: list[int]) -> dict[str, object]:
        return {"Name": name, "Dimensions": dims, "Format/Datatype": dtype}

    layers = [
        {
            "Name": "/MatMul_2+/MatMul_1+/MatMul_myl0_2",
            "LayerType": "gemm",
            "Inputs": [tensor("projection_input", "Half", [16384, 256])],
            "Outputs": [tensor("qkv", "Half", [3, 16384, 128])],
            "TacticName": "sm80_xmma_gemm_f16f16_f16f16_f16",
        },
        {
            "Name": "/MatMul_3_myl0_10",
            "LayerType": "gemm",
            "Inputs": [
                tensor("q", "Float", [4096, 32, 16]),
                tensor("k", "Float", [4096, 16, 32]),
            ],
            "Outputs": [tensor("qk", "Float", [4096, 32, 32])],
            "TacticName": "sm80_xmma_gemm_f32f32_f32f32_f32",
        },
        {
            "Name": "softmax_fusion",
            "LayerType": "kgen",
            "Inputs": [tensor("qk", "Float", [512, 8, 32, 32])],
            "Outputs": [tensor("probability", "Half", [512, 8, 32, 32])],
            "TacticName": "AddSelectCastMaxSubExpSumDiv",
        },
        {
            "Name": "/MatMul_4_myl0_12",
            "LayerType": "gemm",
            "Inputs": [
                tensor("probability", "Half", [4096, 32, 32]),
                tensor("v", "Half", [4096, 32, 16]),
            ],
            "Outputs": [tensor("av", "Half", [4096, 32, 16])],
            "TacticName": "ampere_h16816gemm",
        },
        {
            "Name": "/MatMul_5_myl0_13",
            "LayerType": "gemm",
            "Inputs": [tensor("av", "Half", [16384, 128])],
            "Outputs": [tensor("projected", "Half", [16384, 256])],
            "TacticName": "sm80_xmma_gemm_f16f16_f16f16_f16",
        },
    ]

    result = inspect_f3_deployment_unit_layers(
        layers, heads=8, tokens=32, d_qk=16, d_v=16, embed_dim=256
    )
    assert result["realized_q_projection_precision"] == "FP16"
    assert result["realized_k_projection_precision"] == "FP16"
    assert result["realized_v_projection_precision"] == "FP16"
    assert result["realized_qk_precision"] == "FP32"
    assert result["realized_softmax_precision"] == "FP16"
    assert result["realized_av_precision"] == "FP16"
    assert result["realized_out_projection_precision"] == "FP16"
    assert result["fusion_kind"] == "projection_fusion_with_primitives"
    assert result["fused_mha_detected"] is False
    assert result["requested_realized_match"] is True


def test_full_model_export_exposes_pre_export_transform_without_enabling_it_by_default():
    from search.orchestration.lidar_cobevt_attention_pruning import run_export_build

    parameter = inspect.signature(run_export_build).parameters["pre_export_transform"]
    assert parameter.default is None
