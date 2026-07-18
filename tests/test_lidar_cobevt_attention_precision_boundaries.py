from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


EXPECTED_ROLES = {
    "layernorm",
    "q_projection",
    "k_projection",
    "v_projection",
    "qk_scale",
    "qk_matmul",
    "softmax",
    "av_matmul",
    "output_projection",
    "residual_add",
}


def test_a0_to_a7_profiles_define_every_attention_role_without_defaults():
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        ATTENTION_BOUNDARY_PROFILE_NAMES,
        attention_boundary_profile,
    )

    assert ATTENTION_BOUNDARY_PROFILE_NAMES[:8] == (
        "A0_strict_fp32_reference",
        "A1_qkv_projection_fp16_core_fp32",
        "A2_layernorm_fp16_only",
        "A3_qk_matmul_fp16_only",
        "A4_softmax_fp16_only",
        "A5_av_matmul_fp16_only",
        "A6_output_projection_fp16_only",
        "A7_residual_add_fp16_only",
    )
    for name in ATTENTION_BOUNDARY_PROFILE_NAMES[:8]:
        profile = attention_boundary_profile(name)
        assert set(profile.role_dtypes) == EXPECTED_ROLES
        assert set(profile.role_dtypes.values()) <= {"FP32", "FP16"}
        assert profile.profile_name == name
        assert profile.profile_hash


@pytest.mark.parametrize(
    ("profile_name", "fp16_roles"),
    [
        ("A0_strict_fp32_reference", set()),
        (
            "A1_qkv_projection_fp16_core_fp32",
            {"q_projection", "k_projection", "v_projection"},
        ),
        ("A2_layernorm_fp16_only", {"layernorm"}),
        ("A3_qk_matmul_fp16_only", {"qk_matmul"}),
        ("A4_softmax_fp16_only", {"softmax"}),
        ("A5_av_matmul_fp16_only", {"av_matmul"}),
        ("A6_output_projection_fp16_only", {"output_projection"}),
        ("A7_residual_add_fp16_only", {"residual_add"}),
    ],
)
def test_single_boundary_profiles_change_only_the_named_roles(
    profile_name: str, fp16_roles: set[str]
):
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        attention_boundary_profile,
    )

    profile = attention_boundary_profile(profile_name)

    assert {
        role for role, precision in profile.role_dtypes.items() if precision == "FP16"
    } == fp16_roles
    assert set(profile.output_recovery_roles) == fp16_roles


@pytest.mark.parametrize(
    ("profile_name", "fp16_roles", "recovery_roles"),
    [
        (
            "M1_projection_fp16_core_fp32",
            {"q_projection", "k_projection", "v_projection", "output_projection"},
            {"q_projection", "k_projection", "v_projection", "output_projection"},
        ),
        (
            "M2_projection_qk_av_fp16_softmax_fp32",
            {
                "q_projection",
                "k_projection",
                "v_projection",
                "qk_scale",
                "qk_matmul",
                "av_matmul",
                "output_projection",
            },
            {"qk_matmul", "output_projection"},
        ),
        (
            "M3_projection_av_fp16_qk_softmax_fp32",
            {
                "q_projection",
                "k_projection",
                "v_projection",
                "av_matmul",
                "output_projection",
            },
            {"q_projection", "k_projection", "output_projection"},
        ),
        (
            "M4_projection_boundary_qk_av_fp16",
            {
                "q_projection",
                "k_projection",
                "v_projection",
                "qk_scale",
                "qk_matmul",
                "av_matmul",
                "output_projection",
            },
            {
                "q_projection",
                "k_projection",
                "qk_matmul",
                "output_projection",
            },
        ),
        (
            "M5_projection_softmax_av_add_fp16_qk_fp32",
            {
                "q_projection",
                "k_projection",
                "v_projection",
                "softmax",
                "av_matmul",
                "output_projection",
                "residual_add",
            },
            {"q_projection", "k_projection", "residual_add"},
        ),
    ],
)
def test_evidence_supported_combination_profiles_define_explicit_recovery_boundaries(
    profile_name: str, fp16_roles: set[str], recovery_roles: set[str]
):
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        attention_boundary_profile,
    )

    profile = attention_boundary_profile(profile_name)

    assert {
        role for role, precision in profile.role_dtypes.items() if precision == "FP16"
    } == fp16_roles
    assert set(profile.output_recovery_roles) == recovery_roles


def test_boundary_profile_lookup_rejects_unknown_names():
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        attention_boundary_profile,
    )

    with pytest.raises(ValueError, match="unknown_attention_boundary_profile"):
        attention_boundary_profile("softmax_maybe_fp16")


def test_profile_weighted_precision_changes_attention_projections_only():
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        requested_weighted_precision,
    )

    modules = (
        "backbone_m1.blocks.0.1",
        "fusion_net.layers.0.window_attention.fn.q_proj",
        "fusion_net.layers.0.window_attention.fn.k_proj",
        "fusion_net.layers.0.window_attention.fn.v_proj",
        "fusion_net.layers.0.window_attention.fn.out_proj",
        "fusion_net.layers.0.window_ffd.fn.net.0",
        "cls_head",
    )

    requested = requested_weighted_precision(
        modules, "A1_qkv_projection_fp16_core_fp32"
    )

    assert requested == {
        "backbone_m1.blocks.0.1": "FP32",
        "fusion_net.layers.0.window_attention.fn.q_proj": "FP16",
        "fusion_net.layers.0.window_attention.fn.k_proj": "FP16",
        "fusion_net.layers.0.window_attention.fn.v_proj": "FP16",
        "fusion_net.layers.0.window_attention.fn.out_proj": "FP32",
        "fusion_net.layers.0.window_ffd.fn.net.0": "FP32",
        "cls_head": "FP32",
    }


@pytest.mark.parametrize(
    ("profile_name", "fp16_roles", "recovery_roles"),
    [
        (
            "F1_rest_fp16_projection_fp16_core_fp32",
            {"q_projection", "k_projection", "v_projection", "output_projection"},
            {"q_projection", "k_projection", "v_projection", "output_projection"},
        ),
        (
            "F2_rest_fp16_projection_av_fp16_qk_core_fp32",
            {
                "q_projection",
                "k_projection",
                "v_projection",
                "av_matmul",
                "output_projection",
            },
            {"q_projection", "k_projection", "output_projection"},
        ),
    ],
)
def test_final_profiles_use_rest_fp16_with_explicit_attention_islands(
    profile_name: str, fp16_roles: set[str], recovery_roles: set[str]
):
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        attention_boundary_base_precision,
        attention_boundary_profile,
    )

    profile = attention_boundary_profile(profile_name)

    assert attention_boundary_base_precision(profile_name) == "FP16"
    assert profile.external_weighted_dtype == "FP16"
    assert {
        role for role, precision in profile.role_dtypes.items() if precision == "FP16"
    } == fp16_roles
    assert set(profile.output_recovery_roles) == recovery_roles


def test_final_profile_keeps_nonattention_weighted_groups_fp16():
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        requested_weighted_precision,
    )

    modules = (
        "backbone_m1.blocks.0.1",
        "fusion_net.layers.0.window_attention.fn.q_proj",
        "fusion_net.layers.0.window_attention.fn.qk_not_weighted",
        "fusion_net.layers.0.window_ffd.fn.net.0",
        "cls_head",
    )
    requested = requested_weighted_precision(
        modules, "F1_rest_fp16_projection_fp16_core_fp32"
    )

    assert requested == {
        "backbone_m1.blocks.0.1": "FP16",
        "fusion_net.layers.0.window_attention.fn.q_proj": "FP16",
        "fusion_net.layers.0.window_attention.fn.qk_not_weighted": "FP16",
        "fusion_net.layers.0.window_ffd.fn.net.0": "FP16",
        "cls_head": "FP16",
    }


def _minimal_attention_onnx(path: Path) -> list[SimpleNamespace]:
    base = "/layers.0/window_attention"
    initializers = [
        numpy_helper.from_array(np.eye(4, dtype=np.float32), name=name)
        for name in ("outside_w", "q_w", "k_w", "v_w", "out_w")
    ]
    initializers.extend(
        [
            numpy_helper.from_array(np.ones(4, dtype=np.float32), name="ln_scale"),
            numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="ln_bias"),
            numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="qk_scale"),
            numpy_helper.from_array(np.zeros((1, 1), dtype=np.float32), name="rpe"),
            numpy_helper.from_array(
                np.zeros((1, 1), dtype=np.float32), name="mask_fill"
            ),
            numpy_helper.from_array(
                np.ones((1, 1), dtype=np.bool_), name="condition"
            ),
        ]
    )
    projection_names = {
        "q_projection": "q_canonical",
        "k_projection": "k_canonical",
        "v_projection": "v_canonical",
        "output_projection": "out_canonical",
    }
    nodes = [
        helper.make_node("MatMul", ["input", "outside_w"], ["x"], name="outside_backbone"),
        helper.make_node(
            "LayerNormalization",
            ["x", "ln_scale", "ln_bias"],
            ["ln"],
            name=f"{base}/norm/LayerNormalization",
            axis=-1,
        ),
        helper.make_node("MatMul", ["ln", "q_w"], ["q"], name=projection_names["q_projection"]),
        helper.make_node("MatMul", ["ln", "k_w"], ["k"], name=projection_names["k_projection"]),
        helper.make_node("MatMul", ["ln", "v_w"], ["v"], name=projection_names["v_projection"]),
        helper.make_node("Mul", ["q", "qk_scale"], ["q_scaled"], name=f"{base}/fn/Mul_6"),
        helper.make_node(
            "Einsum",
            ["q_scaled", "k"],
            ["logits"],
            name=f"{base}/fn/Einsum",
            equation="bi,ji->bj",
        ),
        helper.make_node("Add", ["logits", "rpe"], ["biased"], name=f"{base}/fn/Add"),
        helper.make_node(
            "Where",
            ["condition", "mask_fill", "biased"],
            ["masked"],
            name=f"{base}/fn/Where",
        ),
        helper.make_node(
            "Softmax",
            ["masked"],
            ["probability"],
            name=f"{base}/fn/attend/Softmax",
            axis=-1,
        ),
        helper.make_node(
            "Einsum",
            ["probability", "v"],
            ["message"],
            name=f"{base}/fn/Einsum_1",
            equation="bi,ij->bj",
        ),
        helper.make_node(
            "MatMul", ["message", "out_w"], ["update"], name=projection_names["output_projection"]
        ),
        helper.make_node("Add", ["update", "x"], ["output"], name=f"{base}/Add"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "attention_boundary",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, str(path))
    return [
        SimpleNamespace(
            module_path=f"fusion_net.layers.0.window_attention.fn.{name}",
            canonical_node_name=canonical,
        )
        for name, canonical in (
            ("q_proj", projection_names["q_projection"]),
            ("k_proj", projection_names["k_projection"]),
            ("v_proj", projection_names["v_projection"]),
            ("out_proj", projection_names["output_projection"]),
        )
    ]


def _apply_minimal_profile(tmp_path: Path, profile_name: str):
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        apply_attention_boundary_contract,
    )

    source = tmp_path / f"{profile_name}_source.onnx"
    destination = tmp_path / f"{profile_name}_typed.onnx"
    entries = _minimal_attention_onnx(source)
    report = apply_attention_boundary_contract(
        source,
        destination,
        entries,
        profile_name,
        expected_block_count=1,
    )
    return onnx.load(str(destination)), report


def _records_by_role(report):
    return {record["role"]: record for record in report["node_records"]}


@pytest.mark.parametrize(
    ("profile_name", "fp16_roles"),
    [
        (
            "A1_qkv_projection_fp16_core_fp32",
            {"q_projection", "k_projection", "v_projection"},
        ),
        ("A2_layernorm_fp16_only", {"layernorm"}),
        ("A3_qk_matmul_fp16_only", {"qk_matmul"}),
        ("A4_softmax_fp16_only", {"softmax"}),
        ("A5_av_matmul_fp16_only", {"av_matmul"}),
        ("A6_output_projection_fp16_only", {"output_projection"}),
        ("A7_residual_add_fp16_only", {"residual_add"}),
    ],
)
def test_single_boundary_rewrite_isolates_fp16_and_recovers_fp32(
    tmp_path: Path, profile_name: str, fp16_roles: set[str]
):
    model, report = _apply_minimal_profile(tmp_path, profile_name)
    records = _records_by_role(report)

    assert onnx.checker.check_model(model) is None
    assert {role for role, row in records.items() if row["compute_dtype"] == "FP16"} == fp16_roles
    assert all(records[role]["output_dtype"] == "FP32" for role in fp16_roles)
    assert all(records[role]["output_cast_count"] == 1 for role in fp16_roles)
    assert report["external_node_change_count"] == 0
    assert report["missing_role_count"] == 0


def test_a1_projection_outputs_are_explicitly_restored_before_attention_core(
    tmp_path: Path,
):
    model, report = _apply_minimal_profile(
        tmp_path, "A1_qkv_projection_fp16_core_fp32"
    )
    by_name = {node.name: node for node in model.graph.node}
    records = _records_by_role(report)

    for role in ("q_projection", "k_projection", "v_projection"):
        record = records[role]
        assert len(record["input_cast_nodes"]) == 2
        assert len(record["output_cast_nodes"]) == 1
        assert by_name[record["output_cast_nodes"][0]].op_type == "Cast"
    assert records["qk_scale"]["compute_dtype"] == "FP32"
    assert records["qk_matmul"]["compute_dtype"] == "FP32"
    assert records["softmax"]["compute_dtype"] == "FP32"
    assert records["residual_add"]["compute_dtype"] == "FP32"


def test_a0_keeps_entire_attention_and_external_graph_fp32(tmp_path: Path):
    _model, report = _apply_minimal_profile(tmp_path, "A0_strict_fp32_reference")

    assert report["inserted_input_cast_count"] == 0
    assert report["inserted_output_cast_count"] == 0
    assert all(
        row["compute_dtype"] == "FP32" and row["output_dtype"] == "FP32"
        for row in report["node_records"]
    )


def test_existing_graph_description_reads_types_without_rewriting(tmp_path: Path):
    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        describe_existing_attention_boundaries,
    )

    source = tmp_path / "source.onnx"
    entries = _minimal_attention_onnx(source)
    source_hash = source.read_bytes()

    report = describe_existing_attention_boundaries(
        source,
        entries,
        profile_name="legacy_strict_fp32",
        expected_block_count=1,
    )

    assert source.read_bytes() == source_hash
    assert len(report["node_records"]) == 10
    assert all(row["compute_dtype"] == "FP32" for row in report["node_records"])
    assert all(row["output_dtype"] == "FP32" for row in report["node_records"])
    assert report["graph_rewritten"] is False
