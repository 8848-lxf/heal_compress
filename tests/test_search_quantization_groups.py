from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_quantization_groups_use_existing_precision_group_ids() -> None:
    from tracer.precision_coupling_tracer import PrecisionGroup
    from search.quantization_space.group_builder import build_quantization_search_groups

    model = nn.Sequential()
    model.add_module("branch_a", nn.Conv2d(4, 8, 3))
    model.add_module("branch_b", nn.Conv2d(4, 8, 3))
    model.add_module("head", nn.Conv2d(8, 2, 1))
    groups = [
        PrecisionGroup("pg_residual", ["branch_a", "branch_b"], "residual", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg_head", ["head"], "head_constraint", ["fp32", "fp16"], "fp16", True),
    ]

    search_groups = build_quantization_search_groups(model, precision_groups=groups)

    assert [row.group_id for row in search_groups] == ["pg_residual", "pg_head"]
    assert all(not row.group_id.startswith("search::") for row in search_groups)
    assert search_groups[0].module_paths == ("branch_a", "branch_b")
    assert search_groups[0].allowed_precisions == ("FP32", "FP16", "INT8")
    assert search_groups[1].allowed_precisions == ("FP32", "FP16")


def test_group_gene_expands_to_all_members_and_records_fallback() -> None:
    from search.candidate import CandidateGenotype
    from search.quantization_space.types import QuantizationSearchGroup
    from search.quantization_space.legalizer import legalize_group_precision_genes

    groups = [
        QuantizationSearchGroup(
            group_id="pg_residual",
            module_paths=("branch_a", "branch_b"),
            canonical_node_ids=("node_a", "node_b"),
            allowed_precisions=("FP32", "FP16", "INT8"),
            protected=False,
            protection_reason="",
            ordering=0,
            parameter_count=10,
            baseline_macs=100.0,
            metadata={},
        ),
        QuantizationSearchGroup(
            group_id="pg_head",
            module_paths=("head",),
            canonical_node_ids=("node_h",),
            allowed_precisions=("FP32", "FP16"),
            protected=False,
            protection_reason="",
            ordering=1,
            parameter_count=2,
            baseline_macs=10.0,
            metadata={},
        ),
    ]
    genotype = CandidateGenotype(precision_genes={"pg_residual": "INT8", "pg_head": "INT8"})

    result = legalize_group_precision_genes(genotype.precision_genes, groups, default_precision="FP16")

    assert result.stage1_legalized_group_profile["pg_residual"] == "INT8"
    assert result.stage1_legalized_group_profile["pg_head"] == "FP16"
    assert result.fallback_report["pg_head"]["requested_precision"] == "INT8"
    assert result.fallback_report["pg_head"]["realized_precision"] == "FP16"
    expanded = result.expand_to_module_profile()
    assert expanded["branch_a"].realized_precision == "INT8"
    assert expanded["branch_b"].realized_precision == "INT8"
    assert expanded["head"].realized_precision == "FP16"


def test_quantization_group_hash_is_order_stable() -> None:
    from search.quantization_space.types import QuantizationSearchGroup
    from search.quantization_space.codec import quantization_group_profile_hash

    groups = [
        QuantizationSearchGroup("pg_b", ("b",), ("node_b",), ("FP16", "INT8"), False, "", 1, 1, 1.0, {}),
        QuantizationSearchGroup("pg_a", ("a",), ("node_a",), ("FP16", "INT8"), False, "", 0, 1, 1.0, {}),
    ]
    profile = {"pg_b": "INT8", "pg_a": "FP16"}

    left = quantization_group_profile_hash(profile, groups)
    right = quantization_group_profile_hash(dict(reversed(list(profile.items()))), list(reversed(groups)))

    assert left == right


def test_trusted_explicit_qdq_profile_is_fixed_unique_27_layer_control() -> None:
    from search.baselines.original_engines import TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES

    assert len(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES) == 27
    assert len(set(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)) == 27
    assert "shrink_conv.layers.0.double_conv.0" in TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES
    assert "shrink_conv.layers.0.double_conv.2" in TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES
    assert "pyramid_backbone.single_head_2" not in TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES


def test_group_output_contract_separates_concat_output_from_int8_compute() -> None:
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult
    from search.stage2.lidar_pyramid_real_evaluator import _apply_group_output_precision_contract

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="single_head_0",
                canonical_node_name="__canonical__single_head_0__Conv__call00000",
                precision_group="pg_concat_branch",
                requested_precision="int8",
                realized_request_precision="int8",
            )
        ]
    )
    realized = _apply_group_output_precision_contract(
        mapping,
        {"pg_concat_branch": {"output_precision_policy": "FP16"}},
    )

    assert realized.entries[0].realized_request_precision == "int8"
    assert realized.entries[0].realized_output_precision == "fp16"


def test_lidar_pyramid_functional_heads_have_fp16_output_contract() -> None:
    from search.integration.lidar_pyramid_context import _functional_fp16_output_boundary

    for module in (
        "pyramid_backbone.single_head_0",
        "pyramid_backbone.single_head_1",
        "pyramid_backbone.single_head_2",
    ):
        boundary = _functional_fp16_output_boundary(module)
        assert boundary is not None
        assert boundary["merge_kind"] == "functional_sigmoid_weight_merge"
        assert boundary["following_ops"] == ["Sigmoid", "Add", "GridSample"]
        assert boundary["output_qdq_placement"].startswith("no weighted-output Q/DQ")
    assert _functional_fp16_output_boundary("pyramid_backbone.single_head_2") is not None


def test_pfn_linear_int8_override_is_model_specific_and_audited() -> None:
    from search.integration.lidar_pyramid_context import _model_specific_int8_override

    evidence = _model_specific_int8_override("encoder_m1.pillar_vfe.pfn_layers.0.linear")
    assert "canonical_onnx_matmul" in evidence
    assert "per_channel_qdq" in evidence
    assert _model_specific_int8_override("encoder_m1.pillar_vfe") == ""
