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
