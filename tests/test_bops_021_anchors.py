from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _space():
    from search.canonicalization import SearchSpaceSpec
    from search.quantization_space.types import QuantizationSearchGroup

    groups = (
        QuantizationSearchGroup(
            group_id="quant::conv_a",
            module_paths=("conv_a",),
            canonical_node_ids=(),
            allowed_precisions=("FP32", "FP16", "INT8"),
            protected=False,
            protection_reason="",
            ordering=0,
            parameter_count=16,
            baseline_macs=80.0,
        ),
        QuantizationSearchGroup(
            group_id="quant::conv_b",
            module_paths=("conv_b",),
            canonical_node_ids=(),
            allowed_precisions=("FP32", "FP16", "INT8"),
            protected=False,
            protection_reason="",
            ordering=1,
            parameter_count=16,
            baseline_macs=20.0,
        ),
    )
    return SearchSpaceSpec(
        pruning_unit_ids=["prune::a", "prune::b"],
        precision_layer_ids=["conv_a", "conv_b"],
        quantization_groups=groups,
        builder_flags={
            "strongly_typed": True,
            "production_mode": True,
            "plugin_boundary_dtype": "fp32",
        },
    )


def test_all_fp16_theoretical_bops_is_quarter_of_fp32() -> None:
    from search.anchors.bops_021 import theoretical_bops_retention

    assert theoretical_bops_retention("FP16", "FP32") == pytest.approx(0.25)


def test_channel_repair_does_not_modify_layer_bitwidth() -> None:
    from search.anchors.bops_021 import make_all_fp16_genotype, precision_identity_audit
    from search.candidate import CandidateGenotype
    from search.canonicalization import canonicalize_candidate

    space = _space()
    raw = make_all_fp16_genotype(space)
    repaired = CandidateGenotype(
        pruning_genes={"prune::a": 0, "prune::b": 1},
        precision_genes=dict(raw.precision_genes),
    )
    phenotype = canonicalize_candidate(repaired, space)
    audit = precision_identity_audit(
        raw=raw,
        repaired=repaired,
        requested_module_profile=phenotype.requested_precision_profile,
        realized_module_profile=phenotype.realized_precision_profile,
        space=space,
    )

    assert audit["passed"] is True
    assert len(set(audit["profile_hashes"].values())) == 1


def test_no_prune_mixed_anchor_keeps_every_channel() -> None:
    from search.anchors.bops_021 import make_mixed_no_prune_genotype, validate_anchor_semantics

    space = _space()
    genotype = make_mixed_no_prune_genotype(space, int8_group_ids={"quant::conv_b"})
    audit = validate_anchor_semantics("mixed_no_prune", genotype, space)

    assert audit["passed"] is True
    assert set(genotype.pruning_genes.values()) == {1}
    assert genotype.precision_genes["quant::conv_b"] == "INT8"


def test_all_fp16_pruning_anchor_contains_no_int8() -> None:
    from search.anchors.bops_021 import make_all_fp16_pruning_genotype, validate_anchor_semantics

    space = _space()
    genotype = make_all_fp16_pruning_genotype(space, pruned_unit_ids={"prune::a"})
    audit = validate_anchor_semantics("fp16_pruning", genotype, space)

    assert audit["passed"] is True
    assert genotype.pruning_genes["prune::a"] == 0
    assert set(genotype.precision_genes.values()) == {"FP16"}


def test_precision_identity_fails_when_engine_changes_profile() -> None:
    from search.anchors.bops_021 import make_all_fp16_genotype, precision_identity_audit
    from search.canonicalization import canonicalize_candidate

    space = _space()
    raw = make_all_fp16_genotype(space)
    phenotype = canonicalize_candidate(raw, space)
    realized = dict(phenotype.realized_precision_profile)
    realized["conv_b"] = "FP32"
    audit = precision_identity_audit(
        raw=raw,
        repaired=raw,
        requested_module_profile=phenotype.requested_precision_profile,
        realized_module_profile=realized,
        space=space,
    )

    assert audit["passed"] is False
    assert audit["precision_realization_changed"] is True


def test_bops_decomposition_reports_each_transition() -> None:
    from search.anchors.bops_021 import decompose_bops_retention

    report = decompose_bops_retention(
        raw_proxy=0.220,
        repaired_proxy=0.213,
        physical=0.211,
        realized=0.212,
    )

    assert report["delta_bops_channel_repair"] == pytest.approx(-0.007)
    assert report["delta_bops_physical_materialization"] == pytest.approx(-0.002)
    assert report["delta_bops_precision_realization"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("retention", "passed"),
    [(0.204999, False), (0.205, True), (0.21, True), (0.215, True), (0.215001, False)],
)
def test_anchor_realized_bops_gate_is_closed_interval(retention: float, passed: bool) -> None:
    from search.anchors.bops_021 import realized_bops_gate

    assert realized_bops_gate(retention, target=0.21, tolerance=0.005)["passed"] is passed


def test_strongly_typed_realized_profile_must_match_requested() -> None:
    from search.anchors.bops_021 import validate_strongly_typed_profile

    requested = {"conv_a": "FP16", "conv_b": "INT8"}
    assert validate_strongly_typed_profile(
        requested,
        dict(requested),
        builder_flags=_space().builder_flags,
    )["passed"] is True
    assert validate_strongly_typed_profile(
        requested,
        {"conv_a": "FP16", "conv_b": "FP16"},
        builder_flags=_space().builder_flags,
    )["passed"] is False


def test_scatter_plugin_is_not_a_quantization_gene() -> None:
    from search.anchors.bops_021 import validate_plugin_gene_exclusion

    report = validate_plugin_gene_exclusion(_space(), plugin_tokens=("scatter", "pointpillar"))

    assert report["passed"] is True
    assert report["matching_gene_ids"] == []


def test_anchor_output_manifests_must_be_identical() -> None:
    from search.anchors.bops_021 import validate_manifest_consistency

    anchors = {
        "A": {"calibration_manifest_hash": "cal", "validation_manifest_hash": "val"},
        "B": {"calibration_manifest_hash": "cal", "validation_manifest_hash": "val"},
        "C": {"calibration_manifest_hash": "cal", "validation_manifest_hash": "val"},
    }
    assert validate_manifest_consistency(anchors)["passed"] is True
    anchors["C"]["validation_manifest_hash"] = "different"
    assert validate_manifest_consistency(anchors)["passed"] is False


def test_mixed_precision_prefix_uses_mac_not_layer_count() -> None:
    from search.anchors.bops_021 import QuantizationSensitivity, select_mixed_precision_prefix

    rows = [
        QuantizationSensitivity("g_small", ("small",), 5.0, 0.01, 0.01, 1.0, 0.01),
        QuantizationSensitivity("g_large", ("large",), 17.0, 0.02, 0.01, 1.0, 0.01),
        QuantizationSensitivity("g_tail", ("tail",), 78.0, 9.0, 9.0, 1.0, 0.01),
    ]

    selection = select_mixed_precision_prefix(rows, target=0.21, tolerance=0.005)

    assert selection["passed"] is True
    assert selection["selected_group_ids"] == ["g_large", "g_small"]
    assert selection["int8_macs_ratio"] == pytest.approx(0.22)
    assert selection["bops_retention"] == pytest.approx(0.20875)
