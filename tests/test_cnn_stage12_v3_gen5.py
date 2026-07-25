from __future__ import annotations

import pytest


def _domain():
    from search.pruning_space.local_domains import LocalPruningDomain

    return LocalPruningDomain(
        domain_id="backbone::out",
        root_module_path="backbone",
        root_axis="out",
        scope_id="backbone",
        kind="dense",
        original_width=12,
        total_original_width=12,
        ordered_unit_ids=("u0", "u1"),
        legal_widths=(4, 8, 12),
        width_to_pruned_unit_ids={4: ("u0", "u1"), 8: ("u0",), 12: ()},
        unit_root_indices={"u0": (0, 1, 2, 3), "u1": (4, 5, 6, 7)},
        domain_type="cnn_channel",
    )


def _group(group_id: str, protected: bool = False):
    from search.quantization_space.types import QuantizationSearchGroup

    return QuantizationSearchGroup(
        group_id=group_id,
        module_paths=(group_id,),
        canonical_node_ids=(group_id,),
        allowed_precisions=("FP32",) if protected else ("FP32", "FP16", "INT8"),
        protected=protected,
        protection_reason="fixed" if protected else "",
        ordering=0 if not protected else 1,
        parameter_count=4,
        baseline_macs=4.0,
    )


def _space():
    from search.canonicalization import SearchSpaceSpec

    return SearchSpaceSpec(
        pruning_unit_ids=["u0", "u1"],
        precision_layer_ids=["conv", "fixed"],
        pruning_domains=(_domain(),),
        quantization_groups=(_group("conv"), _group("fixed", True)),
        default_precision="FP32",
    )


def test_explicit_gen5_contract_does_not_weaken_default_gen10() -> None:
    from search.ga.stage12_v3 import StrictGAConfig

    assert StrictGAConfig(0.1).generations == 10
    configured = StrictGAConfig(
        0.1, generations=5, generation_contract="formal_gen5"
    )
    assert configured.generations == 5
    with pytest.raises(ValueError, match="generations_contract_mismatch"):
        StrictGAConfig(0.1, generations=4, generation_contract="formal_gen5")


def test_cnn_baseline_contains_only_mutable_loci() -> None:
    from search.ga.cnn_stage12_v3 import baseline_genotype
    from search.ga.stage12_v3 import validate_genotype_schema

    space = _space()
    candidate = baseline_genotype(space)
    assert candidate.pruning_width_genes == {"backbone::out": 12}
    assert candidate.precision_genes == {"conv": "FP32"}
    assert "fixed" not in candidate.precision_genes
    validate_genotype_schema(candidate, space)


def test_cnn_greedy_neighbors_are_decreasing_and_adjacent() -> None:
    from search.ga.cnn_stage12_v3 import baseline_genotype, decreasing_neighbors

    space = _space()
    candidate = baseline_genotype(space)
    rows = decreasing_neighbors(candidate, space)
    by_type = {kind: child for kind, _locus, child in rows}
    assert by_type["structure"].pruning_width_genes["backbone::out"] == 8
    assert by_type["precision"].precision_genes["conv"] == "FP16"


def test_cnn_formal_entrypoint_freezes_five_generations_and_one_seed() -> None:
    source = (__import__("pathlib").Path(__file__).parents[1]
              / "scripts/run_cnn_formal_ga_gen5.py").read_text()
    assert "requires_exactly_5_generations" in source
    assert "single_seed_zero_required" in source
    assert '"StrictStage12V3Runner"' in source
    assert "full1789_executed" in source
