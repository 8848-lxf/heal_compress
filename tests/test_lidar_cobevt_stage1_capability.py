from __future__ import annotations

from pathlib import Path
import copy

import pytest
import torch


CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
CONFIG = CHECKPOINT.with_name("config.yaml")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")


@pytest.fixture(scope="module")
def model():
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    return CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load().model


def test_head_parameter_slices_cover_full_physical_closure(model) -> None:
    from search.model_families.lidar_cobevt.stage1_capability import (
        build_cobevt_head_parameter_slices,
    )

    rows = build_cobevt_head_parameter_slices(model)
    head0 = rows["cobevt::fusion_head::00"]

    assert len(rows) == 8
    assert any(
        row.parameter_name == "shrinker_m1.layers.0.double_conv.2.weight"
        and row.axis == 0
        and len(row.indices) == 32
        for row in head0
    )
    assert any(
        row.parameter_name.endswith("window_attention.fn.to_qkv.weight")
        and row.axis == 0
        and len(row.indices) == 96
        for row in head0
    )
    assert any(
        row.parameter_name == "cls_head.weight"
        and row.axis == 1
        and len(row.indices) == 32
        for row in head0
    )


def test_stage1_space_uses_legal_head_width_and_fp32_fp16_actions(model) -> None:
    from search.model_families.lidar_cobevt.stage1_capability import (
        build_cobevt_stage1_space,
    )
    from search.proxy.fisher_proxy import FisherStatistics

    slices = __import__(
        "search.model_families.lidar_cobevt.stage1_capability",
        fromlist=["build_cobevt_head_parameter_slices"],
    ).build_cobevt_head_parameter_slices(model)
    needed = {row.parameter_name for values in slices.values() for row in values}
    parameters = dict(model.named_parameters())
    statistics = FisherStatistics(
        gradients={name: torch.ones_like(parameters[name]) for name in needed},
        fisher_diag={name: torch.ones_like(parameters[name]) for name in needed},
        manifest_hash="fisher",
    )

    prepared = build_cobevt_stage1_space(
        model,
        statistics=statistics,
        checkpoint_hash="checkpoint",
        fisher_manifest_hash="fisher",
    )
    domain = prepared.inventory.domains[0]

    assert domain.domain_id == "cobevt::fusion_embed_heads"
    assert domain.original_width == 8
    assert domain.legal_keep_widths == (2, 3, 4, 5, 6, 7, 8)
    assert prepared.search_space.structure_gene_type == "legal_keep_width"
    assert all(
        actions == ("FP32", "FP16")
        for actions in prepared.search_space.precision_action_space.values()
    )
    assert "encoder_m1.scatter" not in prepared.search_space.precision_gene_ids


def test_precision_changes_do_not_change_cobevt_decoded_structure(model) -> None:
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.model_families.lidar_cobevt.stage1_capability import (
        build_cobevt_stage1_space,
    )
    from search.proxy.fisher_proxy import FisherStatistics

    slices_module = __import__(
        "search.model_families.lidar_cobevt.stage1_capability",
        fromlist=["build_cobevt_head_parameter_slices"],
    )
    slices = slices_module.build_cobevt_head_parameter_slices(model)
    needed = {row.parameter_name for values in slices.values() for row in values}
    parameters = dict(model.named_parameters())
    statistics = FisherStatistics(
        gradients={name: torch.ones_like(parameters[name]) for name in needed},
        fisher_diag={name: torch.ones_like(parameters[name]) for name in needed},
        manifest_hash="fisher",
    )
    prepared = build_cobevt_stage1_space(
        model,
        statistics=statistics,
        checkpoint_hash="checkpoint",
        fisher_manifest_hash="fisher",
    )
    actions = prepared.search_space.precision_action_space
    fp32 = {group: "FP32" for group in actions}
    fp16 = {group: "FP16" for group in actions}
    width = {"cobevt::fusion_embed_heads": 4}
    first = LegalWidthGenotype(width, fp32)
    second = LegalWidthGenotype(width, fp16)

    decoded_a = prepared.decoder.decode(first.width_genes)
    decoded_b = prepared.decoder.decode(second.width_genes)

    assert decoded_a.structure_hash == decoded_b.structure_hash
    assert decoded_a.pruned_unit_ids == decoded_b.pruned_unit_ids


def test_stage1_structural_parameter_count_matches_materialized_model(model) -> None:
    from search.canonicalization import canonicalize_legal_width_candidate
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.model_families.lidar_cobevt.pruning_recipe import CobevtPruningRecipe
    from search.model_families.lidar_cobevt.stage1_capability import (
        CobevtStructuralSizeProxy,
        build_cobevt_head_parameter_slices,
        build_cobevt_stage1_space,
    )
    from search.proxy.fisher_proxy import FisherStatistics

    slices = build_cobevt_head_parameter_slices(model)
    needed = {row.parameter_name for values in slices.values() for row in values}
    parameters = dict(model.named_parameters())
    statistics = FisherStatistics(
        gradients={name: torch.ones_like(parameters[name]) for name in needed},
        fisher_diag={name: torch.ones_like(parameters[name]) for name in needed},
        manifest_hash="fisher",
    )
    prepared = build_cobevt_stage1_space(
        model,
        statistics=statistics,
        checkpoint_hash="checkpoint",
        fisher_manifest_hash="fisher",
    )
    genotype = LegalWidthGenotype(
        {"cobevt::fusion_embed_heads": 4},
        {group: "FP16" for group in prepared.search_space.precision_action_space},
    )
    phenotype = canonicalize_legal_width_candidate(genotype, prepared.search_space)
    size = CobevtStructuralSizeProxy(model)
    original, predicted = size.structural_parameter_counts(phenotype)

    ranking = tuple(
        int(row.atomic_unit_id.rsplit("::", 1)[1])
        for row in sorted(prepared.second_order_ranking.rows, key=lambda row: row.rank)
    )
    physical_model = copy.deepcopy(model)
    recipe = CobevtPruningRecipe()
    decoded = recipe.decode_fusion_width(
        physical_model,
        keep_width=192,
        ranked_head_ids=ranking,
    )
    report = recipe.materialize_fusion_width(physical_model, decoded)

    assert original == report.original_parameter_count
    assert predicted == report.physical_parameter_count
