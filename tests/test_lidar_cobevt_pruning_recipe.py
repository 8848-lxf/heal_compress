from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn


CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
CONFIG = CHECKPOINT.with_name("config.yaml")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")


@pytest.fixture(scope="module")
def pruned_model_and_report():
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )
    from search.model_families.lidar_cobevt.pruning_recipe import (
        CobevtPruningRecipe,
    )

    model = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load().model
    recipe = CobevtPruningRecipe()
    decoded = recipe.decode_fusion_width(
        model,
        keep_width=192,
        ranked_head_ids=tuple(range(8)),
    )
    report = recipe.materialize_fusion_width(model, decoded)
    return model, report


def test_real_cobevt_fusion_materialization_is_structurally_legal(
    pruned_model_and_report,
):
    model, report = pruned_model_and_report

    assert report.passed
    assert report.original_embed_dim == 256
    assert report.new_embed_dim == 192
    assert report.original_heads == 8
    assert report.new_heads == 6
    assert report.predicted_parameter_count == report.physical_parameter_count
    assert report.physical_parameter_count < report.original_parameter_count
    assert model.shrinker_m1.layers[0].double_conv[2].out_channels == 192
    assert model.cls_head.in_channels == 192
    assert model.reg_head.in_channels == 192
    assert model.dir_head.in_channels == 192


def test_all_attention_qkv_ffn_norm_and_bias_tables_agree(pruned_model_and_report):
    model, _ = pruned_model_and_report

    for name, module in model.named_modules():
        if not name.startswith("fusion_net"):
            continue
        if module.__class__.__name__ == "Attention":
            assert module.heads == 6
        if isinstance(module, nn.LayerNorm):
            assert tuple(module.normalized_shape) == (192,)
        if isinstance(module, nn.Linear):
            assert module.in_features == 192
            assert module.out_features in {192, 576}
        if isinstance(module, nn.Embedding) and name.endswith(
            "relative_position_bias_table"
        ):
            assert module.embedding_dim == 6


def test_structure_hash_is_deterministic_for_same_width_and_ranking():
    from search.model_families.lidar_cobevt.pruning_recipe import (
        CobevtPruningRecipe,
    )

    recipe = CobevtPruningRecipe()
    first = recipe.decode_fusion_width_indices(
        original_width=256,
        keep_width=192,
        dim_head=32,
        ranked_head_ids=tuple(range(8)),
    )
    second = recipe.decode_fusion_width_indices(
        original_width=256,
        keep_width=192,
        dim_head=32,
        ranked_head_ids=tuple(range(8)),
    )

    assert first.structure_hash == second.structure_hash
    assert first.keep_channel_indices == second.keep_channel_indices


def test_real_model_fusion_domain_exposes_only_replayable_legal_widths():
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )
    from search.model_families.lidar_cobevt.pruning_recipe import (
        CobevtPruningRecipe,
    )

    model = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load().model
    domain = CobevtPruningRecipe().fusion_width_domain(model)

    assert domain.domain_id == "cobevt::fusion_embed"
    assert domain.root_module == "shrinker_m1.layers.0.double_conv.2"
    assert domain.original_width == 256
    assert domain.dim_head == 32
    assert domain.legal_keep_widths == (64, 96, 128, 160, 192, 224, 256)
    assert domain.physical_replay_supported is True
    assert domain.protected is False
