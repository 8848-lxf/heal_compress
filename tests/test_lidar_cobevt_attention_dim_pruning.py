from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
CONFIG = CHECKPOINT.with_name("config.yaml")


def _stock_attention():
    if str(HEAL_ROOT) not in sys.path:
        sys.path.insert(0, str(HEAL_ROOT))
    from opencood.models.fuse_modules.swap_fusion_modules import Attention

    torch.manual_seed(7)
    return Attention(
        dim=256,
        dim_head=32,
        dropout=0.0,
        agent_size=2,
        window_size=2,
    ).eval()


def _identity_keep(width: int = 32, heads: int = 8):
    return tuple(tuple(range(width)) for _ in range(heads))


def test_explicit_projection_conversion_is_exact_at_original_width():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        PrunableCobevtAttention,
    )

    original = _stock_attention()
    converted = PrunableCobevtAttention.from_stock_attention(
        original,
        qk_keep_by_head=_identity_keep(),
        vo_keep_by_head=_identity_keep(),
    ).eval()
    x = torch.randn(1, 2, 2, 2, 2, 2, 256)
    mask = torch.ones(1, 2, 2, 2, 2, 1, 2)

    with torch.no_grad():
        expected = original(x, mask)
        actual = converted(x, mask)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
    assert converted.d_qk == 32
    assert converted.d_v == 32
    assert converted.q_proj.weight.shape == (256, 256)
    assert converted.out_proj.weight.shape == (256, 256)


def test_qk_only_physical_pruning_keeps_v_and_out_widths():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        PrunableCobevtAttention,
    )

    original = _stock_attention()
    qk_keep = tuple(tuple(range(0, 32, 2)) for _ in range(8))
    converted = PrunableCobevtAttention.from_stock_attention(
        original,
        qk_keep_by_head=qk_keep,
        vo_keep_by_head=_identity_keep(),
    )

    assert converted.q_proj.weight.shape == (128, 256)
    assert converted.k_proj.weight.shape == (128, 256)
    assert converted.v_proj.weight.shape == (256, 256)
    assert converted.out_proj.weight.shape == (256, 256)
    assert converted.d_qk == 16
    assert converted.d_v == 32
    assert converted.scale == pytest.approx(1.0 / math.sqrt(16.0))
    output = converted(
        torch.randn(1, 2, 2, 2, 2, 2, 256),
        torch.ones(1, 2, 2, 2, 2, 1, 2),
    )
    assert output.shape == (1, 2, 2, 2, 2, 2, 256)
    assert torch.isfinite(output).all()


def test_qk_and_vo_masks_are_coupled_within_family_but_independent_between_families():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        AttentionDimMask,
    )

    qk = tuple(tuple(range(24)) for _ in range(8))
    vo = tuple(tuple(range(8, 32)) for _ in range(8))
    mask = AttentionDimMask(qk_keep_by_head=qk, vo_keep_by_head=vo)

    assert mask.d_qk == 24
    assert mask.d_v == 24
    assert mask.q_keep_by_head == mask.k_keep_by_head
    assert mask.v_keep_by_head == mask.out_input_keep_by_head
    assert mask.qk_keep_by_head != mask.vo_keep_by_head
    assert all(len(values) == 24 for values in mask.qk_keep_by_head)
    assert all(len(values) == 24 for values in mask.vo_keep_by_head)


def test_mask_rejects_inconsistent_per_head_keep_counts():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        AttentionDimMask,
    )

    qk = [tuple(range(24)) for _ in range(8)]
    qk[3] = tuple(range(16))
    with pytest.raises(ValueError, match="qk_keep_count"):
        AttentionDimMask(
            qk_keep_by_head=tuple(qk),
            vo_keep_by_head=tuple(tuple(range(24)) for _ in range(8)),
        )


@pytest.fixture(scope="module")
def real_b1_model():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        materialize_attention_bottleneck,
        uniform_attention_masks,
    )
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    model = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load().model
    masks = uniform_attention_masks(model, d_qk=24, d_v=24)
    report = materialize_attention_bottleneck(model, masks)
    return model, report


def test_real_cobevt_b1_replaces_all_six_attention_modules(real_b1_model):
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        PrunableCobevtAttention,
    )

    model, report = real_b1_model
    attention_rows = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PrunableCobevtAttention)
    ]

    assert report.passed
    assert report.attention_module_count == 6
    assert len(attention_rows) == 6
    assert report.physical_parameter_count < report.original_parameter_count
    for _, module in attention_rows:
        assert module.embed_dim == 256
        assert module.heads == 8
        assert module.d_qk == 24
        assert module.d_v == 24
        assert module.q_proj.weight.shape == (192, 256)
        assert module.k_proj.weight.shape == (192, 256)
        assert module.v_proj.weight.shape == (192, 256)
        assert module.out_proj.weight.shape == (256, 192)


def test_qk_only_and_uniform_masks_have_deterministic_structure_hashes():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        attention_masks_structure_hash,
        uniform_attention_masks,
    )
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    first_model = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load().model
    second_model = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load().model
    first = uniform_attention_masks(first_model, d_qk=16, d_v=32)
    second = uniform_attention_masks(second_model, d_qk=16, d_v=32)

    assert attention_masks_structure_hash(first) == attention_masks_structure_hash(
        second
    )


@pytest.fixture(scope="module")
def real_b2_model():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        materialize_global_embedding_bottleneck,
        stratified_embedding_keep_indices,
        uniform_attention_masks,
    )
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    model = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load().model
    masks = uniform_attention_masks(model, d_qk=24, d_v=24)
    embedding_keep = stratified_embedding_keep_indices(
        heads=8, original_dim_per_head=32, keep_dim_per_head=24
    )
    report = materialize_global_embedding_bottleneck(
        model,
        masks=masks,
        embedding_keep_indices=embedding_keep,
    )
    return model, report


def test_b2_global_embedding_closure_keeps_eight_heads(real_b2_model):
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        PrunableCobevtAttention,
    )

    model, report = real_b2_model
    assert report.passed
    assert report.original_embed_dim == 256
    assert report.new_embed_dim == 192
    assert report.heads == 8
    assert report.physical_parameter_count < report.original_parameter_count
    assert model.shrinker_m1.layers[0].double_conv[2].out_channels == 192
    assert model.cls_head.in_channels == 192
    assert model.reg_head.in_channels == 192
    assert model.dir_head.in_channels == 192
    for name, module in model.named_modules():
        if isinstance(module, PrunableCobevtAttention):
            assert module.embed_dim == 192
            assert module.heads == 8
            assert module.d_qk == 24
            assert module.d_v == 24
        if name.startswith("fusion_net") and isinstance(module, nn.LayerNorm):
            assert tuple(module.normalized_shape) == (192,)


def test_b2_ffn_preserves_hidden_width_but_closes_residual_boundary(real_b2_model):
    model, _ = real_b2_model
    for block in model.fusion_net.layers:
        for ffn in (block.window_ffd.fn, block.grid_ffd.fn):
            assert ffn.net[0].in_features == 192
            assert ffn.net[0].out_features == 256
            assert ffn.net[3].in_features == 256
            assert ffn.net[3].out_features == 192
    assert model.fusion_net.mlp_head[3].in_features == 192
    assert model.fusion_net.mlp_head[3].out_features == 192


def test_stratified_embedding_indices_retain_equal_count_per_head():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        stratified_embedding_keep_indices,
    )

    keep = stratified_embedding_keep_indices(
        heads=8, original_dim_per_head=32, keep_dim_per_head=24
    )
    assert len(keep) == 192
    assert keep[:24] == tuple(range(24))
    assert keep[24:48] == tuple(range(32, 56))
    assert keep[-24:] == tuple(range(224, 248))
