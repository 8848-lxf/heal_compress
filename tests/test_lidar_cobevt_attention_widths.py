from __future__ import annotations

import pytest


def test_attention_legal_widths_are_head_aligned_and_domain_capped():
    from search.model_families.lidar_cobevt.pruning_recipe import (
        legal_fusion_widths,
    )

    widths = legal_fusion_widths(
        original_width=256,
        dim_head=32,
        minimum_retained_ratio=0.2,
        per_domain_max_prune_rate=0.8,
    )

    assert widths == (64, 96, 128, 160, 192, 224, 256)
    assert all(width % 32 == 0 for width in widths)


def test_attention_width_decoder_is_nested_and_precision_independent():
    from search.model_families.lidar_cobevt.pruning_recipe import (
        decode_fusion_width,
    )

    ranking = tuple(range(8))
    keep_192 = decode_fusion_width(256, 192, dim_head=32, ranked_head_ids=ranking)
    keep_160 = decode_fusion_width(256, 160, dim_head=32, ranked_head_ids=ranking)

    assert keep_192.pruned_head_ids == (0, 1)
    assert keep_160.pruned_head_ids == (0, 1, 2)
    assert set(keep_192.pruned_channel_indices).issubset(
        keep_160.pruned_channel_indices
    )
    assert "precision" not in keep_192.to_dict()


@pytest.mark.parametrize("keep_width", [0, 31, 200, 288])
def test_attention_width_decoder_rejects_illegal_widths(keep_width):
    from search.model_families.lidar_cobevt.pruning_recipe import (
        decode_fusion_width,
    )

    with pytest.raises(ValueError):
        decode_fusion_width(
            256,
            keep_width,
            dim_head=32,
            ranked_head_ids=tuple(range(8)),
        )

