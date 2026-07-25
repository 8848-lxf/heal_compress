from __future__ import annotations

import pytest

from scripts.audit_v2xvit_transformer_bops_precision import (
    activation_matmul_bops,
    ffn_macs,
    hgt_relation_attention_macs,
    standard_attention_macs,
    weighted_bops,
)


def test_qkv_projection_macs_include_tokens_heads_and_runtime_groups() -> None:
    row = standard_attention_macs(
        projection_tokens=2 * 3 * 7,
        groups=2 * 3,
        heads=4,
        n_q=7,
        n_k=7,
        d_model=64,
        d_k=8,
        d_v=8,
    )
    expected = 2 * 3 * 7 * 64 * 4 * 8
    assert row["q_projection"] == expected
    assert row["k_projection"] == expected
    assert row["v_projection"] == expected


def test_qk_and_av_macs_are_activation_matmuls() -> None:
    row = standard_attention_macs(
        projection_tokens=14, groups=2, heads=4, n_q=7, n_k=7,
        d_model=64, d_k=8, d_v=12,
    )
    assert row["qk_matmul"] == 2 * 4 * 7 * 7 * 8
    assert row["av_matmul"] == 2 * 4 * 7 * 7 * 12
    assert activation_matmul_bops(row["qk_matmul"], 32, 32) == row["qk_matmul"] * 1024


def test_output_projection_and_ffn_macs() -> None:
    row = standard_attention_macs(
        projection_tokens=21, groups=3, heads=4, n_q=7, n_k=7,
        d_model=64, d_k=8, d_v=12,
    )
    assert row["output_projection"] == 21 * (4 * 12) * 64
    assert ffn_macs(tokens=21, d_model=64, d_ff=256) == {
        "ffn1": 21 * 64 * 256,
        "ffn2": 21 * 64 * 256,
    }


def test_hgt_relation_attention_exposes_both_dh_squared_contractions() -> None:
    row = hgt_relation_attention_macs(
        projection_tokens=2 * 4 * 5 * 2,
        groups=2 * 4 * 5,
        heads=4,
        agents=2,
        d_model=64,
        d_h=8,
    )
    pairs = 2 * 4 * 5 * 4 * 2 * 2
    assert row["qk_relation_transform"] == pairs * 8 * 8
    assert row["message_relation_transform"] == pairs * 8 * 8
    assert row["qk_matmul"] == pairs * 8
    assert row["av_matmul"] == pairs * 8


def test_weighted_and_activation_bops_have_distinct_operand_semantics() -> None:
    assert weighted_bops(100, 8, 16) == 12_800
    assert activation_matmul_bops(100, 16, 8) == 12_800
    with pytest.raises(TypeError):
        # A nonweighted op cannot silently acquire a nonexistent weight bit.
        weighted_bops(100, None, 8)  # type: ignore[arg-type]
