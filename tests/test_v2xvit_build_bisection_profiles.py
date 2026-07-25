from __future__ import annotations

from scripts.generate_v2xvit_build_bisection_profiles import build_profiles


def _fixture():
    groups = {
        "cnn_precision::backbone.blocks.0": "INT8",
        "cnn_precision::shrinker_m1.layers.0": "INT8",
        "transformer_precision::fusion.encoder.layers.0.1.fn::ffn1": "INT8",
        "transformer_precision::fusion.encoder.layers.0.0.layers.0.0.fn::qk_projection": "INT8",
        "transformer_precision::fusion.encoder.layers.0.0.layers.0.1.fn.pwmsa.0::fused_qkv_projection": "INT8",
    }
    old = {
        "genotype": {
            "pruning_width_genes": {"domain": 4},
            "precision_genes": groups,
            "meta": {},
        }
    }
    floor = {"genotypes": {"P8-max-requested": {"precision_genes": groups}}}
    return old, floor


def test_bisection_profiles_preserve_old_physical_widths():
    old, floor = _fixture()
    profiles = build_profiles(old, floor)
    assert all(row["pruning_width_genes"] == {"domain": 4} for row in profiles.values())


def test_bisection_profiles_isolate_precision_families():
    old, floor = _fixture()
    profiles = build_profiles(old, floor)
    assert sum(value == "INT8" for value in profiles["INT8-CNN-only"]["precision_genes"].values()) == 1
    assert sum(value == "INT8" for value in profiles["INT8-shrinker-only"]["precision_genes"].values()) == 1
    assert sum(value == "INT8" for value in profiles["INT8-FFN-only"]["precision_genes"].values()) == 1
    assert sum(value == "INT8" for value in profiles["INT8-Attention-QKVO-only"]["precision_genes"].values()) == 2
    assert set(profiles["S32"]["precision_genes"].values()) == {"FP32"}
    assert set(profiles["S16"]["precision_genes"].values()) == {"FP16"}
