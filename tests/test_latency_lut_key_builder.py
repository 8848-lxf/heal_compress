from __future__ import annotations

from opencood.tools.compression.latency_lut.key_builder import (
    DeploymentUnit,
    build_channel_grid,
    build_lut_keys_for_units,
)


def test_channel_grid_respects_min_keep_ratio_alignment_and_dedupes():
    grid = build_channel_grid(64, [1.0, 0.875, 0.75, 0.5, 0.25], min_keep_ratio=0.5, align=16)

    assert grid == [32, 48, 64]
    assert min(grid) >= 32
    assert all(value % 16 == 0 for value in grid)


def test_key_builder_generates_representative_keys_without_full_cartesian():
    unit = DeploymentUnit(
        unit_id="backbone.stage1",
        module_name="backbone",
        block_name="stage1",
        block_type="conv_block",
        H=100,
        W=352,
        C_base=64,
        kernel_size=3,
        stride=1,
    )
    keys = build_lut_keys_for_units(
        [unit],
        keep_ratios=[1.0, 0.75, 0.5],
        min_keep_ratio=0.5,
        channel_align=16,
        precision_profiles=["TRT_FP16", "TRT_FP32"],
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
    )

    hashes = {key.stable_hash() for key in keys}
    assert len(hashes) == len(keys)
    assert len(keys) < 3 * 3 * 2
    assert {key.precision_profile for key in keys} == {"TRT_FP16", "TRT_FP32"}
    assert all(key.deploy_mode == "single_engine_maxK" and key.fixed_K == 29696 for key in keys)


def test_plugin_key_includes_fixed_k_and_plugin_name():
    unit = DeploymentUnit(
        unit_id="scatter",
        module_name="scatter",
        block_name="scatter",
        block_type="plugin",
        H=100,
        W=352,
        C_base=64,
        fixed_K=29696,
        plugin_name="PointPillarScatterTRT",
    )
    keys = build_lut_keys_for_units(
        [unit],
        keep_ratios=[1.0],
        min_keep_ratio=0.25,
        channel_align=16,
        precision_profiles=["TRT_FP16"],
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
    )

    assert len(keys) == 1
    assert keys[0].plugin_flag is True
    assert keys[0].plugin_name == "PointPillarScatterTRT"
    assert keys[0].fixed_K == 29696
