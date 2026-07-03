from __future__ import annotations

from dataclasses import dataclass

from opencood.tools.compression.root_node_local_pruner import (
    GroupItemSpec,
    ScopeSpec,
    build_coupled_channel_units_from_scope_specs,
)


def test_residual_add_same_channel_index_enters_same_coupled_channel_unit():
    scope = ScopeSpec(
        group_id="group::residual.block.add",
        root_node="backbone.block.conv2",
        root_module="backbone.block.conv2",
        num_channels=4,
        group_type="add",
        items=[
            GroupItemSpec(name="backbone.block.conv2", op_type="Conv", direction="out", reason="main_branch"),
            GroupItemSpec(name="backbone.block.shortcut", op_type="Conv", direction="out", reason="projection_shortcut"),
            GroupItemSpec(name="backbone.block.add", op_type="Add", direction="out", reason="residual_add"),
        ],
    )

    units = build_coupled_channel_units_from_scope_specs([scope])

    ch2 = next(unit for unit in units if unit.root_channel_index == 2)
    member_keys = {(m.module, m.axis, m.index) for m in ch2.members}
    assert ("backbone.block.conv2", "out_channels", 2) in member_keys
    assert ("backbone.block.shortcut", "out_channels", 2) in member_keys
    assert ("backbone.block.add", "add_channel", 2) in member_keys
    assert "residual_add" in ch2.dependency_types
    assert "projection_shortcut" in ch2.dependency_types


def test_concat_branch_offset_maps_concat_output_and_next_conv_input():
    scope = ScopeSpec(
        group_id="group::fusion.concat",
        root_node="fusion.branch_b.conv",
        root_module="fusion.branch_b.conv",
        num_channels=3,
        group_type="cat",
        items=[
            GroupItemSpec(name="fusion.branch_b.conv", op_type="Conv", direction="out", reason="concat_branch_offset", concat_offset=8, branch_id="b"),
            GroupItemSpec(name="fusion.concat", op_type="Concat", direction="out", reason="concat_branch_offset", concat_offset=8, branch_id="b"),
            GroupItemSpec(name="fusion.after_concat", op_type="Conv", direction="in", reason="concat_out_to_next_conv_in", concat_offset=8, branch_id="b"),
        ],
    )

    units = build_coupled_channel_units_from_scope_specs([scope])

    ch1 = next(unit for unit in units if unit.root_channel_index == 1)
    member_keys = {(m.module, m.axis, m.index, m.concat_offset) for m in ch1.members}
    assert ("fusion.branch_b.conv", "out_channels", 1, 8) in member_keys
    assert ("fusion.concat", "concat_output_channel", 9, 8) in member_keys
    assert ("fusion.after_concat", "in_channels", 9, 8) in member_keys
    assert "concat_branch_offset" in ch1.dependency_types
    assert "concat_out_to_next_conv_in" in ch1.dependency_types
