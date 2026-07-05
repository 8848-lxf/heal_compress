from __future__ import annotations

import pytest


def test_grouped_conv_policy_single_choice_default_A():
    from heal_compress.pruning.grouped_conv_policy_registry import (
        GroupedConvPolicyRegistry,
        parse_group_conv_policy_choice,
    )

    registry = GroupedConvPolicyRegistry.default()

    assert registry.default_policy.key == "A"
    assert registry.get("A").name == "flat_output_groups_fixed"
    assert registry.get("B").name == "group_balanced_output_groups_fixed"
    assert registry.get("C").name == "true_group_block_pruning"
    assert registry.get("D").name == "group_coarsening_zero_padded_reblock"

    assert parse_group_conv_policy_choice(None).key == "A"
    assert parse_group_conv_policy_choice("B").key == "B"

    with pytest.raises(ValueError, match="multiple_grouped_conv_policy_in_one_round"):
        parse_group_conv_policy_choice(["A", "C"])


def test_tp_oracle_only_not_model_generation():
    from heal_compress.pruning.grouped_conv_policy_registry import GroupedConvPolicyRegistry

    registry = GroupedConvPolicyRegistry.default()
    for policy in registry.policies.values():
        assert policy.uses_torch_pruning_for_model_generation is False
        assert policy.torch_pruning_allowed_role == "oracle_audit_only"
