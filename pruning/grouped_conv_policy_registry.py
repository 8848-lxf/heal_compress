"""Grouped-conv pruning policy registry for v9.4 one-shot pruner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GroupedConvPolicy:
    key: str
    name: str
    description: str
    uses_torch_pruning_for_model_generation: bool = False
    torch_pruning_allowed_role: str = "oracle_audit_only"


class GroupedConvPolicyRegistry:
    def __init__(self, policies: Iterable[GroupedConvPolicy], default_key: str = "A"):
        self.policies = {policy.key: policy for policy in policies}
        if default_key not in self.policies:
            raise ValueError(f"default grouped conv policy not registered: {default_key}")
        self.default_policy = self.policies[default_key]

    @classmethod
    def default(cls) -> "GroupedConvPolicyRegistry":
        return cls(
            [
                GroupedConvPolicy(
                    "A",
                    "flat_output_groups_fixed",
                    "TP-like grouped Conv2d output-filter pruning with fixed input and fixed groups.",
                ),
                GroupedConvPolicy(
                    "B",
                    "group_balanced_output_groups_fixed",
                    "Output-only grouped Conv2d pruning with equal keep count per original output group.",
                ),
                GroupedConvPolicy(
                    "C",
                    "true_group_block_pruning",
                    "Whole old-group block pruning with synchronized upstream input and downstream output blocks.",
                ),
                GroupedConvPolicy(
                    "D",
                    "group_coarsening_zero_padded_reblock",
                    "Bucket-local output pruning with group coarsening and zero-padded input reblock.",
                ),
            ],
            default_key="A",
        )

    def get(self, key_or_name: str | None) -> GroupedConvPolicy:
        if not key_or_name:
            return self.default_policy
        text = str(key_or_name)
        if text in self.policies:
            return self.policies[text]
        for policy in self.policies.values():
            if text == policy.name:
                return policy
        raise ValueError(f"unsupported_grouped_conv_policy:{key_or_name}")

    def report(self) -> dict:
        return {
            "default_policy": self.default_policy.key,
            "single_policy_per_round": True,
            "policies": [policy.__dict__ for policy in self.policies.values()],
        }


def parse_group_conv_policy_choice(choice: str | Iterable[str] | None) -> GroupedConvPolicy:
    if choice is None:
        return GroupedConvPolicyRegistry.default().default_policy
    if isinstance(choice, str):
        return GroupedConvPolicyRegistry.default().get(choice)
    choices = [str(item) for item in choice if str(item)]
    unique = list(dict.fromkeys(choices))
    if len(unique) > 1:
        raise ValueError("multiple_grouped_conv_policy_in_one_round")
    return GroupedConvPolicyRegistry.default().get(unique[0] if unique else None)
