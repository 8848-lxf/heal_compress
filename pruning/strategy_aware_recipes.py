"""Strategy-aware grouped-conv recipe builders for v9.4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class RecipeRequest:
    module_name: str
    axis: str
    prune_indices: list[int]
    dependency_type: str = ""


@dataclass
class PruneRecipe:
    recipe_id: str
    policy: str
    requests: list[RecipeRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelMember:
    module_name: str
    axis: str
    local_index: int
    source_transform: str = "identity"
    branch_id: str = ""
    concat_offset: int = 0
    residual_add_id: str = ""
    dependency_type: str = ""


@dataclass
class CoupledChannelUnitV94:
    unit_id: str
    root_module: str
    root_axis: str
    root_channel_index: int
    members: list[ChannelMember]


def _old_group_keep_count(channels: int, groups: int, keep_indices: Sequence[int]) -> dict[int, int]:
    per = channels // groups
    counts = {group_id: 0 for group_id in range(groups)}
    for idx in sorted(int(v) for v in keep_indices):
        counts[idx // per] += 1
    return counts


def _reinterpretation_ratio(channels: int, groups: int, keep_indices: Sequence[int]) -> float:
    if not keep_indices:
        return 0.0
    old_per = channels // groups
    new_per = len(keep_indices) // groups
    if new_per <= 0:
        return 0.0
    mismatches = 0
    for new_idx, old_idx in enumerate(sorted(int(v) for v in keep_indices)):
        if old_idx // old_per != new_idx // new_per:
            mismatches += 1
    return mismatches / len(keep_indices)


def build_grouped_bottleneck_recipe(
    *,
    policy: str,
    module_names: dict[str, str],
    channels: int,
    groups: int,
    prune_indices: Sequence[int] | None = None,
    keep_indices: Sequence[int] | None = None,
    prune_group_ids: Sequence[int] | None = None,
    groups_new: int | None = None,
    in_per_group: int | None = None,
    out_per_group: int | None = None,
) -> PruneRecipe:
    policy = str(policy).upper()
    conv1 = module_names["conv1"]
    conv2 = module_names["conv2"]
    bn2 = module_names["bn2"]
    conv3 = module_names["conv3"]
    prune = sorted(int(v) for v in (prune_indices or []))
    keep = sorted(int(v) for v in (keep_indices or [idx for idx in range(channels) if idx not in set(prune)]))

    if policy == "A":
        return PruneRecipe(
            recipe_id=f"A::{conv2}",
            policy="A",
            requests=[
                RecipeRequest(conv2, "out", prune, "grouped_conv_out"),
                RecipeRequest(bn2, "out", prune, "bn_after_grouped_conv"),
                RecipeRequest(conv3, "in", prune, "downstream_in"),
            ],
            metadata={
                "groups_after": groups,
                "groups_changed": False,
                "grouped_keep_pattern_mismatch": len(set(_old_group_keep_count(channels, groups, keep).values())) > 1,
                "reinterpretation_ratio": _reinterpretation_ratio(channels, groups, keep),
            },
        )

    if policy == "B":
        counts = _old_group_keep_count(channels, groups, keep)
        group_balance_pass = len(set(counts.values())) == 1
        return PruneRecipe(
            recipe_id=f"B::{conv2}",
            policy="B",
            requests=[
                RecipeRequest(conv2, "out", prune, "group_balanced_grouped_conv_out"),
                RecipeRequest(bn2, "out", prune, "bn_after_grouped_conv"),
                RecipeRequest(conv3, "in", prune, "downstream_in"),
            ],
            metadata={
                "old_group_keep_count": counts,
                "group_balance_pass": group_balance_pass,
                "reinterpretation_ratio": 0.0 if group_balance_pass else _reinterpretation_ratio(channels, groups, keep),
            },
        )

    if policy == "C":
        in_per = int(in_per_group or channels // groups)
        out_per = int(out_per_group or channels // groups)
        prune_groups = sorted(int(v) for v in (prune_group_ids or []))
        out_prune = [g * out_per + local for g in prune_groups for local in range(out_per)]
        in_prune = [g * in_per + local for g in prune_groups for local in range(in_per)]
        groups_after = groups - len(prune_groups)
        return PruneRecipe(
            recipe_id=f"C::{conv2}",
            policy="C",
            requests=[
                RecipeRequest(conv1, "out", in_prune, "upstream_group_block_out"),
                RecipeRequest(conv2, "in", in_prune, "grouped_conv_input_block"),
                RecipeRequest(conv2, "out", out_prune, "grouped_conv_output_block"),
                RecipeRequest(bn2, "out", out_prune, "bn_after_grouped_conv"),
                RecipeRequest(conv3, "in", out_prune, "downstream_group_block_in"),
            ],
            metadata={
                "groups_before": groups,
                "groups_after": groups_after,
                "in_per_group_after": in_per,
                "out_per_group_after": out_per,
                "reinterpretation_ratio": 0.0,
            },
        )

    if policy == "D":
        if groups_new is None:
            raise ValueError("D requires groups_new")
        if groups_new >= groups or groups % groups_new != 0:
            raise ValueError("invalid_group_coarsening_groups_new")
        in_per = int(in_per_group or channels // groups)
        out_per = int(out_per_group or channels // groups)
        merge_factor = groups // groups_new
        semantic_mismatch_count = 0
        copy_plan = []
        for new_idx, old_idx in enumerate(keep):
            old_group = old_idx // out_per
            target_new_group = old_group // merge_factor
            new_group = new_idx // max(len(keep) // groups_new, 1)
            if new_group != target_new_group:
                semantic_mismatch_count += 1
            offset = (old_group % merge_factor) * in_per
            copy_plan.append(
                {
                    "old_out_idx": old_idx,
                    "new_out_idx": new_idx,
                    "old_group": old_group,
                    "target_new_group": target_new_group,
                    "offset_in_new_group": offset,
                    "copied_slice_range": [offset, offset + in_per],
                }
            )
        return PruneRecipe(
            recipe_id=f"D::{conv2}",
            policy="D",
            requests=[
                RecipeRequest(conv2, "out", prune, "group_coarsening_bucket_local_out"),
                RecipeRequest(bn2, "out", prune, "bn_after_grouped_conv"),
                RecipeRequest(conv3, "in", prune, "downstream_in"),
            ],
            metadata={
                "groups_old": groups,
                "groups_new": groups_new,
                "merge_factor": merge_factor,
                "zero_pad_copy_plan": copy_plan,
                "weight_truncation_count": 0,
                "semantic_mismatch_count": semantic_mismatch_count,
            },
        )
    raise ValueError(f"unsupported_grouped_conv_policy:{policy}")


def build_coupled_channel_units_for_conv_bn_next(
    *,
    root_module: str,
    bn_module: str,
    next_module: str,
    channels: int,
) -> list[CoupledChannelUnitV94]:
    units = []
    for idx in range(int(channels)):
        units.append(
            CoupledChannelUnitV94(
                unit_id=f"{root_module}::out::{idx}",
                root_module=root_module,
                root_axis="out",
                root_channel_index=idx,
                members=[
                    ChannelMember(root_module, "out", idx, dependency_type="conv_out"),
                    ChannelMember(bn_module, "out", idx, dependency_type="conv_out_to_bn"),
                    ChannelMember(next_module, "in", idx, dependency_type="conv_out_to_next_conv_in"),
                ],
            )
        )
    return units
