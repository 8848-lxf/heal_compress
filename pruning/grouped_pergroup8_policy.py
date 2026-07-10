"""Grouped Conv2d per-group 8-aligned pruning policy for v10.8."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass
class GroupedPerGroup8Decision:
    module_name: str
    stage_guess: str
    groups: int
    C_in_before: int
    C_out_before: int
    C_in_after: int
    C_out_after: int
    in_per_group_before: int
    in_per_group_after: int
    out_per_group_before: int
    out_per_group_after: int
    output_pruned: bool
    input_pruned: bool
    per_group_output_keep_indices: dict[int, list[int]] = field(default_factory=dict)
    per_group_input_keep_indices: dict[int, list[int]] = field(default_factory=dict)
    global_keep_indices: list[int] = field(default_factory=list)
    global_prune_indices: list[int] = field(default_factory=list)
    skipped_output_prune_reason: str = ""
    skipped_input_prune_reason: str = ""
    legality_passed: bool = True

    def to_report_row(self) -> dict[str, Any]:
        return asdict(self)


def _stage_guess(module_name: str, per_group: int) -> str:
    low = module_name.lower()
    if "stage0" in low or per_group == 4:
        return "stage0_like"
    if "stage1" in low or per_group == 8:
        return "stage1_like"
    if "stage2" in low or per_group == 16:
        return "stage2_like"
    return "unknown"


def grouped_per_group_aligned_independent_local_pruning(
    *,
    module_name: str,
    scores: torch.Tensor | Sequence[float],
    groups: int,
    raw_prune_per_group: int,
    C_in_before: int | None = None,
    C_out_before: int | None = None,
    max_ch_sparsity: float = 0.60,
    align: int = 8,
) -> GroupedPerGroup8Decision:
    """Resolve grouped output pruning with independent local ranking per group."""

    score_tensor = torch.as_tensor(scores, dtype=torch.float32).detach().cpu().flatten()
    groups = int(groups)
    C_out = int(C_out_before or int(score_tensor.numel()))
    C_in = int(C_in_before or C_out)
    out_per = C_out // groups if groups and C_out % groups == 0 else 0
    in_per = C_in // groups if groups and C_in % groups == 0 else 0
    base = GroupedPerGroup8Decision(
        module_name=module_name,
        stage_guess=_stage_guess(module_name, out_per),
        groups=groups,
        C_in_before=C_in,
        C_out_before=C_out,
        C_in_after=C_in,
        C_out_after=C_out,
        in_per_group_before=in_per,
        in_per_group_after=in_per,
        out_per_group_before=out_per,
        out_per_group_after=out_per,
        output_pruned=False,
        input_pruned=False,
        legality_passed=True,
    )
    if groups <= 0 or out_per <= 0 or int(score_tensor.numel()) != C_out:
        base.skipped_output_prune_reason = "invalid_grouped_shape"
        base.legality_passed = False
        return base
    raw_prune = max(0, min(int(raw_prune_per_group), out_per))
    if raw_prune <= 0:
        base.skipped_output_prune_reason = "no_output_prune_requested"
        return base
    raw_keep = out_per - raw_prune
    aligned_keep = raw_keep - (raw_keep % int(align))
    if aligned_keep < int(align):
        base.skipped_output_prune_reason = "per_group_after_below8" if int(align) == 8 else "per_group_after_below_round_to"
        return base
    final_prune_per_group = out_per - aligned_keep
    if final_prune_per_group / max(out_per, 1) > float(max_ch_sparsity) + 1e-12:
        base.skipped_output_prune_reason = "max_ch_sparsity_exceeded"
        return base

    matrix = score_tensor.view(groups, out_per)
    keep_map: dict[int, list[int]] = {}
    global_keep: list[int] = []
    for group_idx in range(groups):
        _, keep_idx = torch.topk(matrix[group_idx], aligned_keep, largest=True, sorted=False)
        local_keep = sorted(int(v) for v in keep_idx.tolist())
        keep_map[group_idx] = local_keep
        global_keep.extend(group_idx * out_per + local for local in local_keep)
    keep_set = set(global_keep)
    global_prune = [idx for idx in range(C_out) if idx not in keep_set]
    base.per_group_output_keep_indices = keep_map
    base.global_keep_indices = global_keep
    base.global_prune_indices = global_prune
    base.output_pruned = bool(global_prune)
    base.C_out_after = len(global_keep)
    base.out_per_group_after = aligned_keep
    if C_in == C_out and in_per == out_per:
        base.C_in_after = len(global_keep)
        base.in_per_group_after = aligned_keep
        base.input_pruned = bool(global_prune)
        base.per_group_input_keep_indices = {k: list(v) for k, v in keep_map.items()}
    return base


def validate_grouped_input_keep_pergroup8(
    *,
    module_name: str,
    keep_indices: Sequence[int],
    groups: int,
    C_in_before: int,
    align: int = 8,
) -> dict[str, Any]:
    """Validate grouped Conv2d input compaction without changing groups."""

    groups = int(groups)
    C_in_before = int(C_in_before)
    if groups <= 0 or C_in_before % groups:
        return {
            "module_name": module_name,
            "legal": False,
            "skipped_input_prune_reason": "grouped_input_divisibility_violation",
        }
    in_per = C_in_before // groups
    keep = sorted({int(v) for v in keep_indices if 0 <= int(v) < C_in_before})
    per_group: dict[int, list[int]] = {}
    for group_idx in range(groups):
        start = group_idx * in_per
        per_group[group_idx] = sorted(idx - start for idx in keep if start <= idx < start + in_per)
    counts = {len(v) for v in per_group.values()}
    if len(counts) != 1:
        return {
            "module_name": module_name,
            "legal": False,
            "skipped_input_prune_reason": "grouped_input_keep_count_imbalance",
            "per_group_input_keep_indices": per_group,
        }
    keep_per = next(iter(counts), 0)
    if keep_per < int(align):
        return {
            "module_name": module_name,
            "legal": False,
            "skipped_input_prune_reason": "in_per_group_after_below8" if int(align) == 8 else "in_per_group_after_below_round_to",
            "in_per_group_before": in_per,
            "in_per_group_after": keep_per,
            "per_group_input_keep_indices": per_group,
        }
    if keep_per % int(align):
        return {
            "module_name": module_name,
            "legal": False,
            "skipped_input_prune_reason": "in_per_group_after_not_multiple_of8",
            "in_per_group_before": in_per,
            "in_per_group_after": keep_per,
            "per_group_input_keep_indices": per_group,
        }
    return {
        "module_name": module_name,
        "legal": True,
        "skipped_input_prune_reason": "",
        "in_per_group_before": in_per,
        "in_per_group_after": keep_per,
        "per_group_input_keep_indices": per_group,
        "keep_indices": keep,
    }
