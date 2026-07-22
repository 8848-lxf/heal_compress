"""Fail-closed structural contract for unified Q/K/V head dimensions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AttentionFamilyRecord:
    model: str
    family_id: str
    attention_kind: str
    module_paths: tuple[str, ...]
    heads: int
    original_d_h: int
    embed_dim: int
    window_size: int | tuple[int, ...] | None = None
    shared_dependency: str = ""

    def __post_init__(self) -> None:
        if not self.module_paths or len(set(self.module_paths)) != len(self.module_paths):
            raise ValueError("attention_family_module_paths_invalid")
        if min(int(self.heads), int(self.original_d_h), int(self.embed_dim)) <= 0:
            raise ValueError("attention_family_dimensions_invalid")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["module_paths"] = list(self.module_paths)
        return value


@dataclass(frozen=True)
class HeadLocalMask:
    module_path: str
    original_d_h: int
    keep_by_head: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        rows = tuple(tuple(int(value) for value in row) for row in self.keep_by_head)
        object.__setattr__(self, "keep_by_head", rows)
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("head_local_mask_keep_count_mismatch")
        for row in rows:
            if not row or tuple(sorted(set(row))) != row:
                raise ValueError("head_local_mask_indices_not_sorted_unique")
            if row[0] < 0 or row[-1] >= int(self.original_d_h):
                raise ValueError("head_local_mask_index_out_of_range")

    @property
    def target_d_h(self) -> int:
        return len(self.keep_by_head[0])

    @property
    def heads(self) -> int:
        return len(self.keep_by_head)

    def flattened(self) -> tuple[int, ...]:
        return tuple(
            head * int(self.original_d_h) + local
            for head, row in enumerate(self.keep_by_head)
            for local in row
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "original_d_h": int(self.original_d_h),
            "target_d_h": self.target_d_h,
            "heads": self.heads,
            "keep_by_head": [list(row) for row in self.keep_by_head],
            "flattened_keep": list(self.flattened()),
        }


def masks_from_rankings(
    family: AttentionFamilyRecord,
    rankings: Mapping[str, Sequence[Sequence[int]]],
    target_d_h: int,
) -> dict[str, HeadLocalMask]:
    target = int(target_d_h)
    if not 0 < target <= int(family.original_d_h):
        raise ValueError("target_head_dimension_out_of_range")
    result: dict[str, HeadLocalMask] = {}
    for path in family.module_paths:
        if path not in rankings:
            raise ValueError(f"head_dimension_ranking_missing:{path}")
        per_head = tuple(tuple(int(value) for value in row) for row in rankings[path])
        if len(per_head) != int(family.heads):
            raise ValueError(f"head_dimension_ranking_head_count:{path}")
        keep = tuple(tuple(sorted(row[-target:])) for row in per_head)
        if any(len(set(row)) != int(family.original_d_h) for row in per_head):
            raise ValueError(f"head_dimension_ranking_not_permutation:{path}")
        result[path] = HeadLocalMask(path, int(family.original_d_h), keep)
    return result


def audit_nested_masks(masks_by_width: Mapping[int, Mapping[str, HeadLocalMask]]) -> list[dict[str, Any]]:
    widths = sorted(int(value) for value in masks_by_width)
    rows: list[dict[str, Any]] = []
    for low, high in zip(widths, widths[1:]):
        for path, low_mask in masks_by_width[low].items():
            high_mask = masks_by_width[high][path]
            nested = all(set(a) <= set(b) for a, b in zip(low_mask.keep_by_head, high_mask.keep_by_head))
            rows.append({"module_path": path, "lower_d_h": low, "higher_d_h": high, "nested": nested})
    if any(not row["nested"] for row in rows):
        raise RuntimeError("head_dimension_nested_mask_contract_failed")
    return rows


def validate_physical_attention(module: Any, target_d_h: int) -> list[str]:
    issues: list[str] = []
    heads = int(getattr(module, "heads", 0))
    d_qk = int(getattr(module, "d_qk", target_d_h))
    d_v = int(getattr(module, "d_v", target_d_h))
    if d_qk != int(target_d_h) or d_v != int(target_d_h):
        issues.append("qkv_target_dimension_mismatch")
    for name in ("q_proj", "k_proj", "v_proj"):
        value = getattr(module, name, None)
        if value is not None and int(value.out_features) != heads * int(target_d_h):
            issues.append(f"{name}_physical_output_shape_mismatch")
    out = getattr(module, "out_proj", None)
    if out is not None and int(out.in_features) != heads * int(target_d_h):
        issues.append("out_projection_physical_input_shape_mismatch")
    if not math.isclose(float(getattr(module, "scale", 0.0)), int(target_d_h) ** -0.5, rel_tol=0.0, abs_tol=1e-12):
        issues.append("attention_scale_not_updated")
    return issues


__all__ = [
    "AttentionFamilyRecord",
    "HeadLocalMask",
    "audit_nested_masks",
    "masks_from_rankings",
    "stable_hash",
    "validate_physical_attention",
]
