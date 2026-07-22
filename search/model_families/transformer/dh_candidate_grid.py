"""Deterministic dense head-dimension grids for alignment experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class HeadDimensionCandidate:
    d_h: int
    heads: int
    original_d_h: int

    def __post_init__(self) -> None:
        if not 0 < int(self.d_h) <= int(self.original_d_h):
            raise ValueError("invalid_head_dimension_candidate")
        if int(self.heads) <= 0:
            raise ValueError("invalid_attention_head_count")

    @property
    def projection_width(self) -> int:
        return int(self.heads) * int(self.d_h)

    @property
    def alignment_class(self) -> str:
        if self.d_h % 16 == 0:
            return "multiple_of_16"
        if self.d_h % 8 == 0:
            return "multiple_of_8_only"
        if self.d_h % 4 == 0:
            return "multiple_of_4_only"
        if self.d_h % 2 == 0:
            return "multiple_of_2_only"
        return "odd"

    def to_dict(self) -> dict[str, int | str]:
        return {
            **asdict(self),
            "projection_width": self.projection_width,
            "d_h_mod16": self.d_h % 16,
            "d_h_mod8": self.d_h % 8,
            "d_h_mod4": self.d_h % 4,
            "d_h_mod2": self.d_h % 2,
            "projection_width_mod16": self.projection_width % 16,
            "projection_width_mod8": self.projection_width % 8,
            "projection_width_mod4": self.projection_width % 4,
            "alignment_class": self.alignment_class,
        }


def dense_head_dimension_grid(
    original_d_h: int,
    *,
    heads: int,
    low_width_extension: bool = False,
) -> tuple[HeadDimensionCandidate, ...]:
    """Return D0..max(16,ceil(D0/2)), optionally followed by 15..8."""

    original = int(original_d_h)
    lower = max(16, int(math.ceil(original / 2.0)))
    widths = list(range(original, lower - 1, -1))
    if low_width_extension and lower <= 16:
        widths.extend(range(15, 7, -1))
    return tuple(
        HeadDimensionCandidate(d_h=value, heads=int(heads), original_d_h=original)
        for value in widths
    )


def adjacent_microbenchmark_pairs(widths: Iterable[int]) -> tuple[tuple[int, int], ...]:
    values = sorted({int(value) for value in widths}, reverse=True)
    wanted = {(32, 31), (31, 30), (30, 29), (29, 28), (25, 24), (24, 23), (21, 20), (17, 16)}
    available = set(values)
    pairs = [pair for pair in sorted(wanted, reverse=True) if set(pair) <= available]
    if values and values[0] <= 16:
        pairs.extend((high, high - 1) for high in values[:-1] if high - 1 in available)
    if values and values[0] > 32:
        # Apply the same boundary-focused pattern to wider families.
        for high in values[:-1]:
            low = high - 1
            if low not in available:
                continue
            if high == values[0] or high % 8 in {0, 1} or high % 4 in {0, 1}:
                pairs.append((high, low))
    return tuple(dict.fromkeys(pairs))


__all__ = [
    "HeadDimensionCandidate",
    "adjacent_microbenchmark_pairs",
    "dense_head_dimension_grid",
]
