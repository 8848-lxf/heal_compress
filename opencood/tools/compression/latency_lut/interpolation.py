from __future__ import annotations

from typing import Iterable


def _channel_tuple(item: object) -> tuple[int, int]:
    c_in = getattr(item, "C_in", None)
    c_out = getattr(item, "C_out", None)
    return int(c_in or 0), int(c_out or 0)


def conservative_channel_distance(target: tuple[int, int], candidate: tuple[int, int]) -> tuple[int, int, int]:
    """Distance that prefers upward channel matches over downward matches."""
    over = sum(max(0, cand - tgt) for tgt, cand in zip(target, candidate))
    under = sum(max(0, tgt - cand) for tgt, cand in zip(target, candidate))
    abs_dist = sum(abs(cand - tgt) for tgt, cand in zip(target, candidate))
    return (0 if under == 0 else 1, over + under * 4, abs_dist)


def nearest_channel_record(target_key: object, records: Iterable[object]) -> object | None:
    target = _channel_tuple(target_key)
    candidates = list(records)
    if not candidates:
        return None
    return min(candidates, key=lambda rec: conservative_channel_distance(target, _channel_tuple(rec.key)))


def linear_interpolate(
    target: float,
    low_x: float,
    low_y: float,
    high_x: float,
    high_y: float,
) -> float:
    if high_x == low_x:
        return float(high_y)
    alpha = (float(target) - float(low_x)) / (float(high_x) - float(low_x))
    return float(low_y) + alpha * (float(high_y) - float(low_y))
