"""Budget helpers for the formal global selector."""

from __future__ import annotations


def ratio_to_budget(total: int, ratio: float) -> int:
    if total < 0 or not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("invalid total or ratio")
    return int(round(int(total) * float(ratio)))


__all__ = ["ratio_to_budget"]
