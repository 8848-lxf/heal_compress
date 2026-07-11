"""CPU-pure axis-aligned non-maximum suppression."""

from __future__ import annotations

from typing import Sequence


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(left_area + right_area - intersection, 1.0e-12)


def non_maximum_suppression(boxes: Sequence[Sequence[float]], scores: Sequence[float], iou_threshold: float = 0.5) -> list[int]:
    """Return retained indices in descending score order."""

    remaining = sorted(range(len(boxes)), key=lambda index: float(scores[index]), reverse=True)
    kept: list[int] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [index for index in remaining if _iou(boxes[current], boxes[index]) <= float(iou_threshold)]
    return kept
