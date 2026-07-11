"""Small source-independent detection metric helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..types import DetectionMetrics


def compute_detection_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth_count: int,
) -> DetectionMetrics:
    """Compute precision, recall, and ranked AP from matched predictions."""

    ordered = sorted(predictions, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    true_positives = sum(bool(row.get("matched", False)) for row in ordered)
    false_positives = len(ordered) - true_positives
    false_negatives = max(0, int(ground_truth_count) - true_positives)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(int(ground_truth_count), 1)
    running_tp = 0
    precision_at_true_positive = 0.0
    for rank, row in enumerate(ordered, 1):
        if bool(row.get("matched", False)):
            running_tp += 1
            precision_at_true_positive += running_tp / rank
    average_precision = precision_at_true_positive / max(int(ground_truth_count), 1)
    return DetectionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=float(precision),
        recall=float(recall),
        average_precision=float(average_precision),
        ground_truth_count=int(ground_truth_count),
    )
