"""CPU-pure evaluation helpers and generic runtime evaluator."""

from .evaluator import evaluate_engine
from .latency import summarize_latency
from .metrics import compute_detection_metrics
from .nms import non_maximum_suppression

__all__ = ["compute_detection_metrics", "evaluate_engine", "non_maximum_suppression", "summarize_latency"]
