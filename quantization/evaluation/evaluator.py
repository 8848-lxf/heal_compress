"""Generic engine evaluation over caller-prepared inputs."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Mapping

from ..config import EvaluationConfig
from ..types import DetectionMetrics, EvaluationResult
from .latency import summarize_latency
from .metrics import compute_detection_metrics


def evaluate_engine(
    runner: Any,
    prepared_inputs: Iterable[Mapping[str, Any]],
    *,
    config: EvaluationConfig | None = None,
    postprocess: Callable[[Mapping[str, Any]], Any] | None = None,
    metric_adapter: Callable[[list[Any]], tuple[list[Mapping[str, Any]], int]] | None = None,
) -> EvaluationResult:
    """Evaluate an already loaded runner without loading models or datasets."""

    policy = config or EvaluationConfig()
    latencies: list[float] = []
    decoded: list[Any] = []
    failure = ""
    for frame_index, inputs in enumerate(prepared_inputs):
        if policy.max_frames is not None and len(latencies) >= int(policy.max_frames):
            break
        try:
            started = time.perf_counter()
            result = runner.run(inputs)
            outputs, engine_ms = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
            wall_ms = (time.perf_counter() - started) * 1000.0
            if frame_index < int(policy.warmup_frames):
                continue
            latencies.append(float(engine_ms if engine_ms is not None else wall_ms))
            decoded.append(postprocess(outputs) if postprocess is not None else outputs)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
    metrics: DetectionMetrics | None = None
    if metric_adapter is not None and decoded:
        predictions, ground_truth_count = metric_adapter(decoded)
        metrics = compute_detection_metrics(predictions, ground_truth_count)
    return EvaluationResult(
        success=bool(latencies) and not failure,
        evaluated_frames=len(latencies),
        latency=summarize_latency(latencies),
        metrics=metrics,
        outputs=decoded,
        failure_reason=failure,
    )
