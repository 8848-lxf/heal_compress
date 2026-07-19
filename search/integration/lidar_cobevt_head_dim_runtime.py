"""Runtime and numerical evidence for synthetic CoBEVT Attention engines."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Protocol

import torch

from search.model_families.lidar_cobevt.head_dim_capability import (
    HeadDimCandidate,
)
from search.model_families.lidar_cobevt.head_dim_synthetic import (
    build_synthetic_inputs,
)
from search.reporting.cobevt_head_dim_capability import attention_tensor_parity


class _EngineRunner(Protocol):
    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def run_profiled(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]: ...


_DIAGNOSTIC_ROLES = {
    "output": "output",
    "qk_score": "qk_score",
    "softmax_output": "softmax",
    "av_output": "av_output",
}


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("runtime_latency_samples_empty")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, float(quantile)).item())


def _numerical_safety(parity: Iterable[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for row in parity:
        case = str(row.get("input_case", "unknown"))
        role = str(row.get("role", "unknown"))
        if not bool(row.get("finite")):
            failures.append(f"{case}:{role}:nonfinite")
        if role == "output" and float(row.get("cosine_similarity", 0.0)) < 0.99:
            failures.append(f"{case}:{role}:cosine_below_0.99")
        if role == "output" and float(row.get("relative_l2_error", float("inf"))) > 0.1:
            failures.append(f"{case}:{role}:relative_l2_above_0.1")
        if role == "softmax" and float(row.get("js_divergence", float("inf"))) > 0.05:
            failures.append(f"{case}:{role}:js_above_0.05")
    return not failures, failures


def evaluate_synthetic_runtime(
    candidate: HeadDimCandidate,
    *,
    production_runner: _EngineRunner,
    diagnostic_runner: _EngineRunner,
    reference_diagnostic_runner: _EngineRunner,
    input_cases: Iterable[str],
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Evaluate parity with diagnostic engines and latency with production only."""

    if int(warmup_iterations) < 0 or int(measured_iterations) <= 0:
        raise ValueError("invalid_synthetic_runtime_iteration_count")
    reference_candidate = replace(
        candidate, precision_profile="P0_strict_fp32"
    )
    parity_rows: list[dict[str, Any]] = []
    production_diagnostic_parity: list[dict[str, Any]] = []
    production_reference_parity: list[dict[str, Any]] = []
    candidate_inputs_by_case: dict[str, dict[str, torch.Tensor]] = {}
    for input_case in input_cases:
        case = str(input_case)
        reference_inputs = build_synthetic_inputs(
            reference_candidate, input_case=case
        )
        candidate_inputs = build_synthetic_inputs(candidate, input_case=case)
        candidate_inputs_by_case[case] = candidate_inputs
        reference_outputs = reference_diagnostic_runner.run(reference_inputs)
        candidate_outputs = diagnostic_runner.run(candidate_inputs)
        production_outputs = production_runner.run(candidate_inputs)
        if set(_DIAGNOSTIC_ROLES) - set(reference_outputs):
            raise RuntimeError("reference_diagnostic_output_missing")
        if set(_DIAGNOSTIC_ROLES) - set(candidate_outputs):
            raise RuntimeError("candidate_diagnostic_output_missing")
        if "output" not in production_outputs:
            raise RuntimeError("candidate_production_output_missing")
        production_parity = attention_tensor_parity(
            candidate_outputs["output"],
            production_outputs["output"],
            role="output",
        )
        production_parity = {
            "comparison": "production_vs_diagnostic",
            "input_case": case,
            **production_parity,
        }
        production_diagnostic_parity.append(production_parity)
        reference_parity = attention_tensor_parity(
            reference_outputs["output"],
            production_outputs["output"],
            role="output",
        )
        reference_parity = {
            "comparison": "production_vs_same_shape_fp32",
            "input_case": case,
            **reference_parity,
        }
        production_reference_parity.append(reference_parity)
        parity_rows.append(reference_parity)
        for output_name, role in _DIAGNOSTIC_ROLES.items():
            metrics = attention_tensor_parity(
                reference_outputs[output_name],
                candidate_outputs[output_name],
                role=role,
            )
            parity_rows.append(
                {
                    "comparison": "diagnostic_vs_same_shape_fp32",
                    "input_case": case,
                    **metrics,
                }
            )

    timing_case = (
        "random_normal"
        if "random_normal" in candidate_inputs_by_case
        else next(iter(candidate_inputs_by_case))
    )
    timing_inputs = candidate_inputs_by_case[timing_case]
    for _ in range(int(warmup_iterations)):
        production_runner.run_profiled(timing_inputs)
    timings: list[float] = []
    for _ in range(int(measured_iterations)):
        outputs, profile = production_runner.run_profiled(timing_inputs)
        if "output" not in outputs or not bool(torch.isfinite(outputs["output"]).all()):
            raise RuntimeError("synthetic_runtime_output_nonfinite")
        timings.append(float(profile["execute_async_ms"]))
    numerical_safe, numerical_failures = _numerical_safety(parity_rows)
    return {
        "diagnostic_latency_eligible": False,
        "input_cases": sorted(candidate_inputs_by_case),
        "latency_source": "production_engine",
        "latency_status": "screening_shared_gpu",
        "measured_iterations": int(measured_iterations),
        "numerical_failure_reasons": numerical_failures,
        "numerical_safe": numerical_safe,
        "p50_ms": _percentile(timings, 0.50),
        "p90_ms": _percentile(timings, 0.90),
        "p99_ms": _percentile(timings, 0.99),
        "parity": parity_rows,
        "production_diagnostic_parity": production_diagnostic_parity,
        "production_reference_parity": production_reference_parity,
        "runtime_success": True,
        "warmup_iterations": int(warmup_iterations),
    }


__all__ = ["evaluate_synthetic_runtime"]
