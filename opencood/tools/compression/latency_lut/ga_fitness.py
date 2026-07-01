from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .latency_proxy import LatencyEstimate, LatencyProxy


@dataclass
class LatencyFitnessResult:
    P_latency: float
    T_lut_raw: float
    T_calibrated: float
    T_proxy: float
    uncertainty: float
    missing_keys: list[dict[str, Any]]
    unavailable_keys: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "P_latency": self.P_latency,
            "T_lut_raw": self.T_lut_raw,
            "T_calibrated": self.T_calibrated,
            "T_proxy": self.T_proxy,
            "uncertainty": self.uncertainty,
            "missing_keys": self.missing_keys,
            "unavailable_keys": self.unavailable_keys,
        }


def latency_penalty_from_estimate(estimate: LatencyEstimate, *, T_base: float, R_target: float) -> LatencyFitnessResult:
    if T_base <= 0:
        raise ValueError("T_base must be positive")
    violation = max(0.0, float(estimate.latency_ms) / float(T_base) - float(R_target))
    return LatencyFitnessResult(
        P_latency=violation * violation,
        T_lut_raw=float(estimate.latency_lut_raw_ms),
        T_calibrated=float(estimate.latency_calibrated_ms),
        T_proxy=float(estimate.latency_ms),
        uncertainty=float(estimate.uncertainty_ms),
        missing_keys=list(estimate.missing_keys),
        unavailable_keys=list(estimate.unavailable_keys),
    )


def estimate_latency_penalty(
    proxy: LatencyProxy,
    candidate_config: dict[str, Any],
    *,
    T_base: float,
    R_target: float,
) -> LatencyFitnessResult:
    return latency_penalty_from_estimate(proxy.estimate(candidate_config), T_base=T_base, R_target=R_target)

