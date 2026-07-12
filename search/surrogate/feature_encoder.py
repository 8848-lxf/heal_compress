"""Feature encoding for lightweight real-performance predictors."""

from __future__ import annotations

from typing import Any, Mapping

from ..candidate import CandidatePhenotype


def encode_features(phenotype: CandidatePhenotype, proxy_metrics: Mapping[str, Any]) -> dict[str, float]:
    profile = phenotype.realized_precision_profile
    total_layers = max(len(profile), 1)
    return {
        "L_fisher": float(proxy_metrics.get("L_fisher", 0.0) or 0.0),
        "L_sqnr": float(proxy_metrics.get("L_sqnr", 0.0) or 0.0),
        "R_size": float(proxy_metrics.get("R_size", 0.0) or 0.0),
        "R_bops": float(proxy_metrics.get("R_bops", 0.0) or 0.0),
        "pruned_unit_count": float(len(phenotype.pruned_unit_ids)),
        "fp32_layer_count": float(sum(value == "FP32" for value in profile.values())),
        "fp16_layer_count": float(sum(value == "FP16" for value in profile.values())),
        "int8_layer_count": float(sum(value == "INT8" for value in profile.values())),
        "int8_layer_ratio": float(sum(value == "INT8" for value in profile.values()) / total_layers),
        "changed_layer_count": float(sum(value != "FP32" for value in profile.values())),
    }
