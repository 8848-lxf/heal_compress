"""Stage-2 explicit Q/DQ export orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapters.quantization_adapter import FormalQuantizationAdapter
from ..candidate import CandidatePhenotype


class MixedPrecisionExportStage:
    def __init__(self, adapter: FormalQuantizationAdapter | None = None) -> None:
        self.adapter = adapter or FormalQuantizationAdapter()

    def run(
        self,
        *,
        model: Any,
        example_inputs: Sequence[Any] | Mapping[str, Any],
        physical_snapshot: Any,
        phenotype: CandidatePhenotype,
        output_dir: str | Path,
        calibration_scales: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            result = self.adapter.export_qdq(
                model=model,
                example_inputs=example_inputs,
                physical_snapshot=physical_snapshot,
                phenotype=phenotype,
                output_dir=output_dir,
                calibration_scales=calibration_scales or {},
            )
            return {"stage2_status": "ok", **result}
        except Exception as exc:  # noqa: BLE001 - Stage-2 converts failures into archive records
            return {"stage2_status": "qdq_failed", "failure_reason": f"{type(exc).__name__}: {exc}"}


def summarize_qdq_realization(
    *,
    requested_group_profile: Mapping[str, str],
    realized_group_profile: Mapping[str, str],
    realized_canonical_profile: Mapping[str, str],
    qdq_report: Mapping[str, Any],
    int8_macs_ratio: float,
) -> dict[str, Any]:
    records = list(qdq_report.get("records") or [])
    q_count = 0
    dq_count = 0

    def node_count(value: Any) -> int:
        if isinstance(value, (list, tuple, set)):
            return sum(1 for item in value if item)
        return 1 if value else 0

    for row in records:
        data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        q_count += sum(
            node_count(value)
            for key, value in data.items()
            if "quantize" in key.lower() and "dequantize" not in key.lower()
        )
        dq_count += sum(node_count(value) for key, value in data.items() if "dequantize" in key.lower())
    requested_int8_groups = [key for key, value in requested_group_profile.items() if str(value).upper() == "INT8"]
    realized_int8_groups = [key for key, value in realized_group_profile.items() if str(value).upper() == "INT8"]
    realized_int8_layers = [key for key, value in realized_canonical_profile.items() if str(value).upper() == "INT8"]
    return {
        "requested_int8_group_count": len(requested_int8_groups),
        "realized_int8_group_count": len(realized_int8_groups),
        "requested_int8_layer_count": int(qdq_report.get("requested_int8_count", len(realized_int8_layers))),
        "realized_int8_layer_count": len(realized_int8_layers),
        "QuantizeLinear_count": int(q_count),
        "DequantizeLinear_count": int(dq_count),
        "realized_int8_macs_ratio": float(int8_macs_ratio),
        "requested_int8_groups": sorted(requested_int8_groups),
        "realized_int8_groups": sorted(realized_int8_groups),
        "realized_int8_layers": sorted(realized_int8_layers),
    }
