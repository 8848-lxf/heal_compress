"""Adapter over the formal explicit Q/DQ deployment pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..candidate import CandidatePhenotype


def _lower_precision(value: str) -> str:
    return str(value).lower()


class FormalQuantizationAdapter:
    """Export pruned ONNX, legalize precision, and insert explicit Q/DQ."""

    pipeline_name = "formal_explicit_qdq_mixed_precision"

    def __init__(
        self,
        *,
        export_onnx_fn: Callable[..., Any] | None = None,
        profile_builder_fn: Callable[..., Any] | None = None,
        canonical_mapping_fn: Callable[..., Any] | None = None,
        calibration_fn: Callable[..., Any] | None = None,
        qdq_fn: Callable[..., Any] | None = None,
    ) -> None:
        if any(value is None for value in (export_onnx_fn, canonical_mapping_fn, qdq_fn)):
            try:
                from quantization.api import (
                    build_canonical_precision_mapping,
                    collect_calibration_scales,
                    export_pruned_signal_maxk_onnx,
                    insert_explicit_qdq,
                )
                from quantization.precision.calibration import validate_calibration_scales
            except ImportError:
                from heal_compress.quantization.api import (
                    build_canonical_precision_mapping,
                    collect_calibration_scales,
                    export_pruned_signal_maxk_onnx,
                    insert_explicit_qdq,
                )
                from heal_compress.quantization.precision.calibration import validate_calibration_scales
            export_onnx_fn = export_onnx_fn or export_pruned_signal_maxk_onnx
            canonical_mapping_fn = canonical_mapping_fn or build_canonical_precision_mapping
            qdq_fn = qdq_fn or insert_explicit_qdq
            if calibration_fn is None:
                def calibration_fn(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
                    scales = kwargs.get("scales")
                    if scales is not None:
                        validate_calibration_scales(scales, config=kwargs.get("config"))
                        return scales
                    result = collect_calibration_scales(*args, **{key: value for key, value in kwargs.items() if key != "scales"})
                    return result.scales()
        self.export_onnx_fn = export_onnx_fn
        self.profile_builder_fn = profile_builder_fn or self.precision_profile_from_phenotype
        self.canonical_mapping_fn = canonical_mapping_fn
        self.calibration_fn = calibration_fn
        self.qdq_fn = qdq_fn

    def precision_profile_from_phenotype(self, origin_map: Any, phenotype: CandidatePhenotype) -> Any:
        """Convert a search phenotype to the formal PrecisionProfileResult."""

        try:
            from quantization.types import PrecisionAssignment, PrecisionProfileResult
        except ImportError:
            from heal_compress.quantization.types import PrecisionAssignment, PrecisionProfileResult

        origin_modules = sorted(str(row.module_path) for row in origin_map.entries)
        missing = sorted(set(origin_modules) - set(phenotype.precision_profile))
        if missing:
            raise ValueError(f"phenotype lacks precision assignments for canonical modules: {missing}")
        assignments = [
            PrecisionAssignment(
                module_path=module_path,
                precision_group=str((phenotype.metadata.get("module_to_precision_group") or {}).get(module_path, f"pg::{module_path}")),
                requested_precision=_lower_precision(phenotype.precision_profile[module_path].requested_precision),
                ordering=order,
            )
            for order, module_path in enumerate(origin_modules)
        ]
        return PrecisionProfileResult(
            profile_id="search_phenotype",
            assignments=assignments,
            requested_int8_count=sum(row.requested_precision == "int8" for row in assignments),
            requested_int8_ratio=sum(row.requested_precision == "int8" for row in assignments) / max(len(assignments), 1),
            policy_version=phenotype.precision_policy_version,
        )

    def export_qdq(
        self,
        *,
        model: Any,
        example_inputs: Sequence[Any] | Mapping[str, Any],
        physical_snapshot: Any,
        phenotype: CandidatePhenotype,
        output_dir: str | Path,
        onnx_config: Any | None = None,
        naming_config: Any | None = None,
        qdq_config: Any | None = None,
        calibration_scales: Mapping[str, Any] | None = None,
        calibration_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        export = self.export_onnx_fn(
            model,
            example_inputs,
            destination / "exported.onnx",
            physical_snapshot,
            config=onnx_config,
            naming_config=naming_config,
            report_path=destination / "onnx_export_report.json",
        )
        origin_map = export.origin_map
        profile = self.profile_builder_fn(origin_map, phenotype)
        mapping = self.canonical_mapping_fn(origin_map, profile, config=qdq_config)
        scales = self.calibration_fn(scales=calibration_scales or {}, config=None)
        qdq = self.qdq_fn(
            export.onnx_path,
            destination / "qdq.onnx",
            mapping,
            scales=scales,
            config=qdq_config,
            calibration_metadata=calibration_metadata or {},
        )
        return {
            "export": export,
            "origin_map": origin_map,
            "profile": profile,
            "mapping": mapping,
            "scales": scales,
            "qdq": qdq,
        }
