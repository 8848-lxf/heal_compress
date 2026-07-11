"""Independent formal FP16/INT8 ONNX/QDQ/TensorRT deployment package."""

from __future__ import annotations

from importlib import import_module
from typing import Any
import warnings

from .api import (
    apply_canonical_node_names,
    build_canonical_precision_mapping,
    build_onnx_origin_map,
    build_trt_command,
    build_trt_engine,
    compute_detection_metrics,
    evaluate_engine,
    export_pruned_signal_maxk_onnx,
    export_signal_maxk_onnx,
    generate_precision_profile,
    insert_explicit_qdq,
    load_trt_engine,
    run_engine_smoke,
    summarize_latency,
    trace_qdq_root_initializer,
    validate_engine_provenance,
    validate_engine_structure,
    validate_onnx_against_physical_snapshot,
    validate_precision_realization,
    validate_qdq_against_physical_snapshot,
)

__all__ = [
    "QuantizationConfig",
    "apply_canonical_node_names",
    "build_canonical_precision_mapping",
    "build_onnx_origin_map",
    "build_trt_command",
    "build_trt_engine",
    "compute_detection_metrics",
    "evaluate_engine",
    "export_pruned_signal_maxk_onnx",
    "export_signal_maxk_onnx",
    "generate_precision_profile",
    "insert_explicit_qdq",
    "load_trt_engine",
    "run_engine_smoke",
    "summarize_latency",
    "trace_qdq_root_initializer",
    "validate_engine_provenance",
    "validate_engine_structure",
    "validate_onnx_against_physical_snapshot",
    "validate_precision_realization",
    "validate_qdq_against_physical_snapshot",
    # Deprecated compatibility objects.
    "PseudoQuantManager",
    "QDQDeploymentBuilder",
    "pseudo_quantize_weight",
]

from .config import QuantizationConfig


def __getattr__(name: str) -> Any:
    if name in {"PseudoQuantManager", "pseudo_quantize_weight"}:
        warnings.warn(
            f"quantization.{name} is deprecated; use the formal explicit Q/DQ API",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(".pseudo_quant", __name__)
        return getattr(module, name)
    if name == "QDQDeploymentBuilder":
        warnings.warn(
            "quantization.QDQDeploymentBuilder is deprecated; use insert_explicit_qdq",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(".qdq_builder", __name__)
        return getattr(module, name)
    raise AttributeError(name)
