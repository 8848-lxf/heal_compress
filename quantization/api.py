"""Stable public API for formal FP16/INT8 quantization deployment."""

from __future__ import annotations

from .evaluation import compute_detection_metrics, evaluate_engine, summarize_latency
from .export import (
    apply_canonical_node_names,
    build_heal_signal_maxk_export_module,
    build_onnx_origin_map,
    export_pruned_signal_maxk_onnx,
    export_signal_maxk_onnx,
    prepare_signal_maxk_inputs,
    validate_onnx_against_physical_snapshot,
)
from .precision import (
    apply_fp16_merge_output_contract,
    build_canonical_precision_mapping,
    collect_calibration_scales,
    generate_precision_profile,
    insert_explicit_qdq,
    resolve_activation_output_boundary,
    trace_qdq_root_initializer,
    validate_qdq_against_physical_snapshot,
)
from .tensorrt import (
    build_trt_command,
    build_trt_engine,
    load_trt_engine,
    run_engine_smoke,
    validate_engine_provenance,
    validate_engine_structure,
    validate_precision_realization,
)

__all__ = [
    "apply_canonical_node_names",
    "apply_fp16_merge_output_contract",
    "build_canonical_precision_mapping",
    "collect_calibration_scales",
    "build_heal_signal_maxk_export_module",
    "build_onnx_origin_map",
    "build_trt_command",
    "build_trt_engine",
    "compute_detection_metrics",
    "evaluate_engine",
    "export_pruned_signal_maxk_onnx",
    "export_signal_maxk_onnx",
    "generate_precision_profile",
    "insert_explicit_qdq",
    "resolve_activation_output_boundary",
    "prepare_signal_maxk_inputs",
    "load_trt_engine",
    "run_engine_smoke",
    "summarize_latency",
    "trace_qdq_root_initializer",
    "validate_engine_provenance",
    "validate_engine_structure",
    "validate_onnx_against_physical_snapshot",
    "validate_precision_realization",
    "validate_qdq_against_physical_snapshot",
]
