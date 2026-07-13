"""Typed result and artifact schemas for formal quantization deployment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def stable_json_hash(value: Any) -> str:
    """Return the canonical SHA256 of a JSON-compatible value."""

    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResultMixin:
    """JSON-friendly result helper shared by formal artifact dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass
class CanonicalMappingEntry(ResultMixin):
    module_path: str
    module_type: str
    call_index: int
    onnx_op_type: str
    original_node_name: str
    canonical_node_name: str
    weight_initializer: str
    graph_index: int = 0
    groups: int = 1
    channels_per_group: int | None = None
    input_channels_per_group: int | None = None
    output_channels_per_group: int | None = None
    weight_shape: tuple[int, ...] = ()
    root_trace: tuple[dict[str, Any], ...] = ()
    schema_version: str = "canonical-mapping-entry-v1"


@dataclass
class OnnxOriginMapResult(ResultMixin):
    entries: list[CanonicalMappingEntry]
    source_onnx: str = ""
    unresolved_weighted_nodes: list[dict[str, Any]] = field(default_factory=list)
    functional_matmul_nodes: list[str] = field(default_factory=list)
    naming_policy_version: str = "canonical-v2-trt-safe-max68-sha256"
    schema_version: str = "onnx-origin-map-v1"
    origin_map_hash: str = ""

    def __post_init__(self) -> None:
        if not self.origin_map_hash:
            payload = [entry.to_dict() for entry in sorted(self.entries, key=lambda row: (row.call_index, row.graph_index))]
            self.origin_map_hash = stable_json_hash(payload)


@dataclass
class CanonicalRenameResult(ResultMixin):
    input_onnx: str
    output_onnx: str
    renamed_node_count: int
    renamed_nodes: list[dict[str, str]]
    naming_policy_version: str
    output_sha256: str = ""
    schema_version: str = "canonical-rename-report-v1"


@dataclass
class PrecisionAssignment(ResultMixin):
    module_path: str
    precision_group: str
    requested_precision: str
    ordering: int
    protected_precision: str = ""
    fallback_precision: str = ""
    fallback_reason: str = ""


@dataclass
class PrecisionProfileResult(ResultMixin):
    profile_id: str
    assignments: list[PrecisionAssignment]
    requested_int8_count: int
    requested_int8_ratio: float
    policy_version: str
    profile_hash: str = ""
    schema_version: str = "precision-profile-v1"

    def __post_init__(self) -> None:
        if not self.profile_hash:
            payload = {
                "profile_id": self.profile_id,
                "policy_version": self.policy_version,
                "assignments": [row.to_dict() for row in sorted(self.assignments, key=lambda item: item.module_path)],
            }
            self.profile_hash = stable_json_hash(payload)


@dataclass
class CalibrationScaleRecord(ResultMixin):
    """Observed symmetric per-tensor scales for one weighted module."""

    module_path: str
    activation_input_scale: float
    weight_scale: float
    activation_output_scale: float
    activation_input_amax: float
    weight_amax: float
    activation_output_amax: float
    observation_count: int
    zero_point: int = 0
    scale_method: str = "symmetric_absmax_div127_per_tensor_v1"


@dataclass
class CalibrationResult(ResultMixin):
    """Source-independent formal activation/weight calibration result."""

    records: list[CalibrationScaleRecord]
    frame_count: int
    module_count: int
    split: str = "train"
    activation_granularity: str = "per_tensor"
    weight_granularity: str = "per_tensor"
    scale_method: str = "symmetric_absmax_div127_per_tensor_v1"
    schema_version: str = "formal-calibration-result-v1"

    def scales(self) -> dict[str, dict[str, float]]:
        return {
            row.module_path: {
                "activation_input_scale": row.activation_input_scale,
                "weight_scale": row.weight_scale,
                "activation_output_scale": row.activation_output_scale,
            }
            for row in self.records
        }


@dataclass
class CanonicalPrecisionEntry(ResultMixin):
    module_path: str
    canonical_node_name: str
    precision_group: str
    requested_precision: str
    realized_request_precision: str
    original_node_name: str = ""
    weight_initializer: str = ""
    onnx_op_type: str = ""
    call_index: int = 0
    fallback_reason: str = ""
    protected_precision: str = ""
    realized_output_precision: str = ""


@dataclass
class CanonicalPrecisionMappingResult(ResultMixin):
    entries: list[CanonicalPrecisionEntry]
    profile_id: str = ""
    profile_hash: str = ""
    origin_map_hash: str = ""
    policy_version: str = "canonical-precision-mapping-v1"
    mapping_hash: str = ""
    schema_version: str = "canonical-precision-mapping-v1"

    def __post_init__(self) -> None:
        if not self.mapping_hash:
            self.mapping_hash = stable_json_hash([row.to_dict() for row in sorted(self.entries, key=lambda item: item.canonical_node_name)])


@dataclass
class QDQInsertionRecord(ResultMixin):
    module_path: str
    canonical_node_name: str
    weight_initializer: str
    activation_quantize_node: str = ""
    activation_dequantize_node: str = ""
    weight_quantize_node: str = ""
    weight_dequantize_node: str = ""
    output_quantize_nodes: list[str] = field(default_factory=list)
    output_dequantize_nodes: list[str] = field(default_factory=list)
    activation_output_q_inputs: list[str] = field(default_factory=list)
    activation_output_boundary_policy: str = ""
    scale: float = 0.0
    activation_input_scale: float = 0.0
    weight_scale: Any = 0.0
    weight_scale_shape: list[int] = field(default_factory=list)
    weight_axis: int | None = None
    weight_granularity: str = "per_tensor"
    activation_output_scale: float = 0.0
    zero_point: int = 0


@dataclass
class QDQInsertionResult(ResultMixin):
    input_onnx: str
    output_onnx: str
    inserted_layer_count: int
    requested_int8_count: int
    records: list[QDQInsertionRecord]
    fallback_entries: list[dict[str, Any]] = field(default_factory=list)
    calibration_metadata: dict[str, Any] = field(default_factory=dict)
    policy_version: str = "explicit-qdq-canonical-fp16-int8-v1"
    output_sha256: str = ""
    schema_version: str = "qdq-insertion-report-v1"


@dataclass
class QDQRootTrace(ResultMixin):
    compute_node_name: str
    compute_op_type: str
    consumed_weight_tensor: str
    root_initializer: str
    root_initializer_shape: tuple[int, ...]
    trace_chain: list[dict[str, Any]]
    transpose_permutations: list[tuple[int, ...]] = field(default_factory=list)
    trans_b: int = 0
    schema_version: str = "qdq-root-trace-v1"


@dataclass
class ValidationIssue(ResultMixin):
    code: str
    message: str
    module_path: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnnxValidationResult(ResultMixin):
    passed: bool
    checks: list[dict[str, Any]]
    issues: list[ValidationIssue]
    physical_truth_source: str = "physical_structure_snapshot_v2"
    schema_version: str = "onnx-physical-validation-v1"


@dataclass
class OnnxExportResult(ResultMixin):
    onnx_path: str
    input_names: list[str]
    output_names: list[str]
    fixed_k: int
    dynamic_agent_dimension: bool
    checker_passed: bool
    origin_map: OnnxOriginMapResult | None = None
    canonical_rename: CanonicalRenameResult | None = None
    validation: OnnxValidationResult | None = None
    export_report_hash: str = ""
    schema_version: str = "signal-maxk-onnx-export-result-v1"

    def __post_init__(self) -> None:
        if not self.export_report_hash:
            self.export_report_hash = stable_json_hash(
                {
                    "onnx_path": self.onnx_path,
                    "inputs": self.input_names,
                    "outputs": self.output_names,
                    "fixed_k": self.fixed_k,
                    "origin": self.origin_map.origin_map_hash if self.origin_map else "",
                }
            )


@dataclass
class TensorRTCommandResult(ResultMixin):
    command: list[str]
    onnx_path: str
    engine_path: str
    layer_info_path: str
    policy_version: str
    command_hash: str = ""
    schema_version: str = "tensorrt-command-v1"

    def __post_init__(self) -> None:
        if not self.command_hash:
            self.command_hash = stable_json_hash(self.command)


@dataclass
class TensorRTBuildResult(ResultMixin):
    success: bool
    command: TensorRTCommandResult
    returncode: int | None
    elapsed_seconds: float
    engine_hash: str = ""
    log_path: str = ""
    failure_reason: str = ""
    schema_version: str = "tensorrt-build-report-v1"


@dataclass
class EngineStructureValidationResult(ResultMixin):
    passed: bool
    matched_canonical_count: int
    expected_canonical_count: int
    missing_canonical_layers: list[str]
    ambiguous_layers: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    physical_snapshot_schema_version: str = ""
    physical_snapshot_hash: str = ""
    shape_checks: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "engine-structure-validation-v1"


@dataclass
class PrecisionRealizationResult(ResultMixin):
    passed: bool
    requested_int8_count: int
    realized_int8_count: int
    realized_fp16_count: int
    mismatches: list[dict[str, Any]]
    hidden_cast_count: int = 0
    reformat_count: int = 0
    boundary_count: int = 0
    unresolved_layer_count: int = 0
    schema_version: str = "precision-realization-v1"


@dataclass
class ProvenanceValidationResult(ResultMixin):
    passed: bool
    missing_fields: list[str]
    issues: list[ValidationIssue] = field(default_factory=list)
    schema_version: str = "engine-provenance-validation-v1"


@dataclass
class TensorRTEngineHandle(ResultMixin):
    engine_path: str
    engine: Any
    runtime: Any
    logger: Any
    plugin_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"engine_path": self.engine_path, "plugin_path": self.plugin_path, "loaded": self.engine is not None}


@dataclass
class EngineSmokeResult(ResultMixin):
    success: bool
    output_shapes: dict[str, tuple[int, ...]]
    latency_ms: float | None = None
    failure_reason: str = ""
    schema_version: str = "engine-smoke-v1"


@dataclass
class LatencySummary(ResultMixin):
    count: int
    mean_ms: float | None
    p50_ms: float | None
    p90_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    minimum_ms: float | None
    maximum_ms: float | None


@dataclass
class DetectionMetrics(ResultMixin):
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    average_precision: float
    ground_truth_count: int


@dataclass
class EvaluationResult(ResultMixin):
    success: bool
    evaluated_frames: int
    latency: LatencySummary
    metrics: DetectionMetrics | None = None
    outputs: list[Any] = field(default_factory=list)
    failure_reason: str = ""
    schema_version: str = "engine-evaluation-v1"
