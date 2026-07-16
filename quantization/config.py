"""Serializable configuration for formal FP16/INT8 deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class Precision(str, Enum):
    """Supported layer precisions in the formal deployment chain."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


class ConfigMixin:
    """Small dict/YAML-friendly serialization mixin."""

    def to_dict(self) -> dict[str, Any]:
        return _encode(asdict(self))


@dataclass(frozen=True)
class ProjectPaths(ConfigMixin):
    """Caller-provided paths; no machine-specific defaults are embedded."""

    model_config: Path | None = None
    checkpoint: Path | None = None
    output_dir: Path | None = None
    tensorrt_root: Path | None = None
    trtexec_path: Path | None = None
    plugin_path: Path | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectPaths":
        return cls(**{key: Path(item) if item not in (None, "") else None for key, item in value.items()})


@dataclass(frozen=True)
class OnnxExportConfig(ConfigMixin):
    """Signal-maxK ONNX export contract."""

    fixed_k: int = 29696
    opset_version: int = 17
    dynamic_agent_dimension: bool = True
    min_agents: int = 1
    opt_agents: int = 2
    max_agents: int = 2
    input_names: tuple[str, ...] = (
        "voxel_features",
        "voxel_coords",
        "voxel_num_points",
        "pairwise_t_matrix",
        "valid_voxel_mask",
    )
    output_names: tuple[str, ...] = ("cls_preds", "reg_preds", "dir_preds")
    validate_onnx: bool = True
    allow_custom_ops: bool = True
    do_constant_folding: bool = True
    custom_op_domain: str = "trt"
    custom_opset_version: int = 1
    schema_version: str = "signal-maxk-onnx-export-v1"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OnnxExportConfig":
        data = dict(value)
        for key in ("input_names", "output_names"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class CanonicalNamingConfig(ConfigMixin):
    """Deterministic TensorRT-safe ONNX node naming policy."""

    prefix: str = "__canonical__"
    max_name_length: int = 68
    hash_length: int = 10
    policy_version: str = "canonical-v2-trt-safe-max68-sha256"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalNamingConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class PrecisionProfileConfig(ConfigMixin):
    """Deterministic four-profile FP16/INT8 policy."""

    ratios: tuple[tuple[str, float], ...] = (
        ("profile_000", 0.0),
        ("profile_001", 0.2),
        ("profile_002", 0.5),
        ("profile_003", 0.8),
    )
    seed: int = 0
    protected_fp16_patterns: tuple[str, ...] = (
        "deblocks",
        "fpn",
        "single_head",
        "cls_head",
        "reg_head",
        "dir_head",
        "scatter",
        "pillar_vfe",
    )
    policy_version: str = "fp16-int8-four-profile-v1"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrecisionProfileConfig":
        data = dict(value)
        if "ratios" in data:
            data["ratios"] = tuple((str(key), float(ratio)) for key, ratio in data["ratios"])
        if "protected_fp16_patterns" in data:
            data["protected_fp16_patterns"] = tuple(data["protected_fp16_patterns"])
        return cls(**data)

    def ratio_for(self, profile_id: str) -> float:
        try:
            return dict(self.ratios)[str(profile_id)]
        except KeyError as exc:
            raise ValueError(f"unknown precision profile: {profile_id}") from exc


@dataclass(frozen=True)
class CalibrationConfig(ConfigMixin):
    """Metadata contract for Q/DQ calibration scales."""

    split: str = "train"
    frame_count: int = 200
    require_observed_scales: bool = True
    activation_granularity: str = "per_tensor"
    weight_granularity: str = "per_tensor"
    schema_version: str = "calibration-metadata-v1"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class QDQConfig(ConfigMixin):
    """Explicit FP16/INT8 Q/DQ insertion policy."""

    allowed_precisions: tuple[str, ...] = ("fp16", "int8")
    insert_activation_input_qdq: bool = True
    insert_weight_qdq: bool = True
    insert_activation_output_qdq: bool = True
    activation_output_boundary_policy: str = "semantic_post_relu_or_post_merge_v2_unique_pre_activation_chain"
    require_calibration_scales: bool = True
    symmetric: bool = True
    zero_point: int = 0
    weight_granularity: str = "per_channel"
    merge_policy: str = "fp16_merge"
    explicit_fp16_compute_casts: bool = True
    explicit_fp32_compute_casts: bool = True
    grouped_conv_int8_allowed_channels_per_group: tuple[int, ...] = (4, 8, 16, 32)
    policy_version: str = "explicit-qdq-canonical-fp16-int8-v8-strongly-typed-compute-closure"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QDQConfig":
        data = dict(value)
        for key in ("allowed_precisions", "grouped_conv_int8_allowed_channels_per_group"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


def _normalize_shape_profiles(value: Mapping[str, Any]) -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        str(name): {str(kind): tuple(int(dim) for dim in shape) for kind, shape in profile.items()}
        for name, profile in value.items()
    }


@dataclass(frozen=True)
class TensorRTBuildConfig(ConfigMixin):
    """Pure command-generation and opt-in TensorRT build settings."""

    trtexec_path: Path | None = None
    plugin_path: Path | None = None
    workspace_mib: int = 512
    shape_profiles: dict[str, dict[str, tuple[int, ...]]] = field(default_factory=dict)
    timeout_seconds: int = 1800
    precision_constraints: str = "obey"
    enable_fp16: bool = True
    enable_int8: bool = True
    no_tf32: bool = True
    skip_inference: bool = True
    export_layer_info: bool = True
    strongly_typed: bool = False
    policy_version: str = "trt-fp16-int8-explicit-qdq-v2-optional-strong-typing"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorRTBuildConfig":
        data = dict(value)
        for key in ("trtexec_path", "plugin_path"):
            if data.get(key) not in (None, ""):
                data[key] = Path(data[key])
        if "shape_profiles" in data:
            data["shape_profiles"] = _normalize_shape_profiles(data["shape_profiles"])
        return cls(**data)


@dataclass(frozen=True)
class TensorRTValidationConfig(ConfigMixin):
    """Fail-closed structure and precision validation settings."""

    require_all_canonical_layers: bool = True
    reject_ambiguous_metadata: bool = True
    require_physical_snapshot_v2: bool = True
    policy_version: str = "trt-validation-canonical-v1"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorRTValidationConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class EvaluationConfig(ConfigMixin):
    """Runtime evaluation settings without dataset or server paths."""

    warmup_frames: int = 20
    max_frames: int | None = None
    score_threshold: float = 0.0
    iou_threshold: float = 0.5
    schema_version: str = "engine-evaluation-v1"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class QuantizationConfig(ConfigMixin):
    """Top-level round-trippable formal deployment configuration."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    onnx_export: OnnxExportConfig = field(default_factory=OnnxExportConfig)
    canonical_naming: CanonicalNamingConfig = field(default_factory=CanonicalNamingConfig)
    precision_profile: PrecisionProfileConfig = field(default_factory=PrecisionProfileConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    qdq: QDQConfig = field(default_factory=QDQConfig)
    tensorrt_build: TensorRTBuildConfig = field(default_factory=TensorRTBuildConfig)
    tensorrt_validation: TensorRTValidationConfig = field(default_factory=TensorRTValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantizationConfig":
        data = dict(value)
        return cls(
            paths=ProjectPaths.from_dict(data.get("paths", {})),
            onnx_export=OnnxExportConfig.from_dict(data.get("onnx_export", {})),
            canonical_naming=CanonicalNamingConfig.from_dict(data.get("canonical_naming", {})),
            precision_profile=PrecisionProfileConfig.from_dict(data.get("precision_profile", {})),
            calibration=CalibrationConfig.from_dict(data.get("calibration", {})),
            qdq=QDQConfig.from_dict(data.get("qdq", {})),
            tensorrt_build=TensorRTBuildConfig.from_dict(data.get("tensorrt_build", {})),
            tensorrt_validation=TensorRTValidationConfig.from_dict(data.get("tensorrt_validation", {})),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation", {})),
        )

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize to YAML and optionally write it to ``path``."""

        import yaml

        text = yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> "QuantizationConfig":
        """Load YAML text or a YAML file path."""

        import yaml

        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            candidate = Path(source)
            text = candidate.read_text(encoding="utf-8") if "\n" not in source and candidate.is_file() else source
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, Mapping):
            raise TypeError("quantization YAML root must be a mapping")
        return cls.from_dict(payload)


def save_quantization_config(config: QuantizationConfig, path: str | Path) -> Path:
    """Write a formal configuration as YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(destination)
    return destination


def load_quantization_config(path: str | Path) -> QuantizationConfig:
    """Load a formal YAML configuration."""

    return QuantizationConfig.from_yaml(Path(path))
