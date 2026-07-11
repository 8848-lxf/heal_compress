"""Serializable configuration for the formal dependency tracer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def _dump_yaml(payload: Mapping[str, Any], path: str | Path | None = None) -> str:
    """Serialize a config payload without permitting Python object tags."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is optional at import time
        raise RuntimeError("PyYAML is required for YAML configuration round-trip") from exc
    text = yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=True)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def _load_yaml(source: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for YAML configuration round-trip") from exc
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    else:
        candidate = Path(source)
        text = candidate.read_text(encoding="utf-8") if "\n" not in source and candidate.is_file() else source
    value = yaml.safe_load(text)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("tracer YAML configuration must contain a mapping")
    return dict(value)


class TraceBackend(str, Enum):
    """Static graph source used in addition to the representative forward."""

    TORCH_FX = "torch_fx"
    AUTO = "auto"


class UnknownOperationPolicy(str, Enum):
    """Policy for operations without a registered channel transform."""

    FAIL_CHANNEL_CHANGING = "fail_channel_changing"
    FAIL_ALL = "fail_all"
    RECORD_SHAPE_PRESERVING = "record_shape_preserving"


@dataclass(frozen=True)
class DependencyConfig:
    """Rules controlling channel-axis inference and dependency closure."""

    schema_version: str = "dependency-config-v1"
    image_channel_axis: int = 1
    matrix_feature_axis: int = 1
    couple_residual_add_branches: bool = True
    propagate_concat_offsets: bool = True
    propagate_dependency_inputs: bool = True
    treat_depthwise_as_channel_passthrough: bool = True
    unknown_operation_policy: UnknownOperationPolicy = UnknownOperationPolicy.FAIL_CHANNEL_CHANGING

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unknown_operation_policy"] = self.unknown_operation_policy.value
        return payload

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Return safe YAML and optionally write it to ``path``."""

        return _dump_yaml(self.to_dict(), path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DependencyConfig":
        payload = dict(data)
        payload["unknown_operation_policy"] = UnknownOperationPolicy(
            payload.get("unknown_operation_policy", UnknownOperationPolicy.FAIL_CHANNEL_CHANGING.value)
        )
        return cls(**payload)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "DependencyConfig":
        return cls.from_dict(_load_yaml(source))


@dataclass(frozen=True)
class ProtectionConfig:
    """Directional output-contract protection rules.

    Deblock, FPN and detection-head outputs are fixed by default. Their input
    axes remain dependency-prunable when an upstream producer is narrowed.
    """

    schema_version: str = "protection-config-v1"
    deblock_keywords: tuple[str, ...] = ("deblock", "deblocks")
    fpn_keywords: tuple[str, ...] = (
        "fpn_out",
        "pyramid_fpn",
        "fusion_neck",
        "neck.out_conv",
        "neck.final_conv",
    )
    detection_head_keywords: tuple[str, ...] = (
        "cls_head",
        "reg_head",
        "dir_head",
        "heatmap_head",
        "obj_head",
        "box_head",
    )
    fixed_interface_keywords: tuple[str, ...] = (
        "pillar_vfe",
        "pfn_layers",
        "scatter",
    )
    protect_deblock_outputs: bool = True
    protect_fpn_outputs: bool = True
    protect_detection_head_outputs: bool = True
    dependency_input_pruning_for_fixed_outputs: bool = True
    explicit_fixed_output_modules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Return safe YAML and optionally write it to ``path``."""

        return _dump_yaml(self.to_dict(), path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProtectionConfig":
        payload = dict(data)
        for key in (
            "deblock_keywords",
            "fpn_keywords",
            "detection_head_keywords",
            "fixed_interface_keywords",
            "explicit_fixed_output_modules",
        ):
            if key in payload:
                payload[key] = tuple(str(value) for value in payload[key])
        return cls(**payload)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "ProtectionConfig":
        return cls.from_dict(_load_yaml(source))


@dataclass(frozen=True)
class TraceConfig:
    """Top-level formal tracing configuration."""

    graph_schema_version: str = "trace-result-v1"
    backend: TraceBackend = TraceBackend.AUTO
    record_non_weighted_leaf_calls: bool = True
    run_representative_forward: bool = True
    input_call_style: str = "auto"
    fail_on_fx_trace_error: bool = True
    dependency: DependencyConfig = field(default_factory=DependencyConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)

    def __post_init__(self) -> None:
        if self.input_call_style not in {"auto", "single", "args", "kwargs"}:
            raise ValueError(f"unsupported input_call_style: {self.input_call_style}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_schema_version": self.graph_schema_version,
            "backend": self.backend.value,
            "record_non_weighted_leaf_calls": self.record_non_weighted_leaf_calls,
            "run_representative_forward": self.run_representative_forward,
            "input_call_style": self.input_call_style,
            "fail_on_fx_trace_error": self.fail_on_fx_trace_error,
            "dependency": self.dependency.to_dict(),
            "protection": self.protection.to_dict(),
        }

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Return safe YAML and optionally write it to ``path``."""

        return _dump_yaml(self.to_dict(), path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceConfig":
        payload = dict(data)
        payload["backend"] = TraceBackend(payload.get("backend", TraceBackend.AUTO.value))
        payload["dependency"] = DependencyConfig.from_dict(payload.get("dependency", {}))
        payload["protection"] = ProtectionConfig.from_dict(payload.get("protection", {}))
        return cls(**payload)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "TraceConfig":
        return cls.from_dict(_load_yaml(source))


@dataclass(frozen=True)
class ProjectPaths:
    """Caller-supplied project paths; no server-specific defaults are used."""

    project_root: str | None = None
    model_config: str | None = None
    checkpoint: str | None = None
    dataset_root: str | None = None
    output_root: str | None = None
    plugin_library: str | None = None
    trtexec: str | None = None
    schema_version: str = "project-paths-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectPaths":
        return cls(**dict(data))

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Return safe YAML and optionally write it to ``path``."""

        return _dump_yaml(self.to_dict(), path)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "ProjectPaths":
        return cls.from_dict(_load_yaml(source))
