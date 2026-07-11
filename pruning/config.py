"""Serializable configuration for the formal structured-pruning pipeline.

The top-level :class:`PruningConfig` keeps the v10.9 constructor fields while
also exposing the policy objects used by the formal API.  Runtime objects
(models, loaders and callables) deliberately do not belong in these configs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_ROUND_TO = {4, 8, 16, 32, 64, 128}
DEFAULT_ALLOWED_CHANNELS_PER_GROUP = (4, 8, 16, 32, 64, 128, 256, 512)


class _StringEnum(str, Enum):
    """String enum with stable JSON/YAML representation."""


class ImportanceMode(_StringEnum):
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    FIRST_ORDER_TAYLOR = "first_order_taylor"
    SECOND_ORDER_FISHER = "second_order_fisher"


class ImportanceNormalizationStrategy(_StringEnum):
    COUPLED_DEPENDENCY_MEAN_THEN_SCOPE_MEAN_V1 = "coupled_dependency_mean_then_scope_mean_v1"
    NONE = "none"


class ImportanceAggregation(_StringEnum):
    COUPLED_UNIT = "coupled_unit"
    ATOMIC_UNIT = "atomic_unit"


class SelectionStrategy(_StringEnum):
    GLOBAL_ONE_SHOT = "global_one_shot"
    LOCAL_SCOPE = "local_scope"
    CONSTRAINED_GLOBAL = "constrained_global"


class BudgetMode(_StringEnum):
    PARAMETER = "parameter"
    CHANNEL = "channel"


class GroupedConvSelectionPolicy(_StringEnum):
    INDEPENDENT_GROUP_TOPK = "independent_group_topk"
    SHARED_LOCAL_MEAN = "shared_local_mean"
    REMOVE_GROUPS = "remove_groups"


def validate_round_to(value: int) -> int:
    """Validate the legacy dense-channel round-to value."""

    round_to = int(value)
    if round_to not in SUPPORTED_ROUND_TO or round_to <= 0 or (round_to & (round_to - 1)) != 0:
        raise ValueError(
            f"round_to must be one of {sorted(SUPPORTED_ROUND_TO)} and a power of two, got {round_to}"
        )
    return round_to


def _enum(value: Any, enum_type: type[Enum]) -> Enum:
    if isinstance(value, enum_type):
        return value
    return enum_type(str(value))


@dataclass(frozen=True)
class ImportanceNormalizationConfig:
    strategy: ImportanceNormalizationStrategy = (
        ImportanceNormalizationStrategy.COUPLED_DEPENDENCY_MEAN_THEN_SCOPE_MEAN_V1
    )
    epsilon: float = 1e-12
    version: str = "normalization-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", _enum(self.strategy, ImportanceNormalizationStrategy))
        if float(self.epsilon) <= 0:
            raise ValueError("normalization epsilon must be positive")


@dataclass(frozen=True)
class ImportanceConfig:
    mode: ImportanceMode = ImportanceMode.FIRST_ORDER_TAYLOR
    normalization: ImportanceNormalizationConfig = field(default_factory=ImportanceNormalizationConfig)
    aggregation: ImportanceAggregation = ImportanceAggregation.COUPLED_UNIT
    gradient_accumulation: str = "mean"
    implementation_version: str = "first-order-taylor-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _enum(self.mode, ImportanceMode))
        object.__setattr__(self, "aggregation", _enum(self.aggregation, ImportanceAggregation))
        if isinstance(self.normalization, Mapping):
            object.__setattr__(self, "normalization", ImportanceNormalizationConfig(**dict(self.normalization)))
        if self.gradient_accumulation not in {"mean", "sum"}:
            raise ValueError("gradient_accumulation must be 'mean' or 'sum'")

    def __str__(self) -> str:
        return self.mode.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.mode.value == other
        if not isinstance(other, ImportanceConfig):
            return False
        return asdict(self) == asdict(other)


@dataclass(frozen=True)
class SelectionConfig:
    strategy: SelectionStrategy = SelectionStrategy.GLOBAL_ONE_SHOT
    budget_mode: BudgetMode = BudgetMode.PARAMETER
    target_budget: float = 0.0
    per_domain_max_sparsity: float = 0.60
    minimum_retained_channels: int = 4
    deterministic_tie_break: str = "stable_id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", _enum(self.strategy, SelectionStrategy))
        object.__setattr__(self, "budget_mode", _enum(self.budget_mode, BudgetMode))
        if not 0.0 <= float(self.target_budget) <= 1.0:
            raise ValueError("target_budget must be in [0, 1]")
        if not 0.0 <= float(self.per_domain_max_sparsity) <= 1.0:
            raise ValueError("per_domain_max_sparsity must be in [0, 1]")
        if int(self.minimum_retained_channels) <= 0:
            raise ValueError("minimum_retained_channels must be positive")


@dataclass(frozen=True)
class GroupedConvConfig:
    """Grouped widths are explicitly defined in channels *per group*."""

    allowed_channels_per_group: tuple[int, ...] = DEFAULT_ALLOWED_CHANNELS_PER_GROUP
    selection_policy: GroupedConvSelectionPolicy = GroupedConvSelectionPolicy.INDEPENDENT_GROUP_TOPK
    require_equal_channels_per_group: bool = True
    allow_remove_groups: bool = False
    depthwise_special_case: bool = True

    def __post_init__(self) -> None:
        allowed = tuple(sorted({int(value) for value in self.allowed_channels_per_group}))
        if not allowed or any(value <= 0 for value in allowed):
            raise ValueError("allowed_channels_per_group must contain positive widths")
        object.__setattr__(self, "allowed_channels_per_group", allowed)
        object.__setattr__(self, "selection_policy", _enum(self.selection_policy, GroupedConvSelectionPolicy))
        if self.selection_policy is GroupedConvSelectionPolicy.REMOVE_GROUPS and not self.allow_remove_groups:
            raise ValueError("remove_groups requires allow_remove_groups=True")


@dataclass(frozen=True)
class AlignmentConfig:
    dense_conv_channel_alignment: int = 4
    apply_to_dependency_closure: bool = True
    protect_fixed_outputs_from_alignment: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dense_conv_channel_alignment",
            validate_round_to(self.dense_conv_channel_alignment),
        )


@dataclass(frozen=True)
class ProtectionConfig:
    protect_deblock_output: bool = True
    protect_fpn_output: bool = True
    protect_detection_head_output: bool = True
    allow_dependency_input_pruning: bool = True
    fail_on_unresolved_operation: bool = True


@dataclass(frozen=True)
class MaterializationConfig:
    one_shot: bool = True
    transactional_preflight: bool = True
    in_place: bool = False
    validate_after_materialization: bool = True


@dataclass(frozen=True)
class PhysicalArtifactConfig:
    snapshot_schema_version: str = "physical-structure-snapshot-v2"
    hash_schema_version: str = "physical-structure-v2"
    ledger_schema_version: str = "physical-pruning-application-ledger-v2"


@dataclass(frozen=True)
class ProjectPaths:
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
        }


@dataclass(frozen=True)
class PruningConfig:
    """Formal defaults plus backward-compatible v10.9 fields.

    ``target_pruning_ratio`` remains optional so both ``PruningConfig()`` and
    the historical ``PruningConfig(target_pruning_ratio=...)`` are valid.
    """

    target_pruning_ratio: float = 0.0
    target_pruning_mode: str = "param"
    round_to: int = 4
    max_ch_sparsity: float = 0.60
    stage1_min_per_group: int = 8
    stage1_max_ch_sparsity: float = 0.30
    protect_fpn_output: bool = True
    protect_head_output: bool = True
    no_extra_output_protection: bool = True
    fixed_shape_structural_skip: bool = True
    selector: str = "global_one_shot"
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    grouped_conv: GroupedConvConfig = field(default_factory=GroupedConvConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)
    materialization: MaterializationConfig = field(default_factory=MaterializationConfig)
    artifacts: PhysicalArtifactConfig = field(default_factory=PhysicalArtifactConfig)
    schema_version: str = "formal-pruning-config-v1"

    def __post_init__(self) -> None:
        if self.target_pruning_mode not in {"param", "channel"}:
            raise ValueError(f"unsupported target_pruning_mode: {self.target_pruning_mode}")
        if not 0.0 <= float(self.target_pruning_ratio) <= 1.0:
            raise ValueError("target_pruning_ratio must be in [0, 1]")
        object.__setattr__(self, "round_to", validate_round_to(self.round_to))
        if not 0.0 <= float(self.max_ch_sparsity) <= 1.0:
            raise ValueError("max_ch_sparsity must be in [0, 1]")
        if not 0.0 <= float(self.stage1_max_ch_sparsity) <= 1.0:
            raise ValueError("stage1_max_ch_sparsity must be in [0, 1]")
        if int(self.stage1_min_per_group) <= 0:
            raise ValueError("stage1_min_per_group must be positive")

        conversions = (
            ("importance", ImportanceConfig),
            ("selection", SelectionConfig),
            ("grouped_conv", GroupedConvConfig),
            ("alignment", AlignmentConfig),
            ("protection", ProtectionConfig),
            ("materialization", MaterializationConfig),
            ("artifacts", PhysicalArtifactConfig),
        )
        for name, config_type in conversions:
            value = getattr(self, name)
            if name == "importance" and isinstance(value, (str, ImportanceMode)):
                value = ImportanceConfig(mode=value)
            elif isinstance(value, Mapping):
                value = config_type(**dict(value))
            object.__setattr__(self, name, value)

        # Legacy fields remain authoritative when explicitly constructed.
        if self.alignment.dense_conv_channel_alignment != self.round_to:
            object.__setattr__(
                self,
                "alignment",
                AlignmentConfig(
                    dense_conv_channel_alignment=self.round_to,
                    apply_to_dependency_closure=self.alignment.apply_to_dependency_closure,
                    protect_fixed_outputs_from_alignment=self.alignment.protect_fixed_outputs_from_alignment,
                ),
            )
        budget_mode = BudgetMode.PARAMETER if self.target_pruning_mode == "param" else BudgetMode.CHANNEL
        if (
            self.selection.target_budget != self.target_pruning_ratio
            or self.selection.budget_mode is not budget_mode
            or self.selection.per_domain_max_sparsity != self.max_ch_sparsity
        ):
            object.__setattr__(
                self,
                "selection",
                SelectionConfig(
                    strategy=self.selection.strategy,
                    budget_mode=budget_mode,
                    target_budget=self.target_pruning_ratio,
                    per_domain_max_sparsity=self.max_ch_sparsity,
                    minimum_retained_channels=self.selection.minimum_retained_channels,
                    deterministic_tie_break=self.selection.deterministic_tie_break,
                ),
            )
        if self.selector not in {"greedy_global_ranking", "global_one_shot"}:
            raise ValueError(f"unsupported selector: {self.selector}")

    @property
    def importance_mode(self) -> str:
        return self.importance.mode.value

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {key: convert(item) for key, item in asdict(value).items()}
            return value

        return convert(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PruningConfig":
        data = dict(payload)
        nested = {
            "importance": ImportanceConfig,
            "selection": SelectionConfig,
            "grouped_conv": GroupedConvConfig,
            "alignment": AlignmentConfig,
            "protection": ProtectionConfig,
            "materialization": MaterializationConfig,
            "artifacts": PhysicalArtifactConfig,
        }
        for key, config_type in nested.items():
            value = data.get(key)
            if not isinstance(value, Mapping):
                continue
            value = dict(value)
            if key == "importance" and isinstance(value.get("normalization"), Mapping):
                value["normalization"] = ImportanceNormalizationConfig(**dict(value["normalization"]))
            data[key] = config_type(**value)
        return cls(**data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize to YAML and optionally write it to ``path``.

        JSON is used as a standards-compliant YAML subset when PyYAML is not
        installed, keeping release imports free of an optional dependency.
        """

        payload = self.to_dict()
        try:
            import yaml

            text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
        except ImportError:
            text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> "PruningConfig":
        """Load YAML from text or an existing caller-provided path."""

        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            candidate = Path(source)
            text = candidate.read_text(encoding="utf-8") if "\n" not in source and candidate.is_file() else source
        try:
            import yaml

            payload = yaml.safe_load(text)
        except ImportError:
            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("pruning YAML root must be a mapping")
        return cls.from_dict(payload)


__all__ = [
    "AlignmentConfig",
    "BudgetMode",
    "DEFAULT_ALLOWED_CHANNELS_PER_GROUP",
    "GroupedConvConfig",
    "GroupedConvSelectionPolicy",
    "ImportanceAggregation",
    "ImportanceConfig",
    "ImportanceMode",
    "ImportanceNormalizationConfig",
    "ImportanceNormalizationStrategy",
    "MaterializationConfig",
    "PhysicalArtifactConfig",
    "ProjectPaths",
    "ProtectionConfig",
    "PruningConfig",
    "SUPPORTED_ROUND_TO",
    "SelectionConfig",
    "SelectionStrategy",
    "validate_round_to",
]
