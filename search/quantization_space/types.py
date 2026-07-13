"""Types for quantization precision-group search variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..candidate import PrecisionDecision, normalize_precision


@dataclass(frozen=True)
class QuantizationSearchGroup:
    group_id: str
    module_paths: tuple[str, ...]
    canonical_node_ids: tuple[str, ...]
    allowed_precisions: tuple[str, ...]
    protected: bool
    protection_reason: str
    ordering: int
    parameter_count: int
    baseline_macs: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", str(self.group_id))
        object.__setattr__(self, "module_paths", tuple(str(value) for value in self.module_paths))
        object.__setattr__(self, "canonical_node_ids", tuple(str(value) for value in self.canonical_node_ids))
        object.__setattr__(
            self,
            "allowed_precisions",
            tuple(normalize_precision(value) for value in self.allowed_precisions),
        )
        object.__setattr__(self, "protected", bool(self.protected))
        object.__setattr__(self, "protection_reason", str(self.protection_reason or ""))
        object.__setattr__(self, "ordering", int(self.ordering))
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        object.__setattr__(self, "baseline_macs", float(self.baseline_macs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "module_paths": list(self.module_paths),
            "canonical_node_ids": list(self.canonical_node_ids),
            "allowed_precisions": list(self.allowed_precisions),
            "protected": self.protected,
            "protection_reason": self.protection_reason,
            "ordering": self.ordering,
            "parameter_count": self.parameter_count,
            "baseline_macs": self.baseline_macs,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GroupPrecisionLegalization:
    requested_group_profile: dict[str, str]
    stage1_legalized_group_profile: dict[str, str]
    fallback_report: dict[str, dict[str, str]]
    groups: tuple[QuantizationSearchGroup, ...]

    def expand_to_module_profile(self) -> dict[str, PrecisionDecision]:
        profile: dict[str, PrecisionDecision] = {}
        for group in sorted(self.groups, key=lambda row: row.ordering):
            requested = self.requested_group_profile[group.group_id]
            realized = self.stage1_legalized_group_profile[group.group_id]
            reason = self.fallback_report.get(group.group_id, {}).get("fallback_reason", "")
            for module_path in group.module_paths:
                profile[module_path] = PrecisionDecision(requested, realized, reason)
        return {key: profile[key] for key in sorted(profile)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_group_profile": dict(sorted(self.requested_group_profile.items())),
            "stage1_legalized_group_profile": dict(sorted(self.stage1_legalized_group_profile.items())),
            "fallback_report": {key: self.fallback_report[key] for key in sorted(self.fallback_report)},
            "precision_group_expansion": {
                group.group_id: list(group.module_paths)
                for group in sorted(self.groups, key=lambda row: row.ordering)
            },
            "module_to_precision_group": {
                module_path: group.group_id
                for group in sorted(self.groups, key=lambda row: row.ordering)
                for module_path in group.module_paths
            },
            "quantization_group_contracts": {
                group.group_id: {
                    **dict(group.metadata),
                    "member_layers": list(group.module_paths),
                    "requested_precision": self.requested_group_profile[group.group_id],
                    "legalized_precision": self.stage1_legalized_group_profile[group.group_id],
                }
                for group in sorted(self.groups, key=lambda row: row.ordering)
            },
        }
