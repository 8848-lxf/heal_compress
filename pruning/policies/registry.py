"""Composable pruning policy registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import AlignmentConfig, GroupedConvConfig, ProtectionConfig


@dataclass(frozen=True)
class PolicyRegistry:
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    grouped_conv: GroupedConvConfig = field(default_factory=GroupedConvConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)
    policy_version: str = "formal-pruning-policies-v1"


__all__ = ["PolicyRegistry"]
