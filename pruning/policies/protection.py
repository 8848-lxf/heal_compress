"""Directional output protection without freezing dependency inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch.nn as nn

from ..types import DirectionalProtectionPolicy


DEFAULT_FIXED_OUTPUT_TOKENS = (
    "deblock",
    "fpn",
    "cls_head",
    "reg_head",
    "dir_head",
    "classification_head",
    "regression_head",
    "direction_head",
)


def infer_directional_protection(
    module_path: str,
    module: nn.Module,
    *,
    fixed_output_tokens: Sequence[str] = DEFAULT_FIXED_OUTPUT_TOKENS,
) -> DirectionalProtectionPolicy:
    """Infer the formal protection policy for one module path."""

    del module  # Type is intentionally accepted for adapter-specific extension.
    lowered = str(module_path).lower()
    matched = next((token for token in fixed_output_tokens if token in lowered), "")
    if matched:
        return DirectionalProtectionPolicy(
            module_path=str(module_path),
            root_pruning_allowed=False,
            input_dependency_pruning_allowed=True,
            output_dependency_pruning_allowed=False,
            fixed_output_contract=True,
            protection_reason=f"fixed_output_contract:{matched}",
        )
    return DirectionalProtectionPolicy(module_path=str(module_path))


def build_protection_registry(model: nn.Module) -> dict[str, DirectionalProtectionPolicy]:
    """Build deterministic path-to-policy entries for a live model."""

    return {
        name: infer_directional_protection(name, module)
        for name, module in model.named_modules()
    }


def require_direction_allowed(
    policy: DirectionalProtectionPolicy,
    axis: str,
    *,
    dependency_driven: bool,
) -> tuple[bool, str]:
    """Return whether a requested direction is legal under ``policy``."""

    direction = str(axis)
    if direction == "in":
        allowed = dependency_driven and policy.input_dependency_pruning_allowed
        return allowed, "" if allowed else "input_dependency_pruning_protected"
    if direction in {"out", "channel"} and policy.fixed_output_contract:
        return False, policy.protection_reason or "fixed_output_contract"
    if not dependency_driven and not policy.root_pruning_allowed:
        return False, policy.protection_reason or "root_pruning_protected"
    if dependency_driven and not policy.output_dependency_pruning_allowed:
        return False, policy.protection_reason or "output_dependency_pruning_protected"
    return True, ""


__all__ = [
    "DEFAULT_FIXED_OUTPUT_TOKENS",
    "build_protection_registry",
    "infer_directional_protection",
    "require_direction_allowed",
]
