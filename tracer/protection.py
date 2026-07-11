"""Directional channel-protection policy construction."""

from __future__ import annotations

from collections.abc import Sequence

from .config import ProtectionConfig
from .types import ModuleInventoryEntry, ProtectionPolicy


def _matches(path: str, keywords: tuple[str, ...]) -> bool:
    lowered = path.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def build_protection_policies(
    modules: Sequence[ModuleInventoryEntry],
    config: ProtectionConfig | None = None,
) -> list[ProtectionPolicy]:
    """Build one explicit directional policy for every weighted module.

    Output protection never disables dependency-driven input pruning. PFN and
    scatter interfaces remain conservative because their feature/index mapping
    requires an adapter-specific resolver.
    """

    cfg = config or ProtectionConfig()
    explicit = set(cfg.explicit_fixed_output_modules)
    policies: list[ProtectionPolicy] = []
    for module in modules:
        if not module.weighted:
            continue
        path = module.module_path
        reason = ""
        input_allowed = True
        if path in explicit:
            reason = "explicit_fixed_output_contract"
        elif cfg.protect_deblock_outputs and (
            module.module_type.startswith("ConvTranspose") and _matches(path, cfg.deblock_keywords)
        ):
            reason = "deblock_output_contract"
        elif cfg.protect_fpn_outputs and _matches(path, cfg.fpn_keywords):
            reason = "fpn_output_contract"
        elif cfg.protect_detection_head_outputs and _matches(path, cfg.detection_head_keywords):
            reason = "detection_head_output_contract"
        elif _matches(path, cfg.fixed_interface_keywords):
            reason = "fixed_pfn_scatter_interface_contract"
            input_allowed = False
        fixed = bool(reason)
        if fixed and reason != "fixed_pfn_scatter_interface_contract":
            input_allowed = bool(cfg.dependency_input_pruning_for_fixed_outputs)
        policies.append(
            ProtectionPolicy(
                module_path=path,
                module_type=module.module_type,
                root_pruning_allowed=not fixed,
                input_dependency_pruning_allowed=input_allowed,
                output_dependency_pruning_allowed=not fixed,
                fixed_output_contract=fixed,
                protection_reason=reason,
            )
        )
    return policies


def protection_policy_map(policies: Sequence[ProtectionPolicy]) -> dict[str, ProtectionPolicy]:
    """Return a deterministic module-path lookup."""

    return {policy.module_path: policy for policy in policies}

