"""Non-invasive model-family integration layer for HEAL search."""

from .contracts import (
    DeploymentOperatorCapability,
    MergeBoundaryCapability,
    ModelFamilyAudit,
    PluginRequirement,
    PruningDomainCapability,
    WeightedOpCapability,
)
from .heal_v2xvit import HealLidarV2XViTProvider
from .model_provider import HealModelFamilyBundle, load_heal_model_family
from .readiness import ModelFamilySearchReadiness, build_model_family_search_readiness
from .registry import (
    detect_model_family,
    get_model_family,
    register_model_family,
    registered_model_families,
)


register_model_family(HealLidarV2XViTProvider())


__all__ = [
    "DeploymentOperatorCapability",
    "HealLidarV2XViTProvider",
    "HealModelFamilyBundle",
    "MergeBoundaryCapability",
    "ModelFamilyAudit",
    "PluginRequirement",
    "PruningDomainCapability",
    "WeightedOpCapability",
    "detect_model_family",
    "get_model_family",
    "load_heal_model_family",
    "ModelFamilySearchReadiness",
    "build_model_family_search_readiness",
    "register_model_family",
    "registered_model_families",
]
