"""Non-invasive model-family integration layer for HEAL search."""

from .contracts import (
    DeploymentOperatorCapability,
    MergeBoundaryCapability,
    ModelFamilyAudit,
    PluginRequirement,
    PruningDomainCapability,
    WeightedOpCapability,
)
from .heal_lidar_baselines import HealLidarDiscoNetProvider, HealLidarFCooperProvider
from .heal_lidar_deployment import (
    HEAL_LIDAR_BASELINE_INPUT_NAMES,
    HEAL_LIDAR_BASELINE_OUTPUT_NAMES,
    HealLidarBaselineOnnxExport,
    build_heal_lidar_baseline_onnx_mapping,
    build_heal_lidar_baseline_precision_mapping,
    build_heal_lidar_baseline_quantization_groups,
    canonicalize_heal_lidar_baseline_onnx,
    export_heal_lidar_baseline_fixed_k_onnx,
    insert_heal_lidar_baseline_explicit_qdq,
    validate_heal_lidar_fusion_island_realization,
    validate_heal_lidar_precision_realization,
)
from .heal_lidar_pruning import (
    HealLidarBaselinePruningTopology,
    HealLidarPruningDomainSpec,
    build_heal_lidar_baseline_atomic_units,
    materialize_heal_lidar_baseline,
    validate_heal_lidar_baseline_pruning_topology,
)
from .heal_v2xvit import HealLidarV2XViTProvider
from .model_provider import HealModelFamilyBundle, load_heal_model_family
from .onnx_mapping import ModelFamilyOnnxMapping, build_v2xvit_onnx_mapping
from .readiness import ModelFamilySearchReadiness, build_model_family_search_readiness
from .registry import (
    detect_model_family,
    get_model_family,
    register_model_family,
    registered_model_families,
)


register_model_family(HealLidarV2XViTProvider())
register_model_family(HealLidarFCooperProvider())
register_model_family(HealLidarDiscoNetProvider())


__all__ = [
    "DeploymentOperatorCapability",
    "HEAL_LIDAR_BASELINE_INPUT_NAMES",
    "HEAL_LIDAR_BASELINE_OUTPUT_NAMES",
    "HealLidarBaselineOnnxExport",
    "HealLidarBaselinePruningTopology",
    "HealLidarDiscoNetProvider",
    "HealLidarFCooperProvider",
    "HealLidarPruningDomainSpec",
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
    "ModelFamilyOnnxMapping",
    "build_heal_lidar_baseline_atomic_units",
    "build_heal_lidar_baseline_onnx_mapping",
    "build_heal_lidar_baseline_precision_mapping",
    "build_heal_lidar_baseline_quantization_groups",
    "build_model_family_search_readiness",
    "build_v2xvit_onnx_mapping",
    "canonicalize_heal_lidar_baseline_onnx",
    "export_heal_lidar_baseline_fixed_k_onnx",
    "insert_heal_lidar_baseline_explicit_qdq",
    "materialize_heal_lidar_baseline",
    "register_model_family",
    "registered_model_families",
    "validate_heal_lidar_baseline_pruning_topology",
    "validate_heal_lidar_fusion_island_realization",
    "validate_heal_lidar_precision_realization",
]
