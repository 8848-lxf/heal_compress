"""Stage-1 proxy objectives."""

from .joint_weight_taylor import JointWeightTaylorProxy
from .joint_weight_activation_taylor import (
    DomainPerturbationCache,
    DomainPerturbationCacheKey,
    JointOutputTaylorStatistics,
    JointWeightActivationTaylorProxy,
    ModelCandidateOutputProvider,
    TaylorDeploymentUnit,
    collect_joint_output_taylor_statistics,
    score_joint_output_perturbation,
    taylor_units_from_transformer_precision,
)
from .transformer_bops import AttentionWorkload, FFNWorkload, ProjectionFreeAttentionWorkload, TransformerBOPSProxy, UnifiedBOPSProxy, profile_projection_free_attention_workloads, profile_transformer_workloads
from .transformer_latency import TransformerLatencyKey, TransformerLatencyLUT, TransformerLatencyProxy
from .transformer_parameter_slices import build_transformer_unit_parameter_slices

__all__ = [
    "JointWeightTaylorProxy",
    "DomainPerturbationCache",
    "DomainPerturbationCacheKey",
    "JointOutputTaylorStatistics",
    "JointWeightActivationTaylorProxy",
    "ModelCandidateOutputProvider",
    "TaylorDeploymentUnit",
    "collect_joint_output_taylor_statistics",
    "score_joint_output_perturbation",
    "taylor_units_from_transformer_precision",
    "AttentionWorkload",
    "FFNWorkload",
    "ProjectionFreeAttentionWorkload",
    "TransformerBOPSProxy",
    "UnifiedBOPSProxy",
    "profile_transformer_workloads",
    "profile_projection_free_attention_workloads",
    "TransformerLatencyKey",
    "TransformerLatencyLUT",
    "TransformerLatencyProxy",
    "build_transformer_unit_parameter_slices",
]
