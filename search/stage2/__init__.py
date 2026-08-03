"""Stage-2 export and evaluation helpers."""

from .heal_lidar_baseline_real_evaluator import (
    HealLidarBaselineCandidateEvaluator,
    HealLidarBaselineEvaluationConfig,
    HealLidarBaselineRealEvaluator,
)
from .transformer_precision_export import (
    audit_onnx_attention_fp32_contract,
    audit_trt_attention_fp32_contract,
    build_transformer_precision_mapping,
)
from .greedy_anchor_gate import (
    GreedyAnchorGatePolicy,
    apply_greedy_anchor_gate,
    build_greedy_anchor_manifest,
)

__all__ = [
    "HealLidarBaselineCandidateEvaluator",
    "HealLidarBaselineEvaluationConfig",
    "HealLidarBaselineRealEvaluator",
    "audit_onnx_attention_fp32_contract",
    "audit_trt_attention_fp32_contract",
    "build_transformer_precision_mapping",
    "GreedyAnchorGatePolicy",
    "apply_greedy_anchor_gate",
    "build_greedy_anchor_manifest",
]
