"""HEAL LiDAR CoBEVT model-family recipe."""

from .attention_dim_pruning import (
    AttentionDimMask,
    PrunableCobevtAttention,
    materialize_attention_bottleneck,
)
from .input_contract import FixedKContract, derive_fixed_k
from .model_capability import (
    CobevtModelBundle,
    CobevtModelCapability,
    CobevtModelPreflight,
    ScatterCapability,
    validate_scatter_capability,
)

__all__ = [
    "AttentionDimMask",
    "CobevtModelBundle",
    "CobevtModelCapability",
    "CobevtModelPreflight",
    "FixedKContract",
    "PrunableCobevtAttention",
    "ScatterCapability",
    "derive_fixed_k",
    "materialize_attention_bottleneck",
    "validate_scatter_capability",
]
