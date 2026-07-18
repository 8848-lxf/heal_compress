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
from .pruning_recipe import (
    CobevtFusionWidthDomain,
    CobevtPhysicalPruneReport,
    CobevtPruningRecipe,
    DecodedFusionWidth,
    decode_fusion_width,
    legal_fusion_widths,
)

__all__ = [
    "AttentionDimMask",
    "CobevtModelBundle",
    "CobevtModelCapability",
    "CobevtModelPreflight",
    "CobevtFusionWidthDomain",
    "CobevtPhysicalPruneReport",
    "CobevtPruningRecipe",
    "DecodedFusionWidth",
    "FixedKContract",
    "PrunableCobevtAttention",
    "ScatterCapability",
    "derive_fixed_k",
    "decode_fusion_width",
    "legal_fusion_widths",
    "materialize_attention_bottleneck",
    "validate_scatter_capability",
]
