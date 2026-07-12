"""Coupled precision-group search-space helpers."""

from .types import GroupPrecisionLegalization, QuantizationSearchGroup
from .group_builder import build_quantization_search_groups
from .legalizer import legalize_group_precision_genes

__all__ = [
    "GroupPrecisionLegalization",
    "QuantizationSearchGroup",
    "build_quantization_search_groups",
    "legalize_group_precision_genes",
]
