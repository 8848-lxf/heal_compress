"""Coupled precision-group search-space helpers."""

from .types import GroupPrecisionLegalization, QuantizationSearchGroup
from .group_builder import build_quantization_search_groups
from .legalizer import legalize_group_precision_genes

from .transformer_precision import (
    ACTIVATION_PRECISION_STATES,
    SEARCH_PRECISION_STATES,
    RealizedTransformerPrecision,
    TransformerPrecisionUnit,
    assert_transformer_precision_realized,
    build_transformer_precision_units,
    build_transformer_quantization_groups,
    expected_softmax_realization,
    validate_external_precision_profile,
)
from .smoothquant import (
    SMOOTHQUANT_ALPHA_GRID,
    SmoothQuantRecord,
    SmoothQuantRegistry,
    make_smoothquant_record,
    select_smoothquant_alpha,
    smoothquant_scale,
    smoothquant_transform,
)

__all__ = [
    "GroupPrecisionLegalization",
    "QuantizationSearchGroup",
    "build_quantization_search_groups",
    "legalize_group_precision_genes",
    "ACTIVATION_PRECISION_STATES",
    "SEARCH_PRECISION_STATES",
    "RealizedTransformerPrecision",
    "TransformerPrecisionUnit",
    "assert_transformer_precision_realized",
    "build_transformer_precision_units",
    "build_transformer_quantization_groups",
    "expected_softmax_realization",
    "validate_external_precision_profile",
    "SMOOTHQUANT_ALPHA_GRID",
    "SmoothQuantRecord",
    "SmoothQuantRegistry",
    "make_smoothquant_record",
    "select_smoothquant_alpha",
    "smoothquant_scale",
    "smoothquant_transform",
]
