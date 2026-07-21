"""Shared, fail-closed Transformer quantization contracts."""

from .canonical_roles import (
    CANONICAL_TRANSFORMER_ROLES,
    CanonicalRole,
    classify_weighted_module,
)
from .precision_contract import PrecisionContract, precision_profiles

__all__ = [
    "CANONICAL_TRANSFORMER_ROLES",
    "CanonicalRole",
    "PrecisionContract",
    "classify_weighted_module",
    "precision_profiles",
]
