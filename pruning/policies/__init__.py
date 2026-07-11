"""Formal pruning legality and protection policies."""

from .alignment import legalize_dense_keep_count
from .grouped_conv import validate_grouped_conv_shape, validate_remove_groups
from .protection import build_protection_registry, infer_directional_protection, require_direction_allowed
from .registry import PolicyRegistry

__all__ = [
    "PolicyRegistry",
    "build_protection_registry",
    "infer_directional_protection",
    "legalize_dense_keep_count",
    "require_direction_allowed",
    "validate_grouped_conv_shape",
    "validate_remove_groups",
]
