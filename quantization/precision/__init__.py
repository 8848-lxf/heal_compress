"""Formal precision profile, canonical mapping, and Q/DQ APIs."""

from .canonical_mapping import build_canonical_precision_mapping
from .activation_boundary import resolve_activation_output_boundary
from .merge_contract import apply_fp16_merge_output_contract
from .calibration import collect_calibration_scales, validate_calibration_scales
from .profile import generate_precision_profile
from .qdq_inserter import insert_explicit_qdq
from .qdq_trace import trace_qdq_root_initializer
from .validation import validate_qdq_against_physical_snapshot

__all__ = [
    "build_canonical_precision_mapping",
    "resolve_activation_output_boundary",
    "apply_fp16_merge_output_contract",
    "collect_calibration_scales",
    "generate_precision_profile",
    "insert_explicit_qdq",
    "trace_qdq_root_initializer",
    "validate_calibration_scales",
    "validate_qdq_against_physical_snapshot",
]
