from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_strict_fp32_validation_rejects_weighted_fp16_or_int8_layers() -> None:
    from search.baselines.original_engines import validate_baseline_layer_precisions

    report = validate_baseline_layer_precisions(
        "strict_fp32",
        [
            {"Name": "conv0", "LayerType": "Convolution", "Precision": "FP32"},
            {"Name": "conv1", "LayerType": "Convolution", "Precision": "FP16"},
            {"Name": "shape", "LayerType": "Shape", "Precision": "INT32"},
        ],
    )

    assert report["passed"] is False
    assert report["weighted_fp16_count"] == 1
    assert report["weighted_int8_count"] == 0
    assert report["status"] == "strict_fp32_failed"


def test_strict_fp32_allows_mapped_protected_functional_fp16_exception() -> None:
    from search.baselines.original_engines import validate_baseline_layer_precisions

    report = validate_baseline_layer_precisions(
        "strict_fp32",
        [
            {"Name": "conv0", "LayerType": "Convolution", "Precision": "FP32"},
            {"Name": "protected_affine_bmm", "LayerType": "gemm", "Precision": "FP16"},
        ],
        canonical_precision_realization={
            "passed": True,
            "realized_int8_count": 0,
            "realized_fp16_count": 1,
            "unresolved_layer_count": 0,
        },
    )

    assert report["passed"] is True
    assert report["protected_functional_fp16_count"] == 1
    assert report["weighted_fp32_count"] == 1
    assert report["weighted_fp16_count"] == 1


def test_strict_fp16_validation_rejects_int8_and_weighted_fp32_fallback() -> None:
    from search.baselines.original_engines import validate_baseline_layer_precisions

    report = validate_baseline_layer_precisions(
        "strict_fp16",
        [
            {"Name": "conv0", "LayerType": "Convolution", "Precision": "FP16"},
            {"Name": "conv1", "LayerType": "Convolution", "Precision": "FP32"},
            {"Name": "conv2", "LayerType": "Convolution", "Precision": "INT8"},
        ],
    )

    assert report["passed"] is False
    assert report["weighted_fp32_count"] == 1
    assert report["weighted_int8_count"] == 1
    assert report["status"] == "strict_fp16_failed"


def test_maximal_legal_int8_validation_requires_realized_int8() -> None:
    from search.baselines.original_engines import validate_baseline_layer_precisions

    report = validate_baseline_layer_precisions(
        "maximal_legal_int8",
        [
            {"Name": "conv0", "LayerType": "Convolution", "Precision": "FP16"},
            {"Name": "conv1", "LayerType": "Convolution", "Precision": "INT8"},
        ],
    )

    assert report["passed"] is True
    assert report["weighted_int8_count"] == 1
    assert report["status"] == "maximal_legal_int8"


def test_matched_legacy_uses_canonical_70_layer_coverage_after_engine_fusion() -> None:
    from search.baselines.original_engines import validate_baseline_layer_precisions

    report = validate_baseline_layer_precisions(
        "matched_legacy_int8",
        [
            {"Name": f"conv{index}", "LayerType": "Convolution", "Precision": "INT8"}
            for index in range(65)
        ]
        + [
            {"Name": f"fp16{index}", "LayerType": "Convolution", "Precision": "FP16"}
            for index in range(3)
        ],
        canonical_precision_realization={
            "passed": True,
            "realized_int8_count": 67,
            "realized_fp16_count": 3,
            "unresolved_layer_count": 0,
        },
    )

    assert report["passed"] is True
    assert report["weighted_layer_count"] == 70
    assert report["weighted_int8_count"] == 67
    assert report["weighted_fp16_count"] == 3
    assert report["coverage_counting_basis"] == "canonical_precision_realization"
    assert report["raw_engine_weighted_summary"]["weighted_int8_count"] == 65


def test_matched_legacy_rejects_failed_canonical_realization() -> None:
    from search.baselines.original_engines import validate_baseline_layer_precisions

    report = validate_baseline_layer_precisions(
        "matched_legacy_int8",
        [],
        canonical_precision_realization={
            "passed": False,
            "realized_int8_count": 67,
            "realized_fp16_count": 3,
            "unresolved_layer_count": 0,
        },
    )

    assert report["passed"] is False
    assert "matched_legacy_canonical_precision_realization_failed" in report["issues"]


def test_trusted_explicit_qdq_uses_canonical_70_layer_coverage_after_engine_fusion() -> None:
    from search.baselines.original_engines import (
        TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES,
        validate_baseline_layer_precisions,
    )

    expected_int8 = len(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)
    report = validate_baseline_layer_precisions(
        "trusted_explicit_qdq_int8",
        [
            {"Name": f"conv{index}", "LayerType": "Convolution", "Precision": "INT8"}
            for index in range(expected_int8 - 2)
        ]
        + [
            {"Name": f"fp16{index}", "LayerType": "Convolution", "Precision": "FP16"}
            for index in range(43)
        ],
        canonical_precision_realization={
            "passed": True,
            "realized_int8_count": expected_int8,
            "realized_fp16_count": 43,
            "unresolved_layer_count": 0,
        },
    )

    assert report["passed"] is True
    assert report["weighted_layer_count"] == 70
    assert report["weighted_int8_count"] == expected_int8
    assert report["weighted_fp16_count"] == 43
    assert report["coverage_counting_basis"] == "canonical_precision_realization"
    assert report["raw_engine_weighted_summary"]["weighted_int8_count"] == expected_int8 - 2
