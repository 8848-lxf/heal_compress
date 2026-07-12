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
