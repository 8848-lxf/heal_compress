from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _shape(module: str, macs: float, *, c_in: int, c_out: int):
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape

    return RuntimeLayerShape(
        module_path=module,
        call_index=0,
        module_type="Conv2d",
        input_shape=(1, c_in, 1, 1),
        output_shape=(1, c_out, 1, 1),
        c_in=c_in,
        c_out=c_out,
        h_out=1,
        w_out=1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        weight_shape=(c_out, c_in, 1, 1),
        precision_group_id=f"quant::{module}",
        macs=macs,
    )


def _snapshot(rows, parameter_count: int):
    return {
        "snapshot_schema_version": "physical-structure-snapshot-v2",
        "parameter_count": parameter_count,
        "modules": rows,
    }


def test_engine_realized_precision_profile_uses_canonical_inspector_rows() -> None:
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult
    from search.stage2.realized_bops import engine_realized_precision_profile

    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="conv_a",
                canonical_node_name="canonical_conv_a",
                weight_initializer="conv_a.weight",
                precision_group="quant::conv_a",
                requested_precision="int8",
                realized_request_precision="int8",
            ),
            CanonicalPrecisionEntry(
                module_path="conv_b",
                canonical_node_name="canonical_conv_b",
                weight_initializer="conv_b.weight",
                precision_group="quant::conv_b",
                requested_precision="int8",
                realized_request_precision="int8",
            ),
        ]
    )
    layer_info = [
        {"Name": "canonical_conv_a", "LayerType": "Convolution", "Precision": "Int8"},
        {"Name": "canonical_conv_b", "LayerType": "Convolution", "Precision": "Half"},
    ]

    report = engine_realized_precision_profile(layer_info, mapping)

    assert report["passed"] is True
    assert report["realized_precision_profile"] == {"conv_a": "INT8", "conv_b": "FP16"}
    assert report["precision_counts"] == {"FP32": 0, "FP16": 1, "INT8": 1}


def test_realized_bops_uses_physical_macs_engine_precision_and_fp32_reference() -> None:
    from search.stage2.realized_bops import compute_realized_bops

    physical_shapes = [
        _shape("conv_a", 50.0, c_in=5, c_out=10),
        _shape("conv_b", 100.0, c_in=10, c_out=10),
    ]
    baseline_shapes = [
        _shape("conv_a", 100.0, c_in=10, c_out=10),
        _shape("conv_b", 100.0, c_in=10, c_out=10),
    ]
    physical_snapshot = _snapshot(
        [
            {"canonical_module_name": "conv_a", "module_type": "Conv2d", "in_channels": 5, "out_channels": 10, "groups": 1, "parameter_count": 60, "weight_shape": [10, 5, 1, 1]},
            {"canonical_module_name": "conv_b", "module_type": "Conv2d", "in_channels": 10, "out_channels": 10, "groups": 1, "parameter_count": 110, "weight_shape": [10, 10, 1, 1]},
        ],
        parameter_count=170,
    )
    baseline_snapshot = _snapshot(
        [
            {"canonical_module_name": "conv_a", "module_type": "Conv2d", "in_channels": 10, "out_channels": 10, "groups": 1, "parameter_count": 110, "weight_shape": [10, 10, 1, 1]},
            {"canonical_module_name": "conv_b", "module_type": "Conv2d", "in_channels": 10, "out_channels": 10, "groups": 1, "parameter_count": 110, "weight_shape": [10, 10, 1, 1]},
        ],
        parameter_count=220,
    )

    report = compute_realized_bops(
        physical_runtime_shapes=physical_shapes,
        baseline_runtime_shapes=baseline_shapes,
        realized_precision_profile={"conv_a": "INT8", "conv_b": "FP16"},
        physical_snapshot=physical_snapshot,
        baseline_snapshot=baseline_snapshot,
        target_retention=0.14,
        tolerance=0.005,
    )

    assert report["passed"] is True
    assert report["realized_bops"] == pytest.approx(50 * 8 * 8 + 100 * 16 * 16)
    assert report["fp32_reference_bops"] == pytest.approx(200 * 32 * 32)
    assert report["bops_retention"] == pytest.approx(0.140625)
    assert report["physical_params"] == 170
    assert report["parameter_retention"] == pytest.approx(170 / 220)
    assert report["weight_storage_retention"] == pytest.approx((60 * 8 + 110 * 16) / (220 * 32))
    assert report["breakdown"][0]["C_in"] == 5
    assert report["breakdown"][0]["weight_bits"] == 8
    assert report["breakdown"][0]["activation_bits"] == 8


def test_realized_bops_fails_closed_outside_budget() -> None:
    from search.stage2.realized_bops import compute_realized_bops

    snapshot = _snapshot(
        [{"canonical_module_name": "conv", "module_type": "Conv2d", "in_channels": 10, "out_channels": 10, "groups": 1, "parameter_count": 100, "weight_shape": [10, 10, 1, 1]}],
        parameter_count=100,
    )
    report = compute_realized_bops(
        physical_runtime_shapes=[_shape("conv", 100.0, c_in=10, c_out=10)],
        baseline_runtime_shapes=[_shape("conv", 100.0, c_in=10, c_out=10)],
        realized_precision_profile={"conv": "FP16"},
        physical_snapshot=snapshot,
        baseline_snapshot=snapshot,
        target_retention=0.21,
        tolerance=0.005,
    )

    assert report["passed"] is False
    assert report["status"] == "realized_BOPS_out_of_budget"
    assert report["bops_retention"] == pytest.approx(0.25)
