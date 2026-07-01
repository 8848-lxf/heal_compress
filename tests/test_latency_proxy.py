from __future__ import annotations

from opencood.tools.compression.latency_lut.calibration import IdentityCalibrationModel
from opencood.tools.compression.latency_lut.latency_proxy import LatencyProxy
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase
from opencood.tools.compression.latency_lut.schema import LatencyLUTKey, LatencyRecord


def _record(unit: str, precision: str, latency: float) -> LatencyRecord:
    profile = {"FP32": "TRT_FP32", "FP16": "TRT_FP16", "INT8": "TRT_INT8_QDQ"}[precision]
    return LatencyRecord(
        key=LatencyLUTKey(
            deploy_mode="single_engine_maxK",
            fixed_K=29696,
            module_name="backbone",
            block_name=unit,
            block_type="conv_block",
            H=100,
            W=352,
            C_in=64,
            C_out=64,
            kernel_size=3,
            stride=1,
            padding=1,
            batch_size=1,
            precision_profile=profile,
            weight_precision=precision,
            activation_precision="FP16",
            compute_precision=precision,
        ),
        latency_p50_ms=latency,
        latency_p90_ms=latency,
        latency_p95_ms=latency,
        latency_p99_ms=latency,
        latency_mean_ms=latency,
        latency_std_ms=0.02,
        num_warmup=1,
        num_repeat=1,
        timing_method="synthetic",
    )


def test_latency_proxy_estimates_preparsed_candidate_with_boundary_penalty():
    db = LatencyLUTDatabase(default_boundary_penalty_ms=0.05)
    db.add_record(_record("stage1", "FP16", 1.0))
    db.add_record(_record("stage2", "FP32", 2.0))
    proxy = LatencyProxy(db, calibration_model=IdentityCalibrationModel(), kappa=1.0)

    estimate = proxy.estimate(
        {
            "deploy_mode": "single_engine_maxK",
            "fixed_K": 29696,
            "units": [
                {
                    "unit_id": "stage1",
                    "module_name": "backbone",
                    "block_name": "stage1",
                    "block_type": "conv_block",
                    "H": 100,
                    "W": 352,
                    "C_in": 64,
                    "C_out": 64,
                    "precision": "FP16",
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
                {
                    "unit_id": "stage2",
                    "module_name": "backbone",
                    "block_name": "stage2",
                    "block_type": "conv_block",
                    "H": 100,
                    "W": 352,
                    "C_in": 64,
                    "C_out": 64,
                    "precision": "FP32",
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ],
        }
    )

    assert estimate.latency_lut_raw_ms == 3.05
    assert estimate.latency_calibrated_ms == 3.05
    assert estimate.latency_ms > estimate.latency_calibrated_ms
    assert len(estimate.unit_items) == 2
    assert len(estimate.boundary_items) == 1
    assert estimate.missing_keys == []


def test_latency_proxy_rejects_dynamic_bucket_mode():
    proxy = LatencyProxy(LatencyLUTDatabase())
    try:
        proxy.estimate({"deploy_mode": "dynamic_bucket", "fixed_K": 29696, "units": []})
    except ValueError as exc:
        assert "single_engine_maxK" in str(exc)
    else:
        raise AssertionError("dynamic bucket mode must be rejected")


def test_latency_proxy_reports_unavailable_and_plugin_items():
    proxy = LatencyProxy(LatencyLUTDatabase(), kappa=1.0)

    estimate = proxy.estimate(
        {
            "deploy_mode": "single_engine_maxK",
            "fixed_K": 29696,
            "units": [
                {
                    "unit_id": "stage1",
                    "module_name": "backbone",
                    "block_name": "stage1",
                    "block_type": "conv_block",
                    "H": 100,
                    "W": 352,
                    "C_in": 64,
                    "C_out": 64,
                    "precision": "INT8",
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
                {
                    "unit_id": "scatter",
                    "module_name": "scatter",
                    "block_name": "scatter",
                    "block_type": "plugin",
                    "H": 200,
                    "W": 704,
                    "C_in": 64,
                    "C_out": 64,
                    "precision": "FP16",
                    "plugin_name": "PointPillarScatterTRT",
                },
            ],
        }
    )
    payload = estimate.to_dict()

    assert payload["T_lut_raw"] == payload["latency_lut_raw_ms"]
    assert payload["T_proxy"] == payload["latency_ms"]
    assert len(payload["plugin_items"]) == 1
    assert len(payload["unavailable_keys"]) >= 1
    assert payload["calibration_model"] == "identity"
