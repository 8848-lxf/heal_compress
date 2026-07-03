from __future__ import annotations

from opencood.tools.compression.latency_lut.latency_proxy import LatencyProxy
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase
from tests.test_latency_proxy import _record


def test_latency_proxy_outputs_interpretable_decomposition():
    db = LatencyLUTDatabase(default_boundary_penalty_ms=0.05)
    db.add_record(_record("stage1", "FP16", 1.0))
    db.add_record(_record("stage2", "FP32", 2.0))
    proxy = LatencyProxy(db, kappa=0.0)

    payload = proxy.estimate(
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
    ).to_dict()

    assert payload["T_lut_raw"] == 3.05
    assert payload["T_compute_covered"] == 3.0
    assert payload["T_boundary_cast"] == 0.05
    assert payload["T_boundary_qdq"] == 0.0
    assert payload["T_lut_by_precision"]["FP16"] == 1.0
    assert payload["T_lut_by_precision"]["FP32"] == 2.0
    assert payload["covered_unit_count"] == 2
    assert payload["missing_unit_count"] == 0
    assert payload["coverage_ratio_by_units"] == 1.0
    assert payload["exact_key_count"] == 2
    assert payload["coarse_key_count"] == 1
    assert payload["source_by_component"]["T_compute_covered"] == "measured_lut"
    assert payload["calibrator_validated"] is False
    assert payload["ga_integration_allowed"] is False
