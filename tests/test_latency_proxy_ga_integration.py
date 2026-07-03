from __future__ import annotations

import pytest

from opencood.tools.compression.latency_lut.latency_proxy import LatencyProxy
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase
from tests.test_latency_proxy import _record


def test_ga_latency_proxy_does_not_fallback_int8_to_fp16():
    db = LatencyLUTDatabase(default_latency_ms=1.0, default_uncertainty_ms=0.5)
    db.add_record(_record("stage1", "FP16", 0.25))
    proxy = LatencyProxy(db, kappa=1.0)

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
                }
            ],
        }
    )
    payload = estimate.to_dict()

    assert payload["unit_items"][0]["match_type"] == "unavailable"
    assert payload["unit_items"][0]["note"] == "int8_qdq_lut_record_missing"
    assert payload["unit_items"][0]["matched_key_hash"] is None
    assert payload["unit_items"][0]["latency_ms"] != 0.25
    assert payload["unavailable_keys"]
    assert payload["unsupported_precision_regions"] == []
    assert payload["precision_resolution_changes"] == []


def test_ga_latency_proxy_strict_mode_rejects_unavailable_int8():
    proxy = LatencyProxy(LatencyLUTDatabase(), strict_latency=True)

    with pytest.raises(RuntimeError):
        proxy.estimate(
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
                    }
                ],
            }
        )
