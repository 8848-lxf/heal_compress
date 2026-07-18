from __future__ import annotations

import pytest


def _request() -> dict:
    return {
        "model_family": "lidar_cobevt",
        "device": "cuda:2",
        "num_workers": 8,
        "fixed_k": 29696,
        "num_frames": 10,
        "diagnostic_latency_invalid": True,
        "reference_engine_path": "/tmp/a0.plan",
        "candidate_engine_path": "/tmp/a1.plan",
        "output_specs": [
            {
                "block_id": "layers.0.window_attention",
                "role": "q_projection",
                "tensor_name": "q",
            }
        ],
    }


def test_attention_parity_worker_requires_fixed_diagnostic_protocol():
    from search.integration.lidar_cobevt_attention_parity_worker import (
        validate_request,
    )

    validated = validate_request(_request())
    assert validated["device"] == "cuda:2"
    assert validated["fixed_k"] == 29696
    assert validated["num_frames"] == 10
    assert validated["diagnostic_latency_invalid"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device", "cpu", "cobevt_parity_cuda_device_required"),
        ("num_workers", 4, "cobevt_parity_num_workers_must_equal_8"),
        ("fixed_k", 10000, "cobevt_parity_fixed_k_29696_required"),
        ("num_frames", 9, "cobevt_parity_smoke10_required"),
        ("diagnostic_latency_invalid", False, "cobevt_parity_diagnostic_only_required"),
        ("candidate_engine_path", "/tmp/a0.plan", "cobevt_parity_engines_must_differ"),
        ("output_specs", [], "cobevt_parity_output_specs_required"),
    ],
)
def test_attention_parity_worker_fails_closed_for_protocol_drift(
    field: str, value: object, message: str
):
    from search.integration.lidar_cobevt_attention_parity_worker import (
        validate_request,
    )

    request = _request()
    request[field] = value
    with pytest.raises(RuntimeError, match=message):
        validate_request(request)
