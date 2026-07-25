from __future__ import annotations

import pytest
import numpy as np

from search.calibration.v2xvit_train200 import (
    build_train200_contract,
    calibration_reusable,
)
from search.model_family.deployment import (
    _positive_per_channel_amax,
    _validate_calibration_observation_count,
)


def _contract(**changes):
    values = {
        "manifest_hash": "manifest",
        "checkpoint_hash": "checkpoint",
        "physical_hash": "physical",
        "state_dict_shape_hash": "shapes",
        "precision_map_hash": "precision",
        "onnx_hash": "onnx",
        "calibration_algorithm_config_hash": "algorithm",
        "cache_hash": "cache",
        "scale_hash": "scales",
        "processed_frames": 200,
        "skipped_frames": 0,
    }
    values.update(changes)
    return build_train200_contract(**values)


def test_train200_contract_binds_every_structure_precision_and_graph_hash() -> None:
    reference = _contract()
    for key in (
        "manifest_hash", "checkpoint_hash", "physical_hash", "state_dict_shape_hash",
        "precision_map_hash", "onnx_hash", "calibration_algorithm_config_hash",
    ):
        changed = _contract(**{key: f"changed-{key}"})
        assert not calibration_reusable(reference, changed)


def test_train200_contract_requires_exactly_200_processed_and_zero_skipped() -> None:
    with pytest.raises(RuntimeError, match="processed_count"):
        _contract(processed_frames=199)
    with pytest.raises(RuntimeError, match="skipped_frames"):
        _contract(skipped_frames=1)


def test_zero_fp16_weight_channel_gets_positive_fp32_qdq_scale() -> None:
    safe, zero, clamped = _positive_per_channel_amax(np.asarray([0.0, 2.0], dtype=np.float16))
    scale = safe / np.float32(127.0)
    assert safe.dtype == np.float32
    assert zero.tolist() == [True, False]
    assert clamped.tolist() == [True, False]
    assert scale[0] > 0.0


def test_positive_fp32_subnormal_cannot_underflow_qdq_scale() -> None:
    safe, zero, clamped = _positive_per_channel_amax(np.asarray([1.0e-44], dtype=np.float32))
    assert zero.tolist() == [False]
    assert clamped.tolist() == [True]
    assert float((safe / np.float32(127.0))[0]) > 0.0


def test_reused_module_keeps_all_deterministic_calls_per_frame() -> None:
    assert _validate_calibration_observation_count(
        "shared", input_count=400, output_count=400, frame_count=200
    ) == 2
    with pytest.raises(RuntimeError, match="v2xvit_entropy_observation_count"):
        _validate_calibration_observation_count(
            "nondeterministic", input_count=399, output_count=399, frame_count=200
        )
