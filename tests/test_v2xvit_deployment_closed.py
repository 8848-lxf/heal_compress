from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from search.model_family.deployment import (
    _sanitize_weight_channel_amax,
    _validate_calibration_observation_count,
)
from search.stage2.v2xvit_deployment_closed import (
    audit_deployment_closed_profile,
    bind_train200_calibration_identity,
    build_deployment_closed_precision_mapping,
    resolve_origin_module_precision,
)


def _origin():
    return SimpleNamespace(
        origin_map_hash="origin",
        entries=(
            SimpleNamespace(
                module_path="fusion.q_linears.0",
                canonical_node_name="q",
                original_node_name="q0",
                weight_initializer="qw",
                onnx_op_type="MatMul",
                call_index=0,
            ),
            SimpleNamespace(
                module_path="fusion.k_linears.0",
                canonical_node_name="k",
                original_node_name="k0",
                weight_initializer="kw",
                onnx_op_type="MatMul",
                call_index=1,
            ),
        ),
        functional_compute_groups=(),
    )


def _inventory():
    return {
        "rows": [
            {"module_path": "fusion.q_linears.0", "onnx_node": "q", "canonical_role": "q_projection"},
            {"module_path": "fusion.k_linears.0", "onnx_node": "k", "canonical_role": "k_projection"},
            {"module_path": "", "onnx_node": "qk", "canonical_role": "qk_matmul"},
            {"module_path": "", "onnx_node": "softmax", "canonical_role": "softmax"},
        ]
    }


def _nodes():
    return {
        name: SimpleNamespace(input=[f"{name}_in"], output=[f"{name}_out"])
        for name in ("q", "k", "qk", "softmax")
    }


def test_qk_projection_int8_recovers_fp32_and_qk_stays_fp32():
    mapping, requested = build_deployment_closed_precision_mapping(
        origin=_origin(),
        inventory=_inventory(),
        graph_nodes=_nodes(),
        module_precision_profile={"fusion.q_linears.0": "INT8", "fusion.k_linears.0": "INT8"},
        profile_id="test",
    )
    assert {row.realized_request_precision for row in mapping.entries} == {"int8"}
    assert {row.realized_output_precision for row in mapping.entries} == {"fp32"}
    assert mapping.auxiliary_layer_precisions["qk"] == "fp32"
    assert mapping.auxiliary_layer_output_types["qk"] == "fp32"
    assert next(row for row in requested if row["onnx_node"] == "qk")["requested_accumulator"] == "FP32"


def test_softmax_contract_is_explicit_and_not_native_int8_qk():
    mapping, _ = build_deployment_closed_precision_mapping(
        origin=_origin(), inventory=_inventory(), graph_nodes=_nodes(),
        module_precision_profile={"fusion.q_linears.0": "FP16", "fusion.k_linears.0": "FP16"},
        profile_id="test",
    )
    assert mapping.auxiliary_layer_precisions["softmax"] == "fp16"
    assert mapping.auxiliary_layer_precisions["qk"] == "fp32"


def test_missing_or_unknown_precision_locus_fails_closed():
    with pytest.raises(RuntimeError, match="unresolved"):
        resolve_origin_module_precision(_origin(), {"fusion.q_linears.0": "FP16"})
    with pytest.raises(RuntimeError, match="not_exported"):
        resolve_origin_module_precision(
            _origin(),
            {
                "fusion.q_linears.0": "FP16",
                "fusion.k_linears.0": "FP16",
                "missing": "FP16",
            },
        )


def test_train200_identity_binds_all_cache_invalidators(tmp_path: Path):
    manifest = tmp_path / "train200.json"
    checkpoint = tmp_path / "checkpoint.pth"
    onnx = tmp_path / "model.onnx"
    manifest.write_text("manifest")
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"onnx")
    metadata = {
        "frame_count": 200,
        "manifest_hash": "manifest-hash",
        "sample_evidence": [{"ordinal": index} for index in range(200)],
    }
    identity = bind_train200_calibration_identity(
        metadata=metadata,
        scales={"a": {"activation_input_scale": 0.1}},
        train200_manifest=manifest,
        checkpoint=checkpoint,
        physical_structure_hash="physical",
        state_dict_shape_hash="shape",
        precision_map_hash="precision",
        onnx_path=onnx,
        calibration_config={"algorithm": "entropy"},
    )
    assert identity["processed_frames"] == 200
    assert identity["skipped_frames"] == 0
    assert identity["physical_hash"] == "physical"
    assert identity["state_dict_shape_hash"] == "shape"
    assert identity["precision_map_hash"] == "precision"
    assert identity["onnx_hash"]
    assert identity["cache_hash"]


def test_requested_realized_mismatch_fails_closed():
    requested = [{"onnx_node": "q", "requested_precision": "INT8"}]
    realized = [{"onnx_node": "q", "realized_precision": "FP16", "conflict": True}]
    audit = audit_deployment_closed_profile(requested_rows=requested, realized_rows=realized)
    assert not audit["requested_realized_exact"]
    assert audit["silent_fallback"]


def test_exact_zero_weight_channel_gets_positive_exact_zero_scale_floor():
    safe, floor_indices, zero_indices = _sanitize_weight_channel_amax(
        np.asarray([0.25, 0.0, 3.0e-45, 0.5], dtype=np.float32),
        module_path="conv",
    )
    assert zero_indices.tolist() == [1]
    assert floor_indices.tolist() == [1, 2]
    assert safe.tolist() == pytest.approx([0.25, 1.0e-8, 1.0e-8, 0.5])
    assert np.all(safe > 0.0)


def test_nonfinite_weight_channel_still_fails_closed():
    with pytest.raises(RuntimeError, match="v2xvit_entropy_weight_channel_invalid"):
        _sanitize_weight_channel_amax(
            np.asarray([0.25, np.nan], dtype=np.float32), module_path="conv"
        )


def test_reused_module_calibration_counts_processed_frames_not_calls():
    assert _validate_calibration_observation_count(
        "attention", input_count=400, output_count=400, frame_count=200
    ) == 2
    with pytest.raises(RuntimeError, match="v2xvit_entropy_observation_count"):
        _validate_calibration_observation_count(
            "attention", input_count=400, output_count=399, frame_count=200
        )
