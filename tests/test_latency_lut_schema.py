from __future__ import annotations

from opencood.tools.compression.latency_lut.schema import (
    LatencyLUTKey,
    LatencyRecord,
    precision_to_profile,
)


def test_key_hash_is_stable_and_json_serializable():
    key = LatencyLUTKey(
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
        module_name="backbone",
        block_name="stage1",
        block_type="conv_block",
        H=100,
        W=352,
        C_in=64,
        C_out=64,
        kernel_size=3,
        stride=1,
        padding=1,
        batch_size=1,
        precision_profile="TRT_FP16",
        weight_precision="FP16",
        activation_precision="FP16",
        compute_precision="FP16",
    )

    assert key.stable_hash() == LatencyLUTKey.from_dict(key.to_dict()).stable_hash()
    payload = key.to_dict()
    assert payload["deploy_mode"] == "single_engine_maxK"
    assert "bucket" not in "".join(payload.keys()).lower()


def test_latency_record_roundtrip_contains_required_metrics():
    key = LatencyLUTKey(
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
        module_name="head",
        block_name="cls",
        block_type="head_branch",
        H=100,
        W=352,
        C_in=128,
        C_out=2,
        batch_size=1,
        precision_profile="TRT_FP32",
        weight_precision="FP32",
        activation_precision="high_precision",
        compute_precision="FP32",
    )
    record = LatencyRecord(
        key=key,
        latency_p50_ms=0.12,
        latency_p90_ms=0.15,
        latency_p95_ms=0.16,
        latency_p99_ms=0.18,
        latency_mean_ms=0.13,
        latency_std_ms=0.01,
        num_warmup=10,
        num_repeat=20,
        timing_method="cuda_event",
    )

    restored = LatencyRecord.from_dict(record.to_dict())
    assert restored.key.stable_hash() == key.stable_hash()
    assert restored.latency_p95_ms == 0.16
    assert restored.key_hash == key.stable_hash()


def test_precision_to_profile_mapping_is_explicit():
    assert precision_to_profile("FP32") == "TRT_FP32"
    assert precision_to_profile("FP16") == "TRT_FP16"
    assert precision_to_profile("INT8") == "TRT_INT8_QDQ"
