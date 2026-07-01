from __future__ import annotations

from opencood.tools.compression.latency_lut.interpolation import conservative_channel_distance
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase
from opencood.tools.compression.latency_lut.schema import LatencyLUTKey, LatencyRecord


def _key(c_in: int, c_out: int) -> LatencyLUTKey:
    return LatencyLUTKey(
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
        module_name="backbone",
        block_name="stage",
        block_type="conv_block",
        H=100,
        W=352,
        C_in=c_in,
        C_out=c_out,
        kernel_size=3,
        stride=1,
        padding=1,
        batch_size=1,
        precision_profile="TRT_FP16",
        weight_precision="FP16",
        activation_precision="FP16",
        compute_precision="FP16",
    )


def _record(c_in: int, c_out: int, latency: float) -> LatencyRecord:
    return LatencyRecord(
        key=_key(c_in, c_out),
        latency_p50_ms=latency,
        latency_p90_ms=latency,
        latency_p95_ms=latency,
        latency_p99_ms=latency,
        latency_mean_ms=latency,
        latency_std_ms=0.01,
        num_warmup=1,
        num_repeat=1,
        timing_method="synthetic",
    )


def test_exact_lookup():
    db = LatencyLUTDatabase()
    rec = _record(64, 64, 1.0)
    db.add_record(rec)

    item = db.lookup_exact(_key(64, 64))
    assert item is not None
    assert item.match_type == "exact"
    assert item.latency_ms == 1.0


def test_nearest_lookup_uses_conservative_upward_channels():
    db = LatencyLUTDatabase()
    db.add_record(_record(64, 64, 1.0))
    db.add_record(_record(96, 96, 1.5))

    item = db.lookup_nearest(_key(80, 80))
    assert item.match_type == "nearest"
    assert item.latency_ms == 1.5
    assert item.uncertainty_ms > 0.0


def test_interpolate_between_channel_samples():
    db = LatencyLUTDatabase()
    db.add_record(_record(64, 64, 1.0))
    db.add_record(_record(128, 128, 2.0))

    item = db.interpolate(_key(96, 96))
    assert item.match_type == "interpolate"
    assert 1.45 <= item.latency_ms <= 1.55


def test_conservative_distance_prefers_upward_match():
    assert conservative_channel_distance((80, 80), (96, 96)) < conservative_channel_distance((80, 80), (64, 64))
