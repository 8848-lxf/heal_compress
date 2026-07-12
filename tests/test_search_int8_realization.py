from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_forced_int8_candidate_selects_groups_by_macs() -> None:
    from search.candidate import CandidateGenotype
    from search.quantization_space.types import QuantizationSearchGroup
    from search.quantization_space.codec import forced_int8_group_genotype

    groups = [
        QuantizationSearchGroup("pg_a", ("a",), ("node_a",), ("FP32", "FP16", "INT8"), False, "", 0, 1, 70.0, {}),
        QuantizationSearchGroup("pg_b", ("b",), ("node_b",), ("FP32", "FP16"), False, "", 1, 1, 30.0, {}),
    ]

    genotype = forced_int8_group_genotype(groups, minimum_int8_macs_ratio=0.10, default_precision="FP16")

    assert isinstance(genotype, CandidateGenotype)
    assert genotype.precision_genes["pg_a"] == "INT8"
    assert genotype.precision_genes["pg_b"] == "FP16"


def test_qdq_report_counts_int8_nodes() -> None:
    from search.stage2.mixed_precision_export import summarize_qdq_realization

    report = summarize_qdq_realization(
        requested_group_profile={"pg_a": "INT8"},
        realized_group_profile={"pg_a": "INT8"},
        realized_canonical_profile={"a": "INT8", "b": "FP16"},
        qdq_report={"inserted_layer_count": 1, "records": [{"activation_quantize_node": "q", "activation_dequantize_node": "dq"}]},
        int8_macs_ratio=0.25,
    )

    assert report["realized_int8_group_count"] == 1
    assert report["realized_int8_layer_count"] == 1
    assert report["QuantizeLinear_count"] > 0
    assert report["DequantizeLinear_count"] > 0
    assert report["realized_int8_macs_ratio"] == 0.25


def test_evaluation_worker_expands_runner_profile_to_latency_breakdown() -> None:
    import pytest

    from search.integration.evaluation_worker import _latency_row_from_profile

    row = _latency_row_from_profile(
        frame_id=7,
        warmup=False,
        input_prepare_ms=1.25,
        host_to_device_ms=0.75,
        profile={
            "total_runner_ms": 4.5,
            "execute_async_ms": 2.0,
            "synchronize_ms": 0.5,
            "set_input_shape_ms": 0.1,
            "bind_address_ms": 0.2,
            "output_shape_query_ms": 0.3,
            "h2d_copy_ms": 0.4,
            "input_device_copy_ms": 0.6,
            "dtype_cast_ms": 0.0,
            "contiguous_ms": 0.0,
            "input_buffer_reallocated": False,
            "output_buffer_reallocated": False,
        },
        postprocess_ms=3.0,
    )

    assert row["frame_id"] == 7
    assert row["forward_ms"] == 4.5
    assert row["input_prepare_ms"] == 1.25
    assert row["host_to_device_ms"] == 0.75
    assert row["shape_binding_ms"] == pytest.approx(0.6)
    assert row["buffer_allocation_ms"] == 0.0
    assert row["execute_async_ms"] == 2.0
    assert row["device_sync_ms"] == 0.5
    assert row["postprocess_ms"] == 3.0
    assert row["total_ms"] == 7.5


def test_evaluation_worker_reports_latency_distribution_tail_and_cv() -> None:
    import pytest

    from search.integration.evaluation_worker import _latency_distribution

    stats = _latency_distribution([1.0, 2.0, 3.0, 100.0], prefix="forward")

    assert stats["forward_mean_ms"] == pytest.approx(26.5)
    assert stats["forward_p50_ms"] == pytest.approx(2.5)
    assert stats["forward_p99_ms"] == pytest.approx(97.09)
    assert stats["forward_std_ms"] > 0
    assert stats["forward_cv"] > 0
    assert stats["forward_min_ms"] == 1.0
    assert stats["forward_max_ms"] == 100.0
    assert stats["forward_outlier_count"] == 1
