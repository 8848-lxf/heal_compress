import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut import benchmark_cudnn_shape_rulebook_v103 as rb


def test_ordinary_shape_grid_contains_small_nonfriendly_and_friendly():
    rows = rb.generate_ordinary_shape_grid(reduced=True)
    pairs = {(row["C_in"], row["C_out"]) for row in rows}
    channels = {c for pair in pairs for c in pair}

    assert {2, 4}.issubset(channels)
    assert {6, 10}.intersection(channels)
    assert {8, 16, 32, 64}.issubset(channels)
    assert any(row["C_in"] == row["C_out"] for row in rows)
    assert any(row["C_in"] == 16 and row["C_out"] != 16 for row in rows)


def test_grouped_shape_grid_contains_groups_and_per_group_sweep():
    rows = rb.generate_grouped_shape_grid(reduced=True)
    groups = {row["groups"] for row in rows}
    widths = {row["in_per_group"] for row in rows} | {row["out_per_group"] for row in rows}

    assert {2, 4, 8, 16}.issubset(groups)
    assert {2, 4, 6, 8, 16, 32}.issubset(widths)
    assert any(row["C_in"] == row["groups"] * row["in_per_group"] for row in rows)


def test_conv2d_flops_estimate_for_grouped_and_ordinary():
    ordinary = rb.conv2d_flops(batch=1, h=10, w=10, c_in=16, c_out=32, kernel=3, groups=1)
    grouped = rb.conv2d_flops(batch=1, h=10, w=10, c_in=16, c_out=32, kernel=3, groups=4)

    assert ordinary == 1 * 10 * 10 * 32 * 16 * 3 * 3 * 2
    assert grouped == ordinary / 4


def test_shape_classification_rules():
    assert "too_narrow_2_4" in rb.classify_ordinary_shape(4, 4)
    assert "multiple_of_16" in rb.classify_ordinary_shape(16, 16)
    assert "non_multiple_of_8" in rb.classify_ordinary_shape(10, 10)
    assert "per_group_too_narrow_2_4" in rb.classify_grouped_shape(groups=8, in_per_group=4, out_per_group=4)
    assert "per_group_multiple_of_16" in rb.classify_grouped_shape(groups=8, in_per_group=16, out_per_group=16)


def test_benchmark_result_schema_for_dry_row():
    config = {
        "op_type": "ordinary_conv2d",
        "dtype": "fp32",
        "layout": "nchw",
        "batch": 1,
        "H": 10,
        "W": 10,
        "kernel_size": "3x3",
        "stride": 1,
        "padding": 1,
        "C_in": 16,
        "C_out": 16,
        "groups": 1,
        "in_per_group": 16,
        "out_per_group": 16,
        "shape_class": "multiple_of_16",
    }
    row = rb.build_result_row_from_stats(config, {"latency_mean_ms": 1.0, "latency_p50_ms": 1.0, "latency_p90_ms": 1.0, "latency_p95_ms": 1.0, "latency_min_ms": 1.0, "latency_max_ms": 1.0}, cudnn_benchmark=True)

    for key in rb.RESULT_FIELDS:
        assert key in row
    assert row["params"] == 16 * 16 * 3 * 3
    assert row["FLOPs_estimate"] == rb.conv2d_flops(batch=1, h=10, w=10, c_in=16, c_out=16, kernel=3, groups=1)


def test_real_model_shape_extractor_warns_on_missing_inputs(tmp_path):
    shapes, warnings = rb.extract_real_model_shapes([tmp_path / "missing_dir"], max_shapes=10)

    assert shapes == []
    assert warnings
    assert "missing" in warnings[0].lower()
