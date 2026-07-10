import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut import benchmark_conv_shape_efficiency_v102 as bench


def test_relative_speed_to_nearest_friendly_shape_is_populated():
    rows = [
        {
            "op_type": "conv2d",
            "H": 50,
            "W": 176,
            "kernel": "3x3",
            "C_in": 6,
            "C_out": 6,
            "groups": 1,
            "in_per_group": 6,
            "out_per_group": 6,
            "cudnn_benchmark": True,
            "throughput_GFLOPs_per_s": 10.0,
            "latency_per_GFLOP": 0.1,
            "speed_relative_to_nearest_friendly_shape": "",
        },
        {
            "op_type": "conv2d",
            "H": 50,
            "W": 176,
            "kernel": "3x3",
            "C_in": 8,
            "C_out": 8,
            "groups": 1,
            "in_per_group": 8,
            "out_per_group": 8,
            "cudnn_benchmark": True,
            "throughput_GFLOPs_per_s": 20.0,
            "latency_per_GFLOP": 0.05,
            "speed_relative_to_nearest_friendly_shape": "",
        },
    ]

    bench._annotate_relative_speed_to_friendly(rows)

    assert rows[0]["nearest_friendly_shape"] == "C_in=8,C_out=8,groups=1"
    assert rows[0]["speed_relative_to_nearest_friendly_shape"] == 0.5
    assert rows[1]["speed_relative_to_nearest_friendly_shape"] == 1.0


def test_shape_efficiency_summary_answers_core_questions(tmp_path):
    ordinary_rows = [
        {"op_type": "conv2d", "C_in": 2, "C_out": 2, "groups": 1, "in_per_group": 2, "out_per_group": 2, "latency_per_GFLOP": 0.4},
        {"op_type": "conv2d", "C_in": 8, "C_out": 8, "groups": 1, "in_per_group": 8, "out_per_group": 8, "latency_per_GFLOP": 0.1},
        {"op_type": "conv2d", "C_in": 10, "C_out": 10, "groups": 1, "in_per_group": 10, "out_per_group": 10, "latency_per_GFLOP": 0.3},
    ]
    grouped_rows = [
        {"op_type": "grouped_conv2d", "C_in": 16, "C_out": 16, "groups": 8, "in_per_group": 2, "out_per_group": 2, "latency_per_GFLOP": 0.6},
        {"op_type": "grouped_conv2d", "C_in": 64, "C_out": 64, "groups": 8, "in_per_group": 8, "out_per_group": 8, "latency_per_GFLOP": 0.2},
        {"op_type": "grouped_conv2d", "C_in": 80, "C_out": 80, "groups": 8, "in_per_group": 10, "out_per_group": 10, "latency_per_GFLOP": 0.5},
    ]

    out = tmp_path / "summary.md"
    bench._write_summary(out, ordinary_rows, grouped_rows, real_rows=[])

    text = out.read_text()
    assert "C_out=2/4" in text
    assert "Grouped per-group width 2/4" in text
    assert "Next alignment rule recommendation" in text
