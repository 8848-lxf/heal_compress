import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut import benchmark_shape_rulebook_full_v104 as rb
from tools.latency_lut import select_idle_gpu_for_latency as gpu


def test_idle_gpu_selector_parses_and_rejects_busy_requested_gpu():
    gpu_rows = gpu.parse_gpu_query_csv(
        "0, GPU-a, NVIDIA RTX 4090, 1, 1024, 24564, 41, 55.25\n"
        "1, GPU-b, NVIDIA RTX 4090, 42, 12000, 24564, 70, 300.00\n"
    )
    proc_rows = gpu.parse_process_query_csv("GPU-b, 1234, python, 4096\n")
    snapshots = gpu.snapshots_from_query_rows(gpu_rows, proc_rows)

    selected, reason = gpu.choose_idle_gpu(snapshots, max_utilization=5, max_memory_ratio=0.20)
    assert selected is not None
    assert selected.index == 0
    assert "idle" in reason

    selected, reason = gpu.choose_idle_gpu(
        snapshots,
        max_utilization=5,
        max_memory_ratio=0.20,
        requested_index=1,
    )
    assert selected is None
    assert "busy" in reason


def test_ordinary_grid_contains_required_channels_hws_and_kernels():
    rows = rb.generate_ordinary_targeted_grid(real_shapes=[])
    channels = {row["C_in"] for row in rows} | {row["C_out"] for row in rows}
    hws = {(row["H"], row["W"]) for row in rows}
    kernels = {row["kernel_size"] for row in rows}

    assert {2, 4}.issubset(channels)
    assert {6, 10, 12, 14, 18, 20, 22}.issubset(channels)
    assert {8, 16, 32, 64, 128, 256, 512}.issubset(channels)
    assert hws == {(100, 352), (50, 176), (13, 44)}
    assert {"1x1", "3x3"} == kernels
    assert any(row["C_in"] == row["C_out"] == 512 for row in rows)


def test_grouped_general_grid_contains_required_groups_and_widths():
    rows = rb.generate_grouped_general_grid(max_channels=4096)
    groups = {row["groups"] for row in rows}
    in_widths = {row["in_per_group"] for row in rows}
    out_widths = {row["out_per_group"] for row in rows}
    hws = {(row["H"], row["W"]) for row in rows}
    kernels = {row["kernel_size"] for row in rows}

    assert {2, 4, 8, 16, 24, 32, 48, 64}.issubset(groups)
    assert {2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 64}.issubset(in_widths)
    assert {2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 64}.issubset(out_widths)
    assert all(row["C_in"] == row["groups"] * row["in_per_group"] for row in rows)
    assert all(row["C_out"] == row["groups"] * row["out_per_group"] for row in rows)
    assert kernels == {"3x3"}
    assert (50, 176) in hws
    assert len(rows) < 2000


def test_pyramid_c_and_ab_sweeps_cover_required_stage_shapes():
    c_rows = rb.generate_pyramid_c_sweep_grid()
    ab_rows = rb.generate_pyramid_ab_sweep_grid()

    by_stage = {row["stage_name"]: row for row in c_rows}
    assert by_stage["Stage0"]["base_C"] == 128
    assert by_stage["Stage0"]["base_groups"] == 32
    assert by_stage["Stage0"]["per_group"] == 4
    assert by_stage["Stage1"]["base_C"] == 256
    assert by_stage["Stage1"]["per_group"] == 8
    assert by_stage["Stage2"]["base_C"] == 512
    assert by_stage["Stage2"]["per_group"] == 16
    assert {32, 28, 24, 20, 16, 12, 8, 4}.issubset({row["groups_after"] for row in c_rows})

    assert {row["groups"] for row in ab_rows} == {32}
    assert {2, 4, 6, 8, 10, 12, 14, 16, 24, 32}.issubset(
        {row["out_per_group_after"] for row in ab_rows}
    )


def test_decoder_rule_classifier_categories():
    assert rb.classify_decoder_rule(op_type="ordinary_conv2d", c_in=4, c_out=16, groups=1)["category"] == "hard_reject"
    assert rb.classify_decoder_rule(op_type="ordinary_conv2d", c_in=16, c_out=16, groups=1)["category"] == "preferred"
    assert (
        rb.classify_decoder_rule(
            op_type="grouped_conv2d",
            c_in=128,
            c_out=128,
            groups=32,
            in_per_group=4,
            out_per_group=4,
        )["category"]
        == "hard_reject_for_acceleration"
    )
    assert (
        rb.classify_decoder_rule(
            op_type="grouped_conv2d",
            c_in=256,
            c_out=256,
            groups=32,
            in_per_group=8,
            out_per_group=8,
        )["category"]
        == "acceptable_floor"
    )
    assert (
        rb.classify_decoder_rule(
            op_type="grouped_conv2d",
            c_in=512,
            c_out=512,
            groups=32,
            in_per_group=16,
            out_per_group=16,
        )["category"]
        == "preferred"
    )


def test_result_schema_and_skip_report(tmp_path):
    config = rb.make_conv_config(
        op_type="ordinary_conv2d",
        h=13,
        w=44,
        kernel=3,
        c_in=16,
        c_out=16,
        groups=1,
        suite="ordinary",
    )
    row = rb.build_result_row_from_stats(
        config,
        {
            "latency_mean_ms": 1.0,
            "latency_p50_ms": 1.0,
            "latency_p90_ms": 1.2,
            "latency_p95_ms": 1.4,
            "latency_min_ms": 0.9,
            "latency_max_ms": 1.5,
            "latency_std_ms": 0.1,
            "run_median_ms_list": [1.0, 1.1],
            "run_mean_ms_list": [1.0, 1.1],
        },
        dtype="fp32",
        layout="nchw",
        cudnn_benchmark=True,
        allow_tf32=False,
        selected_gpu_index=0,
        gpu_name="NVIDIA RTX 4090",
    )

    for key in rb.RESULT_FIELDS:
        assert key in row
    assert row["C_in_multiple_of_16"] is True
    assert row["params"] == 16 * 16 * 3 * 3

    skip_path = tmp_path / "skipped_shapes_report.csv"
    rb.write_skipped_shapes_report(skip_path, [])
    assert skip_path.read_text(encoding="utf-8").startswith("shape_id,")

    rb.write_skipped_shapes_report(
        skip_path,
        [rb.build_skip_row(config, skip_reason="oom", traceback_text="CUDA out of memory")],
    )
    with skip_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["skip_reason"] == "oom"
    assert "CUDA out of memory" in rows[0]["traceback"]


def test_combination_accounting_is_channel_primary_and_under_limit():
    real_shapes = []
    configs = rb.build_benchmark_plan_configs(real_shapes=real_shapes, max_channels=4096, batch=1)
    main_profiles = rb.build_main_environment_profiles(include_fp32_nchw_main_profile=True)
    diagnostic_env_matrix = rb.build_diagnostic_environment_matrix(
        dtypes=["fp32", "fp16"],
        layouts=["nchw", "channels_last"],
        cudnn_values=[True, False],
        tf32_values=[True, False],
    )
    accounting = rb.build_combination_accounting_report(
        configs,
        dtypes=["fp32", "fp16"],
        layouts=["nchw", "channels_last"],
        cudnn_values=[True, False],
        tf32_values=[True, False],
        expected_row_limit=50000,
    )

    assert main_profiles == [
        {
            "profile_name": "fp16_channels_last_main",
            "dtype": "fp16",
            "layout": "channels_last",
            "cudnn_benchmark": True,
            "allow_tf32": False,
            "allow_tf32_applicable": False,
        },
        {
            "profile_name": "fp32_nchw_main",
            "dtype": "fp32",
            "layout": "nchw",
            "cudnn_benchmark": True,
            "allow_tf32": False,
            "allow_tf32_applicable": True,
        },
    ]
    assert len(diagnostic_env_matrix) == 12
    assert {item["allow_tf32"] for item in diagnostic_env_matrix if item["dtype"] == "fp32"} == {True, False}
    assert {item["allow_tf32"] for item in diagnostic_env_matrix if item["dtype"] == "fp16"} == {False}
    assert accounting["primary_sweep_variables"] == [
        "C_in",
        "C_out",
        "groups",
        "in_per_group",
        "out_per_group",
    ]
    assert accounting["hw_kernel_role"] == "latency_context_only"
    assert accounting["main_environment_profiles"] == main_profiles
    assert accounting["diagnostic_environment_multiplier"] == 12
    assert accounting["tf32_policy"] == "main_fp16_tf32_not_applicable_diagnostic_fp32_both_fp16_false_only"
    assert 5000 <= accounting["main_rulebook_expected_rows"] <= 15000
    assert 500 <= accounting["diagnostic_expected_rows"] <= 2000
    assert accounting["expected_total_rows"] <= 20000
    assert accounting["over_limit"] is False
    assert accounting["suite_config_counts"]["grouped_general"] < 2000
    assert "diagnostic_environment_subset" in accounting["suite_config_counts"]
