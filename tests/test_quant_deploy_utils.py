from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "quant_deploy"))

from exportable_bev_warp import (
    affine_grid_2d_exportable,
    check_exportable_bev_warp_equivalence,
    make_base_grid_2d,
    warp_affine_simple_exportable,
)
from exportable_pillar_vfe import explicit_squeeze_pillar_features
from build_dynamic_fixed_k_int8_engine import _input_files, _trt_dtype_to_numpy
from dump_dynamic_fixed_k_int8_calibration import _covered_combos, _replacement_index_for_combo, _sample_npz_path
from dynamic_single_engine_maxk_common import (
    FIXED_K,
    calibration_cache_path,
    calibration_npz_dir,
    engine_path as single_engine_path,
    onnx_path as single_engine_onnx_path,
    pad_pairwise_to_agent_count,
    profile_from_observed_shapes,
    single_engine_dynamic_axes,
    single_engine_input_names,
)
from quant_deploy_utils import (
    build_trtexec_command,
    create_quant_deploy_run_dirs,
    detect_special_ops_in_onnx_model,
    find_trtexec,
    parse_trtexec_failure,
    parse_precisions,
    write_summary_files,
)
from evaluate_all_deployment_engines_full_val_idle_gpu import eval_modes, mode_report_name, stats as full_val_stats
from select_idle_gpu import GpuProcess, GpuSnapshot, choose_idle_gpu, parse_gpu_query_csv, parse_process_query_csv, wait_for_idle_gpu
from analyze_voxel_k_coverage import ceil_to_multiple, k_distribution_summary, recommended_fixed_k_from_reports
from dump_train_calibration_npz_for_all_strategies import enforce_train_split, file_sha256, write_calibration_manifest


def test_create_quant_deploy_run_dirs_uses_required_layout(tmp_path):
    dirs = create_quant_deploy_run_dirs(
        output_dir=tmp_path / "quant_outputs",
        run_name="lidar_pyramid_fp16_test",
        overwrite=False,
    )

    assert dirs["output_root"] == tmp_path / "quant_outputs" / "lidar_pyramid_fp16_test"
    required_keys = {
        "configs",
        "onnx_fp32",
        "onnx_qdq_int8",
        "engine_fp32",
        "engine_fp16",
        "engine_int8",
        "calibration",
        "calibration_samples",
        "calibration_caches",
        "calibration_reports",
        "benchmark_fp32",
        "benchmark_fp16",
        "benchmark_int8",
        "evaluation_fp32",
        "evaluation_fp16",
        "evaluation_int8",
        "logs_export",
        "logs_build",
        "logs_benchmark",
        "logs_evaluation",
        "summary",
        "debug",
    }
    assert required_keys.issubset(dirs)
    assert all(path.is_dir() for path in dirs.values())
    assert dirs["onnx_fp32"].parts[-3:] == ("artifacts", "onnx", "fp32")
    assert dirs["engine_fp16"].parts[-3:] == ("artifacts", "engines", "fp16")
    assert dirs["benchmark_fp32"].parts[-2:] == ("benchmark", "fp32")


def test_create_quant_deploy_run_dirs_never_uses_tests_outputs(tmp_path):
    forbidden = Path("tests") / "outputs"

    with pytest.raises(ValueError, match="tests/outputs"):
        create_quant_deploy_run_dirs(forbidden, "bad_run", overwrite=False)


def test_create_quant_deploy_run_dirs_appends_timestamp_for_existing_run(tmp_path):
    output_dir = tmp_path / "outputs"
    first = create_quant_deploy_run_dirs(
        output_dir=output_dir,
        run_name="lidar_pyramid_fp16_test",
        overwrite=False,
        timestamp="20260625_213000",
    )
    second = create_quant_deploy_run_dirs(
        output_dir=output_dir,
        run_name="lidar_pyramid_fp16_test",
        overwrite=False,
        timestamp="20260625_213000",
    )

    assert first["output_root"].name == "lidar_pyramid_fp16_test"
    assert second["output_root"].name == "lidar_pyramid_fp16_test_20260625_213000"


def test_parse_precisions_supports_single_and_multi_precision():
    assert parse_precisions(precision="fp16", precisions=None) == ["fp16"]
    assert parse_precisions(precision=None, precisions=["fp32", "fp16"]) == ["fp32", "fp16"]
    assert parse_precisions(precision="fp16", precisions=["fp32", "fp16"]) == ["fp32", "fp16"]

    with pytest.raises(ValueError, match="unsupported precision"):
        parse_precisions(precision="bf16", precisions=None)


def test_build_trtexec_command_for_fp16_practical():
    cmd = build_trtexec_command(
        precision="fp16",
        onnx_path=Path("model.onnx"),
        engine_path=Path("lidar_pyramid_fp16.engine"),
        layerinfo_path=Path("layerinfo_fp16.json"),
    )

    assert "--fp16" in cmd
    assert "--int8" not in cmd
    assert "--precisionConstraints=obey" not in cmd
    assert "--exportLayerInfo=layerinfo_fp16.json" in cmd
    assert "--profilingVerbosity=detailed" in cmd
    assert "--dumpLayerInfo" in cmd


def test_build_trtexec_command_for_strict_fp16():
    cmd = build_trtexec_command(
        precision="fp16",
        onnx_path=Path("model.onnx"),
        engine_path=Path("lidar_pyramid_fp16.engine"),
        layerinfo_path=Path("layerinfo_fp16.json"),
        strict_fp16=True,
    )

    assert "--fp16" in cmd
    assert "--precisionConstraints=obey" in cmd
    assert "--layerPrecisions=*:fp16" in cmd
    assert "--layerOutputTypes=*:fp16" in cmd


def test_build_trtexec_command_for_fp32_no_tf32():
    cmd = build_trtexec_command(
        precision="fp32",
        onnx_path=Path("model.onnx"),
        engine_path=Path("lidar_pyramid_fp32.engine"),
        layerinfo_path=Path("layerinfo_fp32.json"),
        no_tf32=True,
    )

    assert "--fp16" not in cmd
    assert "--int8" not in cmd
    assert "--noTF32" in cmd


def test_build_trtexec_command_for_native_int8_with_calibration_cache():
    cmd = build_trtexec_command(
        precision="int8",
        onnx_path=Path("model.onnx"),
        engine_path=Path("lidar_pyramid_int8.engine"),
        layerinfo_path=Path("layerinfo_int8.json"),
        calib_cache=Path("calib.cache"),
    )

    assert "--int8" in cmd
    assert "--calib=calib.cache" in cmd
    assert "--fp16" not in cmd
    assert "--exportLayerInfo=layerinfo_int8.json" in cmd


def test_dynamic_fixed_k_int8_input_files_match_npz_suffix(tmp_path):
    keep = tmp_path / "sample_000003_N2_bucket3.npz"
    keep.write_bytes(b"npz")
    (tmp_path / "sample_000004_N2_bucket2.npz").write_bytes(b"npz")
    (tmp_path / "sample_000005_N1_bucket3.npz").write_bytes(b"npz")

    assert _input_files(tmp_path, fixed_n=2, bucket_id=3) == [keep]


def test_trt_dtype_to_numpy_maps_calibrator_float_and_int_types():
    class FakeTrt:
        float32 = object()
        float16 = object()
        int32 = object()
        bool = object()

    assert _trt_dtype_to_numpy(FakeTrt.float32, FakeTrt).__name__ == "float32"
    assert _trt_dtype_to_numpy(FakeTrt.float16, FakeTrt).__name__ == "float16"
    assert _trt_dtype_to_numpy(FakeTrt.int32, FakeTrt).__name__ == "int32"
    assert _trt_dtype_to_numpy(FakeTrt.bool, FakeTrt).__name__ == "bool_"


def test_dynamic_fixed_k_calibration_helpers_cover_required_combos(tmp_path):
    rows = [
        {"record_len": 1, "bucket_id": 0},
        {"record_len": 2, "bucket_id": 1},
        {"record_len": 2, "bucket_id": 1},
    ]
    required = {(1, 0), (2, 1), (2, 3)}

    assert _covered_combos(rows, required) == {(1, 0), (2, 1)}
    assert _replacement_index_for_combo(rows) == 2
    assert _sample_npz_path(tmp_path, 2, 2, 3).name == "sample_000002_N2_bucket3.npz"


def test_dynamic_single_engine_maxk_paths_and_input_contract(tmp_path):
    dirs = create_quant_deploy_run_dirs(tmp_path / "outputs", "run", overwrite=False)

    assert single_engine_input_names() == [
        "voxel_features",
        "voxel_coords",
        "voxel_num_points",
        "pairwise_t_matrix",
        "valid_voxel_mask",
    ]
    assert single_engine_onnx_path(dirs).name == "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
    assert single_engine_path(dirs, "fp32").parts[-3:] == (
        "dynamic_agent_single_engine_maxK",
        "fp32",
        "lidar_pyramid_dynamic_agent_single_engine_maxK_fp32.engine",
    )
    assert single_engine_path(dirs, "int8", 200).parts[-3:] == (
        "dynamic_agent_single_engine_maxK",
        "int8_train_calib200",
        "lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib200.engine",
    )
    assert calibration_npz_dir(dirs, 50).name == "dynamic_single_engine_maxK_train_calib50"
    assert calibration_cache_path(dirs, 50).name == "lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib50.cache"


def test_dynamic_single_engine_maxk_profile_uses_observed_dynamic_n_and_fixed_k():
    observed = [
        {"voxel_features": [FIXED_K, 32, 4], "pairwise_t_matrix": [1, 1, 1, 4, 4]},
        {"voxel_features": [FIXED_K, 32, 4], "pairwise_t_matrix": [1, 2, 2, 4, 4]},
        {"voxel_features": [FIXED_K, 32, 4], "pairwise_t_matrix": [1, 2, 2, 4, 4]},
    ]

    profile = profile_from_observed_shapes(observed)

    assert profile["voxel_features"] == {"min": [FIXED_K, 32, 4], "opt": [FIXED_K, 32, 4], "max": [FIXED_K, 32, 4]}
    assert profile["voxel_coords"]["max"] == [FIXED_K, 4]
    assert profile["valid_voxel_mask"]["opt"] == [FIXED_K]
    assert profile["pairwise_t_matrix"] == {
        "min": [1, 1, 1, 4, 4],
        "opt": [1, 2, 2, 4, 4],
        "max": [1, 2, 2, 4, 4],
    }


def test_dynamic_single_engine_maxk_dynamic_axes_keep_k_static_and_n_dynamic():
    axes = single_engine_dynamic_axes(["cls_preds"])

    assert "voxel_features" not in axes
    assert axes["pairwise_t_matrix"] == {1: "num_agents", 2: "num_agents"}
    assert axes["cls_preds"] == {0: "batch"}


def test_dynamic_single_engine_maxk_pairwise_padding_preserves_true_n_payload():
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).numpy()

    padded = pad_pairwise_to_agent_count(pairwise, 2)

    assert padded.shape == (1, 2, 2, 4, 4)
    assert padded.dtype == pairwise.dtype
    assert padded[0, 0, 0, 0, 0] == 1
    assert padded[0, 1, 1, 0, 0] == 1


def test_select_idle_gpu_parses_nvidia_smi_csv():
    gpu_rows = parse_gpu_query_csv(
        "0, GPU-a, NVIDIA H800, 0, 12, 81559\n"
        "1, GPU-b, NVIDIA H800, 98, 42000, 81559\n"
    )
    proc_rows = parse_process_query_csv("GPU-b, 1234, python, 4096\n")

    assert gpu_rows[0]["index"] == 0
    assert gpu_rows[1]["utilization_gpu"] == 98
    assert proc_rows == [GpuProcess(gpu_uuid="GPU-b", pid=1234, process_name="python", used_memory_mb=4096)]


def test_select_idle_gpu_prefers_lowest_memory_then_util():
    snapshots = [
        GpuSnapshot(index=0, uuid="GPU-0", name="H800", utilization_gpu=1, memory_used_mb=1500, memory_total_mb=80000, processes=[]),
        GpuSnapshot(index=1, uuid="GPU-1", name="H800", utilization_gpu=0, memory_used_mb=500, memory_total_mb=80000, processes=[]),
        GpuSnapshot(index=2, uuid="GPU-2", name="H800", utilization_gpu=0, memory_used_mb=100, memory_total_mb=80000, processes=[GpuProcess("GPU-2", 99, "python", 100)]),
    ]

    selected, reason = choose_idle_gpu(snapshots, util_threshold=5, mem_threshold_mb=2000)

    assert selected is not None
    assert selected.index == 1
    assert "lowest memory" in reason


def test_select_idle_gpu_rejects_busy_requested_gpu_without_override():
    snapshots = [
        GpuSnapshot(index=3, uuid="GPU-3", name="H800", utilization_gpu=10, memory_used_mb=3000, memory_total_mb=80000, processes=[]),
    ]

    selected, reason = choose_idle_gpu(snapshots, util_threshold=5, mem_threshold_mb=2000, gpu_index=3)

    assert selected is None
    assert "busy" in reason


def test_full_val_mode_names_and_latency_stats_include_p99():
    names = {mode.key: mode_report_name(mode) for mode in eval_modes()}
    summary = full_val_stats([3.0, 1.0, 2.0])

    assert names["single_engine_maxK_int8_train_calib200"] == "single_engine_maxK_int8_train_calib200_full_val.json"
    assert names["dynamic_bucket_fp16"] == "dynamic_bucket_fp16_full_val.json"
    assert summary["p50"] == 2.0
    assert "p99" in summary


def test_full_val_modes_can_use_fixedk_train_calibration_namespace():
    modes = {mode.key: mode for mode in eval_modes(fixed_k=29696, dynamic_int8_calibration_split="train")}

    assert modes["dynamic_bucket_fp16"].fixed_K == 29696
    assert modes["dynamic_bucket_int8_calib200"].calibration_split == "train"
    assert modes["dynamic_bucket_int8_calib200"].calibration_mode == "train_calib200"
    assert modes["single_engine_maxK_int8_train_calib200"].fixed_K == 29696


def test_wait_for_idle_gpu_records_query_failure_without_hanging(monkeypatch):
    calls = {"count": 0}

    def fail_query():
        calls["count"] += 1
        raise RuntimeError("nvidia-smi timed out after 1s")

    monkeypatch.setattr("select_idle_gpu.query_gpu_snapshots", fail_query)

    with pytest.raises(TimeoutError, match="gpu query failed"):
        wait_for_idle_gpu(wait_timeout_sec=0, poll_interval_sec=1)

    assert calls["count"] == 1


def test_write_summary_files_records_dirs_and_markdown_table(tmp_path):
    dirs = create_quant_deploy_run_dirs(tmp_path / "outputs", "run", overwrite=False)
    summary = {
        "model": "lidar_pyramid",
        "checkpoint": "/ckpt.pth",
        "hypes_yaml": "/config.yaml",
        "output_root": str(dirs["output_root"]),
        "dirs": {"configs": str(dirs["configs"])},
        "onnx_export": {"success": False, "error": "export failed", "export_boundary": "full_model", "opset": 17},
        "precisions": {
            "fp16": {
                "implemented": True,
                "build_success": False,
                "benchmark_success": False,
                "engine_path": None,
                "engine_size_MB": None,
                "forward_p50_ms": None,
                "forward_p90_ms": None,
                "forward_p95_ms": None,
                "fps": None,
                "speedup_vs_fp32": None,
                "error": "build failed",
            }
        },
        "detected_special_ops": {"GridSample": [], "AffineGrid": [], "Scatter": [], "Inverse": [], "unsupported_ops": []},
    }

    paths = write_summary_files(summary, dirs)

    assert paths["json"].exists()
    assert paths["md"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["dirs"]["configs"] == str(dirs["configs"])
    markdown = paths["md"].read_text(encoding="utf-8")
    assert "precision | implemented | build | benchmark" in markdown
    assert "fp16 | yes | no | no" in markdown


def test_find_trtexec_prefers_explicit_path(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit" / "trtexec"
    trt_root = tmp_path / "trt"
    explicit.parent.mkdir()
    (trt_root / "bin").mkdir(parents=True)
    explicit.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (trt_root / "bin" / "trtexec").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    explicit.chmod(0o755)
    (trt_root / "bin" / "trtexec").chmod(0o755)

    found = find_trtexec(trt_root=str(trt_root), explicit_trtexec=str(explicit))

    assert found == str(explicit)


def test_find_trtexec_uses_trt_root_before_path(tmp_path, monkeypatch):
    trt_root = tmp_path / "trt"
    path_dir = tmp_path / "pathbin"
    (trt_root / "targets" / "x86_64-linux-gnu" / "bin").mkdir(parents=True)
    path_dir.mkdir()
    trt_exec = trt_root / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec"
    path_exec = path_dir / "trtexec"
    trt_exec.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    path_exec.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    trt_exec.chmod(0o755)
    path_exec.chmod(0o755)
    monkeypatch.setenv("PATH", str(path_dir))

    found = find_trtexec(trt_root=str(trt_root), explicit_trtexec=None)

    assert found == str(trt_exec)


def test_find_trtexec_falls_back_to_path(tmp_path, monkeypatch):
    path_dir = tmp_path / "pathbin"
    path_dir.mkdir()
    path_exec = path_dir / "trtexec"
    path_exec.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    path_exec.chmod(0o755)
    monkeypatch.setenv("PATH", str(path_dir))
    monkeypatch.delenv("TRT_ROOT", raising=False)

    found = find_trtexec(trt_root=None, explicit_trtexec=None)

    assert found == str(path_exec)


def test_exportable_affine_grid_output_shape_and_dtype():
    theta = torch.eye(2, 3, dtype=torch.float64).unsqueeze(0).repeat(3, 1, 1)

    base = make_base_grid_2d(5, 7, theta.device, theta.dtype, align_corners=False)
    grid = affine_grid_2d_exportable(theta, 5, 7, align_corners=False)

    assert base.shape == (1, 5, 7, 3)
    assert grid.shape == (3, 5, 7, 2)
    assert grid.dtype == torch.float64


def test_exportable_affine_grid_matches_torch_affine_grid():
    theta = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.8, 0.1, 0.2], [-0.1, 0.9, -0.3]],
        ],
        dtype=torch.float32,
    )

    actual = affine_grid_2d_exportable(theta, 6, 8, align_corners=False)
    expected = F.affine_grid(theta, torch.Size((2, 4, 6, 8)), align_corners=False)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_exportable_warp_matches_original_grid_sample_path():
    torch.manual_seed(7)
    src = torch.randn(2, 3, 6, 8)
    theta = torch.tensor(
        [
            [[1.0, 0.0, 0.1], [0.0, 1.0, -0.2]],
            [[0.9, 0.2, 0.0], [-0.2, 0.9, 0.1]],
        ],
        dtype=torch.float32,
    )
    expected_grid = F.affine_grid(theta, torch.Size((2, 3, 6, 8)), align_corners=False)
    expected = F.grid_sample(src, expected_grid, align_corners=False)

    actual = warp_affine_simple_exportable(src, theta, (6, 8), align_corners=False)
    report = check_exportable_bev_warp_equivalence(src=src, theta=theta, dsize=(6, 8), align_corners=False)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert report["max_abs_error"] < 1e-5


def test_special_ops_report_records_aten_and_org_pytorch_nodes():
    class FakeNode:
        def __init__(self, op_type, domain="", name="node"):
            self.op_type = op_type
            self.domain = domain
            self.name = name
            self.output = ["out"]

    class FakeGraph:
        node = [
            FakeNode("GridSample", name="grid"),
            FakeNode("affine_grid_generator", domain="aten", name="affine"),
            FakeNode("CustomPlugin", domain="org.pytorch.aten", name="custom"),
        ]

    class FakeModel:
        graph = FakeGraph()

    report = detect_special_ops_in_onnx_model(FakeModel())

    assert report["GridSample"]
    assert report["AffineGrid"]
    assert report["aten_ops"]
    assert report["org_pytorch_ops"]


def test_special_ops_report_counts_sequence_nodes():
    class FakeNode:
        def __init__(self, op_type, domain="", name="node"):
            self.op_type = op_type
            self.domain = domain
            self.name = name
            self.output = ["out"]
            self.input = []
            self.attribute = []

    class FakeGraph:
        node = [
            FakeNode("SequenceEmpty", name="seq_empty"),
            FakeNode("SequenceInsert", name="seq_insert"),
            FakeNode("SequenceAt", name="seq_at"),
        ]

    class FakeModel:
        graph = FakeGraph()

    report = detect_special_ops_in_onnx_model(FakeModel())

    assert report["SequenceEmpty"]
    assert report["SequenceInsert"]
    assert report["SequenceAt"]
    assert report["sequence_op_count"] == 3


def test_parse_trtexec_failure_extracts_failed_squeeze_node():
    log_text = """
[E] [TRT] ModelImporter.cpp:961: While parsing node number 672 [Squeeze -> "/model/encoder_m1/pillar_vfe/Squeeze_output_0"]:
name: "/model/encoder_m1/pillar_vfe/Squeeze"
op_type: "Squeeze"
[E] [TRT] ModelImporter.cpp:967: ERROR: onnxOpImporters.cpp:5889 In function importSqueeze:
[12] Assertion failed: !isDynamic(shape): Cannot infer squeeze dimensions from a dynamic shape! Please re-export your model with the Squeeze axes input set.
"""

    parsed = parse_trtexec_failure(log_text)

    assert parsed["failed_nodes"][0]["op"] == "Squeeze"
    assert "pillar_vfe/Squeeze" in parsed["failed_nodes"][0]["line"]
    assert any("Cannot infer squeeze dimensions" in item["line"] for item in parsed["unsupported_ops"])


def test_explicit_squeeze_pillar_features_removes_only_axis_one():
    features = torch.randn(4, 1, 8)

    squeezed = explicit_squeeze_pillar_features(features)

    assert squeezed.shape == (4, 8)
    assert torch.equal(squeezed, features[:, 0, :])


def test_explicit_squeeze_pillar_features_keeps_single_voxel_batch_dim():
    features = torch.randn(1, 1, 8)

    squeezed = explicit_squeeze_pillar_features(features)

    assert squeezed.shape == (1, 8)
    assert torch.equal(squeezed, features[:, 0, :])


def test_voxel_k_coverage_summary_uses_required_percentiles_and_top_samples():
    samples = [
        {"sample_idx": 0, "record_len": 1, "original_num_voxels": 10},
        {"sample_idx": 1, "record_len": 2, "original_num_voxels": 20},
        {"sample_idx": 2, "record_len": 2, "original_num_voxels": 30},
        {"sample_idx": 3, "record_len": 1, "original_num_voxels": 40},
    ]

    summary = k_distribution_summary(samples, split="val", old_fixed_k=25)

    assert summary["split"] == "val"
    assert summary["total_samples"] == 4
    assert summary["valid_samples"] == 4
    assert summary["K"]["min"] == 10
    assert summary["K"]["p50"] == 25
    assert summary["K"]["max"] == 40
    assert summary["samples_with_K_gt_24064"] == 2
    assert summary["top_50_largest_K_samples"][0]["original_num_voxels"] == 40


def test_voxel_k_coverage_recommends_ceiled_fixed_k_across_train_and_val():
    train = {"K": {"max": 33001}}
    val = {"K": {"max": 32769}}

    assert ceil_to_multiple(32769, 512) == 33280
    assert recommended_fixed_k_from_reports(train, val, multiple=512) == 33280


def test_train_calibration_manifest_records_hashes_and_rejects_non_train(tmp_path):
    npz = tmp_path / "sample_000000_N1_bucket0.npz"
    npz.write_bytes(b"calibration-bytes")
    row = {
        "path": str(npz),
        "sample_idx": 0,
        "record_len": 1,
        "original_num_voxels": 123,
        "input_shapes": {"voxel_features": [29696, 32, 4]},
        "input_dtypes": {"voxel_features": "float32"},
    }

    with pytest.raises(ValueError, match="train"):
        enforce_train_split("val")

    manifest = write_calibration_manifest(
        npz_dir=tmp_path,
        strategy="dynamic_bucket",
        fixed_k=29696,
        num_samples=1,
        rows=[row],
        config_path="config.yaml",
        checkpoint_path="model.pth",
        calibration_split="train",
    )

    assert manifest["calibration_split"] == "train"
    assert manifest["fixed_K"] == 29696
    assert manifest["files"][0]["sha256"] == file_sha256(npz)
    assert (tmp_path / "manifest.json").exists()
