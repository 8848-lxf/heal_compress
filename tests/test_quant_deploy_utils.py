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
from quant_deploy_utils import (
    INT8_NOT_IMPLEMENTED_MESSAGE,
    build_trtexec_command,
    create_quant_deploy_run_dirs,
    detect_special_ops_in_onnx_model,
    find_trtexec,
    parse_trtexec_failure,
    parse_precisions,
    write_summary_files,
)


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


def test_build_trtexec_command_int8_is_reserved():
    with pytest.raises(NotImplementedError) as exc:
        build_trtexec_command(
            precision="int8",
            onnx_path=Path("model.onnx"),
            engine_path=Path("lidar_pyramid_int8_qdq.engine"),
            layerinfo_path=Path("layerinfo_int8.json"),
        )

    assert INT8_NOT_IMPLEMENTED_MESSAGE in str(exc.value)


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
