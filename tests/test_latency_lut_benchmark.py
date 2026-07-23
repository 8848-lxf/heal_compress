from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from opencood.tools.compression.latency_lut.schema import LatencyLUTKey, LatencyRecord, write_jsonl
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase
from opencood.tools.compression.latency_lut.subgraph_exporter import export_minimal_subgraph
from opencood.tools.compression.latency_lut.tensorRT_benchmark import (
    benchmark_key,
    build_trtexec_command_for_test,
    parse_trtexec_latency,
)
from tools.latency_lut.benchmark_subgraph import filter_keys, success_key_hashes


def _key(
    *,
    module_name: str = "backbone",
    block_name: str = "stage1",
    block_type: str = "conv_block",
    precision_profile: str = "TRT_FP16",
    plugin_flag: bool = False,
    plugin_name: str | None = None,
) -> LatencyLUTKey:
    return LatencyLUTKey(
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
        module_name=module_name,
        block_name=block_name,
        block_type=block_type,
        H=16,
        W=16,
        C_in=16,
        C_out=16,
        kernel_size=3,
        stride=1,
        padding=1,
        batch_size=1,
        precision_profile=precision_profile,
        weight_precision={"TRT_FP32": "FP32", "TRT_FP16": "FP16", "TRT_INT8_QDQ": "INT8"}[precision_profile],
        activation_precision="FP16",
        compute_precision=precision_profile.replace("TRT_", ""),
        plugin_flag=plugin_flag,
        plugin_name=plugin_name,
    )


def test_parse_trtexec_gpu_compute_time_summary():
    log = """
    [06/30/2026-12:00:00] [I] === Performance summary ===
    [06/30/2026-12:00:00] [I] Throughput: 12345 qps
    [06/30/2026-12:00:00] [I] GPU Compute Time: min = 0.112 ms, max = 0.250 ms, mean = 0.140 ms, median = 0.133 ms, percentile(90%) = 0.175 ms, percentile(95%) = 0.190 ms, percentile(99%) = 0.220 ms
    """
    parsed = parse_trtexec_latency(log)

    assert parsed["latency_p50_ms"] == 0.133
    assert parsed["latency_p90_ms"] == 0.175
    assert parsed["latency_p95_ms"] == 0.190
    assert parsed["latency_p99_ms"] == 0.220
    assert parsed["latency_mean_ms"] == 0.140
    assert parsed["latency_std_ms"] == 0.0


def test_benchmark_key_does_not_fallback_int8_without_trtexec(tmp_path: Path):
    record = benchmark_key(
        _key(precision_profile="TRT_INT8_QDQ"),
        output_dir=tmp_path,
        dry_run=False,
        trtexec_path="/does/not/exist/trtexec",
    )

    assert record.status == "skipped_trtexec_not_found"
    assert record.metadata.get("backend") == "tensorrt"
    assert record.latency_p50_ms == 0.0


def test_benchmark_key_skips_plugin_without_fake_latency(tmp_path: Path):
    record = benchmark_key(
        _key(
            module_name="scatter",
            block_name="scatter",
            block_type="plugin",
            plugin_flag=True,
            plugin_name="PointPillarScatterTRT",
        ),
        output_dir=tmp_path,
        dry_run=False,
        trtexec_path="/does/not/exist/trtexec",
        plugin_path=tmp_path / "missing_plugin.so",
    )

    assert record.status == "skipped_plugin_not_available"
    assert "PointPillarScatterTRT" in (record.error_message or "")
    assert record.latency_p50_ms == 0.0


def test_filter_keys_supports_module_precision_limit_and_resume(tmp_path: Path):
    fp16_backbone = _key(module_name="backbone", precision_profile="TRT_FP16")
    fp32_backbone = _key(module_name="backbone", block_name="stage2", precision_profile="TRT_FP32")
    fp16_head = _key(module_name="detection_head", block_name="cls", block_type="head_branch", precision_profile="TRT_FP16")
    keys = [fp16_backbone, fp32_backbone, fp16_head]
    resume_path = tmp_path / "records.jsonl"
    write_jsonl(
        [
            LatencyRecord(
                key=fp16_backbone,
                latency_p50_ms=0.1,
                latency_p90_ms=0.1,
                latency_p95_ms=0.1,
                latency_p99_ms=0.1,
                latency_mean_ms=0.1,
                latency_std_ms=0.0,
                num_warmup=1,
                num_repeat=1,
                timing_method="synthetic",
                status="success",
            )
        ],
        resume_path,
    )

    args = Namespace(
        module_filter=["backbone"],
        precision_filter=["TRT_FP16", "TRT_FP32"],
        limit=None,
        resume=True,
        output=str(resume_path),
    )
    filtered = filter_keys(keys, args, completed_hashes=success_key_hashes(resume_path))

    assert [key.block_name for key in filtered] == ["stage2"]


def test_export_fusion_subgraph_uses_two_inputs(tmp_path: Path):
    onnx = __import__("onnx")
    key = _key(
        module_name="pyramid_fusion",
        block_name="scale1",
        block_type="fusion_block",
        precision_profile="TRT_FP16",
    )
    key = LatencyLUTKey.from_dict(
        {
            **key.to_dict(),
            "H": 8,
            "W": 8,
            "C_in": 24,
            "C_mid": 16,
            "C_out": 16,
            "metadata": {"C_ego": 16, "C_infra": 8, "C_fused": 16},
        }
    )

    exported = export_minimal_subgraph(key, tmp_path, dry_run=False)
    model = onnx.load(exported["onnx_path"])

    assert exported["status"] == "exported"
    assert [item.name for item in model.graph.input] == ["ego", "infrastructure"]


def test_export_pfn_subgraph_keeps_fixed_k_dimension(tmp_path: Path):
    onnx = __import__("onnx")
    key = _key(module_name="pfn", block_name="pfn", block_type="pfn_block", precision_profile="TRT_FP16")
    key = LatencyLUTKey.from_dict(
        {
            **key.to_dict(),
            "C_in": 10,
            "C_out": 32,
            "metadata": {"point_feature_dim": 10},
        }
    )

    exported = export_minimal_subgraph(key, tmp_path, dry_run=False)
    model = onnx.load(exported["onnx_path"])
    dims = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]

    assert exported["status"] == "exported"
    assert dims == [1, 29696, 10]


def test_int8_missing_lut_is_unavailable_not_regular_default():
    db = LatencyLUTDatabase(default_latency_ms=1.0, default_uncertainty_ms=0.5)
    item = db.estimate_key(_key(precision_profile="TRT_INT8_QDQ"))

    assert item.match_type == "unavailable"
    assert item.uncertainty_ms >= 10.0


def test_export_int8_conv_subgraph_contains_qdq_nodes(tmp_path: Path):
    onnx = __import__("onnx")
    key = _key(precision_profile="TRT_INT8_QDQ")
    key.metadata.update(
        {
            "activation_scale": 0.03125,
            "weight_scale": 0.015625,
            "scale_source": "unit_test_calibration",
        }
    )

    exported = export_minimal_subgraph(key, tmp_path, dry_run=False)
    model = onnx.load(exported["onnx_path"])
    op_types = [node.op_type for node in model.graph.node]

    assert "QuantizeLinear" in op_types
    assert "DequantizeLinear" in op_types
    assert exported["metadata"]["scale_source"] == "unit_test_calibration"


def test_export_precision_boundary_subgraph_contains_boundary_metadata(tmp_path: Path):
    onnx = __import__("onnx")
    key = LatencyLUTKey(
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
        module_name="precision_boundary",
        block_name="TRT_FP16_to_TRT_INT8_QDQ",
        block_type="precision_boundary",
        H=8,
        W=8,
        C_in=16,
        C_out=16,
        batch_size=1,
        precision_profile="TRT_FP16",
        weight_precision="FP16",
        activation_precision="FP16",
        compute_precision="FP16",
        src_precision="TRT_FP16",
        dst_precision="TRT_INT8_QDQ",
        tensor_dtype_before="TRT_FP16",
        tensor_dtype_after="TRT_INT8_QDQ",
    )

    exported = export_minimal_subgraph(key, tmp_path, dry_run=False)
    model = onnx.load(exported["onnx_path"])
    op_types = [node.op_type for node in model.graph.node]

    assert exported["metadata"]["boundary_type"] == "TRT_FP16_to_TRT_INT8_QDQ"
    assert exported["metadata"]["uses_qdq"] is True
    assert "QuantizeLinear" in op_types


def test_int8_trtexec_command_requests_int8_and_layer_info(tmp_path: Path):
    key = _key(precision_profile="TRT_INT8_QDQ")

    cmd = build_trtexec_command_for_test(
        trtexec="/opt/tensorrt/bin/trtexec",
        onnx_path=tmp_path / "model.onnx",
        engine_path=tmp_path / "model.engine",
        key=key,
        warmup=1,
        repeat=2,
        device=0,
        min_repeat_ms=None,
        iterations=None,
    )

    assert "--int8" in cmd
    assert "--dumpLayerInfo" in cmd
    assert "--profilingVerbosity=detailed" in cmd
