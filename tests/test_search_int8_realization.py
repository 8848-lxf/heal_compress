from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_onnx_bn_fold_aware_calibration_uses_folded_weight_and_bn_output(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import numpy as np
    import onnx
    import pytest
    import torch
    from onnx import numpy_helper

    from search.integration.calibration_provider import collect_onnx_bn_fold_aware_qdq_scales

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 3, 1, bias=False)
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.bn(self.conv(value))

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = Block()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.block(value)

    model = Model().eval()
    with torch.no_grad():
        model.block.conv.weight.copy_(torch.tensor([[[[0.25]], [[-0.5]]], [[[0.75]], [[0.1]]], [[[-0.2]], [[0.4]]]]))
        model.block.bn.weight.copy_(torch.tensor([2.0, 0.5, 3.0]))
        model.block.bn.bias.copy_(torch.tensor([0.3, -0.2, 0.1]))
        model.block.bn.running_mean.copy_(torch.tensor([0.4, -0.1, 0.2]))
        model.block.bn.running_var.copy_(torch.tensor([0.25, 4.0, 0.5]))
    batches = [torch.tensor([[[[1.0, -2.0]], [[0.5, 3.0]]]]), torch.tensor([[[[-1.5, 0.25]], [[2.0, -0.75]]]])]
    onnx_path = tmp_path / "folded.onnx"
    torch.onnx.export(model, batches[0], onnx_path, opset_version=13, do_constant_folding=True)
    graph = onnx.load(str(onnx_path))
    conv_node = next(node for node in graph.graph.node if node.op_type == "Conv")
    assert not any(node.op_type == "BatchNormalization" for node in graph.graph.node)
    initializer_name = str(conv_node.input[1])
    origin_map = SimpleNamespace(
        entries=[
            SimpleNamespace(
                module_path="block.conv",
                canonical_node_name=str(conv_node.name),
                weight_initializer=initializer_name,
            )
        ]
    )

    scales, details = collect_onnx_bn_fold_aware_qdq_scales(
        model=model,
        batches=batches,
        module_paths=["block.conv"],
        forward_fn=lambda inner, batch: inner(batch),
        onnx_path=onnx_path,
        origin_map=origin_map,
        activation_calibration_method="absmax",
    )

    initializers = {row.name: numpy_helper.to_array(row) for row in graph.graph.initializer}
    expected_weight = float(np.max(np.abs(initializers[initializer_name]))) / 127.0
    expected_input = max(float(batch.abs().amax()) for batch in batches) / 127.0
    with torch.no_grad():
        expected_output = max(float(model(batch).abs().amax()) for batch in batches) / 127.0
    row = scales["block.conv"]
    assert row["weight_scale"] == pytest.approx(expected_weight)
    assert row["activation_input_scale"] == pytest.approx(expected_input)
    assert row["activation_output_scale"] == pytest.approx(expected_output)
    assert details["output_module_paths"] == {"block.conv": "block.bn"}

    channel_scales, channel_details = collect_onnx_bn_fold_aware_qdq_scales(
        model=model,
        batches=batches,
        module_paths=["block.conv"],
        forward_fn=lambda inner, batch: inner(batch),
        onnx_path=onnx_path,
        origin_map=origin_map,
        weight_granularity="per_channel",
        activation_calibration_method="absmax",
    )
    expected_channel = np.max(np.abs(initializers[initializer_name]), axis=(1, 2, 3)) / 127.0
    assert channel_scales["block.conv"]["weight_scale"] == pytest.approx(expected_channel.tolist())
    assert channel_scales["block.conv"]["weight_axis"] == 0
    assert channel_scales["block.conv"]["weight_scale_shape"] == [3]
    assert channel_details["weight_granularity"] == "per_channel"
    assert channel_details["weight_axes"] == {"block.conv": 0}

    entropy_scales, entropy_details = collect_onnx_bn_fold_aware_qdq_scales(
        model=model,
        batches=batches,
        module_paths=["block.conv"],
        forward_fn=lambda inner, batch: inner(batch),
        onnx_path=onnx_path,
        origin_map=origin_map,
        weight_granularity="per_channel",
        activation_calibration_method="entropy",
        histogram_bins=256,
    )
    assert entropy_details["activation_calibration_method"] == "entropy"
    assert entropy_details["passes"] == 2
    assert 0.0 < entropy_scales["block.conv"]["activation_input_scale"] <= expected_input
    assert 0.0 < entropy_scales["block.conv"]["activation_output_scale"] <= expected_output


def test_calibration_metadata_uses_each_canonical_node_not_last_node(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import onnx
    import torch

    from search.integration.calibration_provider import collect_onnx_bn_fold_aware_qdq_scales

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Conv2d(2, 3, 1)
            self.second = torch.nn.Conv2d(3, 4, 1)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.second(torch.relu(self.first(value)))

    model = Model().eval()
    sample = torch.randn(1, 2, 3, 3)
    onnx_path = tmp_path / "two_nodes.onnx"
    torch.onnx.export(model, sample, onnx_path, opset_version=13, do_constant_folding=True)
    graph = onnx.load(str(onnx_path))
    nodes = [node for node in graph.graph.node if node.op_type == "Conv"]
    assert len(nodes) == 2
    origin = SimpleNamespace(
        entries=[
            SimpleNamespace(
                module_path=module,
                canonical_node_name=str(node.name),
                weight_initializer=str(node.input[1]),
            )
            for module, node in zip(("first", "second"), nodes)
        ]
    )

    scales, _details = collect_onnx_bn_fold_aware_qdq_scales(
        model=model,
        batches=[sample],
        module_paths=["first", "second"],
        forward_fn=lambda inner, batch: inner(batch),
        onnx_path=onnx_path,
        origin_map=origin,
        weight_granularity="per_channel",
        activation_calibration_method="absmax",
    )

    assert scales["first"]["activation_input_tensor"] == str(nodes[0].input[0])
    assert scales["first"]["activation_output_tensor"] == str(nodes[0].output[0])
    assert scales["second"]["activation_input_tensor"] == str(nodes[1].input[0])
    assert scales["second"]["activation_output_tensor"] == str(nodes[1].output[0])


def test_exact_npz_calibration_manifest_verifies_tensor_files(tmp_path: Path) -> None:
    import hashlib
    import json

    import numpy as np
    import pytest
    import torch

    from search.integration.calibration_provider import (
        FIXED_K_CALIBRATION_INPUT_NAMES,
        load_fixed_k_calibration_npz_batches,
    )

    fixed_k = 4
    sample = tmp_path / "sample_000000_N1.npz"
    np.savez_compressed(
        sample,
        voxel_features=np.arange(fixed_k * 2 * 4, dtype=np.float32).reshape(fixed_k, 2, 4),
        voxel_coords=np.arange(fixed_k * 4, dtype=np.int32).reshape(fixed_k, 4),
        voxel_num_points=np.arange(fixed_k, dtype=np.int32),
        pairwise_t_matrix=np.eye(4, dtype=np.float32).reshape(1, 1, 1, 4, 4),
        valid_voxel_mask=np.ones(fixed_k, dtype=np.bool_),
    )
    digest = hashlib.sha256(sample.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strategy": "single_engine_maxK",
                "calibration_split": "train",
                "fixed_K": fixed_k,
                "num_samples": 1,
                "input_names": list(FIXED_K_CALIBRATION_INPUT_NAMES),
                "train_dataset_indices": [0],
                "files": [{"name": sample.name, "path": str(sample), "sha256": digest, "bytes": sample.stat().st_size}],
            }
        ),
        encoding="utf-8",
    )

    batches, provenance = load_fixed_k_calibration_npz_batches(
        manifest,
        num_batches=1,
        fixed_k=fixed_k,
        device=torch.device("cpu"),
    )

    assert len(batches) == 1
    assert set(batches[0]) == set(FIXED_K_CALIBRATION_INPUT_NAMES)
    assert torch.equal(batches[0]["voxel_num_points"], torch.arange(fixed_k, dtype=torch.int32))
    assert provenance["files_verified"] is True
    assert provenance["sample_count"] == 1
    assert len(provenance["tensor_manifest_hash"]) == 64

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="calibration_npz_file_hash_mismatch"):
        load_fixed_k_calibration_npz_batches(
            manifest,
            num_batches=1,
            fixed_k=fixed_k,
            device=torch.device("cpu"),
        )


def test_tensorrt_entropy_cache_requires_exact_boundaries_and_keeps_per_channel_weights(
    tmp_path: Path,
) -> None:
    import struct
    from types import SimpleNamespace

    import numpy as np
    import onnx
    import pytest
    from onnx import TensorProto, helper, numpy_helper

    from search.integration.calibration_provider import qdq_scales_from_tensorrt_entropy_cache

    weight = np.asarray(
        [
            [[[1.0]], [[-2.0]], [[0.5]]],
            [[[4.0]], [[-1.0]], [[0.25]]],
        ],
        dtype=np.float32,
    )
    node = helper.make_node(
        "Conv",
        ["input_tensor", "weight"],
        ["output_tensor"],
        name="canonical_conv",
    )
    graph = helper.make_graph(
        [node, helper.make_node("Relu", ["output_tensor"], ["relu_tensor"], name="canonical_relu")],
        "cache_test",
        [helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("relu_tensor", TensorProto.FLOAT, [1, 2, 2, 2])],
        [numpy_helper.from_array(weight, name="weight")],
    )
    onnx_path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), onnx_path)
    cache_path = tmp_path / "calibration.cache"
    cache_path.write_text(
        "\n".join(
            [
                "TRT-100900-EntropyCalibration2",
                f"input_tensor: {struct.pack('!f', 0.125).hex()}",
                f"output_tensor: {struct.pack('!f', 0.25).hex()}",
                f"relu_tensor: {struct.pack('!f', 0.375).hex()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    origin = SimpleNamespace(
        entries=[
            SimpleNamespace(
                module_path="conv",
                canonical_node_name="canonical_conv",
                weight_initializer="weight",
            )
        ]
    )

    scales, details = qdq_scales_from_tensorrt_entropy_cache(
        onnx_path=onnx_path,
        origin_map=origin,
        module_paths=["conv"],
        cache_path=cache_path,
        weight_granularity="per_channel",
    )

    assert scales["conv"]["activation_input_scale"] == pytest.approx(0.125)
    assert scales["conv"]["activation_output_scale"] == pytest.approx(0.375)
    assert scales["conv"]["activation_output_tensor"] == "relu_tensor"
    assert scales["conv"]["activation_output_boundary_resolution"] == "post_relu_semantic_boundary"
    assert scales["conv"]["weight_axis"] == 0
    assert scales["conv"]["weight_scale"] == pytest.approx([2.0 / 127.0, 4.0 / 127.0])
    assert scales["conv"]["activation_scale_source"].endswith("exact_tensor_match")
    assert details["exact_activation_scale_match_count"] == 2

    cache_path.write_text(
        "TRT-100900-EntropyCalibration2\n"
        f"input_tensor: {struct.pack('!f', 0.125).hex()}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tensorrt_entropy_exact_tensor_match_missing"):
        qdq_scales_from_tensorrt_entropy_cache(
            onnx_path=onnx_path,
            origin_map=origin,
            module_paths=["conv"],
            cache_path=cache_path,
            weight_granularity="per_channel",
        )


def test_tensorrt_entropy_calibration_profile_uses_manifest_agent_distribution() -> None:
    from search.integration.tensorrt_entropy_calibration_worker import _profile

    profile = _profile([1, 2, 2, 2], 29696)

    assert profile["pairwise_t_matrix"]["min"] == [1, 1, 1, 4, 4]
    assert profile["pairwise_t_matrix"]["opt"] == [1, 2, 2, 4, 4]
    assert profile["pairwise_t_matrix"]["max"] == [1, 2, 2, 4, 4]
    assert profile["voxel_features"]["opt"] == [29696, 32, 4]

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
        qdq_report={
            "inserted_layer_count": 1,
            "requested_int8_count": 1,
            "records": [
                {
                    "activation_quantize_node": "q-in",
                    "activation_dequantize_node": "dq-in",
                    "weight_quantize_node": "q-weight",
                    "weight_dequantize_node": "dq-weight",
                    "output_quantize_nodes": ["q-out-0", "q-out-1"],
                    "output_dequantize_nodes": ["dq-out-0", "dq-out-1"],
                }
            ],
        },
        int8_macs_ratio=0.25,
    )

    assert report["realized_int8_group_count"] == 1
    assert report["realized_int8_layer_count"] == 1
    assert report["QuantizeLinear_count"] == 4
    assert report["DequantizeLinear_count"] == 4
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


def test_all_keep_model_identity_is_tensor_exact_and_detects_changes() -> None:
    import copy

    import torch

    from search.stage2.lidar_pyramid_real_evaluator import _all_keep_model_identity

    original = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1), torch.nn.BatchNorm2d(3)).eval()
    physical = copy.deepcopy(original)
    assert _all_keep_model_identity(original, physical)["passed"] is True

    with torch.no_grad():
        physical[0].weight.view(-1)[0].add_(1.0)
    report = _all_keep_model_identity(original, physical)
    assert report["passed"] is False
    assert "0.weight" in report["mismatched_tensor_keys"]
