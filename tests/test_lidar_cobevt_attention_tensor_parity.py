from __future__ import annotations

import math
from pathlib import Path

import torch


def _make_diagnostic_source(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["q"], name="q_node"),
            helper.make_node("Identity", ["q"], ["logits"], name="qk_node"),
            helper.make_node("Identity", ["logits"], ["output"], name="out_node"),
        ],
        "attention_diagnostic",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        value_info=[
            helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4]),
        ],
    )
    onnx.save(helper.make_model(graph), str(path))


def test_append_attention_diagnostic_outputs_preserves_source_and_tags_outputs(
    tmp_path: Path,
):
    import hashlib
    import onnx

    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        append_attention_diagnostic_outputs,
    )

    source = tmp_path / "typed.onnx"
    destination = tmp_path / "diagnostic.onnx"
    _make_diagnostic_source(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    report = append_attention_diagnostic_outputs(
        source,
        destination,
        [
            {"block_id": "layers.0.window", "role": "q_projection", "tensor_name": "q"},
            {"block_id": "layers.0.window", "role": "qk_matmul", "tensor_name": "logits"},
        ],
    )

    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert source != destination
    assert report["diagnostic_latency_invalid"] is True
    assert report["source_onnx_sha256"] == source_hash
    assert [row["tensor_name"] for row in report["outputs"]] == ["q", "logits"]
    assert [value.name for value in onnx.load(str(destination)).graph.output] == [
        "output",
        "q",
        "logits",
    ]


def test_append_attention_diagnostic_outputs_fails_closed_for_missing_tensor(
    tmp_path: Path,
):
    import pytest

    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        append_attention_diagnostic_outputs,
    )

    source = tmp_path / "typed.onnx"
    _make_diagnostic_source(source)

    with pytest.raises(ValueError, match="attention_diagnostic_tensor_missing"):
        append_attention_diagnostic_outputs(
            source,
            tmp_path / "diagnostic.onnx",
            [{"block_id": "layers.0.window", "role": "softmax", "tensor_name": "missing"}],
        )


def test_append_attention_diagnostic_outputs_validates_parser_compatible_scatter(
    tmp_path: Path,
):
    import onnx
    from onnx import TensorProto, helper

    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        append_attention_diagnostic_outputs,
    )

    source = tmp_path / "typed_parser.onnx"
    destination = tmp_path / "diagnostic.onnx"
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "PointPillarScatterTRT",
                    ["features", "coordinates", "record_len"],
                    ["scatter"],
                    name="scatter",
                ),
                helper.make_node("Identity", ["scatter"], ["output"], name="output"),
            ],
            "parser_compatible_scatter",
            [
                helper.make_tensor_value_info("features", TensorProto.FLOAT, [4, 64]),
                helper.make_tensor_value_info("coordinates", TensorProto.INT32, [4, 4]),
                helper.make_tensor_value_info("record_len", TensorProto.INT32, [1]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 2, 2])],
            value_info=[
                helper.make_tensor_value_info("scatter", TensorProto.FLOAT, [1, 64, 2, 2])
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, str(source))

    report = append_attention_diagnostic_outputs(
        source,
        destination,
        [{"block_id": "global", "role": "scatter", "tensor_name": "scatter"}],
    )

    saved = onnx.load(str(destination))
    scatter = next(node for node in saved.graph.node if node.op_type == "PointPillarScatterTRT")
    assert scatter.domain == ""
    assert report["known_custom_op_schema_bypass"] == ["PointPillarScatterTRT"]


def test_attention_diagnostic_output_specs_include_softmax_input_and_public_outputs():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        attention_diagnostic_output_specs,
    )

    records = []
    for role in (
        "layernorm",
        "q_projection",
        "k_projection",
        "v_projection",
        "qk_matmul",
        "softmax",
        "av_matmul",
        "output_projection",
        "residual_add",
    ):
        records.append(
            {
                "block_id": "layers.0.window_attention",
                "role": role,
                "input_tensors_before": (
                    ["attention_update", "residual_input"]
                    if role == "residual_add"
                    else [f"{role}_input"]
                ),
                "output_tensors_before": [f"{role}_output"],
            }
        )

    specs = attention_diagnostic_output_specs(
        {"node_records": records},
        fused_bev_tensor="fused_bev",
        head_input_tensor="head_input",
    )

    by_role = {row["role"]: row["tensor_name"] for row in specs}
    assert by_role["layernorm"] == "layernorm_output"
    assert by_role["q_projection"] == "q_projection_output"
    assert by_role["scaled_qk_logits"] == "softmax_input"
    assert by_role["softmax"] == "softmax_output"
    assert by_role["residual_attention_update"] == "attention_update"
    assert by_role["residual_input"] == "residual_input"
    assert by_role["residual_add"] == "residual_add_output"
    assert by_role["fused_bev"] == "fused_bev"
    assert by_role["head_input"] == "head_input"


def test_attention_diagnostic_output_specs_fail_closed_for_incomplete_role_record():
    import pytest

    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        attention_diagnostic_output_specs,
    )

    with pytest.raises(ValueError, match="attention_diagnostic_role_incomplete"):
        attention_diagnostic_output_specs(
            {
                "node_records": [
                    {
                        "block_id": "layers.0.window_attention",
                        "role": "softmax",
                        "input_tensors_before": [],
                        "output_tensors_before": ["softmax_output"],
                    }
                ]
            }
        )


def test_attention_diagnostic_shards_cover_candidate_outputs_once_and_keep_residual_aux_reference_only():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        attention_diagnostic_output_shards,
    )

    roles = (
        "layernorm",
        "q_projection",
        "k_projection",
        "v_projection",
        "qk_matmul",
        "scaled_qk_logits",
        "softmax",
        "av_matmul",
        "output_projection",
        "residual_attention_update",
        "residual_input",
        "residual_add",
        "fused_bev",
        "head_input",
    )
    specs = [
        {"block_id": "layers.0.window_attention", "role": role, "tensor_name": role}
        for role in roles
    ]

    shards = attention_diagnostic_output_shards(specs)
    candidate_roles = [
        row["role"] for shard in shards for row in shard["output_specs"]
    ]
    reference_only_roles = [
        row["role"] for shard in shards for row in shard["reference_only_specs"]
    ]

    assert [shard["shard_id"] for shard in shards] == ["pre", "qk", "post"]
    assert len(candidate_roles) == len(set(candidate_roles)) == 12
    assert set(reference_only_roles) == {
        "residual_attention_update",
        "residual_input",
    }
    assert "residual_add" in candidate_roles
    assert not (set(candidate_roles) & set(reference_only_roles))


def test_tensor_error_metrics_are_exact_for_identical_and_shifted_values():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        tensor_error_metrics,
    )

    reference = torch.tensor([[1.0, 2.0, 3.0]])
    identical = tensor_error_metrics(reference, reference.clone())
    shifted = tensor_error_metrics(reference, reference + 1.0)

    assert identical["cosine_similarity"] == 1.0
    assert identical["relative_l2_error"] == 0.0
    assert identical["maximum_absolute_error"] == 0.0
    assert shifted["maximum_absolute_error"] == 1.0
    assert shifted["mean_absolute_error"] == 1.0
    assert shifted["reference_dtype"] == "torch.float32"
    assert shifted["candidate_dtype"] == "torch.float32"


def test_metric_accumulation_stays_on_cuda_and_uses_float32():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        metric_accumulation_spec,
    )

    cuda_spec = metric_accumulation_spec("cuda")
    cpu_spec = metric_accumulation_spec("cpu")

    assert cuda_spec == {"device_policy": "preserve", "dtype": torch.float32}
    assert cpu_spec == {"device_policy": "preserve", "dtype": torch.float64}


def test_tensor_error_metrics_count_nonfinite_values_without_hiding_them():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        tensor_error_metrics,
    )

    metrics = tensor_error_metrics(
        torch.tensor([1.0, float("nan"), float("inf")]),
        torch.tensor([1.0, 2.0, float("-inf")]),
    )

    assert metrics["reference_nan_count"] == 1
    assert metrics["reference_inf_count"] == 1
    assert metrics["candidate_nan_count"] == 0
    assert metrics["candidate_inf_count"] == 1
    assert metrics["finite"] is False


def test_qk_metrics_detect_argmax_rank_sign_and_topk_changes():
    from search.model_families.lidar_cobevt.attention_tensor_parity import qk_metrics

    reference = torch.tensor([[4.0, 3.0, 2.0, -1.0]])
    candidate = torch.tensor([[3.0, 4.0, -2.0, -1.0]])

    metrics = qk_metrics(reference, candidate, topk_values=(1, 2, 4))

    assert metrics["top1_index_agreement"] == 0.0
    assert metrics["topk_overlap_1"] == 0.0
    assert metrics["topk_overlap_2"] == 1.0
    assert metrics["topk_overlap_4"] == 1.0
    assert metrics["sign_flip_ratio"] == 0.25
    assert metrics["row_max_absolute_error"] == 0.0
    assert metrics["row_rank_correlation"] < 1.0


def test_softmax_metrics_report_normalization_entropy_and_divergence():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        softmax_metrics,
    )

    reference = torch.tensor([[0.7, 0.2, 0.1]])
    candidate = torch.tensor([[0.6, 0.3, 0.1]])

    metrics = softmax_metrics(reference, candidate, topk_values=(1, 2))

    assert metrics["reference_row_sum_max_deviation"] < 1e-7
    assert metrics["candidate_row_sum_max_deviation"] < 1e-7
    assert metrics["attention_argmax_agreement"] == 1.0
    assert metrics["topk_overlap_1"] == 1.0
    assert metrics["kl_fp32_to_candidate"] > 0.0
    assert metrics["js_divergence"] > 0.0
    assert math.isfinite(metrics["entropy_delta"])


def test_residual_metrics_measure_small_update_retention():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        residual_metrics,
    )

    residual = torch.tensor([1000.0, 1000.0])
    update = torch.tensor([0.1, 0.2])
    fp32_output = residual + update
    candidate_output = residual.clone()

    metrics = residual_metrics(residual, update, fp32_output, candidate_output)

    assert metrics["attention_to_residual_l2_ratio"] < 0.001
    assert metrics["candidate_update_retention_ratio"] == 0.0
    assert metrics["candidate_output_equals_residual_ratio"] == 1.0
    assert metrics["residual_output_relative_l2_error"] > 0.0


def test_failure_frame_selection_is_stable_for_equal_errors():
    from search.model_families.lidar_cobevt.attention_tensor_parity import (
        select_failure_frames,
    )

    rows = [
        {"frame_id": "b", "maximum_absolute_error": 2.0},
        {"frame_id": "a", "maximum_absolute_error": 2.0},
        {"frame_id": "c", "maximum_absolute_error": 1.0},
        {"frame_id": "d", "maximum_absolute_error": 3.0},
    ]

    assert [row["frame_id"] for row in select_failure_frames(rows, count=3)] == [
        "d",
        "a",
        "b",
    ]
