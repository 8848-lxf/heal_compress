from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from search.model_families.lidar_cobevt.attention_dim_pruning import (
    PrunableCobevtAttention,
)


def _attention() -> PrunableCobevtAttention:
    module = PrunableCobevtAttention(
        embed_dim=8,
        heads=2,
        d_qk=4,
        d_v=4,
        window_size=(2, 2),
        relative_position_rows=64,
    ).eval()
    module.relative_position_index = torch.arange(64).reshape(8, 8)
    return module


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(1, 2, 1, 1, 2, 2, 8)
    mask = torch.ones(1, 1, 1, 2, 2, 1, 2, dtype=torch.bool)
    return x, mask


def test_microbenchmark_specs_cover_uniform_and_qk_only_widths():
    from search.model_families.lidar_cobevt.attention_microbenchmark import (
        attention_microbenchmark_specs,
    )

    pairs = {(row.d_qk, row.d_v) for row in attention_microbenchmark_specs()}

    assert {(width, width) for width in (8, 12, 16, 20, 24, 28, 32)} <= pairs
    assert {(24, 32), (16, 32), (32, 16)} <= pairs


def test_explicit_int8_attention_export_contains_real_qdq(tmp_path: Path):
    import onnx

    from search.model_families.lidar_cobevt.attention_microbenchmark import (
        ExplicitQDQAttentionGraph,
    )

    x, mask = _inputs()
    graph = ExplicitQDQAttentionGraph.from_attention(
        _attention(), use_mask_rpe=True
    ).eval()
    graph.calibrate(x, mask)
    output = graph(x, mask)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    destination = tmp_path / "attention_qdq.onnx"
    torch.onnx.export(
        graph,
        (x, mask),
        destination,
        input_names=("activation", "attention_mask"),
        output_names=("output",),
        opset_version=17,
    )
    model = onnx.load(str(destination))
    node_types = [node.op_type for node in model.graph.node]

    assert node_types.count("QuantizeLinear") >= 9
    assert node_types.count("DequantizeLinear") >= 9
    assert "Softmax" in node_types


def test_tensorrt_runtime_loader_fails_closed_when_library_is_missing(
    tmp_path: Path,
):
    from search.orchestration.lidar_cobevt_attention_microbenchmark import (
        load_tensorrt_runtime,
    )

    with pytest.raises(FileNotFoundError, match="libnvinfer.so.10"):
        load_tensorrt_runtime(tmp_path)


def test_engine_audit_recognizes_mha_fusion_and_metadata_nodes(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_microbenchmark import (
        audit_attention_engine,
    )

    layer_info = tmp_path / "layer_info.json"
    layer_info.write_text(
        json.dumps(
            {
                "Layers": [
                    {
                        "Name": "_gemm_mha_v2_myl0_4",
                        "LayerType": "kgen",
                        "TacticName": "_gemm_mha_v2_tactic",
                        "Metadata": (
                            "[ONNX Layer: /qk_matmul/MatMul]"
                            "[ONNX Layer: /softmax/Softmax]"
                            "[ONNX Layer: /av_matmul/MatMul]"
                        ),
                        "Inputs": [{"Format/Datatype": "Int8"}],
                        "Outputs": [{"Format/Datatype": "Int8"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = audit_attention_engine(layer_info)

    assert audit["mha_fused"] is True
    assert audit["qk_matmul_precision_layer_count"] == 1
    assert audit["softmax_precision_layer_count"] == 1
    assert audit["av_matmul_precision_layer_count"] == 1
    assert audit["qk_matmul_precision"] == ["Int8"]
