from __future__ import annotations

from pathlib import Path

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
