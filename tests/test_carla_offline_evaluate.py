from carla_integration.offline_evaluate import (
    _distribution,
    _postprocess_outputs_fp32,
    mathematical_voxel_capacity,
)


def test_voxel_capacity_uses_complete_grid_not_dataset_max_k():
    capacity = mathematical_voxel_capacity(
        [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5], [0.4, 0.4, 5.0]
    )
    assert capacity == 131072
    assert capacity != 29696


def test_latency_distribution():
    result = _distribution([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["p50"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0


def test_mixed_precision_detection_heads_are_normalized_for_postprocess():
    import torch

    outputs = {
        "cls_preds": torch.ones(1, dtype=torch.float16),
        "reg_preds": torch.ones(1, dtype=torch.float16),
        "dir_preds": torch.ones(1, dtype=torch.float32),
    }
    normalized = _postprocess_outputs_fp32(outputs)
    assert set(normalized) == set(outputs)
    assert all(value.dtype == torch.float32 for value in normalized.values())
