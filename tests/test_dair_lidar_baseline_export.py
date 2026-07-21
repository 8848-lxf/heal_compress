from __future__ import annotations

import torch


def _ego(record_len: int) -> dict:
    count = 3
    return {
        "inputs_m1": {
            "voxel_features": torch.arange(count * 4, dtype=torch.float32).reshape(count, 1, 4),
            "voxel_coords": torch.tensor(
                [[0, 0, 0, 0], [0, 0, 0, 1], [max(0, record_len - 1), 0, 1, 0]],
                dtype=torch.int64,
            ),
            "voxel_num_points": torch.ones(count, dtype=torch.int64),
        },
        "record_len": torch.tensor([record_len]),
        "pairwise_t_matrix": torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(
            1, record_len, record_len, 1, 1
        ),
    }


def test_prepare_baseline_inputs_freezes_k_and_masks_agents() -> None:
    from search.model_family.export.heal_lidar_baselines import (
        HealLidarBaselineExportPolicy,
        prepare_heal_lidar_baseline_inputs,
    )

    policy = HealLidarBaselineExportPolicy(fixed_k=8, max_agents=2)
    prepared = prepare_heal_lidar_baseline_inputs(_ego(1), policy=policy)
    assert tuple(prepared) == (
        "voxel_features",
        "voxel_coords",
        "voxel_num_points",
        "pairwise_t_matrix",
        "valid_voxel_mask",
        "agent_mask",
    )
    assert prepared["voxel_features"].shape == (8, 1, 4)
    assert prepared["pairwise_t_matrix"].shape == (1, 2, 2, 4, 4)
    assert prepared["valid_voxel_mask"].sum().item() == 3
    assert prepared["agent_mask"].tolist() == [[1.0, 0.0]]


def test_masked_max_and_attention_ignore_padded_agent() -> None:
    from search.model_family.export.heal_lidar_baselines import _attention_fuse, _max_fuse

    valid = torch.tensor([1.0, 0.0])
    feature = torch.tensor([[[[2.0]], [[3.0]]], [[[100.0]], [[100.0]]]])
    max_fused = _max_fuse(feature, valid)
    assert torch.equal(max_fused, feature[:1])

    padded = feature.clone()
    padded[1] = 1000.0
    attention = _attention_fuse(padded, valid)
    assert torch.allclose(attention, feature[:1])


def test_baseline_policy_rejects_invalid_contract() -> None:
    from search.model_family.export.heal_lidar_baselines import HealLidarBaselineExportPolicy

    for kwargs in ({"fixed_k": 0}, {"fixed_k": 8, "max_agents": 0}):
        try:
            HealLidarBaselineExportPolicy(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid baseline export policy was accepted")


def test_physical_snapshot_uses_explicit_model_family() -> None:
    from search.model_family.deployment import build_physical_structure_snapshot_v2

    snapshot = build_physical_structure_snapshot_v2(
        torch.nn.Sequential(torch.nn.Linear(4, 3)).eval(),
        model_family="lidar_test_family",
    )
    assert snapshot["model_family"] == "lidar_test_family"
    assert snapshot["modules"][0]["weight_shape"] == [3, 4]
