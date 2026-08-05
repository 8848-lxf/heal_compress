from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from point_frontend.gpu_voxelization import (
    DeferredRawPointPreprocessor,
    DeterministicGpuVoxelizer,
    defer_dataset_voxelization,
    mathematical_voxel_capacity,
    voxelize_ego_batch,
)


def _hypes():
    return {
        "heter": {
            "modality_setting": {
                "m1": {
                    "preprocess": {
                        "core_method": "SpVoxelPreprocessor",
                        "cav_lidar_range": [
                            -102.4,
                            -51.2,
                            -3.5,
                            102.4,
                            51.2,
                            1.5,
                        ],
                        "args": {
                            "voxel_size": [0.4, 0.4, 5.0],
                            "max_points_per_voxel": 32,
                        },
                    }
                }
            }
        }
    }


def test_deferred_preprocessor_preserves_agent_point_tensors():
    processor = DeferredRawPointPreprocessor()
    first = processor.preprocess(np.ones((3, 4), dtype=np.float32))
    second = processor.preprocess(np.zeros((2, 4), dtype=np.float32))
    result = processor.collate_batch({"raw_points": [first["raw_points"], second["raw_points"]]})
    assert len(result["raw_points"]) == 2
    assert all(torch.is_tensor(value) for value in result["raw_points"])
    assert [tuple(value.shape) for value in result["raw_points"]] == [(3, 4), (2, 4)]


def test_dataset_defer_replaces_only_requested_lidar_preprocessor():
    original = object()
    dataset = SimpleNamespace(
        sensor_type_dict={"m1": "lidar"}, pre_processor_m1=original
    )
    contract = defer_dataset_voxelization(dataset, _hypes(), modality="m1")
    assert isinstance(dataset.pre_processor_m1, DeferredRawPointPreprocessor)
    assert contract["cpu_voxelization_in_dataloader"] is False
    assert mathematical_voxel_capacity(
        [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5], [0.4, 0.4, 5.0]
    ) == 131072


def test_voxelize_ego_batch_restores_standard_heal_contract():
    expected = {
        "voxel_features": torch.ones((2, 32, 4)),
        "voxel_coords": torch.zeros((2, 4), dtype=torch.int32),
        "voxel_num_points": torch.ones((2,), dtype=torch.int32),
    }

    class FakeVoxelizer:
        def voxelize(self, rows):
            assert len(rows) == 2
            return expected, {
                "total_voxel_count": 2,
                "total_saturated_voxel_count": 0,
            }, 0.25

    ego = {
        "record_len": torch.tensor([2]),
        "inputs_m1": {"raw_points": [torch.ones((3, 4)), torch.ones((4, 4))]},
    }
    audit, elapsed = voxelize_ego_batch(ego, FakeVoxelizer(), modality="m1")
    assert ego["inputs_m1"] is expected
    assert audit["total_voxel_count"] == 2
    assert elapsed == 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_voxelizer_is_deterministic_for_saturated_full_grid():
    config = {
        "core_method": "SpVoxelPreprocessor",
        "cav_lidar_range": [0, 0, 0, 2, 2, 1],
        "args": {
            "voxel_size": [1, 1, 1],
            "max_points_per_voxel": 32,
            "num_point_features": 4,
        },
    }
    rows = []
    expected = {}
    for point_index in range(40):
        for y in range(2):
            for x in range(2):
                intensity = float(1000 * y + 100 * x + point_index)
                rows.append([x + 0.1, y + 0.1, 0.1, intensity])
                expected.setdefault((0, y, x), []).append(intensity)
    points = torch.tensor(rows, dtype=torch.float32, device="cuda")
    voxelizer = DeterministicGpuVoxelizer(config, "cuda:0")

    outputs = [voxelizer.voxelize([points]) for _ in range(5)]
    reference = outputs[0][0]
    for batch, audit, _elapsed in outputs:
        assert audit["total_voxel_count"] == 4
        assert audit["total_saturated_voxel_count"] == 4
        assert torch.equal(reference["voxel_coords"], batch["voxel_coords"])
        assert torch.equal(reference["voxel_features"], batch["voxel_features"])
        assert torch.equal(reference["voxel_num_points"], batch["voxel_num_points"])

    for voxel_index, coordinate in enumerate(reference["voxel_coords"].cpu()):
        key = tuple(int(value) for value in coordinate[1:])
        selected = reference["voxel_features"][voxel_index, :, 3].cpu().tolist()
        assert selected == expected[key][:32]
