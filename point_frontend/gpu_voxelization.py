"""GPU hard-voxelization shared by CARLA and DAIR-V2X evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


def mathematical_voxel_capacity(
    lidar_range: Sequence[float], voxel_size: Sequence[float]
) -> int:
    """Return the complete grid capacity, independent of dataset voxel counts."""

    extent = np.asarray(lidar_range[3:6], dtype=np.float64) - np.asarray(
        lidar_range[:3], dtype=np.float64
    )
    grid = np.rint(extent / np.asarray(voxel_size, dtype=np.float64)).astype(
        np.int64
    )
    if np.any(grid <= 0):
        raise ValueError(f"invalid voxel grid: {grid.tolist()}")
    return int(np.prod(grid, dtype=np.int64))


def _preprocess_config(
    hypes: Mapping[str, Any], modality: str
) -> MutableMapping[str, Any]:
    try:
        config = hypes["heter"]["modality_setting"][modality]["preprocess"]
    except KeyError as exc:
        raise KeyError(f"missing LiDAR preprocess config for modality {modality}") from exc
    if str(config.get("core_method")) != "SpVoxelPreprocessor":
        raise ValueError(
            f"GPU voxelization requires SpVoxelPreprocessor, got {config.get('core_method')}"
        )
    return copy.deepcopy(config)


class DeferredRawPointPreprocessor:
    """Keep HEAL's shuffled/masked points raw until the CUDA evaluation process."""

    def preprocess(self, points: np.ndarray) -> Mapping[str, np.ndarray]:
        values = np.ascontiguousarray(points, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(f"expected point tensor [P,4], got {values.shape}")
        return {"raw_points": values}

    @staticmethod
    def collate_batch(batch: Any) -> Mapping[str, list[Any]]:
        import torch

        rows: list[Any] = []
        items = batch if isinstance(batch, list) else [batch]
        for item in items:
            if not isinstance(item, Mapping) or "raw_points" not in item:
                raise ValueError("deferred voxelization batch lacks raw_points")
            values = item["raw_points"]
            values = values if isinstance(values, list) else [values]
            for points in values:
                tensor = points if torch.is_tensor(points) else torch.from_numpy(points)
                rows.append(tensor.float().contiguous())
        if not rows:
            raise ValueError("deferred voxelization received no point clouds")
        return {"raw_points": rows}


@dataclass(frozen=True)
class GpuVoxelizationConfig:
    lidar_range: tuple[float, ...]
    voxel_size: tuple[float, ...]
    max_points_per_voxel: int
    point_feature_count: int
    grid_size_xyz: tuple[int, ...]
    per_agent_capacity: int


class DeterministicGpuVoxelizer:
    """Run variable-K hard voxelization with deterministic CUDA tensor ops."""

    def __init__(
        self,
        preprocess_config: Mapping[str, Any],
        device: Any,
    ) -> None:
        import torch

        if torch.device(device).type != "cuda":
            raise ValueError("DeterministicGpuVoxelizer requires a CUDA device")
        arguments = preprocess_config["args"]
        lidar_range = tuple(float(value) for value in preprocess_config["cav_lidar_range"])
        voxel_size = tuple(float(value) for value in arguments["voxel_size"])
        extent = np.asarray(lidar_range[3:6], dtype=np.float64) - np.asarray(
            lidar_range[:3], dtype=np.float64
        )
        grid_size_xyz = tuple(
            int(value)
            for value in np.rint(
                extent / np.asarray(voxel_size, dtype=np.float64)
            ).astype(np.int64)
        )
        self.config = GpuVoxelizationConfig(
            lidar_range=lidar_range,
            voxel_size=voxel_size,
            max_points_per_voxel=int(arguments["max_points_per_voxel"]),
            point_feature_count=int(arguments.get("num_point_features", 4)),
            grid_size_xyz=grid_size_xyz,
            per_agent_capacity=mathematical_voxel_capacity(lidar_range, voxel_size),
        )
        if self.config.point_feature_count != 4:
            raise ValueError(
                f"HEAL LiDAR GPU frontend requires four point features, got "
                f"{self.config.point_feature_count}"
            )
        self.device = torch.device(device)
        self._lower_xyz = torch.tensor(
            lidar_range[:3], dtype=torch.float32, device=self.device
        )
        self._voxel_size_xyz = torch.tensor(
            voxel_size, dtype=torch.float32, device=self.device
        )
        self._grid_size_xyz = torch.tensor(
            grid_size_xyz, dtype=torch.int64, device=self.device
        )

    @classmethod
    def from_hypes(
        cls,
        hypes: Mapping[str, Any],
        device: Any,
        *,
        modality: str = "m1",
    ) -> "DeterministicGpuVoxelizer":
        return cls(
            _preprocess_config(hypes, modality),
            device,
        )

    def runtime_contract(self) -> Mapping[str, Any]:
        return {
            "backend": "torch_cuda_deterministic_hard_voxel",
            "device": str(self.device),
            "lidar_range": list(self.config.lidar_range),
            "voxel_size": list(self.config.voxel_size),
            "grid_size_xyz": list(self.config.grid_size_xyz),
            "max_points_per_voxel": self.config.max_points_per_voxel,
            "per_agent_mathematical_capacity": self.config.per_agent_capacity,
            "dataset_derived_max_k": False,
            "deterministic_for_fixed_input_order": True,
            "saturated_voxel_point_selection": "input_order_first_n",
        }

    def _voxelize_one(self, values: Any) -> tuple[Any, Any, Any, Any]:
        import torch

        coordinates_xyz = torch.floor(
            (values[:, :3] - self._lower_xyz) / self._voxel_size_xyz
        ).to(torch.int64)
        valid = ((coordinates_xyz >= 0) & (
            coordinates_xyz < self._grid_size_xyz
        )).all(dim=1)
        point_indices = torch.nonzero(valid, as_tuple=False).flatten()
        coordinates_xyz = coordinates_xyz[point_indices]
        if int(point_indices.shape[0]) <= 0:
            raise RuntimeError("empty voxel cloud after range filtering")

        size_x, size_y, _ = self.config.grid_size_xyz
        voxel_ids = (
            coordinates_xyz[:, 0]
            + int(size_x)
            * (coordinates_xyz[:, 1] + int(size_y) * coordinates_xyz[:, 2])
        )
        point_count = int(voxel_ids.shape[0])
        # The second term makes every key unique, so CUDA sorting is independent
        # of tie ordering while retaining HEAL's input-order first-N rule.
        composite_keys = voxel_ids * (point_count + 1) + torch.arange(
            point_count, dtype=torch.int64, device=self.device
        )
        order = torch.argsort(composite_keys)
        sorted_voxel_ids = voxel_ids[order]
        unique_voxel_ids, full_counts = torch.unique_consecutive(
            sorted_voxel_ids, return_counts=True
        )
        voxel_count = int(unique_voxel_ids.shape[0])
        if voxel_count > self.config.per_agent_capacity:
            raise RuntimeError(
                "GPU voxelizer exceeded the mathematical grid capacity: "
                f"K={voxel_count}>{self.config.per_agent_capacity}"
            )

        starts = torch.cumsum(full_counts, dim=0) - full_counts
        ranks = torch.arange(
            point_count, dtype=torch.int64, device=self.device
        ) - torch.repeat_interleave(starts, full_counts)
        keep = ranks < self.config.max_points_per_voxel
        group_indices = torch.repeat_interleave(
            torch.arange(voxel_count, dtype=torch.int64, device=self.device),
            full_counts,
        )[keep]
        kept_ranks = ranks[keep]
        kept_points = values[point_indices[order[keep]]]
        voxels = torch.zeros(
            (
                voxel_count,
                self.config.max_points_per_voxel,
                self.config.point_feature_count,
            ),
            dtype=values.dtype,
            device=self.device,
        )
        voxels[group_indices, kept_ranks] = kept_points

        x = unique_voxel_ids.remainder(int(size_x))
        yz = torch.div(unique_voxel_ids, int(size_x), rounding_mode="floor")
        y = yz.remainder(int(size_y))
        z = torch.div(yz, int(size_y), rounding_mode="floor")
        coordinates_zyx = torch.stack((z, y, x), dim=1).to(torch.int32)
        counts = full_counts.clamp_max(self.config.max_points_per_voxel).to(
            torch.int32
        )
        return voxels, coordinates_zyx, counts, full_counts

    def voxelize(
        self, point_clouds: Sequence[Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], float]:
        import torch

        if not point_clouds:
            raise ValueError("GPU voxelizer requires at least one point cloud")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rows = []
        point_counts = []
        voxel_counts = []
        saturated_counts = []
        start.record()
        for agent_index, points in enumerate(point_clouds):
            if not torch.is_tensor(points):
                raise TypeError("GPU voxelizer expects point clouds as CUDA tensors")
            values = points.to(device=self.device, dtype=torch.float32).contiguous()
            if values.ndim != 2 or int(values.shape[1]) != self.config.point_feature_count:
                raise ValueError(f"expected point tensor [P,4], got {tuple(values.shape)}")
            if not values.is_cuda:
                raise RuntimeError("point cloud did not reach CUDA before voxelization")
            voxels, coordinates, counts, full_counts = self._voxelize_one(values)
            voxel_count = int(voxels.shape[0])
            if voxel_count <= 0:
                raise RuntimeError(f"empty voxel cloud for agent {agent_index}")
            prefix = torch.full(
                (voxel_count, 1),
                int(agent_index),
                dtype=coordinates.dtype,
                device=self.device,
            )
            rows.append((voxels, torch.cat((prefix, coordinates), dim=1), counts))
            point_counts.append(int(values.shape[0]))
            voxel_counts.append(voxel_count)
            saturated_counts.append(
                int(
                    (full_counts >= self.config.max_points_per_voxel)
                    .sum()
                    .item()
                )
            )
        batch = {
            "voxel_features": torch.cat([row[0] for row in rows], dim=0).contiguous(),
            "voxel_coords": torch.cat([row[1] for row in rows], dim=0)
            .to(torch.int32)
            .contiguous(),
            "voxel_num_points": torch.cat([row[2] for row in rows], dim=0)
            .to(torch.int32)
            .contiguous(),
        }
        end.record()
        end.synchronize()
        audit = {
            "input_point_counts": point_counts,
            "voxel_counts": voxel_counts,
            "total_voxel_count": int(sum(voxel_counts)),
            "saturated_voxel_counts": saturated_counts,
            "total_saturated_voxel_count": int(sum(saturated_counts)),
            "per_agent_mathematical_capacity": self.config.per_agent_capacity,
        }
        return batch, audit, float(start.elapsed_time(end))


def defer_dataset_voxelization(
    dataset: Any,
    hypes: Mapping[str, Any],
    *,
    modality: str = "m1",
) -> Mapping[str, Any]:
    """Replace one HEAL CPU voxelizer with a raw-point collation boundary."""

    _preprocess_config(hypes, modality)
    sensor_types = getattr(dataset, "sensor_type_dict", {})
    if sensor_types and str(sensor_types.get(modality)) != "lidar":
        raise ValueError(f"modality {modality} is not a LiDAR endpoint")
    attribute = f"pre_processor_{modality}"
    if not hasattr(dataset, attribute):
        raise AttributeError(f"HEAL dataset lacks {attribute}")
    setattr(dataset, attribute, DeferredRawPointPreprocessor())
    return {
        "deferred": True,
        "modality": modality,
        "dataset_preprocessor_attribute": attribute,
        "cpu_voxelization_in_dataloader": False,
    }


def voxelize_ego_batch(
    ego_batch: MutableMapping[str, Any],
    voxelizer: DeterministicGpuVoxelizer,
    *,
    modality: str = "m1",
) -> tuple[Mapping[str, Any], float]:
    """Materialize deferred raw points into the standard HEAL batch contract."""

    key = f"inputs_{modality}"
    source = ego_batch.get(key)
    if not isinstance(source, Mapping) or "raw_points" not in source:
        raise RuntimeError(f"HEAL batch lacks deferred raw points at {key}")
    raw_points = source["raw_points"]
    if not isinstance(raw_points, Sequence):
        raise TypeError("deferred raw_points must be a sequence")
    record_len = int(ego_batch["record_len"][0].item())
    if len(raw_points) != record_len:
        raise RuntimeError(
            f"raw point agent count mismatch: {len(raw_points)} != {record_len}"
        )
    voxel_batch, audit, elapsed_ms = voxelizer.voxelize(raw_points)
    ego_batch[key] = voxel_batch
    return audit, elapsed_ms
