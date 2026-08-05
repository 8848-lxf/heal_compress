"""Coordinate conversions between CARLA and the DAIR-trained HEAL model."""

from __future__ import annotations

from typing import Iterable

import numpy as np


HANDEDNESS = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def _matrix4(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("transform contains non-finite values")
    return matrix


def carla_transform_to_right_handed(carla_matrix: object) -> np.ndarray:
    """Convert a CARLA local-to-world matrix from left- to right-handed axes."""

    matrix = _matrix4(carla_matrix)
    return HANDEDNESS @ matrix @ HANDEDNESS


def canonical_sensor_to_world(carla_matrix: object, z_offset: float) -> np.ndarray:
    """Return the canonical sensor-to-world transform used by HEAL.

    CARLA points are first converted with ``y = -y`` and then shifted by
    ``z_offset = physical_height_above_road - canonical_DAIR_height``.  The
    returned transform maps those shifted points back into right-handed world
    coordinates.  Using a local offset avoids coupling point height to a
    CARLA map's absolute world elevation.
    """

    raw_sensor_to_world = carla_transform_to_right_handed(carla_matrix)
    canonical_to_raw = np.eye(4, dtype=np.float64)
    canonical_to_raw[2, 3] = -float(z_offset)
    return raw_sensor_to_world @ canonical_to_raw


def canonicalize_lidar(points: object, z_offset: float) -> np.ndarray:
    """Convert CARLA ``[x,y,z,intensity]`` points to a DAIR-like local frame."""

    source = np.asarray(points)
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError(f"expected Nx3 or Nx4 point cloud, got {source.shape}")
    result = source.astype(np.float32, copy=True)
    result[:, 1] *= -1.0
    result[:, 2] += float(z_offset)
    if not np.isfinite(result).all():
        raise ValueError("point cloud contains non-finite values")
    return result


def pairwise_transforms(sensor_to_world: Iterable[object]) -> np.ndarray:
    """Build ``pairwise[i,j] = T_j_world^-1 T_i_world`` (agent i to j)."""

    transforms = [_matrix4(value) for value in sensor_to_world]
    if not transforms:
        raise ValueError("at least one sensor transform is required")
    count = len(transforms)
    result = np.empty((count, count, 4, 4), dtype=np.float32)
    for source in range(count):
        for target in range(count):
            result[source, target] = np.linalg.solve(
                transforms[target], transforms[source]
            ).astype(np.float32)
    return result


def right_handed_world_points_to_sensor(
    world_points: object,
    sensor_to_world: object,
) -> np.ndarray:
    """Project right-handed world XYZ points into a canonical sensor frame."""

    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 world points, got {points.shape}")
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    projected = np.linalg.solve(_matrix4(sensor_to_world), homogeneous.T).T
    return projected[:, :3].astype(np.float32)


def boxes_to_corners(boxes: object) -> np.ndarray:
    """Convert ``[x,y,z,l,w,h,yaw]`` boxes to HEAL-compatible corners."""

    values = np.asarray(boxes, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"expected Nx7 boxes, got {values.shape}")
    template = np.asarray(
        [
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
        ],
        dtype=np.float32,
    ) / 2.0
    local = values[:, None, 3:6] * template[None, :, :]
    cosine = np.cos(values[:, 6])
    sine = np.sin(values[:, 6])
    x = local[:, :, 0].copy()
    y = local[:, :, 1].copy()
    local[:, :, 0] = x * cosine[:, None] - y * sine[:, None]
    local[:, :, 1] = x * sine[:, None] + y * cosine[:, None]
    return local + values[:, None, :3]


def transform_points(points: object, source_to_target: object) -> np.ndarray:
    """Transform point XYZ with a homogeneous source-to-target matrix."""

    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"expected Nx3 or Nx4 points, got {values.shape}")
    homogeneous = np.concatenate(
        [values[:, :3].astype(np.float64), np.ones((values.shape[0], 1))], axis=1
    )
    return (homogeneous @ _matrix4(source_to_target).T)[:, :3].astype(np.float32)


def point_counts_in_boxes(points: object, boxes: object) -> np.ndarray:
    """Count points inside oriented boxes represented by HEAL's eight corners."""

    xyz = np.asarray(points, dtype=np.float32)
    corners = np.asarray(boxes, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected Nx3 points, got {xyz.shape}")
    if corners.size == 0:
        return np.empty((0,), dtype=np.int64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError(f"expected Mx8x3 boxes, got {corners.shape}")
    result = np.empty(corners.shape[0], dtype=np.int64)
    for index, box in enumerate(corners):
        origin = box[0]
        edges = np.stack([box[3] - origin, box[1] - origin, box[4] - origin])
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= 0):
            raise ValueError(f"degenerate box at index {index}")
        projected = (xyz - origin) @ (edges / lengths[:, None]).T
        inside = np.logical_and(
            projected >= -1e-4, projected <= lengths + 1e-4
        ).all(axis=1)
        result[index] = int(inside.sum())
    return result
