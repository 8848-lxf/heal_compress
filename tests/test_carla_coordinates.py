import numpy as np

from carla_integration.coordinates import (
    HANDEDNESS,
    boxes_to_corners,
    canonical_sensor_to_world,
    canonicalize_lidar,
    pairwise_transforms,
    point_counts_in_boxes,
    transform_points,
)
from carla_integration.collect_scenes import _crop_horizontal_fov


def _transform(x=0.0, y=0.0, z=0.0, yaw_degrees=0.0):
    yaw = np.deg2rad(yaw_degrees)
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def test_canonical_points_and_transform_preserve_world_position():
    carla_sensor_to_world = _transform(10.0, 4.0, 8.0, 30.0)
    raw = np.asarray([[2.0, 3.0, -7.5, 0.8]], dtype=np.float32)
    offset = 7.5 - 1.675
    canonical = canonicalize_lidar(raw, offset)
    canonical_to_world = canonical_sensor_to_world(carla_sensor_to_world, offset)

    canonical_world = canonical_to_world @ np.r_[canonical[0, :3], 1.0]
    raw_world_lh = carla_sensor_to_world @ np.r_[raw[0, :3], 1.0]
    expected_world_rh = HANDEDNESS @ raw_world_lh
    np.testing.assert_allclose(canonical_world, expected_world_rh, atol=1e-6)


def test_pairwise_transform_maps_source_point_to_target():
    first = canonical_sensor_to_world(_transform(5.0, 2.0, 2.0, 15.0), 0.0)
    second = canonical_sensor_to_world(_transform(-3.0, 7.0, 7.5, -20.0), 5.825)
    pairwise = pairwise_transforms([first, second])
    point_first = np.asarray([3.0, -1.0, 0.2, 1.0])
    world = first @ point_first
    expected_second = np.linalg.solve(second, world)
    np.testing.assert_allclose(pairwise[0, 1] @ point_first, expected_second, atol=1e-5)
    np.testing.assert_allclose(pairwise[0, 0], np.eye(4), atol=1e-6)


def test_boxes_to_corners_uses_heal_perimeter_order():
    corners = boxes_to_corners([[0, 0, 0, 4, 2, 2, 0]])
    expected_bottom = np.asarray([[2, -1], [2, 1], [-2, 1], [-2, -1]])
    np.testing.assert_allclose(corners[0, :4, :2], expected_bottom)
    np.testing.assert_allclose(corners[0, :4, 2], -1)
    np.testing.assert_allclose(corners[0, 4:, 2], 1)


def test_software_horizontal_fov_crop_keeps_forward_sector():
    points = np.asarray(
        [[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1], [-1, 0, 0, 1]],
        dtype=np.float32,
    )
    cropped = _crop_horizontal_fov(points, 100.0)
    np.testing.assert_array_equal(cropped, points[:2])


def test_transform_and_visible_point_counts_for_rotated_box():
    matrix = _transform(3.0, -2.0, 1.0, 90.0)
    transformed = transform_points([[1.0, 0.0, 0.0]], matrix)
    np.testing.assert_allclose(transformed, [[3.0, -1.0, 1.0]], atol=1e-6)

    box = boxes_to_corners([[0, 0, 0, 4, 2, 2, np.pi / 4]])
    points = np.asarray(
        [[0, 0, 0], [1, 0, 0], [4, 4, 0], [0, 0, 2]], dtype=np.float32
    )
    np.testing.assert_array_equal(point_counts_in_boxes(points, box), [2])
