import numpy as np

from carla_integration.intensity_calibration import (
    IntensityCalibration,
    _Histogram,
    filtered_intensity,
)


def test_histogram_quantiles_and_endpoint_map_are_monotonic():
    histogram = _Histogram(1024)
    histogram.add(np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    quantiles = histogram.quantiles(np.asarray([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(quantiles[[0, 2]], [0.0, 1.0])
    assert 0.49 < quantiles[1] < 0.51

    calibration = IntensityCalibration(
        {
            "schema_version": "heal-carla-intensity-calibration-v1",
            "endpoints": {
                "vehicle": {
                    "source_quantiles": [0.0, 0.5, 1.0],
                    "target_quantiles": [0.0, 0.1, 0.2],
                },
                "infrastructure": {
                    "source_quantiles": [0.0, 0.5, 1.0],
                    "target_quantiles": [0.0, 0.01, 0.02],
                },
            },
        }
    )
    points = np.asarray([[0, 0, 0, 0.25], [0, 0, 0, 0.75]], dtype=np.float32)
    mapped = calibration.apply(points, "vehicle")
    np.testing.assert_allclose(mapped[:, 3], [0.05, 0.15], atol=1e-6)
    np.testing.assert_array_equal(points[:, 3], [0.25, 0.75])


def test_filtered_intensity_uses_roi_and_vehicle_ego_mask():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.1],
            [5.0, 0.0, 0.0, 0.2],
            [50.0, 0.0, 0.0, 0.3],
        ],
        dtype=np.float32,
    )
    limits = [-10, -10, -2, 10, 10, 2]
    np.testing.assert_allclose(
        filtered_intensity(points, limits, remove_ego=True), [0.2]
    )
    np.testing.assert_allclose(
        filtered_intensity(points, limits, remove_ego=False), [0.1, 0.2]
    )
