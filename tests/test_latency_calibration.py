from __future__ import annotations

from opencood.tools.compression.latency_lut.calibration import (
    CalibrationSample,
    IdentityCalibrationModel,
    LinearCalibrationModel,
)


def test_identity_calibration_returns_raw_latency():
    model = IdentityCalibrationModel()
    assert model.predict(3.2, {"num_units": 2}) == 3.2


def test_linear_calibration_fits_simple_samples():
    samples = [
        CalibrationSample(candidate_id="a", predicted_lut_ms=1.0, real_engine_p50_ms=2.0, features={"num_units": 1}),
        CalibrationSample(candidate_id="b", predicted_lut_ms=2.0, real_engine_p50_ms=4.0, features={"num_units": 2}),
        CalibrationSample(candidate_id="c", predicted_lut_ms=3.0, real_engine_p50_ms=6.0, features={"num_units": 3}),
    ]
    model = LinearCalibrationModel.fit(samples)

    assert abs(model.predict(4.0, {"num_units": 4}) - 8.0) < 1e-6
    payload = model.to_dict()
    restored = LinearCalibrationModel.from_dict(payload)
    assert abs(restored.predict(4.0, {"num_units": 4}) - 8.0) < 1e-6
