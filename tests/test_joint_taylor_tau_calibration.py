from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _row(
    candidate: str,
    *,
    map_value: float,
    loss: float,
    prune: float,
    bops: float,
    precision: str = "strict_fp16",
    valid: bool = True,
) -> dict[str, object]:
    return {
        "candidate_hash": candidate,
        "physical_hash": f"physical-{candidate}",
        "precision_profile_hash": f"precision-{candidate}",
        "engine_hash": f"engine-{candidate}",
        "mAP": map_value,
        "L_joint": loss,
        "R_prune": prune,
        "R_BOPS": bops,
        "precision_variant": precision,
        "valid_for_tau": valid,
        "calibration_manifest_hash": "calibration",
        "validation_manifest_hash": "validation",
    }


def test_tau_uses_absolute_point_one_map_drop_and_safe_tie_breaks() -> None:
    from search.proxy.tau_calibration import calibrate_tau

    rows = [
        _row("safe-low-prune", map_value=0.625, loss=4.0, prune=0.30, bops=0.30),
        _row("safe-high-prune", map_value=0.625, loss=5.0, prune=0.40, bops=0.29),
        _row("unsafe", map_value=0.624999, loss=6.0, prune=0.50, bops=0.25),
    ]

    result = calibrate_tau(rows, mAP_reference=0.725, code_commit="commit")

    assert result.mAP_min_allowed == pytest.approx(0.625)
    assert result.max_allowed_absolute_mAP_drop == pytest.approx(0.1)
    assert result.safe_anchor_candidate_hash == "safe-high-prune"
    assert result.safe_anchor_delta_mAP == pytest.approx(0.1)
    assert result.safe_anchor_L_joint == pytest.approx(5.0)
    assert result.tau == pytest.approx(5.0 / math.log(2.0))
    assert result.calibration_passed is True


def test_tau_tie_breaks_by_lower_bops_then_stable_candidate_hash() -> None:
    from search.proxy.tau_calibration import calibrate_tau

    rows = [
        _row("z", map_value=0.63, loss=2.0, prune=0.4, bops=0.25),
        _row("b", map_value=0.63, loss=3.0, prune=0.4, bops=0.20),
        _row("a", map_value=0.63, loss=4.0, prune=0.4, bops=0.20),
    ]

    result = calibrate_tau(rows, mAP_reference=0.725)

    assert result.safe_anchor_candidate_hash == "a"


def test_tau_fails_closed_without_positive_loss_safe_anchor() -> None:
    from search.proxy.tau_calibration import TauCalibrationError, calibrate_tau

    with pytest.raises(TauCalibrationError, match="no_positive_loss_safe_anchor"):
        calibrate_tau(
            [
                _row("original", map_value=0.725, loss=0.0, prune=0.0, bops=0.25),
                _row("unsafe", map_value=0.60, loss=2.0, prune=0.1, bops=0.24),
            ],
            mAP_reference=0.725,
        )


def test_tau_marks_underresolved_when_every_nonzero_anchor_is_safe() -> None:
    from search.proxy.tau_calibration import calibrate_tau

    rows = [
        _row(str(index), map_value=0.72 - index * 0.005, loss=float(index), prune=index / 10, bops=0.25)
        for index in range(1, 8)
    ]

    result = calibrate_tau(rows, mAP_reference=0.725)

    assert result.tau_boundary_underresolved is True


def test_tau_distribution_reports_landmarks_and_saturation() -> None:
    from search.proxy.tau_calibration import task_score_distribution

    diagnostics = task_score_distribution([0.0, 1.0, 2.0, 1000.0], tau=1.0)

    assert diagnostics["count"] == 4
    assert diagnostics["S_task_max"] == pytest.approx(1.0)
    assert diagnostics["S_task_min"] == pytest.approx(math.exp(-80.0))
    assert diagnostics["S_task_ge_0_99_count"] == 1
    assert diagnostics["S_task_le_0_01_count"] == 1
    assert diagnostics["saturated_count"] == 1


def test_proxy_scale_is_written_read_only_and_hash_stable(tmp_path: Path) -> None:
    from search.proxy.tau_calibration import calibrate_tau, proxy_scale_hash, write_proxy_scale

    scale = calibrate_tau(
        [_row("safe", map_value=0.63, loss=2.0, prune=0.4, bops=0.2)],
        mAP_reference=0.725,
        code_commit="commit",
    )
    path = tmp_path / "proxy_scale.json"

    written_hash = write_proxy_scale(path, scale)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert written_hash == proxy_scale_hash(payload)
    assert payload["mapping"] == "exponential"
    assert payload["formula"] == "exp(-L_joint/tau)"
    assert payload["max_allowed_absolute_mAP_drop"] == pytest.approx(0.1)
    assert os.stat(path).st_mode & 0o222 == 0


def test_tau_is_not_recomputed_from_generation_statistics() -> None:
    from search.proxy.tau_calibration import validate_fixed_proxy_scale

    baseline = {"tau": 3.0, "mapping": "exponential", "formula": "exp(-L_joint/tau)"}

    assert validate_fixed_proxy_scale(baseline, baseline, generation=9)["passed"] is True
    with pytest.raises(RuntimeError, match="proxy_scale_changed_during_ga"):
        validate_fixed_proxy_scale(
            baseline,
            {**baseline, "tau": 3.1},
            generation=1,
        )
