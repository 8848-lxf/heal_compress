from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _rows() -> list[dict[str, object]]:
    return [
        {"phenotype_hash": f"p{index}", "L_joint_raw": float(index)}
        for index in range(1, 11)
    ]


def test_joint_loss_scale_uses_unique_positive_nearest_rank_p90(
    tmp_path: Path,
) -> None:
    from search.proxy.joint_loss_scale import (
        calibrate_joint_loss_scale,
        load_joint_loss_scale,
        write_joint_loss_scale,
    )

    payload = calibrate_joint_loss_scale(
        [*_rows(), {"phenotype_hash": "p9", "L_joint_raw": 9.0}],
        code_commit="abc",
        created_at="2026-07-17T00:00:00+08:00",
    )

    assert payload["value"] == pytest.approx(9.0)
    assert payload["member_count"] == 10
    assert payload["quantile"] == pytest.approx(0.90)
    assert payload["quantile_method"] == "nearest_rank"
    assert payload["mapping"] == "linear_fixed_scale"
    assert payload["formula"] == "J1=-0.8*(L_joint/L_scale)+0.2*R_prune"
    path = write_joint_loss_scale(tmp_path / "joint_loss_scale.json", payload)
    assert path == tmp_path / "joint_loss_scale.json"
    assert os.stat(path).st_mode & 0o222 == 0
    assert (
        load_joint_loss_scale(path)["joint_loss_scale_hash"]
        == payload["joint_loss_scale_hash"]
    )


@pytest.mark.parametrize("loss", [0.0, -1.0, float("inf"), float("nan")])
def test_joint_loss_scale_rejects_empty_positive_pool(loss: float) -> None:
    from search.proxy.joint_loss_scale import calibrate_joint_loss_scale

    with pytest.raises(
        ValueError, match="joint_loss_scale_positive_finite_pool_required"
    ):
        calibrate_joint_loss_scale(
            [{"phenotype_hash": "x", "L_joint_raw": loss}]
        )


def test_joint_loss_scale_rejects_conflicting_duplicate_phenotype() -> None:
    from search.proxy.joint_loss_scale import calibrate_joint_loss_scale

    with pytest.raises(
        ValueError, match="joint_loss_scale_duplicate_loss_conflict:same"
    ):
        calibrate_joint_loss_scale(
            [
                {"phenotype_hash": "same", "L_joint_raw": 1.0},
                {"phenotype_hash": "same", "L_joint_raw": 2.0},
            ]
        )


def test_joint_loss_scale_loader_rejects_writable_file(tmp_path: Path) -> None:
    from search.proxy.joint_loss_scale import (
        calibrate_joint_loss_scale,
        load_joint_loss_scale,
        write_joint_loss_scale,
    )

    path = write_joint_loss_scale(
        tmp_path / "joint_loss_scale.json",
        calibrate_joint_loss_scale(_rows()),
    )
    path.chmod(0o644)

    with pytest.raises(RuntimeError, match="joint_loss_scale_must_be_read_only"):
        load_joint_loss_scale(path)


def test_joint_loss_scale_loader_rejects_hash_mismatch(tmp_path: Path) -> None:
    from search.proxy.joint_loss_scale import (
        calibrate_joint_loss_scale,
        load_joint_loss_scale,
        write_joint_loss_scale,
    )

    path = write_joint_loss_scale(
        tmp_path / "joint_loss_scale.json",
        calibrate_joint_loss_scale(_rows()),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["value"] = 99.0
    path.chmod(0o644)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(RuntimeError, match="joint_loss_scale_hash_mismatch"):
        load_joint_loss_scale(path)
