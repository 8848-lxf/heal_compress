from __future__ import annotations

from pathlib import Path

from tools.latency_lut.build_mixed_precision_engine import FALLBACK_STATUS, build_mixed_engine, parse_args


def test_mixed_precision_builder_requires_verified_layer_mapping(tmp_path: Path):
    args = parse_args(
        [
            "--onnx",
            str(tmp_path / "model.onnx"),
            "--engine",
            str(tmp_path / "model.engine"),
            "--candidate",
            str(tmp_path / "candidate.json"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    report = build_mixed_engine(args)

    assert report["success"] is False
    assert report["status"] == FALLBACK_STATUS
    assert report["failed_stage"] == "layer_mapping"
    assert report["precision_verification"] is None
