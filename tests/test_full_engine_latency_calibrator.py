from __future__ import annotations

import json
from pathlib import Path

from tools.latency_lut.train_full_engine_latency_calibrator import train_calibrator


def test_train_full_engine_latency_calibrator_uses_ridge_when_samples_are_few(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    rows = [
        {
            "candidate_id": "a",
            "valid_for_calibration": True,
            "T_real_p50": 3.0,
            "T_lut_raw": 0.3,
            "T_compute_covered": 0.2,
            "T_boundary_cast": 0.01,
            "T_boundary_qdq": 0.02,
            "T_plugin_or_scatter": 0.05,
            "T_grid_sample_or_geometry": 0.0,
            "T_elementwise_merge": 0.0,
            "T_memory_reformat": 0.0,
            "T_fixed_overhead": 0.0,
            "num_precision_switches": 1,
            "observed_fp32_layers": 1,
            "observed_fp16_layers": 10,
            "observed_int8_layers": 2,
            "num_cast_inserted": 1,
            "num_qdq_nodes_inserted": 4,
        },
        {
            "candidate_id": "b",
            "valid_for_calibration": True,
            "T_real_p50": 4.0,
            "T_lut_raw": 0.5,
            "T_compute_covered": 0.4,
            "T_boundary_cast": 0.02,
            "T_boundary_qdq": 0.03,
            "T_plugin_or_scatter": 0.05,
            "T_grid_sample_or_geometry": 0.0,
            "T_elementwise_merge": 0.0,
            "T_memory_reformat": 0.0,
            "T_fixed_overhead": 0.0,
            "num_precision_switches": 2,
            "observed_fp32_layers": 2,
            "observed_fp16_layers": 9,
            "observed_int8_layers": 2,
            "num_cast_inserted": 2,
            "num_qdq_nodes_inserted": 4,
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model_path = tmp_path / "model.json"
    report_path = tmp_path / "report.md"

    report = train_calibrator(
        dataset,
        model_path,
        report_path,
        model="auto",
        ridge_alpha=1.0,
    )

    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert report["valid_samples"] == 2
    assert model_payload["model_type"] == "least_squares_linear"
    assert model_payload["regularization"] == "ridge"
    assert model_payload["tiny_mlp"]["status"] == "skipped"
    assert "T_compute_covered" in model_payload["feature_names"]
    assert "residuals" in model_payload
