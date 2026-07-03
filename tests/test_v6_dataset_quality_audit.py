from __future__ import annotations

import json
from pathlib import Path

from tools.latency_lut.audit_v6_dataset_quality import audit_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_audit_v6_dataset_reports_gate_and_residuals(tmp_path: Path) -> None:
    decomp = tmp_path / "decomp.json"
    decomp.write_text(
        json.dumps(
            {
                "T_compute_covered": 0.2,
                "T_boundary_cast": 0.01,
                "T_boundary_qdq": 0.02,
                "T_plugin_or_scatter": 0.1,
                "T_grid_sample_or_geometry": 0.1,
                "T_elementwise_merge": 0.01,
                "T_memory_reformat": 0.01,
                "T_fixed_overhead": 0.1,
                "matched_lut_keys": [{"k": 1}, {"k": 2}],
                "coarse_keys": [],
                "missing_keys": [],
                "unavailable_keys": [],
                "uncertainty": 0.0,
            }
        ),
        encoding="utf-8",
    )
    closure = tmp_path / "closure.json"
    closure.write_text(json.dumps({"success": True, "num_dtype_mismatches_fixed": 2, "num_cast_inserted": 1, "qdq_pattern_preserved": True}), encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"valid": True, "num_conv_gemm_checked": 3, "num_merge_checked": 2, "num_qdq_patterns_checked": 1, "num_errors": 0, "errors": []}), encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "candidate_id": "cand_a__prof_0",
                "base_structure_candidate_id": "cand_a",
                "precision_profile_id": "prof_0",
                "is_width_changed_subnet": True,
                "is_pruned": True,
                "is_pruned_mixed": True,
                "route2_explicit_precision": True,
                "dtype_closure_valid": True,
                "has_fp32": True,
                "has_fp16": True,
                "has_int8_qdq": True,
                "precision_profile": {"default": "FP16", "overrides": {"layer0.layer0.0.conv2.Conv": "INT8", "cls_head.Conv": "FP32"}},
                "observed_fp32_layers": 2,
                "observed_fp16_layers": 4,
                "observed_int8_layers": 1,
                "precision_verification_failures": [],
                "T_lut_raw": 0.54,
                "T_real_p50": 3.0,
                "T_real_p90": 3.2,
                "mAP": 0.8,
                "num_changed_conv_layers": 5,
                "param_keep_ratio": 0.8,
                "missing_keys": [],
                "unavailable_keys": [],
                "lut_decomposition_path": str(decomp),
                "dtype_closure_report_path": str(closure),
                "dtype_closure_validation_report_path": str(validation),
            }
        ],
    )
    report = audit_dataset(dataset=dataset, results_dir=tmp_path, decomposition_dir=tmp_path, profiles=tmp_path / "missing.json")
    assert report["basic"]["num_labels"] == 1
    assert report["basic"]["num_with_fp32_fp16_int8_qdq"] == 1
    assert report["latency_alignment"]["raw_lut_mae_ms"] == 2.46
    assert report["dtype_closure"]["dtype_closure_valid_labels"] == 1
    assert report["final_decision"]["can_train_latency_proxy_next"] is False
