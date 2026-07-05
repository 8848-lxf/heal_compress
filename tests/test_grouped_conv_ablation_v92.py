from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.latency_lut.run_grouped_conv_ablation_v92_fix_a_profile_b_c import (
    REQUIRED_OUTPUT_FILES,
    validate_v92_output_bundle,
)


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# ok\n", encoding="utf-8")


def _complete_bundle(root: Path) -> None:
    for name in REQUIRED_OUTPUT_FILES:
        path = root / name
        if name.endswith(".json"):
            _json(path, {})
        elif name.endswith(".jsonl"):
            _jsonl(path, [])
        elif name.endswith(".csv"):
            _csv(path, [{"placeholder": 1}])
        elif name.endswith(".md"):
            _md(path)
    _json(
        root / "a_fix_tp_equivalence_audit_report.json",
        {
            "num_layers_checked": 3,
            "all_equivalent": True,
            "layers": [
                {"module_name": "layer0.conv2", "equivalent": True, "divergence_reason": ""},
                {"module_name": "layer1.conv2", "equivalent": True, "divergence_reason": ""},
                {"module_name": "layer2.conv2", "equivalent": True, "divergence_reason": ""},
            ],
        },
    )
    _csv(
        root / "a_fixed_full_model_summary.csv",
        [
            {
                "policy": "flat_output_groups_fixed",
                "target_prune_ratio": "0.05",
                "forward_smoke_status": "forward_passed",
                "eval_status": "success",
                "latency_status": "success",
                "AP_0.3": "0.8",
                "mAP": "0.7",
                "latency_ms_p50": "10.0",
            },
            {
                "policy": "flat_output_groups_fixed",
                "target_prune_ratio": "0.10",
                "forward_smoke_status": "forward_passed",
                "eval_status": "success",
                "latency_status": "success",
                "AP_0.3": "0.79",
                "mAP": "0.69",
                "latency_ms_p50": "9.5",
            },
        ],
    )
    _json(root / "full_model_forward_smoke_report.json", [{"experiment_id": "a_fixed_ratio_0.05", "forward_smoke_status": "forward_passed"}])
    _json(root / "full_model_eval_short_report.json", [{"experiment_id": "a_fixed_ratio_0.05", "eval_status": "success", "AP_0.3": 0.8, "mAP": 0.7}])
    _json(root / "full_model_latency_report.json", [{"experiment_id": "a_fixed_ratio_0.05", "latency_status": "success", "latency_ms_p50": 10.0}])
    _csv(root / "c_group_block_aligned_sweep_summary.csv", [{"variant": "C2", "attempted": "true"}])
    _json(root / "d_reblock_design_audit.json", {"full_model_executed": False, "should_wait_for_a_closure": True})


def test_v92_validator_requires_all_outputs(tmp_path):
    _complete_bundle(tmp_path)
    (tmp_path / "b_speed_diagnosis_report.json").unlink()

    report = validate_v92_output_bundle(tmp_path)

    assert not report["valid"]
    assert "missing_required_file:b_speed_diagnosis_report.json" in report["errors"]


def test_v92_validator_requires_a_equivalence_pass(tmp_path):
    _complete_bundle(tmp_path)
    _json(tmp_path / "a_fix_tp_equivalence_audit_report.json", {"num_layers_checked": 3, "all_equivalent": False, "layers": []})

    report = validate_v92_output_bundle(tmp_path)

    assert not report["valid"]
    assert "a_tp_equivalence_not_closed" in report["errors"]


def test_v92_validator_requires_a_full_model_eval_and_latency(tmp_path):
    _complete_bundle(tmp_path)
    _csv(
        tmp_path / "a_fixed_full_model_summary.csv",
        [
            {
                "policy": "flat_output_groups_fixed",
                "target_prune_ratio": "0.05",
                "forward_smoke_status": "forward_passed",
                "eval_status": "skipped",
                "latency_status": "success",
                "AP_0.3": "",
                "mAP": "",
                "latency_ms_p50": "0",
            }
        ],
    )

    report = validate_v92_output_bundle(tmp_path)

    assert not report["valid"]
    assert "a_fixed_forward_passed_eval_missing:0.05" in report["errors"]
    assert "a_fixed_forward_passed_latency_invalid:0.05" in report["errors"]
