from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.latency_lut.run_grouped_conv_ablation_v91_audit_and_c_resolver import (
    REQUIRED_OUTPUT_FILES,
    validate_v91_output_bundle,
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


def _complete_bundle(root: Path) -> None:
    for name in REQUIRED_OUTPUT_FILES:
        if name.endswith(".json"):
            _json(root / name, {})
        elif name.endswith(".jsonl"):
            _jsonl(root / name, [])
        elif name.endswith(".csv"):
            _csv(root / name, [{"placeholder": 1}])
        elif name.endswith(".md"):
            (root / name).write_text("# ok\n", encoding="utf-8")
    _csv(
        root / "v88_v89_ap_latency_tradeoff_summary.csv",
        [
            {
                "source_version": "v89",
                "policy": "group_balanced_output_groups_fixed",
                "target_prune_ratio": 0.05,
                "AP_0.3": 0.86,
                "mAP": 0.81,
                "latency_ms_p50": 19.4,
                "speedup_vs_baseline": 1.02,
            }
        ],
    )
    _json(
        root / "a_tp_equivalence_audit_report.json",
        {
            "num_layers_checked": 3,
            "layers": [
                {"module_name": "layer0.conv2", "tp_executed": True, "a_executed": True, "equivalent": True, "divergence_reason": ""},
                {"module_name": "layer1.conv2", "tp_executed": True, "a_executed": True, "equivalent": False, "divergence_reason": "idx_mismatch"},
                {"module_name": "layer2.conv2", "tp_executed": False, "a_executed": True, "equivalent": False, "divergence_reason": "tp_depgraph_failed"},
            ],
        },
    )
    _json(
        root / "c_group_block_resolver_report.json",
        {
            "num_blocks_checked": 3,
            "blocks": [
                {"block_name": "b0", "grouped_conv": "b0.conv2", "resolver_status": "success", "dependency_complete": True}
            ],
        },
    )
    _csv(
        root / "c_group_block_full_model_summary.csv",
        [
            {
                "experiment_id": "c_ratio_0.05",
                "target_prune_ratio": 0.05,
                "forward_smoke_status": "forward_passed",
                "eval_status": "success",
                "latency_status": "success",
                "AP_0.3": 0.8,
                "mAP": 0.7,
                "latency_ms_p50": 10.0,
                "dependency_complete": True,
                "in_out_block_sync_pass": True,
            }
        ],
    )
    _json(root / "full_model_forward_smoke_report.json", [{"experiment_id": "c_ratio_0.05", "forward_smoke_status": "forward_passed"}])
    _json(root / "full_model_eval_short_report.json", [{"experiment_id": "c_ratio_0.05", "eval_status": "success", "AP_0.3": 0.8, "mAP": 0.7}])
    _json(root / "full_model_latency_report.json", [{"experiment_id": "c_ratio_0.05", "latency_status": "success", "latency_ms_p50": 10.0}])


def test_v91_validator_requires_all_outputs(tmp_path):
    _complete_bundle(tmp_path)
    (tmp_path / "a_tp_equivalence_summary.md").unlink()

    report = validate_v91_output_bundle(tmp_path)

    assert not report["valid"]
    assert "missing_required_file:a_tp_equivalence_summary.md" in report["errors"]


def test_v91_validator_requires_three_a_tp_layers(tmp_path):
    _complete_bundle(tmp_path)
    _json(tmp_path / "a_tp_equivalence_audit_report.json", {"num_layers_checked": 2, "layers": []})

    report = validate_v91_output_bundle(tmp_path)

    assert not report["valid"]
    assert "a_tp_equivalence_layers_lt_3" in report["errors"]


def test_v91_validator_requires_eval_and_latency_for_forward_passed(tmp_path):
    _complete_bundle(tmp_path)
    _json(tmp_path / "full_model_eval_short_report.json", [{"experiment_id": "c_ratio_0.05", "eval_status": "skipped"}])
    _json(tmp_path / "full_model_latency_report.json", [{"experiment_id": "c_ratio_0.05", "latency_status": "success", "latency_ms_p50": 0}])

    report = validate_v91_output_bundle(tmp_path)

    assert not report["valid"]
    assert "forward_passed_eval_missing:c_ratio_0.05" in report["errors"]
    assert "forward_passed_latency_invalid:c_ratio_0.05" in report["errors"]
