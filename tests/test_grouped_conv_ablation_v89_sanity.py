from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.latency_lut.run_grouped_conv_ablation_v89_sanity import (
    REQUIRED_OUTPUT_FILES,
    SMALL_RATIOS,
    validate_v89_output_bundle,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _complete_bundle(root: Path) -> None:
    for name in REQUIRED_OUTPUT_FILES:
        if name.endswith(".json"):
            _write_json(root / name, {})
        elif name.endswith(".jsonl"):
            _write_jsonl(root / name, [])
        elif name.endswith(".csv"):
            _write_csv(root / name, [{"placeholder": "1"}])
        elif name.endswith(".md"):
            (root / name).write_text("# ok\n", encoding="utf-8")
    _write_json(
        root / "rewrite_equivalence_report.json",
        {
            "rewrite_0pct_status": "success",
            "baseline_AP_0.3": 0.8,
            "rewrite_AP_0.3": 0.79,
            "baseline_mAP": 0.7,
            "rewrite_mAP": 0.69,
            "ap_equivalent": True,
        },
    )
    _write_csv(
        root / "small_ratio_sweep_summary.csv",
        [
            {
                "policy": "group_balanced_output_groups_fixed",
                "score_mode": "l1",
                "target_prune_ratio": ratio,
                "model_path": str(root / f"small_{ratio}.pth"),
                "forward_smoke_status": "forward_passed",
                "eval_status": "success",
                "latency_status": "success",
                "AP_0.3": 0.5,
                "mAP": 0.4,
                "latency_ms_p50": 10.0,
            }
            for ratio in SMALL_RATIOS
        ],
    )
    _write_csv(
        root / "single_layer_sensitivity_summary.csv",
        [
            {
                "module_name": f"layer{i}.conv2",
                "target_prune_ratio": 0.25,
                "forward_smoke_status": "forward_passed",
                "eval_status": "success",
                "AP_0.3": 0.4,
                "mAP": 0.3,
                "AP_drop": i,
            }
            for i in range(16)
        ],
    )
    _write_json(
        root / "full_model_forward_smoke_report.json",
        [{"experiment_id": "x", "forward_smoke_status": "forward_passed"}],
    )
    _write_json(
        root / "full_model_eval_short_report.json",
        [{"experiment_id": "x", "eval_status": "success", "AP_0.3": 0.4, "mAP": 0.3}],
    )
    _write_json(
        root / "full_model_latency_report.json",
        [{"experiment_id": "x", "latency_status": "success", "latency_ms_p50": 10.0}],
    )


def test_v89_validator_requires_zero_percent_equivalence(tmp_path):
    _complete_bundle(tmp_path)
    _write_json(
        tmp_path / "rewrite_equivalence_report.json",
        {
            "rewrite_0pct_status": "success",
            "baseline_AP_0.3": 0.8,
            "rewrite_AP_0.3": 0.1,
            "baseline_mAP": 0.7,
            "rewrite_mAP": 0.1,
            "ap_equivalent": False,
        },
    )

    report = validate_v89_output_bundle(tmp_path)

    assert not report["valid"]
    assert "rewrite_0pct_not_equivalent" in report["errors"]


def test_v89_validator_requires_all_small_ratios_and_16_single_layers(tmp_path):
    _complete_bundle(tmp_path)
    _write_csv(tmp_path / "small_ratio_sweep_summary.csv", [{"target_prune_ratio": 0.05}])
    _write_csv(tmp_path / "single_layer_sensitivity_summary.csv", [{"module_name": "only_one", "eval_status": "success"}])

    report = validate_v89_output_bundle(tmp_path)

    assert not report["valid"]
    assert any(err.startswith("missing_small_ratio:") for err in report["errors"])
    assert "single_layer_sensitivity_expected_16_got_1" in report["errors"]


def test_v89_validator_rejects_missing_eval_or_zero_latency_for_forward_passed(tmp_path):
    _complete_bundle(tmp_path)
    _write_json(
        tmp_path / "full_model_eval_short_report.json",
        [{"experiment_id": "x", "eval_status": "skipped", "AP_0.3": "", "mAP": ""}],
    )
    _write_json(
        tmp_path / "full_model_latency_report.json",
        [{"experiment_id": "x", "latency_status": "success", "latency_ms_p50": 0.0}],
    )

    report = validate_v89_output_bundle(tmp_path)

    assert not report["valid"]
    assert "forward_passed_eval_missing:x" in report["errors"]
    assert "forward_passed_latency_invalid:x" in report["errors"]
