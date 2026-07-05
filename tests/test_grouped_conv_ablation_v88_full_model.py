from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.latency_lut.run_grouped_conv_ablation_v88_full_model import (
    L1_MAIN_POLICIES,
    MAIN_RATIOS,
    REQUIRED_OUTPUT_FILES,
    validate_v88_output_bundle,
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


def _make_complete_bundle(root: Path) -> None:
    rows: list[dict] = []
    smoke: list[dict] = []
    eval_rows: list[dict] = []
    latency_rows: list[dict] = []
    failure_rows: list[dict] = []
    artifacts: list[dict] = []
    per_layer: list[dict] = []

    for policy in L1_MAIN_POLICIES:
        for ratio in MAIN_RATIOS:
            model_path = root / "models" / f"{policy}__l1__ratio_{ratio}.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"placeholder")
            status = "forward_passed" if policy in {
                "flat_output_groups_fixed",
                "group_balanced_output_groups_fixed",
                "group_coarsening_zero_padded_reblock",
            } else "forward_failed"
            rows.append(
                {
                    "policy": policy,
                    "score_mode": "l1",
                    "target_prune_ratio": ratio,
                    "model_path": str(model_path),
                    "plan_status": "success",
                    "rewrite_status": "success" if status == "forward_passed" else "failed",
                    "forward_smoke_status": status,
                    "eval_status": "success" if status == "forward_passed" else "skipped_forward_failed",
                    "latency_status": "success" if status == "forward_passed" else "skipped_forward_failed",
                    "AP_0.3": 0.1 if status == "forward_passed" else "",
                    "mAP": 0.2 if status == "forward_passed" else "",
                    "latency_ms_p50": 12.3 if status == "forward_passed" else "",
                    "failure_reason": "" if status == "forward_passed" else "dependency_incomplete_after_resolver_attempt",
                }
            )
            smoke.append(
                {
                    "policy": policy,
                    "score_mode": "l1",
                    "target_prune_ratio": ratio,
                    "model_path": str(model_path) if status == "forward_passed" else "",
                    "forward_smoke_status": status,
                    "traceback": "" if status == "forward_passed" else "resolver tried and failed",
                }
            )
            if status == "forward_passed":
                eval_rows.append(
                    {
                        "policy": policy,
                        "score_mode": "l1",
                        "target_prune_ratio": ratio,
                        "model_path": str(model_path),
                        "eval_status": "success",
                        "AP_0.3": 0.1,
                        "mAP": 0.2,
                    }
                )
                latency_rows.append(
                    {
                        "policy": policy,
                        "score_mode": "l1",
                        "target_prune_ratio": ratio,
                        "model_path": str(model_path),
                        "latency_status": "success",
                        "latency_backend": "pytorch",
                        "latency_ms_p50": 12.3,
                    }
                )
                artifacts.append(
                    {
                        "policy": policy,
                        "score_mode": "l1",
                        "target_prune_ratio": ratio,
                        "model_path": str(model_path),
                        "artifact_type": "torch_checkpoint",
                    }
                )
            per_layer.append(
                {
                    "policy": policy,
                    "score_mode": "l1",
                    "target_prune_ratio": ratio,
                    "module_name": "backbone.block.conv2",
                    "resolver_attempted": True,
                    "full_model_rewrite_attempted": True,
                    "single_layer_only": False,
                    "legality_status": "ok" if status == "forward_passed" else "dependency_incomplete",
                }
            )
            if policy == "true_group_block_pruning":
                failure_rows.append(
                    {
                        "policy": policy,
                        "score_mode": "l1",
                        "target_prune_ratio": ratio,
                        "module_name": "backbone.block.conv2",
                        "stage": "resolver",
                        "failure_reason": "unsupported_group_block_pattern",
                        "resolver_attempted": True,
                    }
                )

    for filename in REQUIRED_OUTPUT_FILES:
        path = root / filename
        if filename.endswith(".json"):
            _write_json(path, [] if filename != "pruned_model_artifacts.json" else artifacts)
        elif filename.endswith(".jsonl"):
            _write_jsonl(path, failure_rows if filename == "failure_cases.jsonl" else per_layer)
        elif filename.endswith(".csv"):
            _write_csv(path, rows)
        elif filename.endswith(".md"):
            path.write_text("# summary\n", encoding="utf-8")

    _write_json(root / "full_model_forward_smoke_report.json", smoke)
    _write_json(root / "full_model_eval_short_report.json", eval_rows)
    _write_json(root / "full_model_latency_report.json", latency_rows)
    _write_json(root / "pruned_model_artifacts.json", artifacts)


def test_v88_validator_requires_all_output_files(tmp_path):
    _make_complete_bundle(tmp_path)
    (tmp_path / "resolver_report.json").unlink()

    report = validate_v88_output_bundle(tmp_path)

    assert not report["valid"]
    assert "missing_required_file:resolver_report.json" in report["errors"]


def test_v88_validator_requires_twelve_l1_main_records(tmp_path):
    _make_complete_bundle(tmp_path)
    rows = list(csv.DictReader((tmp_path / "full_model_ablation_summary.csv").open(encoding="utf-8")))
    rows = rows[:-1]
    _write_csv(tmp_path / "full_model_ablation_summary.csv", rows)

    report = validate_v88_output_bundle(tmp_path)

    assert not report["valid"]
    assert any(err.startswith("missing_l1_main_record:") for err in report["errors"])


def test_v88_validator_rejects_fake_eval_or_zero_latency_for_forward_passed(tmp_path):
    _make_complete_bundle(tmp_path)
    rows = list(csv.DictReader((tmp_path / "full_model_ablation_summary.csv").open(encoding="utf-8")))
    rows[0]["AP_0.3"] = ""
    rows[0]["mAP"] = "not_evaluated"
    rows[0]["latency_ms_p50"] = "0"
    _write_csv(tmp_path / "full_model_ablation_summary.csv", rows)

    report = validate_v88_output_bundle(tmp_path)

    assert not report["valid"]
    assert any("missing_eval_for_forward_passed" in err for err in report["errors"])
    assert any("invalid_latency_for_forward_passed" in err for err in report["errors"])


def test_v88_validator_requires_c_resolver_attempts_and_d_full_model_not_single_layer(tmp_path):
    _make_complete_bundle(tmp_path)
    _write_jsonl(
        tmp_path / "per_layer_grouped_conv_details.jsonl",
        [
            {
                "policy": "group_coarsening_zero_padded_reblock",
                "score_mode": "l1",
                "target_prune_ratio": 0.5,
                "module_name": "conv",
                "resolver_attempted": False,
                "full_model_rewrite_attempted": False,
                "single_layer_only": True,
            }
        ],
    )
    _write_jsonl(tmp_path / "failure_cases.jsonl", [])

    report = validate_v88_output_bundle(tmp_path)

    assert not report["valid"]
    assert "true_group_block_pruning_resolver_not_attempted" in report["errors"]
    assert "group_coarsening_zero_padded_reblock_only_single_layer_smoke" in report["errors"]
