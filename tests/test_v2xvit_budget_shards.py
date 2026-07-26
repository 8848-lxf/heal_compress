from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from scripts.run_v2xvit_six_budget_formal_ga_gen10 import (
    TARGETS,
    _parse_targets,
    _shard_suffix,
)
from scripts.run_v2xvit_ga_final_latency import load_budget_summary
from scripts.summarize_v2xvit_six_budget_results import build_summary
from scripts.watch_and_evaluate_v2xvit_final_budget import run as run_watcher
from scripts.watch_and_run_v2xvit_six_budget_latency import completion_state


def test_budget_shard_parser_preserves_frozen_budget_subset() -> None:
    assert _parse_targets("0.10") == (0.10,)
    assert _parse_targets("0.10,0.05") == (0.10, 0.05)
    assert tuple(TARGETS) == (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)


@pytest.mark.parametrize("value", ["", "0.12", "0.10,0.10"])
def test_budget_shard_parser_rejects_empty_unsupported_or_duplicate(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_targets(value)


def test_budget_shard_report_suffix_isolated_and_validated() -> None:
    assert _shard_suffix("") == ""
    assert _shard_suffix("budget010") == "_budget010"
    with pytest.raises(RuntimeError, match="invalid_shard_id"):
        _shard_suffix("../shared")


def test_formal_stage2_uses_fixed300_screen_and_fixed500_generation_winner() -> None:
    from scripts.run_v2xvit_six_budget_formal_ga_gen10 import (
        GENERATION_WINNER_EVALUATION_FRAMES,
        GENERATION_WINNER_EVALUATION_PROTOCOL,
        GENERATION_WINNER_EVALUATION_WARMUP_FRAMES,
        STAGE2_EVALUATION_FRAMES,
        STAGE2_EVALUATION_PROTOCOL,
        STAGE2_EVALUATION_WARMUP_FRAMES,
    )

    assert STAGE2_EVALUATION_FRAMES == 300
    assert STAGE2_EVALUATION_WARMUP_FRAMES == 100
    assert STAGE2_EVALUATION_PROTOCOL == "top5_fixed300_warmup100_screening"
    assert GENERATION_WINNER_EVALUATION_FRAMES == 500
    assert GENERATION_WINNER_EVALUATION_WARMUP_FRAMES == 200
    assert (
        GENERATION_WINNER_EVALUATION_PROTOCOL
        == "generation_winner_fixed500_warmup200"
    )


def test_fixed500_watcher_fails_closed_when_formal_budget_failed(tmp_path) -> None:
    failure = tmp_path / "ga/budget_010/failure.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"failure":"phenotype_drift"}\n', encoding="utf-8")
    args = SimpleNamespace(
        output_root=tmp_path,
        label="010",
        control="ga",
        physical_gpu=4,
        model_config=tmp_path / "config.yaml",
        heal_root=tmp_path,
        tensorrt_root=tmp_path,
        plugin=tmp_path / "plugin.so",
        fixed500_manifest=tmp_path / "fixed500.json",
        fixed_k=27904,
        poll_seconds=0.001,
    )
    assert run_watcher(args) == 2
    status = tmp_path / "reports/final_fixed500_watcher_010_ga.json"
    assert "formal_budget_failed" in status.read_text(encoding="utf-8")


def test_final_latency_uses_completed_per_budget_summary(tmp_path) -> None:
    summary = tmp_path / "ga/budget_030/seed_0/budget_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        '{"greedy_anchor":{"complete_phenotype_hash":"g"},'
        '"final_winner":{"complete_phenotype_hash":"a"}}\n',
        encoding="utf-8",
    )
    assert load_budget_summary(tmp_path, "030")["final_winner"]["complete_phenotype_hash"] == "a"
    with pytest.raises(RuntimeError, match="formal_budget_summary_missing:025"):
        load_budget_summary(tmp_path, "025")


def test_final_latency_rejects_incomplete_budget_summary(tmp_path) -> None:
    summary = tmp_path / "ga/budget_030/seed_0/budget_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"greedy_anchor":{}}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="formal_budget_summary_incomplete:030"):
        load_budget_summary(tmp_path, "030")


def test_six_budget_latency_watcher_requires_every_valid_fixed500(tmp_path) -> None:
    for label in ("030", "025", "020", "015", "010", "005"):
        summary = tmp_path / f"ga/budget_{label}/seed_0/budget_summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            '{"greedy_anchor":{"complete_phenotype_hash":"g"},'
            '"final_winner":{"complete_phenotype_hash":"a"}}\n',
            encoding="utf-8",
        )
        for control, candidate_hash in (("greedy", "g"), ("ga", "a")):
            evaluation = (
                tmp_path
                / f"{control}_final_fixed500_partial/budget_{label}/{candidate_hash}/evaluation.json"
            )
            evaluation.parent.mkdir(parents=True)
            evaluation.write_text(
                '{"status":"ok","num_evaluated_frames":500,"num_skipped_frames":0}\n',
                encoding="utf-8",
            )
    assert completion_state(tmp_path)["ready"] is True
    broken = tmp_path / "ga_final_fixed500_partial/budget_005/a/evaluation.json"
    broken.write_text(
        '{"status":"ok","num_evaluated_frames":499,"num_skipped_frames":1}\n',
        encoding="utf-8",
    )
    state = completion_state(tmp_path)
    assert state["ready"] is False
    assert "fixed500_invalid:005:ga" in state["failures"]


def test_six_budget_summary_is_explicitly_partial_without_formal_results(tmp_path) -> None:
    baseline = tmp_path / "baseline_fixed500/B0/evaluation.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        '{"AP@0.3":0.7,"AP@0.5":0.6,"AP@0.7":0.4,"mAP":0.5666666667,'
        '"num_evaluated_frames":500,"num_skipped_frames":0,"eval_manifest_hash":"m"}\n',
        encoding="utf-8",
    )
    result = build_summary(tmp_path)
    assert result["status"] == "partial"
    assert result["baseline_evaluated"] == 500
    assert result["budgets"]["030"]["status"] == "pending"
