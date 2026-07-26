from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from scripts.run_v2xvit_six_budget_formal_ga_gen10 import (
    TARGETS,
    _parse_targets,
    _shard_suffix,
)
from scripts.watch_and_evaluate_v2xvit_final_budget import run as run_watcher


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
