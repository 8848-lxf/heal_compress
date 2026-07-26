from __future__ import annotations

import argparse

import pytest

from scripts.run_v2xvit_six_budget_formal_ga_gen10 import (
    TARGETS,
    _parse_targets,
    _shard_suffix,
)


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
