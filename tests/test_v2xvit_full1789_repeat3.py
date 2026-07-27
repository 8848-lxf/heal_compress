from __future__ import annotations

import pytest

from scripts.run_v2xvit_sixbudget_full1789_repeat3 import summarize_repetitions


def _row(index: int) -> dict[str, float | int]:
    return {
        "repeat": index,
        "AP@0.3": 0.7 + index * 0.001,
        "AP@0.5": 0.6 + index * 0.001,
        "AP@0.7": 0.4 + index * 0.001,
        "mAP": 0.5 + index * 0.001,
        "forward_p50_ms": 10.0 + index,
    }


def test_repeat3_summary_uses_all_three_repetitions() -> None:
    result = summarize_repetitions([_row(1), _row(2), _row(3)])
    assert result["mAP_mean"] == pytest.approx(0.502)
    assert result["mAP_std"] == pytest.approx(0.001)
    assert len(result["repetitions"]) == 3


def test_repeat3_summary_fails_closed_on_missing_repeat() -> None:
    with pytest.raises(RuntimeError, match="repeat_count_mismatch"):
        summarize_repetitions([_row(1), _row(2)])
