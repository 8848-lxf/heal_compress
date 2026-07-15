from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_historical_mask_legality_diagnostic_never_repairs_mask() -> None:
    from search.audits.prune_rate_reachability import diagnose_historical_mask

    selected = ["old0", "old1"]
    result = diagnose_historical_mask(
        selected,
        current_unit_ids={"old0"},
        protected_unit_ids={"old0"},
        rejection_reasons={"old1": "missing_unit_mapping"},
    )

    assert result["input_unit_ids"] == selected
    assert result["output_unit_ids"] == selected
    assert result["repair_applied"] is False
    assert result["passed"] is False
    assert result["rejections"] == {
        "old0": "protected",
        "old1": "missing_unit_mapping",
    }


def test_accuracy_collapse_is_not_classified_as_structural_failure() -> None:
    from search.audits.prune_rate_reachability import classify_replay_failure

    assert (
        classify_replay_failure(
            physical_export_success=True,
            forward_success=True,
            evaluation_success=True,
            map_value=0.067819,
            map_reference=0.718853,
            collapse_absolute_drop=0.1,
        )
        == "ACCURACY_COLLAPSE"
    )
    assert (
        classify_replay_failure(
            physical_export_success=False,
            forward_success=False,
            evaluation_success=False,
            map_value=None,
            map_reference=0.718853,
        )
        == "STRUCTURAL_FAILURE"
    )
