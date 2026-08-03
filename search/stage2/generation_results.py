"""Per-generation Stage-2 admission and winner artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, default=str)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_generation_stage2_results(
    generation_dir: str | Path,
    *,
    round_index: int,
    generation_index: int,
    bops_target: float | None,
    selection_report: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    expected_screening_frames: int,
) -> dict[str, Any]:
    """Choose one generation winner while preserving every rejection reason."""

    destination = Path(generation_dir)
    rows = [dict(row) for row in candidate_rows]
    selected_count = int(selection_report.get("selected_count", len(rows)) or 0)
    if selected_count != len(rows):
        raise RuntimeError(
            "generation_stage2_result_count_mismatch:"
            f"{len(rows)}!={selected_count}:generation={generation_index}"
        )

    failures: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None
    if selected_count == 0:
        status = "no_stage2_candidate"
        reason = str(
            selection_report.get("no_candidate_reason", "no_stage2_candidate")
        )
    elif selected_count == 1:
        row = rows[0]
        gate_passed = row.get("accuracy_gate_passed") is not False
        if str(row.get("status", "")) == "ok" and gate_passed:
            winner = {
                **row,
                "evaluation_500_skipped": True,
                "generation_winner_reason": "sole_stage2_candidate",
            }
            status = "single_candidate_direct_winner"
            reason = ""
        else:
            rejection = (
                [str(row.get("accuracy_gate_rejection", "accuracy_gate_failed"))]
                if not gate_passed
                else []
            )
            failures.append({**row, "generation_admission_rejection": rejection})
            status = "no_deployable_stage2_candidate"
            reason = str(
                row.get(
                    "accuracy_gate_rejection",
                    row.get("failure_reason", row.get("status", "deployment_failed")),
                )
            )
    else:
        admitted: list[dict[str, Any]] = []
        for row in rows:
            evaluated = int(
                row.get("num_evaluated_frames", row.get("evaluated", -1)) or 0
            )
            skipped = int(
                row.get("num_skipped_frames", row.get("skipped", -1)) or 0
            )
            try:
                f2 = float(row.get("F2", float("inf")))
            except (TypeError, ValueError):
                f2 = float("inf")
            rejection_reasons = []
            if str(row.get("status", "")) != "ok":
                rejection_reasons.append(
                    str(row.get("failure_reason", row.get("status", "stage2_failed")))
                )
            if evaluated != int(expected_screening_frames):
                rejection_reasons.append(
                    f"evaluated_frames_{evaluated}_expected_{int(expected_screening_frames)}"
                )
            if skipped != 0:
                rejection_reasons.append(f"skipped_frames_{skipped}_expected_0")
            if not math.isfinite(f2):
                rejection_reasons.append("finite_F2_required")
            if row.get("accuracy_gate_passed") is False:
                rejection_reasons.append(
                    str(row.get("accuracy_gate_rejection", "accuracy_gate_failed"))
                )
            if rejection_reasons:
                failures.append(
                    {
                        **row,
                        "generation_admission_rejection": rejection_reasons,
                    }
                )
            else:
                admitted.append(row)
        winner = (
            min(
                admitted,
                key=lambda row: (
                    float(row["F2"]),
                    str(row.get("candidate_hash", "")),
                ),
            )
            if admitted
            else None
        )
        if winner is None:
            status = "no_successful_500_frame_candidate"
            reason = "all_selected_candidates_failed_stage2_admission"
        else:
            winner = {
                **winner,
                "evaluation_500_skipped": False,
                "generation_winner_reason": "minimum_weighted_AP_latency_F2",
            }
            status = "generation_winner_selected"
            reason = ""

    report = {
        "round_index": int(round_index),
        "generation_index": int(generation_index),
        "bops_target": bops_target,
        "status": status,
        "failure_reason": reason,
        "selected_count": selected_count,
        "candidate_count": len(rows),
        "successful_candidate_count": sum(
            str(row.get("status", "")) == "ok" for row in rows
        ),
        "evaluated_500_count": sum(
            int(row.get("num_evaluated_frames", row.get("evaluated", 0)) or 0)
            == int(expected_screening_frames)
            for row in rows
        ),
        "expected_screening_frames": int(expected_screening_frames),
        "selection_report": dict(selection_report),
        "candidates": rows,
        "failures": failures,
        "winner": winner,
    }
    _write_json(destination / "generation_stage2_results.json", report)
    _write_csv(destination / "generation_stage2_results.csv", rows)
    _write_json(destination / "generation_selection_report.json", dict(selection_report))
    if winner is not None:
        _write_json(destination / "generation_winner.json", winner)
    else:
        _write_json(
            destination / "generation_failure.json",
            {
                "round_index": int(round_index),
                "generation_index": int(generation_index),
                "status": status,
                "failure_reason": reason,
                "selection_report": dict(selection_report),
                "failures": failures,
            },
        )
    return report
