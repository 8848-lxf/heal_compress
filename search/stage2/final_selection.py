"""Final selection helpers for completed round winners."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def collect_round_winners(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for round_dir in sorted(root.glob("round_[0-9][0-9][0-9]")):
        best_path = round_dir / "round_best_candidate.json"
        if not best_path.is_file():
            continue
        row = _read_json(best_path)
        row["round_index"] = int(round_dir.name.split("_")[-1])
        row["round_dir"] = str(round_dir)
        rows.append(row)
    return sorted(rows, key=lambda row: (float(row.get("F2", float("inf"))), int(row.get("round_index", 0))))


def _float_value(row: dict[str, Any], name: str, default: float = float("inf")) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_value(row: dict[str, Any], name: str, default: int = 0) -> int:
    try:
        return int(row.get(name, default))
    except (TypeError, ValueError):
        return default


def _final_candidate_rejection_reason(row: dict[str, Any], *, budget: float, expected_frames: int, seen_signatures: set[str]) -> str | None:
    if _float_value(row, "R_BOPS_realized") > float(budget):
        return "realized_bops_over_final_budget"
    if _int_value(row, "pruned_unit_count") <= 0 and _int_value(row, "realized_int8_layer_count") <= 0:
        return "control_only"
    if _int_value(row, "num_evaluated_frames") != int(expected_frames):
        return "full_validation_frame_count_mismatch"
    if _int_value(row, "latency_measured_frames", _int_value(row, "num_evaluated_frames")) != int(expected_frames):
        return "latency_frame_count_mismatch"
    if _int_value(row, "num_skipped_frames") != 0:
        return "skipped_frames_nonzero"
    signature = str(row.get("deployment_signature", ""))
    if not signature:
        return "missing_deployment_signature"
    if signature in seen_signatures:
        return "duplicate_deployment_signature"
    return None


def select_final_winner_from_full_validation(
    rows: list[dict[str, Any]],
    *,
    budget: float = 0.185,
    expected_frames: int = 1789,
) -> dict[str, Any]:
    """Select the final compressed winner from full-validation rows."""

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for row in rows:
        item = dict(row)
        reason = _final_candidate_rejection_reason(item, budget=budget, expected_frames=expected_frames, seen_signatures=seen_signatures)
        if reason is None:
            item["final_winner_pool_status"] = "eligible"
            eligible.append(item)
            seen_signatures.add(str(item.get("deployment_signature", "")))
        else:
            item["final_winner_pool_status"] = reason
            rejected.append(item)
    if not eligible:
        return {
            "status": "no_final_candidate_meets_budget",
            "winner": None,
            "eligible_count": 0,
            "rejected": rejected,
        }
    winner = sorted(eligible, key=lambda row: (_float_value(row, "F2_full"), str(row.get("candidate_hash", ""))))[0]
    return {
        "status": "ok",
        "winner": winner,
        "eligible_count": len(eligible),
        "eligible": eligible,
        "rejected": rejected,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["round_index", "candidate_hash", "status", "F1", "F2", "mAP", "forward_p50_ms", "artifact_dir", "round_dir"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_final_selection(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    final_dir = root / "final_selection"
    winners = collect_round_winners(root)
    if not winners:
        raise RuntimeError("no_round_winners_for_final_selection")
    best = winners[0]
    _write_csv(final_dir / "round_winners.csv", winners)
    _write_json(final_dir / "round_winners.json", {"winners": winners})
    _write_csv(final_dir / "full_validation_results.csv", winners)
    _write_json(
        final_dir / "full_validation_results.json",
        {
            "validation_scope": "stage2_300_frame_results_only",
            "full_validation_complete": False,
            "reason": "current evaluator has no separate full-validation candidate reevaluation path",
            "winners": winners,
        },
    )
    manifest = {
        "candidate_hash": best.get("candidate_hash", ""),
        "round_index": best.get("round_index"),
        "F1": best.get("F1"),
        "F2": best.get("F2"),
        "mAP": best.get("mAP"),
        "forward_p50_ms": best.get("forward_p50_ms"),
        "artifact_dir": best.get("artifact_dir", ""),
        "selection_scope": "round_winners_stage2_300_frame_F2",
        "full_validation_complete": False,
    }
    _write_json(final_dir / "final_best_manifest.json", manifest)
    round_dir = root / f"round_{int(best.get('round_index', 0)):03d}"
    copies = {
        "round_best_pruned_model.pth": "final_best_pruned_model.pth",
        "round_best_pruned.onnx": "final_best_pruned.onnx",
        "round_best_qdq.onnx": "final_best_qdq.onnx",
        "round_best.engine.plan": "final_best.engine.plan",
        "round_best_evaluation_300.json": "final_best_full_validation.json",
    }
    for src_name, dst_name in copies.items():
        src = round_dir / src_name
        if src.is_file():
            shutil.copy2(src, final_dir / dst_name)
    return {"final_best": manifest, "winners": winners}
