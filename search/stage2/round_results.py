"""Round-level Stage-2 result aggregation."""

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


def _score_value(score: dict[str, Any], name: str) -> float | None:
    value = score.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_round_stage2_results(run_dir: str | Path, *, round_index: int = 0) -> list[dict[str, Any]]:
    root = Path(run_dir)
    round_dir = root / f"round_{int(round_index):03d}"
    manifest = _read_json(round_dir / "repaired_top5_manifest.json")
    rows: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates", []) or []:
        candidate_hash = str(candidate.get("repaired_phenotype_hash", ""))
        candidate_dir = round_dir / "stage2" / candidate_hash
        score_path = candidate_dir / "stage2_score.json"
        score = _read_json(score_path) if score_path.is_file() else {"status": "missing_stage2_score", "F2": float("inf")}
        artifact_dir = str(score.get("artifact_dir") or candidate_dir)
        rows.append(
            {
                "round_index": int(round_index),
                "candidate_rank": int(candidate.get("candidate_rank", len(rows))),
                "candidate_hash": candidate_hash,
                "artifact_dir": artifact_dir,
                "status": str(score.get("status", "missing")),
                "F1": float(candidate.get("repaired_F1", score.get("F1", float("inf")))),
                "F2": _score_value(score, "F2") if _score_value(score, "F2") is not None else float("inf"),
                "mAP": _score_value(score, "mAP"),
                "AP@0.3": _score_value(score, "AP@0.3"),
                "AP@0.5": _score_value(score, "AP@0.5"),
                "AP@0.7": _score_value(score, "AP@0.7"),
                "forward_p50_ms": _score_value(score, "forward_p50_ms"),
                "L_map_real": _score_value(score, "L_map_real"),
                "R_latency_real": _score_value(score, "R_latency_real"),
                "physical_hash": str(score.get("physical_hash", "")),
                "deployment_hash": str(score.get("deployment_hash", "")),
                "eval_hash": str(score.get("eval_hash", "")),
                "engine_hash": str(score.get("engine_hash", "")),
            }
        )
    return sorted(rows, key=lambda row: (float(row["F2"]), int(row["candidate_rank"]), row["candidate_hash"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "round_index",
        "candidate_rank",
        "candidate_hash",
        "status",
        "F1",
        "F2",
        "mAP",
        "AP@0.3",
        "AP@0.5",
        "AP@0.7",
        "forward_p50_ms",
        "L_map_real",
        "R_latency_real",
        "physical_hash",
        "deployment_hash",
        "eval_hash",
        "engine_hash",
        "artifact_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_markdown(path: Path, rows: list[dict[str, Any]], winner: dict[str, Any]) -> None:
    lines = [
        "# Round Stage-2 Top-5 Results",
        "",
        f"- winner: `{winner.get('candidate_hash', '')}`",
        f"- winner F2: `{winner.get('F2', '')}`",
        "",
        "| rank | candidate | status | F1 | F2 | mAP | p50 ms |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate_rank']} | `{row['candidate_hash']}` | {row['status']} | "
            f"{row['F1']} | {row['F2']} | {row.get('mAP')} | {row.get('forward_p50_ms')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_winner_artifacts(round_dir: Path, winner: dict[str, Any]) -> None:
    source = Path(str(winner["artifact_dir"]))
    copies = {
        ("pruned_checkpoint.pth",): "round_best_pruned_model.pth",
        ("pruned_fp32.onnx",): "round_best_pruned.onnx",
        ("pruned_qdq.onnx",): "round_best_qdq.onnx",
        ("engine.plan",): "round_best.engine.plan",
        ("evaluation_300.json", "evaluation.json"): "round_best_evaluation_300.json",
    }
    for source_names, dst_name in copies.items():
        src = next((source / name for name in source_names if (source / name).is_file()), None)
        if src is None:
            raise FileNotFoundError(f"winner_artifact_missing:{source}:{'|'.join(source_names)}")
        shutil.copy2(src, round_dir / dst_name)


def write_round_stage2_results(run_dir: str | Path, *, round_index: int = 0) -> dict[str, Any]:
    root = Path(run_dir)
    round_dir = root / f"round_{int(round_index):03d}"
    rows = collect_round_stage2_results(root, round_index=round_index)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError(f"no_successful_stage2_candidates:round_{int(round_index):03d}")
    winner = ok_rows[0]
    _write_csv(round_dir / "stage2_top5_results.csv", rows)
    _write_json(round_dir / "stage2_top5_results.json", {"round_index": int(round_index), "winner": winner, "candidates": rows})
    _write_markdown(round_dir / "stage2_top5_results.md", rows, winner)
    _write_json(round_dir / "round_best_candidate.json", winner)
    _write_json(round_dir / "round_best_F1_F2.json", {"candidate_hash": winner["candidate_hash"], "F1": winner["F1"], "F2": winner["F2"]})
    _copy_winner_artifacts(round_dir, winner)
    return {"round_index": int(round_index), "winner": winner, "candidates": rows}
