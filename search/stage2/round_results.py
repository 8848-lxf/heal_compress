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
        physical_path = candidate_dir / "physical_hash.json"
        physical = _read_json(physical_path) if physical_path.is_file() else {}
        artifact_dir = str(score.get("artifact_dir") or candidate_dir)
        parameter_base = float(
            physical.get(
                "parameter_count_base",
                candidate.get("Parameter_Count_Base") or 0.0,
            )
            or 0.0
        )
        parameter_after = float(
            physical.get(
                "parameter_count_pruned",
                candidate.get("Parameter_Count_After") or 0.0,
            )
            or 0.0
        )
        r_size = _score_value(candidate, "R_Size")
        r_bops = _score_value(candidate, "R_BOPS")
        latency_ratio = _score_value(score, "R_latency_real")
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
                "BOPS_target": _score_value(candidate, "BOPS_Target"),
                "R_BOPS_vs_FP32": r_bops,
                "BOPS_abs_delta": _score_value(candidate, "BOPS_Abs_Delta"),
                "BOPS_compression_x": (
                    1.0 / r_bops if r_bops is not None and r_bops > 0.0 else None
                ),
                "R_Size_vs_FP32": r_size,
                "mixed_weight_compression_x": (
                    1.0 / r_size if r_size is not None and r_size > 0.0 else None
                ),
                "parameter_count_base": parameter_base,
                "parameter_count_after": parameter_after,
                "parameter_pruning_rate": (
                    1.0 - parameter_after / parameter_base
                    if parameter_base > 0.0
                    else _score_value(candidate, "Parameter_Pruning_Rate")
                ),
                "parameter_compression_x": (
                    parameter_base / parameter_after
                    if parameter_base > 0.0 and parameter_after > 0.0
                    else None
                ),
                "measured_speedup_vs_FP32": (
                    1.0 / latency_ratio
                    if latency_ratio is not None and latency_ratio > 0.0
                    else None
                ),
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
        "BOPS_target",
        "R_BOPS_vs_FP32",
        "BOPS_abs_delta",
        "BOPS_compression_x",
        "R_Size_vs_FP32",
        "mixed_weight_compression_x",
        "parameter_count_base",
        "parameter_count_after",
        "parameter_pruning_rate",
        "parameter_compression_x",
        "measured_speedup_vs_FP32",
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
        ("pruned_checkpoint.pth", "physical/pruned_checkpoint.pth"): "round_best_pruned_model.pth",
        ("pruned_fp32.onnx", "export/physical_fp32.onnx"): "round_best_pruned.onnx",
        ("pruned_qdq.onnx", "qdq/explicit_qdq.onnx"): "round_best_qdq.onnx",
        ("engine.plan", "deployment/candidate.plan"): "round_best.engine.plan",
        (
            "evaluation_300.json",
            "evaluation.json",
            "evaluation/evaluation.json",
        ): "round_best_evaluation_300.json",
    }
    for source_names, dst_name in copies.items():
        src = next((source / name for name in source_names if (source / name).is_file()), None)
        if src is None:
            raise FileNotFoundError(f"winner_artifact_missing:{source}:{'|'.join(source_names)}")
        shutil.copy2(src, round_dir / dst_name)


def write_round_stage2_results(
    run_dir: str | Path,
    *,
    round_index: int = 0,
    allow_no_success: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)
    round_dir = root / f"round_{int(round_index):03d}"
    rows = collect_round_stage2_results(root, round_index=round_index)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        if not allow_no_success:
            raise RuntimeError(f"no_successful_stage2_candidates:round_{int(round_index):03d}")
        _write_csv(round_dir / "stage2_top5_results.csv", rows)
        _write_json(
            round_dir / "stage2_top5_results.json",
            {
                "round_index": int(round_index),
                "winner": None,
                "candidates": rows,
                "status": "no_successful_stage2_candidate",
            },
        )
        _write_markdown(round_dir / "stage2_top5_results.md", rows, {})
        _write_json(
            round_dir / "round_stage2_failure.json",
            {
                "round_index": int(round_index),
                "status": "no_successful_stage2_candidate",
                "candidate_count": len(rows),
                "failure_reasons": [
                    row.get("failure_reason", row.get("status", "")) for row in rows
                ],
            },
        )
        return {
            "round_index": int(round_index),
            "winner": None,
            "candidates": rows,
            "status": "no_successful_stage2_candidate",
        }
    winner = ok_rows[0]
    _write_csv(round_dir / "stage2_top5_results.csv", rows)
    _write_json(round_dir / "stage2_top5_results.json", {"round_index": int(round_index), "winner": winner, "candidates": rows})
    _write_markdown(round_dir / "stage2_top5_results.md", rows, winner)
    _write_json(round_dir / "round_best_candidate.json", winner)
    _write_json(round_dir / "round_best_F1_F2.json", {"candidate_hash": winner["candidate_hash"], "F1": winner["F1"], "F2": winner["F2"]})
    _copy_winner_artifacts(round_dir, winner)
    return {"round_index": int(round_index), "winner": winner, "candidates": rows}
