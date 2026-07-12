from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_round_stage2_results_selects_lowest_f2_and_copies_winner_artifacts(tmp_path: Path) -> None:
    from search.stage2.round_results import write_round_stage2_results

    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_000"
    round_dir.mkdir(parents=True)
    manifest = {
        "candidates": [
            {"candidate_rank": 0, "repaired_phenotype_hash": "slow", "repaired_F1": 0.1},
            {"candidate_rank": 1, "repaired_phenotype_hash": "fast", "repaired_F1": 0.2},
        ]
    }
    (round_dir / "repaired_top5_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for candidate_hash, f2 in [("slow", 2.0), ("fast", 1.0)]:
        candidate_dir = round_dir / "stage2" / candidate_hash
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "stage2_score.json").write_text(
            json.dumps({"status": "ok", "F2": f2, "mAP": 0.7, "forward_p50_ms": 3.0}),
            encoding="utf-8",
        )
        (candidate_dir / "pruned_checkpoint.pth").write_bytes(f"model-{candidate_hash}".encode())
        (candidate_dir / "pruned_fp32.onnx").write_bytes(f"onnx-{candidate_hash}".encode())
        (candidate_dir / "pruned_qdq.onnx").write_bytes(f"qdq-{candidate_hash}".encode())
        (candidate_dir / "engine.plan").write_bytes(f"engine-{candidate_hash}".encode())
        (candidate_dir / "evaluation_300.json").write_text(json.dumps({"candidate": candidate_hash}), encoding="utf-8")

    result = write_round_stage2_results(run_dir, round_index=0)

    assert result["winner"]["candidate_hash"] == "fast"
    assert (round_dir / "stage2_top5_results.csv").is_file()
    assert (round_dir / "stage2_top5_results.json").is_file()
    assert (round_dir / "stage2_top5_results.md").is_file()
    assert json.loads((round_dir / "round_best_candidate.json").read_text())["candidate_hash"] == "fast"
    assert (round_dir / "round_best_pruned_model.pth").read_bytes() == b"model-fast"
    assert (round_dir / "round_best_pruned.onnx").read_bytes() == b"onnx-fast"
    assert (round_dir / "round_best_qdq.onnx").read_bytes() == b"qdq-fast"
    assert (round_dir / "round_best.engine.plan").read_bytes() == b"engine-fast"
    assert json.loads((round_dir / "round_best_F1_F2.json").read_text()) == {
        "candidate_hash": "fast",
        "F1": 0.2,
        "F2": 1.0,
    }


def test_round_stage2_results_copies_winner_from_cached_artifact_dir(tmp_path: Path) -> None:
    from search.stage2.round_results import write_round_stage2_results

    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_001"
    round_dir.mkdir(parents=True)
    cached_dir = run_dir / "round_000" / "stage2" / "cached"
    cached_dir.mkdir(parents=True)
    for name in ["pruned_checkpoint.pth", "pruned_fp32.onnx", "pruned_qdq.onnx", "engine.plan"]:
        (cached_dir / name).write_bytes(f"cached-{name}".encode())
    (cached_dir / "evaluation_300.json").write_text(json.dumps({"cached": True}), encoding="utf-8")
    (round_dir / "repaired_top5_manifest.json").write_text(
        json.dumps({"candidates": [{"candidate_rank": 0, "repaired_phenotype_hash": "cached", "repaired_F1": 0.1}]}),
        encoding="utf-8",
    )
    candidate_dir = round_dir / "stage2" / "cached"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "stage2_score.json").write_text(
        json.dumps({"status": "ok", "F2": 0.5, "artifact_dir": str(cached_dir)}),
        encoding="utf-8",
    )

    write_round_stage2_results(run_dir, round_index=1)

    assert (round_dir / "round_best_pruned_model.pth").read_bytes() == b"cached-pruned_checkpoint.pth"
