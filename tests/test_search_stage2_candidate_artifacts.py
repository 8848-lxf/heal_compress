from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_candidate_summary_artifacts_include_evaluation_objective_and_hashes(tmp_path: Path) -> None:
    from search.candidate import CandidatePhenotype
    from search.stage2.candidate_artifacts import write_candidate_summary_artifacts
    from search.stage2.objective import Stage2ObjectiveConfig

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "evaluation.json").write_text(json.dumps({"status": "ok", "mAP": 0.5, "forward_p50_ms": 2.0}), encoding="utf-8")
    (candidate_dir / "engine.plan").write_bytes(b"engine")
    (candidate_dir / "pruned_fp32.onnx").write_bytes(b"onnx")
    (candidate_dir / "pruned_qdq.onnx").write_bytes(b"qdq")
    score = {"status": "ok", "F2": 0.2, "L_map_real": 0.0, "R_latency_real": 1.0}

    write_candidate_summary_artifacts(
        candidate_dir,
        candidate_hash="abc",
        phenotype=CandidatePhenotype(),
        stage2_score=score,
        objective_config=Stage2ObjectiveConfig(eta_map=0.8, eta_latency=0.2, latency_metric="forward_p50_ms", tau_ap=0.02),
        stage1_manifest_record={"repaired_F1": 0.1},
    )

    manifest = json.loads((candidate_dir / "candidate_manifest.json").read_text())
    objective = json.loads((candidate_dir / "stage2_objective_report.json").read_text())
    hashes = json.loads((candidate_dir / "artifact_hashes.json").read_text())

    assert manifest["candidate_hash"] == "abc"
    assert manifest["stage1_manifest_record"]["repaired_F1"] == 0.1
    assert json.loads((candidate_dir / "evaluation_300.json").read_text())["mAP"] == 0.5
    assert objective["formula"] == "eta_AP * L_AP + eta_latency * R_latency"
    assert objective["F2"] == 0.2
    assert "engine.plan" in hashes["artifacts"]


def test_candidate_summary_artifacts_can_skip_existing_files_for_resume(tmp_path: Path) -> None:
    from search.candidate import CandidatePhenotype
    from search.stage2.candidate_artifacts import write_candidate_summary_artifacts
    from search.stage2.objective import Stage2ObjectiveConfig

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "evaluation.json").write_text(json.dumps({"status": "ok", "mAP": 0.5}), encoding="utf-8")
    write_candidate_summary_artifacts(
        candidate_dir,
        candidate_hash="abc",
        phenotype=CandidatePhenotype(),
        stage2_score={"status": "ok", "F2": 0.2},
        objective_config=Stage2ObjectiveConfig(),
    )
    target = candidate_dir / "candidate_manifest.json"
    before = target.stat().st_mtime_ns

    write_candidate_summary_artifacts(
        candidate_dir,
        candidate_hash="abc",
        phenotype=CandidatePhenotype(pruned_unit_ids=["new"]),
        stage2_score={"status": "ok", "F2": 0.3},
        objective_config=Stage2ObjectiveConfig(),
        overwrite=False,
    )

    assert target.stat().st_mtime_ns == before
    assert json.loads(target.read_text())["phenotype"]["pruned_unit_ids"] == []
