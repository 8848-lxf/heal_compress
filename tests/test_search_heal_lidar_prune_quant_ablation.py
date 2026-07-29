from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def _phenotype(*, pruned: list[str], precision: str):
    from search.candidate import CandidatePhenotype, PrecisionDecision

    contract = {
        "q0": {
            "requested_precision": precision,
            "legalized_precision": precision,
            "realized_precision": precision,
        }
    }
    return CandidatePhenotype(
        pruned_unit_ids=pruned,
        precision_profile={
            "backbone.conv": PrecisionDecision(precision, precision, "")
        },
        pruning_policy_version="test-pruning",
        precision_policy_version="test-precision",
        metadata={
            "group_keep_map_by_scope": {"scope": [1, 2]},
            "group_prune_map_by_scope": {"scope": [0]},
            "requested_group_profile": {"q0": precision},
            "stage1_legalized_group_profile": {"q0": precision},
            "quantization_group_contracts": contract,
        },
    )


def test_family_ablation_matrix_preserves_one_axis_and_deduplicates_builds() -> None:
    from search.ablation.heal_lidar_prune_quant import build_family_ablation_matrix

    phenotype = _phenotype(pruned=["unit::0"], precision="INT8")
    sources = [
        {
            "family_id": "heal_lidar_fcooper",
            "method": method,
            "budget": budget,
            "actual_bops": budget,
            "candidate_hash": method,
            "artifact_dir": f"/{method}",
            "phenotype": phenotype.to_dict(),
            "phenotype_hash": "source-phenotype",
            "engine_path": f"/{method}/candidate.plan",
            "engine_sha256": f"engine-{method}",
        }
        for method, budget in (("ga", 0.30), ("greedy", 0.25))
    ]

    rows = build_family_ablation_matrix(sources)
    by_key = {(row["method"], row["variant"]): row for row in rows}

    assert len(rows) == 6
    assert by_key[("ga", "prune_quant")]["requires_engine_build"] is False
    assert by_key[("greedy", "prune_quant")]["requires_engine_build"] is False
    assert by_key[("ga", "prune_only")]["requires_engine_build"] is True
    assert by_key[("greedy", "prune_only")]["requires_engine_build"] is False
    assert by_key[("greedy", "prune_only")]["engine_reuse_row_id"] == by_key[("ga", "prune_only")]["row_id"]
    assert by_key[("ga", "quant_only")]["requires_engine_build"] is True
    assert by_key[("greedy", "quant_only")]["requires_engine_build"] is False
    assert all(row["requires_fresh_evaluation"] for row in rows)

    prune_only = by_key[("ga", "prune_only")]["phenotype"]
    assert prune_only["pruned_unit_ids"] == ["unit::0"]
    assert prune_only["realized_precision_profile"] == {"backbone.conv": "FP32"}
    assert prune_only["metadata"]["heal_lidar_auxiliary_precision"] == "FP32"
    assert by_key[("ga", "prune_only")]["heal_lidar_auxiliary_precision"] == "FP32"
    quant_only = by_key[("ga", "quant_only")]["phenotype"]
    assert quant_only["pruned_unit_ids"] == []
    assert quant_only["realized_precision_profile"] == {"backbone.conv": "INT8"}
    assert "group_keep_map_by_scope" not in quant_only["metadata"]
    assert by_key[("ga", "quant_only")]["heal_lidar_auxiliary_precision"] == "FP16"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _accepted_artifact(root: Path, phenotype, *, candidate_hash: str) -> None:
    engine = root / "deployment/candidate.plan"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(candidate_hash.encode("utf-8"))
    engine_hash = hashlib.sha256(engine.read_bytes()).hexdigest()
    _write_json(root / "phenotype.json", phenotype.to_dict())
    _write_json(root / "deployment/precision_realization_acceptance.json", {"passed": True})
    _write_json(
        root / "candidate_stage2_result.json",
        {"status": "ok", "engine_sha256": engine_hash},
    )


def test_collect_family_candidates_uses_ga_stage2_and_greedy_stage2_full(tmp_path: Path) -> None:
    from search.ablation.heal_lidar_prune_quant import (
        collect_authoritative_family_candidates,
    )

    family = "heal_lidar_disco"
    ga = tmp_path / "ga"
    greedy = tmp_path / "greedy"
    context = {"family_id": family, "fixed_k": 29696, "checkpoint_hash": "checkpoint"}
    _write_json(ga / "context_report.json", context)
    _write_json(greedy / "context_report.json", context)
    phenotype = _phenotype(pruned=["unit::0"], precision="FP16")
    budgets = (0.10, 0.20)
    greedy_rows = []
    for index, budget in enumerate(budgets):
        ga_artifact = tmp_path / f"ga_artifact_{index}"
        greedy_artifact = tmp_path / f"greedy_artifact_{index}"
        _accepted_artifact(ga_artifact, phenotype, candidate_hash=f"ga-{index}")
        _accepted_artifact(greedy_artifact, phenotype, candidate_hash=f"greedy-{index}")
        _write_json(
            ga / f"round_{index:03d}/round_best_candidate.json",
            {
                "BOPS_target": budget,
                "R_BOPS_vs_FP32": budget + 0.001,
                "candidate_hash": f"ga-{index}",
                "artifact_dir": str(ga_artifact),
            },
        )
        greedy_rows.append(
            {
                "budgets": [budget],
                "candidate_hash": f"greedy-{index}",
                "artifact_dir": str(greedy_artifact),
                "proxy_metrics": {"R_bops_vs_fp32": budget - 0.001},
            }
        )
    _write_json(
        greedy / "greedy/full_validation_results.json", {"candidates": greedy_rows}
    )

    rows = collect_authoritative_family_candidates(
        family_id=family,
        ga_root=ga,
        greedy_root=greedy,
        repository_root=tmp_path,
        budgets=budgets,
    )

    assert [(row["method"], row["budget"]) for row in rows] == [
        ("ga", 0.20),
        ("ga", 0.10),
        ("greedy", 0.20),
        ("greedy", 0.10),
    ]
    assert all(Path(row["engine_path"]).name == "candidate.plan" for row in rows)
    assert all(row["stage2_result"]["status"] == "ok" for row in rows)


def test_collect_formal_family_candidates_uses_current_joint_result_layout(
    tmp_path: Path,
) -> None:
    from search.ablation.heal_lidar_prune_quant import (
        collect_formal_family_candidates,
    )

    family = "heal_lidar_fcooper"
    root = tmp_path / "formal"
    _write_json(
        root / "context_report.json",
        {"family_id": family, "fixed_k": 29696, "checkpoint_hash": "checkpoint"},
    )
    phenotype = _phenotype(pruned=["unit::0"], precision="INT8")
    results = {}
    for method, key in (("ga", "final_winner"), ("greedy", "greedy_anchor")):
        artifact = tmp_path / f"{method}_artifact"
        _accepted_artifact(artifact, phenotype, candidate_hash=method)
        results[key] = {
            "status": "ok",
            "requested_realized_exact": True,
            "complete_phenotype_hash": f"{method}-hash",
            "metadata": {"raw": {"source_artifact_dir": str(artifact)}},
        }
    _write_json(root / "reports/formal_ga_results.json", {"results": {"005": results}})

    rows = collect_formal_family_candidates(
        family_id=family,
        formal_root=root,
        repository_root=tmp_path,
        budgets=(0.05,),
    )

    assert [(row["method"], row["budget"]) for row in rows] == [
        ("ga", 0.05),
        ("greedy", 0.05),
    ]
    assert all(
        row["actual_bops_source"] == "pending_exact_phenotype_recompute"
        for row in rows
    )


def test_build_all_reuses_one_context_and_safely_resumes(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_heal_lidar_prune_quant_ablation as runner

    run_dir = tmp_path / "ablation"
    run_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "baselines:\n  strict_fp32_engine: /unused/fp32.plan\n"
        "full_validation:\n  num_frames: 1789\n  warmup_frames: 200\n  latency_rounds: 3\n",
        encoding="utf-8",
    )
    phenotype = _phenotype(pruned=["unit::0"], precision="FP16")
    rows = []
    for index in range(2):
        phenotype_path = run_dir / f"phenotype_{index}.json"
        _write_json(phenotype_path, phenotype.to_dict())
        row_id = f"owner-{index}"
        rows.append(
            {
                "row_id": row_id,
                "variant": "prune_only" if index == 0 else "quant_only",
                "requires_engine_build": True,
                "build_owner_row_id": row_id,
                "artifact_dir": str(run_dir / f"artifact_{index}"),
                "phenotype_path": str(phenotype_path),
                "deployment_config_signature": f"signature-{index}",
            }
        )
    rows.append(
        {
            **rows[0],
            "row_id": "reuse-owner-0",
            "requires_engine_build": False,
            "build_owner_row_id": "owner-0",
        }
    )
    _write_json(
        run_dir / "ablation_manifest.json",
        {"family_id": "heal_lidar_fcooper", "rows": rows},
    )
    calls = {"context": 0, "evaluator": 0, "build": 0}

    def fake_context(*_args, **_kwargs):
        calls["context"] += 1
        return object()

    class FakeEvaluator:
        def __init__(self, **_kwargs):
            calls["evaluator"] += 1

        def build_candidate_artifacts(
            self, _phenotype, *, output_dir, candidate_hash
        ):
            calls["build"] += 1
            artifact = Path(output_dir)
            artifact.mkdir(parents=True, exist_ok=True)
            engine = artifact / "deployment/candidate.plan"
            engine.parent.mkdir(parents=True, exist_ok=True)
            engine.write_bytes(candidate_hash.encode("utf-8"))
            result = {
                "status": "ok",
                "engine_path": str(engine),
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "evaluation_invoked": False,
            }
            _write_json(artifact / "candidate_build_result.json", result)
            return result

    monkeypatch.setattr(runner, "_build_context", fake_context)
    monkeypatch.setattr(runner, "HealLidarBaselineCandidateEvaluator", FakeEvaluator)
    args = SimpleNamespace(
        run_dir=run_dir,
        config=config_path,
        gpu_id=4,
    )

    first = runner._build_all(args)
    second = runner._build_all(args)

    assert first["status"] == "complete"
    assert first["completed_owner_count"] == 2
    assert second["status"] == "complete"
    assert second["initial_resume_hit_count"] == 2
    assert calls == {"context": 1, "evaluator": 1, "build": 2}
    progress = json.loads(
        (run_dir / "build_all_gpu_4_progress.json").read_text(encoding="utf-8")
    )
    assert progress["same_gpu_serial"] is True
    assert progress["evaluation_invoked"] is False
