from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakePool:
    parallelism = 4

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tasks: list[dict] = []

    def map_tasks(self, tasks):
        self.tasks.extend(tasks)
        results = []
        for task in tasks:
            candidate = str(task["candidate_hash"])
            if task["task_protocol"] == "build_smoke":
                engine = self.root / f"{candidate}.plan"
                engine.write_bytes(b"engine")
                results.append(
                    {
                        "candidate_hash": candidate,
                        "status": "ok",
                        "engine_path": str(engine),
                        "engine_hash": f"engine-{candidate}",
                        "physical_hash": f"physical-{candidate}",
                        "deployment_hash": f"deployment-{candidate}",
                        "deployment_identity": f"identity-{candidate}",
                        "precision_identity_passed": True,
                        "requested_precision_profile_hash": "precision",
                        "realized_precision_profile_hash": "precision",
                        "raw_precision_gene_hash": "precision",
                        "repaired_precision_gene_hash": "precision",
                    }
                )
            else:
                metadata = dict(task.get("deployment_metadata", {}))
                results.append(
                    {
                        **metadata,
                        "candidate_hash": candidate,
                        "status": "ok",
                        "mAP": 0.71,
                        "AP@0.7": 0.57,
                        "forward_p50_ms": 4.0,
                        "num_evaluated_frames": 1789,
                        "num_skipped_frames": 0,
                        "precision_identity_passed": True,
                    }
                )
        return results


def _endpoint(budget: float, candidate: str) -> dict:
    return {
        "target_bops": budget,
        "candidate_hash": candidate,
        "phenotype_hash": f"phenotype-{candidate}",
        "phenotype": {
            "pruned_unit_ids": [],
            "precision_profile": {},
            "metadata": {"phenotype_hash": f"phenotype-{candidate}"},
        },
        "genotype": {
            "width_genes": {},
            "precision_genes": {},
        },
        "metrics": {"R_BOPS": budget, "R_param": 0.9},
    }


def _winner(root: Path, candidate: str, generation: int, budget: float = 0.20):
    engine = root / f"winner-{candidate}.plan"
    engine.write_bytes(b"engine")
    return {
        "candidate_hash": candidate,
        "generation": generation,
        "budget": budget,
        "engine_path": str(engine),
        "engine_hash": f"engine-{candidate}",
        "physical_hash": f"physical-{candidate}",
        "deployment_hash": f"deployment-{candidate}",
        "deployment_identity": f"identity-{candidate}",
        "R_BOPS": budget,
        "R_param": 0.9,
        "raw_precision_gene_hash": "precision",
        "repaired_precision_gene_hash": "precision",
        "requested_precision_profile_hash": "precision",
        "realized_precision_profile_hash": "precision",
        "precision_identity_passed": True,
    }


def test_greedy_builds_only_one_unique_endpoint_per_budget_lineage(
    tmp_path: Path,
) -> None:
    from search.orchestration.legal_width_stage2 import (
        run_greedy_endpoint_full_validation,
    )

    pool = _FakePool(tmp_path)
    result = run_greedy_endpoint_full_validation(
        endpoints=[
            _endpoint(0.05, "a"),
            _endpoint(0.10, "a"),
            _endpoint(0.15, "b"),
        ],
        stage2_pool=pool,
        run_dir=tmp_path,
        required_evaluated_frames=1789,
        required_skipped_frames=0,
    )

    assert result["unique_deployment_count"] == 2
    assert result["budget_lineage_count"] == 3
    assert sum(task["task_protocol"] == "build_smoke" for task in pool.tasks) == 2
    assert sum(task["task_protocol"] == "full_validation" for task in pool.tasks) == 2


def test_all_unique_generation_winners_are_full_validated_once(
    tmp_path: Path,
) -> None:
    from search.orchestration.legal_width_stage2 import (
        run_generation_winner_full_validation,
    )

    pool = _FakePool(tmp_path)
    result = run_generation_winner_full_validation(
        generation_winners=[
            _winner(tmp_path, "a", generation=1),
            _winner(tmp_path, "a", generation=2),
            _winner(tmp_path, "b", generation=3),
        ],
        stage2_pool=pool,
        run_dir=tmp_path,
        required_evaluated_frames=1789,
        required_skipped_frames=0,
    )

    assert result["unique_deployment_count"] == 2
    assert result["lineage_reference_count"] == 3
    assert all(task["task_protocol"] == "full_validation" for task in pool.tasks)
    assert len(pool.tasks) == 2
    by_candidate = {
        row["candidate_hash"]: row for row in result["successful_candidates"]
    }
    assert len(by_candidate["a"]["lineage_references"]) == 2


def test_external_greedy_full_validation_is_reused_without_tasks(
    tmp_path: Path,
) -> None:
    from search.orchestration.legal_width_stage2 import (
        load_external_greedy_full_validation,
    )

    rows = []
    endpoints = []
    for candidate in ("a", "b"):
        engine = tmp_path / f"{candidate}.plan"
        engine.write_bytes(b"engine")
        endpoints.append({"candidate_hash": candidate})
        rows.append(
            {
                "candidate_hash": candidate,
                "status": "ok",
                "engine_path": str(engine),
                "full_validation_success": True,
                "evaluated_frames": 1789,
                "skipped_frames": 0,
                "precision_identity_passed": True,
                "mAP": 0.7,
            }
        )
    path = tmp_path / "greedy_full_validation.json"
    path.write_text(
        json.dumps(
            {
                "successful_count": 2,
                "successful_candidates": rows,
                "results": rows,
                "build_task_count": 2,
                "full_validation_task_count": 2,
            }
        ),
        encoding="utf-8",
    )

    result = load_external_greedy_full_validation(path, endpoints=endpoints)

    assert result["external_reuse"] is True
    assert result["current_run_build_task_count"] == 0
    assert result["current_run_full_validation_task_count"] == 0
    assert result["successful_count"] == 2
    assert result["external_manifest_sha256"]
