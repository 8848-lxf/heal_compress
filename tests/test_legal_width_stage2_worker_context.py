from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_legal_width_worker_requires_global_pruning_context() -> None:
    from search.stage2.candidate_worker import _requires_global_pruning_context

    assert _requires_global_pruning_context(
        {"pruning": {"gene_type": "legal_keep_width"}}
    )
    assert _requires_global_pruning_context(
        {"joint_taylor_anchor_sweep": {"enabled": True}}
    )
    assert not _requires_global_pruning_context(
        {"pruning": {"gene_type": "coupled_channel_keep_mask"}}
    )


def test_evaluation_only_task_reuses_existing_engine(tmp_path: Path) -> None:
    from search.stage2.candidate_worker import _evaluate_task

    engine = tmp_path / "engine.plan"
    engine.write_bytes(b"engine")

    class Evaluator:
        def _evaluate_engine(self, engine_path, output_dir):
            assert Path(engine_path) == engine
            assert Path(output_dir).name == "full"
            return {
                "status": "ok",
                "mAP": 0.71,
                "num_evaluated_frames": 1789,
                "num_skipped_frames": 0,
            }

    precision_hash = "precision-hash"
    result = _evaluate_task(
        Evaluator(),
        {
            "candidate_hash": "candidate",
            "phenotype": {
                "pruned_unit_ids": [],
                "precision_profile": {},
                "metadata": {},
            },
            "output_dir": str(tmp_path / "full"),
            "evaluation_only_engine_path": str(engine),
            "deployment_metadata": {
                "raw_precision_gene_hash": precision_hash,
                "repaired_precision_gene_hash": precision_hash,
                "requested_precision_profile_hash": precision_hash,
                "realized_precision_profile_hash": precision_hash,
                "precision_identity_passed": True,
            },
        },
        4,
    )

    assert result["status"] == "ok"
    assert result["precision_identity_passed"] is True
    assert result["engine_path"] == str(engine)
