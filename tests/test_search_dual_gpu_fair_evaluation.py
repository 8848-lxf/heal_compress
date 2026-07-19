from __future__ import annotations

import json
from pathlib import Path

import pytest

from search.integration.dual_gpu_fair_evaluation import (
    ablation_contribution_rows,
    aggregate_repeated_results,
    build_evaluation_inventory,
    build_method_ablation_inventory,
    cross_gpu_differences,
    evaluation_frame_latency_rows,
    ensure_evaluation_only_output,
    summarize_results,
    validate_evaluation_result,
    verify_existing_engine,
)


REPORTS = (
    "physical_validation.json",
    "physical_plan_validation.json",
    "engine_structure_validation.json",
    "precision_realization_validation.json",
    "merge_precision_realization.json",
    "production_qdq_boundary_audit.json",
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _artifact(path: Path) -> None:
    import hashlib

    path.mkdir(parents=True, exist_ok=True)
    engine = path / "engine.plan"
    engine.write_bytes(b"immutable-engine")
    _json(
        path / "deployment_manifest.json",
        {
            "engine_hash": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "deployment_hash": "deployment",
            "physical_hash": "physical",
        },
    )
    for name in REPORTS:
        _json(path / name, {"passed": True})
    _json(path / "realized_precision_profile.json", {"a": "INT8", "b": "FP16"})


def test_inventory_orders_fp32_then_ga_then_greedy(tmp_path: Path) -> None:
    root = tmp_path / "ablation"
    baseline = root / "full_validation/baselines/original_strict_fp32"
    _artifact(baseline)
    candidates = []
    for method in ("greedy", "ga"):
        for budget in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            artifact = tmp_path / method / f"{budget:.2f}"
            _artifact(artifact)
            candidates.append(
                {
                    "method": method,
                    "budget": budget,
                    "actual_bops": budget,
                    "source_artifact_dir": str(artifact),
                    "candidate_hash": f"{method}-{budget}",
                }
            )
    _json(root / "ablation_manifest.json", {"candidates": candidates})
    rows = build_evaluation_inventory(ablation_root=root)
    assert [row["item_id"] for row in rows] == [
        "fp32_original",
        "ga_bops_0.30",
        "ga_bops_0.25",
        "ga_bops_0.20",
        "ga_bops_0.15",
        "ga_bops_0.10",
        "ga_bops_0.05",
        "greedy_bops_0.30",
        "greedy_bops_0.25",
        "greedy_bops_0.20",
        "greedy_bops_0.15",
        "greedy_bops_0.10",
        "greedy_bops_0.05",
    ]


def test_existing_engine_requires_hash_and_all_acceptance_reports(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _artifact(artifact)
    result = verify_existing_engine({"artifact_dir": str(artifact)})
    assert result["engine_size_bytes"] > 0
    assert result["int8_count"] == 1
    (artifact / "merge_precision_realization.json").write_text(
        json.dumps({"passed": False}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="acceptance_failed"):
        verify_existing_engine({"artifact_dir": str(artifact)})


def test_evaluation_protocol_and_evaluation_only_guard(tmp_path: Path) -> None:
    result = {
        "status": "ok",
        "num_evaluated_frames": 2,
        "num_skipped_frames": 0,
        "dataloader_num_workers": 8,
        "reset_after_warmup": True,
        "eval_manifest_hash": "manifest",
        "cuda_postprocess_audit": {"passed": True},
        "latency_rows": [{"warmup": True}, {"warmup": False}, {"warmup": False}],
    }
    validate_evaluation_result(
        result, num_frames=2, warmup_frames=1, manifest_hash="manifest"
    )
    (tmp_path / "evaluation.json").write_text("{}", encoding="utf-8")
    ensure_evaluation_only_output(tmp_path)
    (tmp_path / "unexpected.plan").write_bytes(b"no")
    with pytest.raises(RuntimeError, match="build_artifact"):
        ensure_evaluation_only_output(tmp_path)


def test_summary_uses_fp32_from_same_gpu() -> None:
    common = {
        "sequence_index": 0,
        "actual_bops": 1.0,
        "AP@0.3": 1.0,
        "AP@0.5": 1.0,
        "AP@0.7": 1.0,
        "mAP": 1.0,
        "forward_p90_ms": 1.0,
        "forward_p99_ms": 1.0,
        "int8_count": 0,
        "fp16_count": 0,
        "fp32_count": 1,
        "num_evaluated_frames": 1,
        "num_skipped_frames": 0,
        "evaluated_frame_ids": ["frame"],
        "engine_sha256": "hash",
        "gpu_cleanup": {"passed": True},
    }
    rows = [
        {**common, "gpu_id": 0, "item_id": "fp32", "method": "fp32", "budget": None, "forward_p50_ms": 10.0},
        {**common, "gpu_id": 0, "item_id": "ga", "method": "ga", "budget": 0.3, "forward_p50_ms": 5.0},
        {**common, "gpu_id": 1, "item_id": "fp32", "method": "fp32", "budget": None, "forward_p50_ms": 12.0},
        {**common, "gpu_id": 1, "item_id": "ga", "method": "ga", "budget": 0.3, "forward_p50_ms": 4.0},
    ]
    summary = summarize_results(rows)
    assert summary[1]["speedup_vs_same_gpu_fp32"] == 2.0
    assert summary[3]["speedup_vs_same_gpu_fp32"] == 3.0
    cross = cross_gpu_differences(summary)
    assert len(cross) == 2
    assert cross[0]["same_engine_hash"]
    assert cross[1]["p50_ratio_gpu1_vs_gpu0"] == 0.8


def test_method_ablation_inventory_uses_source_pq_and_local_p_q(tmp_path: Path) -> None:
    root = tmp_path / "ablation"
    baseline = root / "full_validation/baselines/original_strict_fp32"
    _artifact(baseline)
    rows = []
    for budget in (0.30, 0.25, 0.20, 0.15, 0.10, 0.05):
        for variant in ("prune_quant", "prune_only", "quant_only"):
            local = root / "candidates/ga" / f"{budget:.2f}" / variant
            source = tmp_path / "source" / f"{budget:.2f}"
            _artifact(local)
            _artifact(source)
            rows.append(
                {
                    "method": "ga",
                    "budget": budget,
                    "actual_bops": budget,
                    "variant": variant,
                    "artifact_dir": str(local),
                    "source_artifact_dir": str(source),
                    "source_candidate_hash": "hash",
                    "row_id": f"{budget}-{variant}",
                }
            )
    _json(root / "ablation_results.json", {"rows": rows})
    inventory = build_method_ablation_inventory(ablation_root=root, method="ga")
    assert len(inventory) == 19
    assert inventory[1]["variant"] == "prune_quant"
    assert inventory[1]["artifact_dir"].startswith(str(tmp_path / "source"))
    assert inventory[2]["variant"] == "prune_only"
    assert inventory[2]["artifact_dir"].startswith(str(root / "candidates"))


def test_ablation_contribution_uses_assigned_fp32_baseline() -> None:
    def row(variant: str, value: float) -> dict[str, object]:
        return {
            "assigned_method": "ga",
            "variant": variant,
            "gpu_id": 0,
            "budget": None if variant == "fp32" else 0.3,
            "actual_bops": 1.0 if variant == "fp32" else 0.3,
            "mAP": value,
            "forward_p50_ms": 10.0,
            "speedup_vs_same_gpu_fp32": 1.0,
        }

    result = ablation_contribution_rows(
        [
            row("fp32", 0.70),
            row("prune_quant", 0.65),
            row("prune_only", 0.68),
            row("quant_only", 0.66),
        ]
    )[0]
    assert result["prune_quant_delta_vs_fp32"] == pytest.approx(-0.05)
    assert result["pq_interaction_mAP"] == pytest.approx(0.01)


def test_per_frame_latency_excludes_warmup_and_preserves_components() -> None:
    rows = evaluation_frame_latency_rows(
        {
            "num_evaluated_frames": 2,
            "latency_rows": [
                {"frame_id": "warmup", "warmup": True, "forward_ms": 1, "postprocess_ms": 2, "total_ms": 3},
                {"frame_id": "a", "warmup": False, "forward_ms": 4, "postprocess_ms": 5, "total_ms": 9},
                {"frame_id": "b", "warmup": False, "forward_ms": 6, "postprocess_ms": 7, "total_ms": 13},
            ],
        },
        metadata={"engine": "test"},
    )
    assert [row["frame_id"] for row in rows] == ["a", "b"]
    assert rows[0]["forward_ms"] == 4.0
    assert rows[0]["postprocess_ms"] == 5.0
    assert rows[0]["total_ms"] == 9.0


def test_repeat_aggregation_reports_mean_and_std() -> None:
    rows = []
    for repeat_index, value in enumerate((1.0, 2.0)):
        row = {
            "assigned_method": "ga",
            "variant": "fp32",
            "budget": None,
            "gpu_id": 0,
            "repeat_index": repeat_index,
            "engine_sha256": "engine",
            "frame_order_hash": "frames",
            "actual_bops": 1.0,
        }
        for metric in (
            "AP@0.3", "AP@0.5", "AP@0.7", "mAP", "forward_mean_ms",
            "forward_p50_ms", "forward_p90_ms", "forward_p99_ms",
            "postprocess_mean_ms", "postprocess_p50_ms", "postprocess_p90_ms",
            "postprocess_p99_ms", "total_mean_ms", "total_p50_ms",
            "total_p90_ms", "total_p99_ms", "speedup_vs_same_gpu_fp32",
        ):
            row[metric] = value
        rows.append(row)
    result = aggregate_repeated_results(rows, repeat_count=2)[0]
    assert result["mAP_across_runs_mean"] == 1.5
    assert result["mAP_across_runs_std"] == 0.5
