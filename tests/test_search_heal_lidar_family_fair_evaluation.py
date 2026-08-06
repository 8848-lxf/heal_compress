from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from search.integration.heal_lidar_family_fair_evaluation import (
    ablation_contribution_rows,
    aggregate_contribution_rows,
    aggregate_five_repeat_results,
    build_family_evaluation_inventories,
    evaluate_existing_family_engine,
    evaluation_frame_latency_rows,
    load_resumable_family_evaluation,
    summarize_repeat_results,
    validate_family_evaluation_result,
)
from scripts.run_heal_lidar_family_split_gpu_fair_evaluation import (
    _gpu_pools,
    _seed_method_results,
)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _engine(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _build_inventory_fixture(root: Path) -> tuple[Path, Path, Path]:
    baseline = root / "baseline.plan"
    _engine(baseline, b"fp32")
    baseline_checkpoint = root / "baseline.pth"
    _engine(baseline_checkpoint, b"fp32-checkpoint")
    rows = []
    for method in ("ga", "greedy"):
        for variant in ("prune_quant", "prune_only", "quant_only"):
            artifact = root / "artifacts" / method / variant
            engine = artifact / "deployment/candidate.plan"
            digest = _engine(engine, f"{method}-{variant}".encode())
            checkpoint = artifact / "physical/pruned_checkpoint.pth"
            _engine(checkpoint, f"{method}-{variant}-checkpoint".encode())
            phenotype = artifact / "phenotype.json"
            _json(
                phenotype,
                {
                    "realized_precision_profile": {
                        "a": "INT8" if variant != "prune_only" else "FP32",
                        "b": "FP16" if variant != "prune_only" else "FP32",
                    }
                },
            )
            parameter_result = {
                "status": "ok",
                "physical_parameter_count_before": 100,
                "physical_parameter_count_after": 80,
                "physical_parameter_pruning_ratio": 0.2,
            }
            if variant != "prune_quant":
                parameter_result["evaluation_invoked"] = False
            _json(
                artifact
                / (
                    "candidate_stage2_result.json"
                    if variant == "prune_quant"
                    else "candidate_build_result.json"
                ),
                parameter_result,
            )
            rows.append(
                {
                    "row_id": f"{method}-{variant}",
                    "family_id": "heal_lidar_fcooper",
                    "method": method,
                    "budget": 0.30,
                    "actual_bops": 0.301,
                    "variant": variant,
                    "phenotype_path": str(phenotype),
                    "artifact_dir": str(artifact),
                    "engine_path": str(engine),
                    "engine_sha256": digest,
                }
            )
    _json(root / "engine_inventory.json", {"family_id": "heal_lidar_fcooper", "rows": rows})
    return root, baseline, baseline_checkpoint


def test_family_inventory_uses_engine_inventory_and_artifact_metadata(
    tmp_path: Path,
) -> None:
    root, baseline, baseline_checkpoint = _build_inventory_fixture(tmp_path / "build")
    inventories = build_family_evaluation_inventories(
        build_root=root,
        baseline_engine_path=baseline,
        baseline_checkpoint_path=baseline_checkpoint,
        family_id="heal_lidar_fcooper",
        budgets=(0.30,),
    )
    assert list(inventories) == ["ga", "greedy"]
    assert [row["variant"] for row in inventories["ga"]] == [
        "fp32",
        "prune_quant",
        "prune_only",
        "quant_only",
    ]
    assert inventories["ga"][0]["fp32_count"] == 2
    assert inventories["ga"][1]["int8_count"] == 1
    assert inventories["ga"][1]["fp16_count"] == 1
    assert inventories["ga"][1]["parameter_count_base"] == 100
    assert inventories["ga"][1]["parameter_count_pruned"] == 80
    assert inventories["ga"][1]["parameter_reduction"] == 0.2


def test_family_inventory_rejects_engine_hash_mismatch(tmp_path: Path) -> None:
    root, baseline, baseline_checkpoint = _build_inventory_fixture(tmp_path / "build")
    payload = json.loads((root / "engine_inventory.json").read_text(encoding="utf-8"))
    payload["rows"][0]["engine_sha256"] = "bad"
    _json(root / "engine_inventory.json", payload)
    with pytest.raises(RuntimeError, match="engine_hash_mismatch"):
        build_family_evaluation_inventories(
            build_root=root,
            baseline_engine_path=baseline,
            baseline_checkpoint_path=baseline_checkpoint,
            family_id="heal_lidar_fcooper",
            budgets=(0.30,),
        )


def _evaluation_result() -> dict[str, Any]:
    return {
        "status": "ok",
        "AP@0.3": 0.7,
        "AP@0.5": 0.6,
        "AP@0.7": 0.5,
        "mAP": 0.6,
        "forward_mean_ms": 3.0,
        "forward_p50_ms": 3.0,
        "forward_p90_ms": 3.2,
        "forward_p99_ms": 3.3,
        "postprocess_mean_ms": 2.0,
        "postprocess_p50_ms": 2.0,
        "postprocess_p90_ms": 2.2,
        "postprocess_p99_ms": 2.3,
        "total_mean_ms": 5.0,
        "total_p50_ms": 5.0,
        "total_p90_ms": 5.2,
        "total_p99_ms": 5.3,
        "num_warmup_frames": 1,
        "num_evaluated_frames": 2,
        "num_skipped_frames": 0,
        "warmup_frame_ids": ["a"],
        "evaluated_frame_ids": ["a", "b"],
        "skipped_frame_ids": [],
        "fixed_k": None,
        "input_contract": "heal_post_scatter_dynamic_frontend_v1",
        "runtime_max_k_dependency": False,
        "eval_manifest_hash": "manifest",
        "reset_after_warmup": True,
        "dataloader_num_workers": 8,
        "cuda_postprocess_audit": {"passed": True},
        "latency_rows": [
            {
                "frame_id": "a",
                "phase": "warmup",
                "forward_ms": 1.0,
                "postprocess_ms": 2.0,
            },
            {
                "frame_id": "a",
                "phase": "evaluation",
                "forward_ms": 3.0,
                "postprocess_ms": 4.0,
            },
            {
                "frame_id": "b",
                "phase": "evaluation",
                "forward_ms": 5.0,
                "postprocess_ms": 6.0,
            },
        ],
    }


def test_family_protocol_accepts_phase_schema_and_csv_excludes_warmup() -> None:
    result = _evaluation_result()
    validate_family_evaluation_result(
        result,
        num_frames=2,
        warmup_frames=1,
        manifest_hash="manifest",
    )
    rows = evaluation_frame_latency_rows(result, metadata={"test": True})
    assert [row["frame_id"] for row in rows] == ["a", "b"]
    assert [row["total_ms"] for row in rows] == [7.0, 11.0]


def test_existing_family_engine_evaluation_locks_and_cleans(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "candidate.plan"
    digest = _engine(engine, b"immutable")
    checkpoint = tmp_path / "candidate.pth"
    checkpoint_digest = _engine(checkpoint, b"frontend")
    manifest = tmp_path / "manifest.json"
    _json(
        manifest,
        {
            "manifest_hash": "manifest",
            "reset_after_warmup": True,
            "warmup_frame_ids": ["a"],
            "evaluation_frame_ids": ["a", "b"],
        },
    )
    calls = []

    def evaluator(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _evaluation_result()

    result = evaluate_existing_family_engine(
        source={
            "family_id": "heal_lidar_disco",
            "assigned_method": "ga",
            "sequence_index": 1,
            "item_id": "ga_bops_0.30_prune_quant",
            "variant": "prune_quant",
            "budget": 0.30,
            "actual_bops": 0.301,
            "engine_path": str(engine),
            "engine_sha256": digest,
            "frontend_checkpoint_path": str(checkpoint),
            "frontend_checkpoint_sha256": checkpoint_digest,
        },
        gpu_id=4,
        repeat_index=0,
        output_dir=tmp_path / "evaluation",
        model_config=tmp_path / "config.yaml",
        heal_root=tmp_path,
        tensorrt_root=tmp_path,
        plugin_path=tmp_path / "plugin.so",
        eval_manifest_path=manifest,
        num_frames=2,
        warmup_frames=1,
        evaluator=evaluator,
        snapshotter=lambda gpu: {
            "gpu_id": gpu,
            "memory_used_mib": 0,
        },
        cleaner=lambda **kwargs: {"passed": True, "after": {"memory_used_mib": 0}},
    )
    assert calls[0]["physical_gpu_id"] == 4
    assert calls[0]["dataloader_num_workers"] == 8
    assert calls[0]["input_contract"] == "heal_post_scatter_dynamic_frontend_v1"
    assert calls[0]["fixed_k"] is None
    assert calls[0]["checkpoint_path"] == checkpoint.resolve()
    assert result["source_engine_unchanged"]
    csv_rows = (tmp_path / "evaluation/per_frame_latency.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(csv_rows) == 3
    assert engine.read_bytes() == b"immutable"

    _json(tmp_path / "evaluation/evaluation.json", _evaluation_result())
    resumed = load_resumable_family_evaluation(
        source={
            "family_id": "heal_lidar_disco",
            "assigned_method": "ga",
            "sequence_index": 1,
            "item_id": "ga_bops_0.30_prune_quant",
            "variant": "prune_quant",
            "budget": 0.30,
            "actual_bops": 0.301,
            "engine_path": str(engine),
            "engine_sha256": digest,
            "frontend_checkpoint_path": str(checkpoint),
            "frontend_checkpoint_sha256": checkpoint_digest,
        },
        gpu_id=4,
        repeat_index=0,
        output_dir=tmp_path / "evaluation",
        eval_manifest_path=manifest,
        num_frames=2,
        warmup_frames=1,
    )
    assert resumed["resume_hit"] is True
    assert resumed["mAP"] == 0.6


def test_resume_fails_closed_on_partial_directory(tmp_path: Path) -> None:
    engine = tmp_path / "candidate.plan"
    digest = _engine(engine, b"immutable")
    checkpoint = tmp_path / "candidate.pth"
    checkpoint_digest = _engine(checkpoint, b"frontend")
    manifest = tmp_path / "manifest.json"
    _json(manifest, {"manifest_hash": "manifest", "reset_after_warmup": True})
    partial = tmp_path / "partial"
    partial.mkdir()
    _json(partial / "evaluation.json", _evaluation_result())
    with pytest.raises(RuntimeError, match="resume_incomplete_evaluation"):
        load_resumable_family_evaluation(
            source={
                "family_id": "heal_lidar_fcooper",
                "assigned_method": "greedy",
                "sequence_index": 0,
                "item_id": "greedy_fp32_original",
                "variant": "fp32",
                "engine_path": str(engine),
                "engine_sha256": digest,
                "frontend_checkpoint_path": str(checkpoint),
                "frontend_checkpoint_sha256": checkpoint_digest,
            },
            gpu_id=3,
            repeat_index=0,
            output_dir=partial,
            eval_manifest_path=manifest,
            num_frames=2,
            warmup_frames=1,
        )


def test_seed_method_results_validates_and_hardlinks_complete_stream(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "candidate.plan"
    digest = _engine(engine, b"immutable")
    checkpoint = tmp_path / "candidate.pth"
    checkpoint_digest = _engine(checkpoint, b"frontend")
    manifest = tmp_path / "eval_manifest.json"
    _json(
        manifest,
        {
            "manifest_hash": "manifest",
            "reset_after_warmup": True,
            "warmup_frame_ids": ["a"],
            "evaluation_frame_ids": ["a", "b"],
        },
    )
    source = {
        "family_id": "heal_lidar_disco",
        "assigned_method": "greedy",
        "sequence_index": 0,
        "item_id": "greedy_fp32_original",
        "variant": "fp32",
        "budget": None,
        "actual_bops": 1.0,
        "engine_path": str(engine),
        "engine_sha256": digest,
        "frontend_checkpoint_path": str(checkpoint),
        "frontend_checkpoint_sha256": checkpoint_digest,
    }
    source_root = tmp_path / "source_run"
    source_dir = (
        source_root
        / "repeat_00/gpu_5_greedy/00_greedy_fp32_original"
    )
    evaluate_existing_family_engine(
        source=source,
        gpu_id=5,
        repeat_index=0,
        output_dir=source_dir,
        model_config=tmp_path / "config.yaml",
        heal_root=tmp_path,
        tensorrt_root=tmp_path,
        plugin_path=tmp_path / "plugin.so",
        eval_manifest_path=manifest,
        num_frames=2,
        warmup_frames=1,
        evaluator=lambda **kwargs: _evaluation_result(),
        snapshotter=lambda gpu: {"gpu_id": gpu, "memory_used_mib": 0},
        cleaner=lambda **kwargs: {"passed": True, "after": {"memory_used_mib": 0}},
    )
    _json(source_dir / "evaluation.json", _evaluation_result())
    _json(
        source_root / "family_fair_evaluation_manifest.json",
        {
            "family_id": "heal_lidar_disco",
            "gpu_assignment": {"ga": 2, "greedy": 5},
            "protocol": {
                "num_frames": 2,
                "warmup_frames": 1,
                "latency_rounds": 3,
                "dataloader_num_workers": 8,
                "cuda_postprocess": True,
                "fixed_k": None,
                "input_contract": "heal_post_scatter_dynamic_frontend_v1",
                "max_agents": 2,
                "eval_manifest_hash": "manifest",
                "eval_manifest_file_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
            },
            "inventories": {"greedy": [source]},
        },
    )
    args = Namespace(
        seed_method_root=[f"greedy={source_root}"],
        family_id="heal_lidar_disco",
        eval_manifest=manifest,
        num_frames=2,
        warmup_frames=1,
        latency_rounds=3,
        max_agents=2,
        repeat_count=1,
    )
    destination_root = tmp_path / "destination_run"
    seeded = _seed_method_results(
        args=args,
        run_dir=destination_root,
        inventories={"greedy": [source]},
        gpu_assignment={"greedy": 5},
    )
    destination = (
        destination_root
        / "repeat_00/gpu_5_greedy/00_greedy_fp32_original"
    )
    assert seeded["greedy"]["validated_and_seeded_count"] == 1
    assert (destination / "evaluation.json").is_file()
    assert (destination / "evaluation.json").stat().st_ino == (
        source_dir / "evaluation.json"
    ).stat().st_ino
    assert (destination_root / "seeded_method_resume.json").is_file()


def test_gpu_pools_allow_whole_repeat_shards_and_reject_overlap() -> None:
    args = Namespace(
        ga_gpu=2,
        ga_extra_gpu=[3],
        greedy_gpu=5,
        greedy_extra_gpu=[],
    )
    assert _gpu_pools(args) == {"ga": [2, 3], "greedy": [5]}
    args.greedy_extra_gpu = [3]
    with pytest.raises(RuntimeError, match="requires_distinct_gpus"):
        _gpu_pools(args)
    args.seed_method_root = ["greedy=/tmp/complete-greedy"]
    assert _gpu_pools(args) == {"ga": [2, 3], "greedy": [5, 3]}


def test_seed_method_prefix_stops_before_first_rejected_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "eval_manifest.json"
    _json(
        manifest,
        {"manifest_hash": "manifest", "reset_after_warmup": True},
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    engine = tmp_path / "engine.plan"
    digest = _engine(engine, b"engine")
    inventory = [
        {
            "family_id": "heal_lidar_disco",
            "assigned_method": "ga",
            "sequence_index": index,
            "item_id": item_id,
            "variant": "fp32" if index == 0 else "prune_quant",
            "engine_path": str(engine),
            "engine_sha256": digest,
        }
        for index, item_id in enumerate(("ga_fp32_original", "ga_bops_0.30_prune_quant"))
    ]
    source_root = tmp_path / "interrupted"
    accepted = source_root / "repeat_00/gpu_4_ga/00_ga_fp32_original"
    accepted.mkdir(parents=True)
    (accepted / "accepted.json").write_text("{}", encoding="utf-8")
    _json(
        source_root / "family_fair_evaluation_manifest.json",
        {
            "family_id": "heal_lidar_disco",
            "gpu_assignment": {"ga": 4, "greedy": 5},
            "protocol": {
                "num_frames": 2,
                "warmup_frames": 1,
                "latency_rounds": 3,
                "dataloader_num_workers": 8,
                "cuda_postprocess": True,
                "fixed_k": None,
                "input_contract": "heal_post_scatter_dynamic_frontend_v1",
                "max_agents": 2,
                "eval_manifest_hash": "manifest",
                "eval_manifest_file_sha256": manifest_sha,
            },
            "inventories": {"ga": inventory},
        },
    )

    def fake_loader(**kwargs: Any) -> dict[str, Any]:
        path = Path(kwargs["output_dir"])
        if "01_ga_bops_0.30_prune_quant" in str(path):
            raise RuntimeError("family_resume_cleanup_not_accepted")
        if not path.is_dir():
            raise RuntimeError("missing")
        return {"resume_hit": True}

    monkeypatch.setattr(
        "scripts.run_heal_lidar_family_split_gpu_fair_evaluation."
        "load_resumable_family_evaluation",
        fake_loader,
    )
    args = Namespace(
        seed_method_root=[],
        seed_method_prefix_root=[f"ga={source_root}"],
        family_id="heal_lidar_disco",
        eval_manifest=manifest,
        num_frames=2,
        warmup_frames=1,
        latency_rounds=3,
        max_agents=2,
        repeat_count=1,
    )
    destination = tmp_path / "recovery"
    seeded = _seed_method_results(
        args=args,
        run_dir=destination,
        inventories={"ga": inventory},
        gpu_assignment={"ga": 4},
    )
    assert seeded["ga"]["seed_mode"] == "accepted_prefix"
    assert seeded["ga"]["validated_and_seeded_count"] == 1
    assert seeded["ga"]["first_unaccepted"]["work_index"] == 1
    assert (destination / "repeat_00/gpu_4_ga/00_ga_fp32_original").is_dir()
    assert not (
        destination
        / "repeat_00/gpu_4_ga/01_ga_bops_0.30_prune_quant"
    ).exists()


def _summary_input(repeat: int, variant: str, value: float) -> dict[str, Any]:
    row = {
        "family_id": "heal_lidar_fcooper",
        "assigned_method": "ga",
        "gpu_id": 0,
        "repeat_index": repeat,
        "sequence_index": 0 if variant == "fp32" else 1,
        "item_id": variant,
        "variant": variant,
        "budget": None if variant == "fp32" else 0.30,
        "actual_bops": 1.0 if variant == "fp32" else 0.30,
        "parameter_count_base": 100,
        "parameter_count_pruned": 80,
        "parameter_reduction": 0.2,
        "int8_count": 0,
        "fp16_count": 0,
        "fp32_count": 2,
        "num_evaluated_frames": 2,
        "num_skipped_frames": 0,
        "evaluated_frame_ids": ["a", "b"],
        "engine_sha256": f"engine-{variant}",
        "source_engine_unchanged": True,
        "gpu_cleanup": {"passed": True},
    }
    for metric in (
        "AP@0.3",
        "AP@0.5",
        "AP@0.7",
        "mAP",
        "forward_mean_ms",
        "forward_p50_ms",
        "forward_p90_ms",
        "forward_p99_ms",
        "postprocess_mean_ms",
        "postprocess_p50_ms",
        "postprocess_p90_ms",
        "postprocess_p99_ms",
        "total_mean_ms",
        "total_p50_ms",
        "total_p90_ms",
        "total_p99_ms",
    ):
        row[metric] = value
    return row


def test_repeat_aggregation_and_contribution() -> None:
    raw = []
    for repeat in range(2):
        raw.extend(
            (
                _summary_input(repeat, "fp32", 10.0 + repeat),
                _summary_input(repeat, "prune_quant", 5.0 + repeat),
                _summary_input(repeat, "prune_only", 8.0 + repeat),
                _summary_input(repeat, "quant_only", 7.0 + repeat),
            )
        )
    summary = summarize_repeat_results(raw)
    aggregate = aggregate_five_repeat_results(summary, repeat_count=2)
    pq = next(row for row in aggregate if row["variant"] == "prune_quant")
    assert pq["mAP_across_runs_mean"] == 5.5
    assert pq["mAP_across_runs_std"] == 0.5
    contributions = ablation_contribution_rows(summary)
    assert len(contributions) == 2
    assert contributions[0]["pq_interaction_mAP"] == pytest.approx(0.0)
    combined = aggregate_contribution_rows(contributions, repeat_count=2)
    assert combined[0]["prune_quant_delta_vs_fp32_across_runs_mean"] == -5.0
