from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_repaired_physical_validation_requires_exact_unit_indices_and_group_maps() -> None:
    from pruning.types import PhysicalPruningPlan, PhysicalPruningPlanEntry
    from search.adapters.pruning_adapter import FormalPruningAdapter
    from search.candidate import CandidatePhenotype
    from search.stage2.physical_validation import validate_repaired_physical_plan

    unit = SimpleNamespace(
        stable_id="g8",
        protected=False,
        scope_id="scope",
        root_module_path="gconv",
        root_axis="out",
        root_indices=[8],
        source_coupled_unit_ids=["cu8"],
        channel_cost=1,
        parameter_cost=1,
        constraints={"grouped_conv": True, "depthwise": False, "grouped_module_path": "gconv"},
        members=[],
    )
    phenotype = CandidatePhenotype(
        pruned_unit_ids=["g8"],
        metadata={
            "group_keep_map_by_scope": {"scope": {0: [0, 1, 2, 3], 1: [0, 2, 4, 6]}},
            "group_prune_map_by_scope": {"scope": {0: [4, 5, 6, 7], 1: [1, 3, 5, 7]}},
        },
    )
    request = FormalPruningAdapter(
        build_plan_fn=lambda *a, **k: None,
        legalize_plan_fn=lambda *a, **k: None,
        materialize_fn=lambda *a, **k: None,
        snapshot_fn=lambda *a, **k: None,
        hash_fn=lambda *a, **k: None,
        validate_fn=lambda *a, **k: None,
    ).request_from_phenotype(phenotype, [unit])
    plan = PhysicalPruningPlan(
        entries=[
            PhysicalPruningPlanEntry(
                module_path="gconv",
                axis="out",
                prune_indices=[8],
                keep_indices=[0, 1, 2, 3, 4, 5, 6, 7],
                original_axis_size=9,
                source_request_ids=["search_bundle::0000"],
                group_keep_map={0: [0, 1, 2, 3], 1: [0, 2, 4, 6]},
                group_prune_map={0: [4, 5, 6, 7], 1: [1, 3, 5, 7]},
                metadata={"scope_ids": ["scope"]},
            )
        ],
        source_request=request,
    )

    report = validate_repaired_physical_plan(phenotype, request, plan)

    assert report["passed"] is True
    assert report["repaired_mask_to_request_verified"] is True
    assert report["repaired_mask_to_physical_plan_verified"] is True
    assert report["group_keep_map_frozen_verified"] is True


def test_repaired_physical_validation_fails_closed_on_plan_index_repair() -> None:
    from pruning.types import PhysicalPruningPlan, PhysicalPruningPlanEntry
    from search.candidate import CandidatePhenotype
    from search.stage2.physical_validation import validate_repaired_physical_plan
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    phenotype = CandidatePhenotype(pruned_unit_ids=["u0"])
    request = SamplingPruningRequest(
        entries=[SamplingPruningEntry("r0", "scope", "conv", "out", [0], source_atomic_unit_ids=["u0"])],
        selected_atomic_unit_ids=["u0"],
    )
    plan = PhysicalPruningPlan(
        entries=[
            PhysicalPruningPlanEntry(
                module_path="conv",
                axis="out",
                prune_indices=[0, 1],
                keep_indices=[2, 3],
                original_axis_size=4,
                source_request_ids=["r0"],
                metadata={"scope_ids": ["scope"]},
            )
        ],
        source_request=request,
    )

    report = validate_repaired_physical_plan(phenotype, request, plan)

    assert report["passed"] is False
    assert "repaired_physical_plan_mismatch" in report["issues"]
    assert report["request_vs_plan_prune_indices_equal"] is False


def test_physical_artifact_writer_writes_required_aliases_for_cache_hits(tmp_path: Path) -> None:
    import torch
    from pruning.types import SamplingPruningRequest
    from search.stage2.lidar_pyramid_real_evaluator import _write_physical_artifact_files

    class Dictable:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return dict(self.payload)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    output_dir = tmp_path / "candidate"
    _write_physical_artifact_files(
        output_dir=output_dir,
        request=SamplingPruningRequest(),
        plan=Dictable({"entries": []}),
        ledger=Dictable({"operations": []}),
        snapshot=Dictable({"modules": []}),
        validation=Dictable({"passed": True}),
        model=Model(),
        checkpoint_hash="ckpt",
        physical_hash_value="physical",
        parameter_count_base=10,
        parameter_count_pruned=10,
        plan_validation={"passed": True},
    )

    for name in [
        "sampling_pruning_request.json",
        "physical_pruning_plan.json",
        "physical_plan_validation.json",
        "physical_structure_snapshot.json",
        "physical_validation.json",
        "materialization_report.json",
        "physical_widths.csv",
        "pruned_checkpoint.pth",
        "pruned_state_dict.pth",
        "physical_hash.json",
    ]:
        assert (output_dir / name).is_file(), name


def test_physical_selection_key_includes_frozen_group_maps() -> None:
    from search.candidate import CandidatePhenotype
    from search.stage2.lidar_pyramid_real_evaluator import _physical_selection_key

    first = CandidatePhenotype(
        pruned_unit_ids=["u0"],
        metadata={"group_keep_map_by_scope": {"scope": {0: [0, 1, 2, 3]}}},
    )
    second = CandidatePhenotype(
        pruned_unit_ids=["u0"],
        metadata={"group_keep_map_by_scope": {"scope": {0: [0, 2, 4, 6]}}},
    )

    assert _physical_selection_key(first, "ckpt") != _physical_selection_key(second, "ckpt")


def test_tensorrt_cache_identity_ignores_volatile_environment() -> None:
    from search.integration.runtime_environment import TensorRTEnvironment
    from search.stage2.lidar_pyramid_real_evaluator import _tensorrt_cache_identity

    first = TensorRTEnvironment(
        Path("/opt/TensorRT"),
        Path("/opt/TensorRT/bin/trtexec"),
        Path("/tmp/plugin.so"),
        conda_env="modelopt",
        env={"PWD": "/workspace/a", "PATH": "/a/bin", "LD_LIBRARY_PATH": "/a/lib"},
    )
    second = TensorRTEnvironment(
        Path("/opt/TensorRT"),
        Path("/opt/TensorRT/bin/trtexec"),
        Path("/tmp/plugin.so"),
        conda_env="modelopt",
        env={"PWD": "/workspace/b", "PATH": "/b/bin", "LD_LIBRARY_PATH": "/b/lib"},
    )

    assert first.to_dict()["env_hash"] != second.to_dict()["env_hash"]
    assert _tensorrt_cache_identity(first) == _tensorrt_cache_identity(second)


def test_original_baseline_reuses_existing_eval_when_cache_key_is_new(tmp_path: Path) -> None:
    import json

    from search.cache.real_eval_cache import RealEvalCache
    from search.integration.runtime_environment import TensorRTEnvironment
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator

    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.run_dir = tmp_path
    evaluator.context = SimpleNamespace(
        checkpoint_hash="ckpt",
        eval_manifest_hash="manifest",
        physical_gpu_id=0,
        tensorrt=TensorRTEnvironment(
            Path("/opt/TensorRT"),
            Path("/opt/TensorRT/bin/trtexec"),
            Path("/tmp/plugin.so"),
            conda_env="modelopt",
            env={"PATH": "/volatile"},
        ),
    )
    evaluator.num_frames = 300
    evaluator.warmup_frames = 30
    evaluator.latency_rounds = 3
    evaluator.real_cache = RealEvalCache(tmp_path / "archives" / "real_eval_archive.jsonl")

    baseline_dir = tmp_path / "baselines" / "original_strict_fp16"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "engine.plan").write_bytes(b"engine")
    (baseline_dir / "evaluation.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (baseline_dir / "baseline_eval.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "baseline_precision": "strict_fp16",
                "mAP": 0.7,
                "forward_p50_ms": 2.5,
                "engine_hash": "engine",
            }
        ),
        encoding="utf-8",
    )

    def fail_deploy(*_args, **_kwargs):
        raise AssertionError("baseline should have been reused from disk")

    evaluator._deploy_and_evaluate = fail_deploy  # type: ignore[method-assign]

    result = evaluator.evaluate_original_baseline("strict_fp16", full_validation=False)

    assert result["status"] == "ok"
    assert result["cache_hit"] is True
    assert result["cache_source"] == "existing_baseline_eval"
    assert evaluator.real_cache.get(result["cache_key"]) is not None


def test_physical_width_report_checks_only_axes_changed_by_pruning(tmp_path: Path) -> None:
    import csv

    from search.stage2.lidar_pyramid_real_evaluator import _write_physical_widths_csv

    output = tmp_path / "physical_widths.csv"
    snapshot = {
        "modules": [
            {
                "canonical_module_name": "cls_head",
                "module_type": "Conv2d",
                "in_channels": 256,
                "out_channels": 2,
                "groups": 1,
            },
            {
                "canonical_module_name": "single_head",
                "module_type": "Conv2d",
                "in_channels": 108,
                "out_channels": 1,
                "groups": 1,
            },
        ]
    }
    plan = {
        "entries": [
            {
                "module_path": "single_head",
                "axis": "in",
                "original_axis_size": 128,
                "keep_indices": list(range(108)),
            }
        ]
    }

    _write_physical_widths_csv(output, snapshot_payload=snapshot, plan_payload=plan)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = {row["module path"]: row for row in csv.DictReader(handle)}
    assert rows["cls_head"]["alignment status"] == "not_channel_pruned"
    assert rows["single_head"]["alignment status"] == "dense_width_aligned"


def test_cached_origin_map_reloads_typed_entries(tmp_path: Path) -> None:
    import json

    from search.stage2.lidar_pyramid_real_evaluator import _load_origin_map_result

    path = tmp_path / "origin_map.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "module_path": "conv",
                        "module_type": "Conv2d",
                        "call_index": 0,
                        "onnx_op_type": "Conv",
                        "original_node_name": "node",
                        "canonical_node_name": "canonical",
                        "weight_initializer": "weight",
                    }
                ],
                "source_onnx": "model.onnx",
            }
        ),
        encoding="utf-8",
    )

    result = _load_origin_map_result(path)

    assert len(result.entries) == 1
    assert result.entries[0].module_path == "conv"
    assert result.origin_map_hash


def test_full_validation_reuses_existing_candidate_engine_without_rebuild(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    from search.candidate import CandidatePhenotype
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
    from search.stage2.objective import Stage2ObjectiveConfig

    source = tmp_path / "stage2_500" / "candidate"
    source.mkdir(parents=True)
    engine = source / "engine.plan"
    engine.write_bytes(b"immutable-engine")
    engine_hash = hashlib.sha256(engine.read_bytes()).hexdigest()
    (source / "deployment_manifest.json").write_text(
        json.dumps(
            {
                "engine_hash": engine_hash,
                "deployment_hash": "deployment",
                "physical_hash": "physical",
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "physical_validation.json",
        "physical_plan_validation.json",
        "engine_structure_validation.json",
        "precision_realization_validation.json",
        "merge_precision_realization.json",
        "production_qdq_boundary_audit.json",
    ):
        (source / name).write_text(
            json.dumps({"status": "ok", "passed": True}),
            encoding="utf-8",
        )
    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.objective_config = Stage2ObjectiveConfig(
        latency_metric="forward_p50_ms"
    )
    calls = {"evaluation": 0, "deployment": 0}

    def evaluate_engine(path, output_dir):
        calls["evaluation"] += 1
        assert Path(path) == engine
        assert Path(output_dir) != source
        return {
            "status": "ok",
            "mAP": 0.61,
            "forward_p50_ms": 3.0,
            "evaluated_frames": 1789,
            "skipped_frames": 0,
        }

    def fail_deploy(*_args, **_kwargs):
        calls["deployment"] += 1
        raise AssertionError("full validation must not rebuild deployment")

    evaluator._evaluate_engine = evaluate_engine  # type: ignore[method-assign]
    evaluator._deploy_and_evaluate = fail_deploy  # type: ignore[method-assign]
    evaluator._stage2_reference_baseline = lambda: {  # type: ignore[method-assign]
        "status": "ok",
        "mAP": 0.62,
        "forward_p50_ms": 2.0,
    }

    result = evaluator.reevaluate_existing_candidate_engine(
        CandidatePhenotype(),
        source_artifact_dir=source,
        output_dir=tmp_path / "full" / "candidate",
        candidate_hash="candidate",
    )

    assert result["status"] == "ok"
    assert result["engine_hash"] == engine_hash
    assert result["engine_rebuilt"] is False
    assert result["physical_rebuilt"] is False
    assert result["onnx_rebuilt"] is False
    assert result["calibration_rebuilt"] is False
    assert result["qdq_rebuilt"] is False
    assert calls == {"evaluation": 1, "deployment": 0}
    assert engine.read_bytes() == b"immutable-engine"
