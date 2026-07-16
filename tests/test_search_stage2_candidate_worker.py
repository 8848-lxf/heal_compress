from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_candidate_worker_context_uses_assigned_physical_gpu_and_controller_allowlist(
    tmp_path: Path,
) -> None:
    from search.stage2.candidate_worker import _context_kwargs

    request = {
        "gpu_id": 5,
        "controller_pid": os.getpid(),
        "worker_dir": str(tmp_path),
        "checkpoint": "/checkpoint.pth",
        "config": {
            "model": {"config": "/model.yaml"},
            "runtime": {
                "gpu_id": "5",
                "heal_root": "/heal",
                "tensorrt_root": "/trt",
                "plugin_path": "/plugin.so",
                "tensorrt_env": "modelopt",
                "allow_foreign_gpu_processes": False,
                "max_gpu_utilization_pct": 20,
            },
            "search": {"max_pruning_units": 96},
            "pruning": {
                "gene_type": "coupled_channel_keep_mask",
                "grouped_conv": {
                    "position_mode": "independent_group_topk",
                    "default_channels_per_group": 8,
                    "allowed_channels_per_group": [4, 8],
                },
            },
            "precision": {"default": "FP16"},
            "proxy": {
                "fisher_calibration_batches": 8,
                "quant_calibration_batches": 200,
                "quant_calibration_npz_manifest": "/train200.json",
                "quant_activation_calibration_backend": "tensorrt_entropy_calibration2",
                "quant_calibration_force_rebuild": True,
            },
            "stage2": {
                "num_frames": 300,
                "warmup_frames": 30,
                "reset_after_warmup": True,
            },
            "stage2_parallel": {
                "allow_controller_process_on_stage1_gpu": True,
            },
        },
    }

    kwargs = _context_kwargs(request)

    assert kwargs["gpu_id"] == "5"
    assert kwargs["exclude_gpu_ids"] == []
    assert kwargs["allowed_gpu_pids"] == {os.getpid()}
    assert kwargs["num_frames"] == 300
    assert kwargs["quant_calibration_batches"] == 200


def test_candidate_worker_does_not_allow_controller_on_other_gpu(tmp_path: Path) -> None:
    from search.stage2.candidate_worker import _context_kwargs

    request = {
        "gpu_id": 4,
        "controller_pid": 123,
        "worker_dir": str(tmp_path),
        "checkpoint": "/checkpoint.pth",
        "config": {
            "model": {"config": "/model.yaml"},
            "runtime": {"gpu_id": "5"},
            "search": {},
            "pruning": {},
            "precision": {},
            "proxy": {},
            "stage2": {},
        },
    }

    assert _context_kwargs(request)["allowed_gpu_pids"] == set()


def test_candidate_worker_preflight_uses_same_strict_pid_policy(tmp_path: Path) -> None:
    from search.stage2.candidate_worker import _gpu_isolation_kwargs

    request = {
        "gpu_id": 5,
        "controller_pid": 321,
        "worker_dir": str(tmp_path),
        "checkpoint": "/checkpoint.pth",
        "config": {
            "model": {},
            "runtime": {
                "gpu_id": "5",
                "allow_foreign_gpu_processes": False,
                "max_gpu_utilization_pct": 20,
            },
            "stage2_parallel": {
                "allow_controller_process_on_stage1_gpu": True,
            },
        },
    }

    policy = _gpu_isolation_kwargs(request)

    assert policy["allowed_pids"] == {321}
    assert policy["allow_foreign_processes"] is False
    assert policy["max_gpu_utilization_pct"] == 20


def test_candidate_worker_requires_raw_repaired_requested_realized_hash_identity(
    tmp_path: Path,
) -> None:
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.stage2.candidate_worker import _evaluate_task

    class Evaluator:
        def evaluate_candidate_two_level(self, *_args, **_kwargs):
            return {
                "status": "ok",
                "F2": 0.1,
                "precision_identity_passed": True,
                "requested_precision_profile_hash": "same",
                "realized_precision_profile_hash": "same",
            }

    phenotype = CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile={"layer": PrecisionDecision("FP16", "FP16", "")},
    )
    base_task = {
        "candidate_hash": "candidate",
        "phenotype": phenotype.to_dict(),
        "output_dir": str(tmp_path),
        "smoke_frames": 10,
        "smoke_warmup_frames": 10,
        "raw_precision_gene_hash": "same",
        "repaired_precision_gene_hash": "same",
    }

    accepted = _evaluate_task(Evaluator(), base_task, 4)
    rejected = _evaluate_task(
        Evaluator(),
        {**base_task, "repaired_precision_gene_hash": "changed"},
        4,
    )

    assert accepted["precision_identity_passed"] is True
    assert accepted["status"] == "ok"
    assert rejected["precision_identity_passed"] is False
    assert rejected["status"] == "precision_profile_hash_mismatch"
    assert rejected["F2"] == float("inf")


def test_candidate_worker_routes_build_smoke_without_formal_evaluation(
    tmp_path: Path,
) -> None:
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.stage2.candidate_worker import _evaluate_task

    calls: list[str] = []

    class Evaluator:
        def build_and_smoke_candidate(self, *_args, **_kwargs):
            calls.append("build_smoke")
            return {
                "status": "ok",
                "precision_identity_passed": True,
                "requested_precision_profile_hash": "same",
                "realized_precision_profile_hash": "same",
            }

        def evaluate_existing_engine(self, *_args, **_kwargs):
            raise AssertionError("build_smoke must not run formal evaluation")

    phenotype = CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile={"layer": PrecisionDecision("FP16", "FP16", "")},
    )
    result = _evaluate_task(
        Evaluator(),
        {
            "task_protocol": "build_smoke",
            "task_cache_key": "build-key",
            "candidate_hash": "candidate",
            "phenotype": phenotype.to_dict(),
            "output_dir": str(tmp_path),
            "smoke_frames": 10,
            "smoke_warmup_frames": 10,
            "raw_precision_gene_hash": "same",
            "repaired_precision_gene_hash": "same",
        },
        4,
    )

    assert calls == ["build_smoke"]
    assert result["task_protocol"] == "build_smoke"
    assert result["task_cache_key"] == "build-key"
    assert result["status"] == "ok"


def test_candidate_worker_routes_evaluation_without_rebuilding_engine(
    tmp_path: Path,
) -> None:
    from search.stage2.candidate_worker import _evaluate_task

    calls: list[str] = []

    class Evaluator:
        def build_and_smoke_candidate(self, *_args, **_kwargs):
            raise AssertionError("evaluation-only task must not rebuild engine")

        def evaluate_existing_engine(self, engine_path, **_kwargs):
            calls.append(str(engine_path))
            return {
                "status": "ok",
                "precision_identity_passed": True,
                "requested_precision_profile_hash": "same",
                "realized_precision_profile_hash": "same",
            }

    result = _evaluate_task(
        Evaluator(),
        {
            "task_protocol": "full_validation",
            "task_cache_key": "full-key",
            "candidate_hash": "candidate",
            "engine_path": "/engine.plan",
            "output_dir": str(tmp_path),
            "deployment_metadata": {
                "raw_precision_gene_hash": "same",
                "repaired_precision_gene_hash": "same",
            },
        },
        5,
    )

    assert calls == ["/engine.plan"]
    assert result["task_protocol"] == "full_validation"
    assert result["task_cache_key"] == "full-key"
    assert result["status"] == "ok"


def test_candidate_worker_reference_protocol_needs_no_phenotype(tmp_path: Path) -> None:
    from search.stage2.candidate_worker import _evaluate_task

    class Evaluator:
        def evaluate_original_baseline(self, precision, *, full_validation=False):
            assert precision == "strict_fp32"
            assert full_validation is False
            return {
                "status": "ok",
                "mAP": 0.73,
                "forward_p50_ms": 10.0,
                "engine_hash": "engine",
                "eval_hash": "evaluation",
            }

    result = _evaluate_task(
        Evaluator(),
        {
            "task_protocol": "reference_strict_fp32",
            "task_cache_key": "reference-key",
            "candidate_hash": "strict-fp32-reference",
            "output_dir": str(tmp_path),
        },
        4,
    )

    assert result["status"] == "ok"
    assert result["reference_precision"] == "strict_fp32"
    assert result["reference_hash"]
    assert result["task_protocol"] == "reference_strict_fp32"


def test_candidate_worker_rejects_unknown_formal_protocol(tmp_path: Path) -> None:
    import pytest

    from search.stage2.candidate_worker import _evaluate_task

    with pytest.raises(ValueError, match="unknown_stage2_task_protocol"):
        _evaluate_task(
            object(),
            {
                "task_protocol": "evaluate_123",
                "task_cache_key": "unknown-key",
                "candidate_hash": "candidate",
                "output_dir": str(tmp_path),
            },
            4,
        )
