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
