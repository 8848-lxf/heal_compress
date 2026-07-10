from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_v12_deblock_contract_metadata_marks_historical_pruning() -> None:
    from tools.latency_lut.prepare_v12_combined_lut_dataset import summarize_surface_contract_from_manifest

    manifest = {
        "module_channel_before_after": [
            {
                "module_name": "pyramid_backbone.deblocks.0.0",
                "module_type": "ConvTranspose2d",
                "before": {"in_channels": 256, "out_channels": 128},
                "after": {"in_channels": 192, "out_channels": 80},
            },
            {
                "module_name": "cls_head",
                "module_type": "Conv2d",
                "before": {"in_channels": 128, "out_channels": 2},
                "after": {"in_channels": 96, "out_channels": 2},
            },
        ]
    }

    summary = summarize_surface_contract_from_manifest(manifest, source_subset="historical_v11")

    assert summary["deblock_output_pruned"] is True
    assert summary["deblock_output_pruned_modules"] == ["pyramid_backbone.deblocks.0.0"]
    assert summary["deblock_output_pruning_source"] == "historical_v11_or_opt_in"
    assert summary["head_output_changed"] is False


def test_v12_deblock_contract_metadata_requires_new_v12_outputs_unchanged() -> None:
    from tools.latency_lut.prepare_v12_combined_lut_dataset import summarize_surface_contract_from_manifest

    manifest = {
        "module_channel_before_after": [
            {
                "module_name": "pyramid_backbone.deblocks.0.0",
                "module_type": "ConvTranspose2d",
                "before": {"in_channels": 256, "out_channels": 128},
                "after": {"in_channels": 192, "out_channels": 128},
            },
            {
                "module_name": "pyramid_backbone.deblocks.0.1",
                "module_type": "BatchNorm2d",
                "before": {"num_features": 128},
                "after": {"num_features": 128},
            },
            {
                "module_name": "pyramid_backbone.shrink_conv.0",
                "module_type": "Conv2d",
                "before": {"in_channels": 384, "out_channels": 128},
                "after": {"in_channels": 384, "out_channels": 96},
            },
        ]
    }

    summary = summarize_surface_contract_from_manifest(manifest, source_subset="new_v12_deblock_protected")

    assert summary["deblock_output_pruned"] is False
    assert summary["new_v12_deblock_contract_passed"] is True
    assert summary["deblock_output_pruning_source"] == "none_default_protected"


def test_v12_lut_sample_label_requires_300_real_validation_frames(tmp_path: Path) -> None:
    from tools.latency_lut.run_v12_lut_engine_eval_workers import build_lut_sample_label

    subnet_dir = tmp_path / "subnets/subnet_000"
    profile_dir = subnet_dir / "profile_003"
    profile_dir.mkdir(parents=True)
    (subnet_dir / "onnx").mkdir()
    (subnet_dir / "onnx/model_signal_maxk.onnx").write_bytes(b"onnx")
    (profile_dir / "engine.plan").write_bytes(b"engine")
    _write_json(
        subnet_dir / "pruning_manifest.json",
        {
            "subnet_id": "subnet_000",
            "structure_hash": "structure",
            "shape_hash": "shape",
            "uses_taylor_ranking": False,
            "random_sampling_method": "deployment_aware_random_without_taylor_ranking",
            "actual_param_prune_ratio": 0.4,
            "actual_channel_prune_ratio": 0.3,
            "achieved_global_bops_prune_ratio": 0.5,
            "module_channel_before_after": [],
            "source_subset": "new_v12_deblock_protected",
        },
    )
    _write_json(profile_dir / "mixed_precision_profile.json", {"requested_int8_group_ratio": 0.8, "layer_precision_assignment": {"conv": "int8"}})
    _write_json(profile_dir / "qdq_insert_report.json", {"success": True, "inserted_qdq_nodes": [{}, {}]})
    _write_json(profile_dir / "engine_structure_check_report.json", {"structure_check_passed": True})
    _write_json(
        profile_dir / "engine_precision_realization_report.json",
        {
            "precision_realization_passed": True,
            "int8_realized_layer_count": 1,
            "int8_compute_fp16_boundary_count": 0,
            "precision_realization_mismatch_count": 0,
        },
    )
    _write_json(profile_dir / "trt_smoke_report.json", {"success": True})
    eval_report = {
        "eval_success": True,
        "evaluated_frames": 300,
        "synthetic_used": False,
        "validation_dataloader_used": True,
        "skipped_frames": 1,
        "ap": {"AP@0.03": 0.1, "AP@0.30": 0.2, "AP@0.50": 0.3, "AP@0.70": 0.4, "mAP": 0.25},
        "latency_summary": {"forward_mean_ms": 3.0, "forward_p50_ms": 2.0, "forward_p90_ms": 4.0},
    }
    _write_json(profile_dir / "eval_report.json", eval_report)

    label = build_lut_sample_label(
        subnet_dir=subnet_dir,
        profile_dir=profile_dir,
        subnet_id="subnet_000",
        profile_id="profile_003",
        source_subset="new_v12_deblock_protected",
        required_eval_frames=300,
    )

    assert label["dataset_version"] == "v12_combined_lut_dataset_300frames"
    assert label["label_available"] is True
    assert label["evaluated_frames"] == 300
    assert label["synthetic_used"] is False
    assert label["validation_dataloader_used"] is True
    assert label["forward_latency_p90_ms"] == 4.0


def test_v12_worker_queue_uses_300_frame_completion_and_lock_files(tmp_path: Path) -> None:
    from tools.latency_lut.run_v12_lut_engine_eval_workers import build_pending_jobs

    dataset = tmp_path / "dataset"
    complete = dataset / "subnets/subnet_000/profile_000"
    pending = dataset / "subnets/subnet_000/profile_001"
    running = dataset / "subnets/subnet_001/profile_000"
    for path in (complete, pending, running):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(dataset / "subnets/subnet_000/pruning_manifest.json", {"subnet_id": "subnet_000", "structure_hash": "h0"})
    _write_json(dataset / "subnets/subnet_001/pruning_manifest.json", {"subnet_id": "subnet_001", "structure_hash": "h1"})
    _write_json(complete / "lut_sample_label.json", {"label_available": True, "evaluated_frames": 300})
    (complete / ".done").write_text("ok", encoding="utf-8")
    (running / ".running.lock").write_text("locked", encoding="utf-8")

    args = argparse.Namespace(
        dataset_dir=str(dataset),
        precision_profiles_per_subnet=2,
        resume=True,
        skip_existing_success=True,
        eval_frame_count=300,
    )

    jobs, _ = build_pending_jobs(args)

    assert [(job.subnet_id, job.profile_id) for job in jobs] == [("subnet_000", "profile_001"), ("subnet_001", "profile_001")]


def test_gate_dryrun_protects_deblock_output_dependency_domain() -> None:
    from tools.latency_lut.run_v11_random_deployment_aware_gate_dryrun import _apply_deblock_output_domain_protection

    convt = nn.ConvTranspose2d(64, 128, 2)
    group = types.SimpleNamespace(
        group_id="domain_deblock",
        protected=False,
        protected_reason="",
        items=[
            types.SimpleNamespace(
                name="pyramid_backbone.deblocks.0.0",
                module=convt,
                direction="out",
            )
        ],
    )

    report = _apply_deblock_output_domain_protection(
        [group],
        argparse.Namespace(protect_deblock_output="true", allow_deblock_output_pruning="false"),
    )

    assert group.protected is True
    assert group.protected_reason == "requires_deblock_output_pruning_but_deblock_output_protected"
    assert report["protected_deblock_output_domains"] == ["pyramid_backbone.deblocks.0.0"]
    assert report["skipped_domains"][0]["skipped_reason"] == "requires_deblock_output_pruning_but_deblock_output_protected"
