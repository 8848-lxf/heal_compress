from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_experiment(root: Path) -> None:
    smoke = {
        "split": "val",
        "frame_ids": ["s0", "s1"],
        "warmup_frame_ids": ["w0", "w1"],
        "evaluation_frame_ids": ["s0", "s1"],
        "warmup_frames": 2,
        "num_frames": 2,
        "reset_after_warmup": True,
        "evaluation_offset": 2,
        "manifest_hash": "smoke-hash",
    }
    fixed = {
        "split": "val",
        "frame_ids": [f"f{index}" for index in range(60)],
        "warmup_frame_ids": ["w0", "w1"],
        "evaluation_frame_ids": [f"f{index}" for index in range(60)],
        "warmup_frames": 2,
        "num_frames": 60,
        "reset_after_warmup": True,
        "evaluation_offset": 2,
        "manifest_hash": "fixed-hash",
    }
    _write_json(root / "manifests/smoke10_manifest.json", smoke)
    _write_json(root / "manifests/fixed500_manifest.json", fixed)
    _write_json(root / "candidate_masks/baseline_d32.json", {"mask": "identity"})
    _write_json(
        root / "experiment_config.json",
        {
            "fixed_k": 29696,
            "fixed_k_validated": True,
            "fixed_k_contract": {
                "deployment_topology": "single_engine_fixed_k",
                "fixed_k": 29696,
                "overflow_count": 0,
                "validated_scope": "full_validation",
            },
            "candidates": [
                {
                    "candidate_id": "baseline_d32",
                    "d_qk": 32,
                    "d_v": 32,
                    "embed_dim": 256,
                    "experiment": "baseline",
                    "heads": 8,
                    "mask_path": str(root / "candidate_masks/baseline_d32.json"),
                    "variant": "unpruned_explicit_projection",
                },
                {"candidate_id": "qk_only_d24"},
            ],
            "manifests": {
                "smoke10": {
                    "path": str(root / "manifests/smoke10_manifest.json"),
                    "manifest_hash": "smoke-hash",
                },
                "fixed500": {
                    "path": str(root / "manifests/fixed500_manifest.json"),
                    "manifest_hash": "fixed-hash",
                },
            },
        },
    )


def test_prepare_boundary_audit_isolates_baseline_and_creates_fixed50(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        prepare_boundary_audit_run,
    )

    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_experiment(source)

    result = prepare_boundary_audit_run(
        source_output=source,
        output_dir=output,
        checkpoint_hash="checkpoint-hash",
        config_hash="config-hash",
        plugin_hash="plugin-hash",
        code_commit="code-commit",
    )

    experiment = json.loads((output / "experiment_config.json").read_text())
    assert result["fixed_k"] == 29696
    assert result["deployment_topology"] == "single_engine_fixed_k"
    assert [row["candidate_id"] for row in experiment["candidates"]] == [
        "baseline_d32"
    ]
    assert Path(experiment["candidates"][0]["mask_path"]).parent == (
        output / "candidate_masks"
    )
    fixed50 = json.loads((output / "manifests/fixed50_manifest.json").read_text())
    assert fixed50["evaluation_frame_ids"] == [f"f{index}" for index in range(50)]
    assert fixed50["warmup_frame_ids"] == ["w0", "w1"]
    assert fixed50["num_frames"] == 50
    assert len(list((output / "profiles").iterdir())) == 8
    assert not list(output.rglob("*.plan"))


def test_prepare_boundary_audit_rejects_nonempty_destination(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        prepare_boundary_audit_run,
    )

    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_experiment(source)
    output.mkdir()
    (output / "foreign.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boundary_audit_output_not_empty"):
        prepare_boundary_audit_run(
            source_output=source,
            output_dir=output,
            checkpoint_hash="checkpoint-hash",
            config_hash="config-hash",
            plugin_hash="plugin-hash",
            code_commit="code-commit",
        )


@pytest.mark.parametrize(
    ("delta_map", "expected"),
    [
        (-0.050000, "catastrophic"),
        (-0.049999, "ambiguous"),
        (-0.010001, "ambiguous"),
        (-0.010000, "safe"),
        (0.001, "safe"),
    ],
)
def test_smoke_delta_classification_uses_exact_protocol_thresholds(
    delta_map: float, expected: str
):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        classify_smoke_delta,
    )

    assert classify_smoke_delta(delta_map) == expected


def test_freshness_signature_changes_with_every_owned_identity():
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        boundary_freshness_signature,
    )

    fields = {
        "code_commit": "code",
        "checkpoint_hash": "checkpoint",
        "config_hash": "config",
        "plugin_hash": "plugin",
        "profile_hash": "profile",
        "fixed_k": 29696,
        "smoke_manifest_hash": "smoke",
        "fixed500_manifest_hash": "fixed500",
        "tensorrt_version": "10.9",
        "cuda_version": "11.8",
    }
    baseline = boundary_freshness_signature(**fields)
    assert baseline == boundary_freshness_signature(**fields)
    for key, value in fields.items():
        changed = dict(fields)
        changed[key] = value + "-other" if isinstance(value, str) else value + 256
        assert boundary_freshness_signature(**changed) != baseline
    assert len(baseline) == 64
    assert set(baseline) <= set("0123456789abcdef")
