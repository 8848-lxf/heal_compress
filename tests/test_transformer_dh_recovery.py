from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from search.orchestration.lidar_transformer_dh_phase_a_accept import accept_phase_a
from search.orchestration.lidar_transformer_dh_recovery import (
    ExperimentState,
    lock_is_stale,
)
from search.reporting.transformer_dh_alignment_final import _latency_summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _phase_a_tree(root: Path) -> None:
    manifest_hashes = {}
    semantic = {}
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        path = root / "evaluation" / "manifests" / model / "fixed500.json"
        semantic[model] = f"semantic-{model}"
        _write(path, {"manifest_hash": semantic[model]})
        manifest_hashes[f"{model}/fixed500.json"] = _sha(path)
    _write(root / "run_manifest.json", {"manifest_hashes": manifest_hashes})
    profiles = ("P32", "P16", "P8")
    ordinal = 0
    for model_index, model in enumerate(("lidar_cobevt", "lidar_v2xvit")):
        count = 55
        for local in range(count):
            family = f"family_{model_index}_{local:03d}"
            d_h = local + 1
            structure_dir = root / "structures" / model / family / f"dh_{d_h:03d}"
            onnx = structure_dir / "base_fp32_canonical.onnx"
            onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx.write_bytes(f"onnx-{model}-{local}".encode())
            structure_hash = f"structure-{model}-{local}"
            _write(
                structure_dir / "structure_result.json",
                {
                    "status": "ok",
                    "model": model,
                    "family_id": family,
                    "d_h": d_h,
                    "structure_hash": structure_hash,
                    "state_dict_shape_hash": f"shape-{model}-{local}",
                    "physical_forward_finite": True,
                    "onnx_checker_passed": True,
                    "onnx_shape_inference_passed": True,
                    "onnx_sha256": _sha(onnx),
                },
            )
            for profile in profiles:
                directory = root / "engines" / model / family / f"dh_{d_h:03d}" / profile
                engine = directory / "engine.plan"
                engine.parent.mkdir(parents=True, exist_ok=True)
                engine.write_bytes(f"engine-{ordinal}".encode())
                engine_hash = _sha(engine)
                _write(
                    directory / "baseline_result.json",
                    {
                        "status": "ok",
                        "model": model,
                        "family_id": family,
                        "d_h": d_h,
                        "profile": profile,
                        "strongly_typed": True,
                        "requested_realized_conflict_count": 0,
                        "engine_sha256": engine_hash,
                    },
                )
                _write(directory / "engine_alignment_audit.json", {"fallback_hint": False})
                manifest_path = root / "evaluation" / "manifests" / model / "fixed500.json"
                _write(
                    directory / "evaluation" / "fixed500" / "evaluation_acceptance.json",
                    {
                        "status": "ok",
                        "evaluated": 500,
                        "skipped": 0,
                        "engine_sha256": engine_hash,
                        "structure_hash": structure_hash,
                        "manifest_hash": semantic[model],
                        "manifest_sha256": _sha(manifest_path),
                    },
                )
                ordinal += 1


def test_phase_a_certificate_accepts_110_330_330(tmp_path: Path) -> None:
    _phase_a_tree(tmp_path)
    result = accept_phase_a(tmp_path)
    assert result["status"] == "accepted_from_existing_artifacts"
    assert result["structures"]["accepted"] == 110
    assert result["engines"]["unique_hashes"] == 330
    assert result["fixed500"]["accepted"] == 330
    assert result["attempt_lineage_complete"] is False


@pytest.mark.parametrize(
    "mutation",
    ("missing_fixed500", "evaluated_not_500", "skip_nonzero", "duplicate_engine", "precision_conflict"),
)
def test_phase_a_certificate_fails_closed(tmp_path: Path, mutation: str) -> None:
    _phase_a_tree(tmp_path)
    builds = sorted(tmp_path.glob("engines/lidar_*/*/dh_*/*/baseline_result.json"))
    first = builds[0]
    fixed = first.parent / "evaluation" / "fixed500" / "evaluation_acceptance.json"
    if mutation == "missing_fixed500":
        fixed.unlink()
    elif mutation in {"evaluated_not_500", "skip_nonzero"}:
        value = json.loads(fixed.read_text(encoding="utf-8"))
        value["evaluated" if mutation == "evaluated_not_500" else "skipped"] = 499 if mutation == "evaluated_not_500" else 1
        _write(fixed, value)
    elif mutation == "precision_conflict":
        value = json.loads(first.read_text(encoding="utf-8"))
        value["requested_realized_conflict_count"] = 1
        _write(first, value)
    else:
        second = builds[1]
        second_engine = second.parent / "engine.plan"
        second_engine.write_bytes((first.parent / "engine.plan").read_bytes())
        value = json.loads(second.read_text(encoding="utf-8"))
        value["engine_sha256"] = _sha(second_engine)
        _write(second, value)
        second_fixed = second.parent / "evaluation" / "fixed500" / "evaluation_acceptance.json"
        fixed_value = json.loads(second_fixed.read_text(encoding="utf-8"))
        fixed_value["engine_sha256"] = _sha(second_engine)
        _write(second_fixed, fixed_value)
    with pytest.raises(RuntimeError):
        accept_phase_a(tmp_path)


def test_stopped_wrapper_lock_is_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "search.orchestration.lidar_transformer_dh_recovery._process",
        lambda pid: {
            "exists": True,
            "pid": pid,
            "uid": os.getuid(),
            "state": "T",
            "active": False,
            "command": str(tmp_path),
        },
    )
    stale, reason = lock_is_stale({"pid": 123}, run_root=tmp_path)
    assert stale
    assert "not_active" in reason


def test_active_valid_task_lock_cannot_be_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = ExperimentState(tmp_path)
    lock = state.locks / "phase-candidate.json"
    _write(lock, {"pid": 123})
    monkeypatch.setattr(
        "search.orchestration.lidar_transformer_dh_recovery._process",
        lambda pid: {
            "exists": True,
            "pid": pid,
            "uid": os.getuid(),
            "state": "S",
            "active": True,
            "command": f"worker --run-root {tmp_path}",
        },
    )
    with pytest.raises(RuntimeError, match="active_scheduler_lock"):
        with state.attempt(
            phase="phase", candidate_id="candidate", artifact_output_path=tmp_path / "x"
        ):
            pass


def test_phase_gates_and_evidence_contracts_are_explicit() -> None:
    formal = Path(
        "search/orchestration/lidar_transformer_dh_formal_latency.py"
    ).read_text(encoding="utf-8")
    joint = Path("search/orchestration/lidar_transformer_dh_joint.py").read_text(
        encoding="utf-8"
    )
    micro = Path(
        "search/orchestration/lidar_transformer_dh_microbenchmark.py"
    ).read_text(encoding="utf-8")
    phase_b = Path("search/orchestration/lidar_transformer_dh_phase_b.py").read_text(
        encoding="utf-8"
    )
    phase_a_body = formal.split("def run_phase_a_formal_latency", 1)[1].split(
        "def run_phase_b_formal_latency", 1
    )[0]
    phase_b_body = formal.split("def run_phase_b_formal_latency", 1)[1]
    assert "require_phase_b_fixed500" not in phase_a_body
    assert "require_phase_b_fixed500" in phase_b_body
    assert 'cache_namespace=f"dh_joint_{model_name}_{stable_hash(dict(targets))[:16]}_P8_gpu{physical_gpu}"' in joint
    assert '"fresh_joint_calibration": True' in joint
    assert '"full_engine_predictor": False' in micro
    assert '"unit_latency_additive": False' in micro
    assert "delta_predicted_additive" in phase_b
    assert "interaction_joint" in phase_b
    assert "control_4_d_h" in formal and "control_8_d_h" in formal
    assert "speedup_vs_same_profile_baseline" in formal


def test_phase_b_never_reuses_phase_a_engine_or_structure_paths() -> None:
    joint = Path("search/orchestration/lidar_transformer_dh_joint.py").read_text(
        encoding="utf-8"
    )
    assert '"joint" / joint_id(targets)' in joint
    assert "prepare_joint_structure" in joint
    assert '"state_dict_shape_hash"' in joint


def test_formal_search_and_pyramid_are_not_imported() -> None:
    for path in Path("search/orchestration").glob("lidar_transformer_dh_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "heal-unified-search-h800" not in source or path.name == "lidar_transformer_dh_provenance.py"
        assert "lidar_pyramid_search" not in source


def test_final_latency_summary_requires_baseline_and_both_aligned_controls() -> None:
    common = {
        "model": "lidar_cobevt",
        "attention_family": "family",
        "profile": "P16",
        "gpu_uuid": "GPU-x",
        "warmup": 200,
        "iterations": 2000,
        "repeats": 5,
    }
    rows = [
        {
            **common,
            "d_h": 32,
            "baseline_replay": True,
            "replay_index": 0,
            "latency_beneficial": False,
            "alignment_class": "multiple_of_8",
        },
        {
            **common,
            "d_h": 31,
            "baseline_replay": False,
            "replay_index": 1,
            "latency_beneficial": True,
            "beneficial_vs_control4": True,
            "beneficial_vs_control8": True,
            "alignment_class": "non4",
            "p50_ms": 9.0,
            "speedup_vs_same_profile_baseline": 1.1,
            "control_4_d_h": 32,
            "control_8_d_h": 32,
            "speedup_vs_control4": 1.1,
            "speedup_vs_control8": 1.1,
        },
        {
            **common,
            "d_h": 32,
            "baseline_replay": True,
            "replay_index": 2,
            "latency_beneficial": False,
            "alignment_class": "multiple_of_8",
        },
    ]
    result = _latency_summary(rows, phase="phase_a")
    assert result["canonical_candidate_rows"] == 2
    assert result["strict_nonaligned_beneficial_rows"] == 1
