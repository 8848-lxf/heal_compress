"""Fail-closed acceptance of the completed Transformer d_h Phase-A artifacts.

This module is deliberately read-only with respect to Phase-A structures,
engines, and evaluations.  Its only output is an evidence certificate under
the run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


EXPECTED_STRUCTURES = 110
EXPECTED_ENGINES = 330
EXPECTED_FIXED500 = 330
PROFILES = {"P32", "P16", "P8"}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required_artifact_missing:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required_artifact_not_object:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(workdir: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=workdir, text=True).strip()


def _phase_a_structure_paths(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (run_root / "structures").glob("lidar_*/*/dh_*/structure_result.json")
        if "joint" not in path.parts
    )


def _phase_a_engine_paths(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (run_root / "engines").glob("lidar_*/*/dh_*/*/baseline_result.json")
        if "joint" not in path.parts and path.parent.name in PROFILES
    )


def accept_phase_a(
    run_root: Path,
    *,
    expected_structures: int = EXPECTED_STRUCTURES,
    expected_engines: int = EXPECTED_ENGINES,
    expected_fixed500: int = EXPECTED_FIXED500,
    write: bool = True,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    run_manifest = _read(run_root / "run_manifest.json")
    workdir = Path(__file__).resolve().parents[2]
    errors: list[str] = []

    structures = _phase_a_structure_paths(run_root)
    if len(structures) != expected_structures:
        errors.append(f"structure_count:{len(structures)}!={expected_structures}")
    structure_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    structure_hashes: list[str] = []
    for path in structures:
        row = _read(path)
        key = (str(row.get("model")), str(row.get("family_id")), int(row.get("d_h", -1)))
        if key in structure_by_key:
            errors.append(f"duplicate_structure_key:{key}")
        structure_by_key[key] = row
        structure_hash = str(row.get("structure_hash", ""))
        structure_hashes.append(structure_hash)
        if row.get("status") != "ok":
            errors.append(f"structure_not_ok:{key}:{row.get('status')}")
        if not structure_hash or not row.get("state_dict_shape_hash"):
            errors.append(f"structure_hash_incomplete:{key}")
        if not bool(row.get("physical_forward_finite")):
            errors.append(f"structure_forward_not_finite:{key}")
        if not bool(row.get("onnx_checker_passed")) or not bool(row.get("onnx_shape_inference_passed")):
            errors.append(f"structure_onnx_not_accepted:{key}")
        onnx_path = path.parent / "base_fp32_canonical.onnx"
        if not onnx_path.is_file() or _sha256(onnx_path) != str(row.get("onnx_sha256", "")):
            errors.append(f"structure_onnx_hash_mismatch:{key}")
    if len(set(structure_hashes)) != len(structure_hashes) or "" in structure_hashes:
        errors.append("structure_hashes_not_unique_and_complete")

    engines = _phase_a_engine_paths(run_root)
    if len(engines) != expected_engines:
        errors.append(f"engine_count:{len(engines)}!={expected_engines}")
    engine_hashes: list[str] = []
    fixed_count = 0
    manifest_match = True
    fallback_count = 0
    precision_conflicts = 0
    observed_profiles: set[str] = set()
    for path in engines:
        row = _read(path)
        model = str(row.get("model"))
        family = str(row.get("family_id"))
        d_h = int(row.get("d_h", -1))
        profile = str(row.get("profile"))
        key = (model, family, d_h)
        observed_profiles.add(profile)
        if key not in structure_by_key:
            errors.append(f"engine_structure_missing:{key}:{profile}")
            continue
        if row.get("status") != "ok" or not bool(row.get("strongly_typed")):
            errors.append(f"engine_not_accepted:{key}:{profile}:{row.get('status')}")
        conflict_count = int(row.get("requested_realized_conflict_count", -1))
        precision_conflicts += max(conflict_count, 0)
        if conflict_count != 0:
            errors.append(f"precision_conflict:{key}:{profile}:{conflict_count}")
        engine_path = path.parent / "engine.plan"
        engine_hash = str(row.get("engine_sha256", ""))
        engine_hashes.append(engine_hash)
        if not engine_path.is_file() or not engine_path.stat().st_size or _sha256(engine_path) != engine_hash:
            errors.append(f"engine_hash_mismatch:{key}:{profile}")
        alignment_path = path.parent / "engine_alignment_audit.json"
        alignment = _read(alignment_path)
        if bool(alignment.get("fallback_hint")):
            fallback_count += 1
            errors.append(f"engine_fallback:{key}:{profile}")
        fixed_path = path.parent / "evaluation" / "fixed500" / "evaluation_acceptance.json"
        fixed = _read(fixed_path)
        fixed_count += 1
        if fixed.get("status") != "ok" or int(fixed.get("evaluated", -1)) != 500 or int(fixed.get("skipped", -1)) != 0:
            errors.append(
                f"fixed500_not_complete:{key}:{profile}:"
                f"{fixed.get('status')}:{fixed.get('evaluated')}:{fixed.get('skipped')}"
            )
        if str(fixed.get("engine_sha256", "")) != engine_hash:
            errors.append(f"fixed500_engine_hash_mismatch:{key}:{profile}")
        if str(fixed.get("structure_hash", "")) != str(structure_by_key[key].get("structure_hash", "")):
            errors.append(f"fixed500_structure_hash_mismatch:{key}:{profile}")
        manifest_path = run_root / "evaluation" / "manifests" / model / "fixed500.json"
        manifest = _read(manifest_path)
        semantic_hash = str(manifest.get("manifest_hash", ""))
        relative = str(manifest_path.relative_to(run_root / "evaluation" / "manifests"))
        file_hash = _sha256(manifest_path)
        expected_file_hash = str(run_manifest.get("manifest_hashes", {}).get(relative, ""))
        matches = (
            str(fixed.get("manifest_hash", "")) == semantic_hash
            and str(fixed.get("manifest_sha256", "")) == file_hash
            and file_hash == expected_file_hash
        )
        manifest_match &= matches
        if not matches:
            errors.append(f"fixed500_manifest_hash_mismatch:{key}:{profile}")

    if fixed_count != expected_fixed500:
        errors.append(f"fixed500_count:{fixed_count}!={expected_fixed500}")
    if len(set(engine_hashes)) != len(engine_hashes) or "" in engine_hashes:
        errors.append("engine_hashes_not_unique_and_complete")
    if observed_profiles != PROFILES:
        errors.append(f"precision_profiles:{sorted(observed_profiles)}!={sorted(PROFILES)}")

    current_head = _git(workdir, "rev-parse", "HEAD")
    certificate = {
        "schema_version": "h800-transformer-dh-phase-a-certificate-v1",
        "phase": "A",
        "status": "accepted_from_existing_artifacts" if not errors else "rejected",
        "run_root": str(run_root),
        "structures": {
            "expected": expected_structures,
            "accepted": len(structures) if not any(value.startswith("structure_") for value in errors) else None,
            "unique_hashes": len(set(structure_hashes)),
        },
        "engines": {
            "expected": expected_engines,
            "accepted": len(engines) if not any(value.startswith("engine_") for value in errors) else None,
            "unique_hashes": len(set(engine_hashes)),
        },
        "fixed500": {
            "expected": expected_fixed500,
            "accepted": fixed_count if not any(value.startswith("fixed500_") for value in errors) else None,
            "evaluated_each": 500,
            "skipped_each": 0,
        },
        "precision_conflicts": precision_conflicts,
        "fallbacks": fallback_count,
        "profiles": sorted(observed_profiles),
        "manifest_hashes_match": manifest_match,
        "structure_hashes_complete": bool(structure_hashes) and "" not in structure_hashes,
        "engine_hashes_complete": bool(engine_hashes) and "" not in engine_hashes,
        "phase_a_artifact_commit": "d0964c39cd350239e9e831b737bed551e17ad313",
        "phase_a_artifact_commit_evidence": (
            "initial implementation commit; per-artifact commit was not recorded, so later "
            "report-only commits must not be attributed to every Phase-A artifact"
        ),
        "microbenchmark_fix_commit": "6cfde0a126b940be1aa8558725b60fc7bd49cb4c",
        "current_branch_head": current_head,
        "attempt_lineage_complete": False,
        "attempt_lineage_limitation": "interrupted/retried attempt IDs were not preserved by Phase-A",
        "eligible_for_phase_a_formal_latency": not errors,
        "eligible_for_phase_b": not errors,
        "errors": errors,
    }
    if write:
        destination = run_root / "phase_a_completion_certificate.json"
        destination.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if errors:
        raise RuntimeError("phase_a_artifact_acceptance_failed:" + ";".join(errors[:20]))
    return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    certificate = accept_phase_a(Path(args.run_root))
    print(json.dumps(certificate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
