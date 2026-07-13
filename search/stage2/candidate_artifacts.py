"""Candidate-level Stage-2 summary artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..candidate import CandidatePhenotype
from .objective import Stage2ObjectiveConfig


SUMMARY_ARTIFACTS = [
    "sampling_pruning_request.json",
    "physical_pruning_plan.json",
    "physical_plan_validation.json",
    "physical_structure_snapshot.json",
    "physical_validation.json",
    "physical_widths.csv",
    "pruned_checkpoint.pth",
    "pruned_state_dict.pth",
    "pruned_fp32.onnx",
    "pruned_qdq.onnx",
    "engine.plan",
    "engine_layer_info.json",
    "engine_structure_validation.json",
    "precision_realization_validation.json",
    "evaluation.json",
    "stage2_score.json",
]

CACHE_COPY_ARTIFACTS = [name for name in SUMMARY_ARTIFACTS if name != "stage2_score.json"]


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _plain(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(candidate_dir: str | Path, names: list[str] | None = None) -> dict[str, Any]:
    root = Path(candidate_dir)
    artifacts: dict[str, Any] = {}
    for name in names or SUMMARY_ARTIFACTS:
        path = root / name
        if not path.is_file():
            continue
        stat = path.stat()
        artifacts[name] = {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "sha256": _sha256(path),
        }
    return {"candidate_dir": str(root), "artifacts": artifacts}


def _copy_cached_artifacts(root: Path, stage2_score: dict[str, Any], *, overwrite: bool) -> str:
    source_text = str(stage2_score.get("artifact_dir") or "")
    if not source_text:
        return ""
    source = Path(source_text)
    try:
        if source.resolve() == root.resolve():
            return str(source)
    except OSError:
        return str(source)
    if not source.is_dir():
        return str(source)
    for name in CACHE_COPY_ARTIFACTS:
        src = source / name
        dst = root / name
        if not src.is_file():
            continue
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return str(source)


def write_candidate_summary_artifacts(
    candidate_dir: str | Path,
    *,
    candidate_hash: str,
    phenotype: CandidatePhenotype,
    stage2_score: dict[str, Any],
    objective_config: Stage2ObjectiveConfig,
    stage1_manifest_record: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> None:
    root = Path(candidate_dir)
    artifact_source_dir = _copy_cached_artifacts(root, stage2_score, overwrite=overwrite)
    evaluation_path = root / "evaluation.json"
    if evaluation_path.is_file():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _write_json(root / "evaluation_300.json", evaluation, overwrite=overwrite)
    manifest = {
        "candidate_hash": str(candidate_hash),
        "status": str(stage2_score.get("status", "")),
        "phenotype": phenotype.to_dict(),
        "stage1_manifest_record": dict(stage1_manifest_record or {}),
        "artifact_dir": str(root),
        "artifact_source_dir": artifact_source_dir or str(root),
        "required_artifacts": list(SUMMARY_ARTIFACTS),
    }
    _write_json(root / "candidate_manifest.json", manifest, overwrite=overwrite)
    objective = {
        "formula": "eta_AP * L_AP + eta_latency * R_latency",
        "eta_AP": float(objective_config.eta_map),
        "eta_latency": float(objective_config.eta_latency),
        "tau_AP": objective_config.tau_ap,
        "latency_metric": objective_config.latency_metric,
        "L_AP": stage2_score.get("L_map_real"),
        "R_latency": stage2_score.get("R_latency_real"),
        "F2": stage2_score.get("F2"),
        "status": stage2_score.get("status"),
        "accuracy_reference": "original_strict_fp32_required_by_contract",
        "latency_reference": "original_strict_fp16_required_by_contract",
    }
    _write_json(root / "stage2_objective_report.json", objective, overwrite=overwrite)
    _write_json(root / "artifact_hashes.json", artifact_hashes(root), overwrite=overwrite)
