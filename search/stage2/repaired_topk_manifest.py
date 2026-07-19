"""Build manifests and per-candidate configs from repaired Stage-1 Top-K."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..candidate import CandidatePhenotype
from ..hashing import canonical_json_hash


METRIC_FIELDS = {
    "R_Fisher": ("R_Fisher", "L_fisher"),
    "L_SQNR": ("L_SQNR", "L_sqnr"),
    "R_Size": ("R_Size", "R_size"),
    "R_BOPS": ("R_BOPS", "R_bops"),
    "P_BOPS": ("P_BOPS", "P_bops"),
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def _phenotype_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return CandidatePhenotype.from_dict(payload).to_dict()


def _profile_from_phenotype(payload: dict[str, Any], field: str) -> dict[str, str]:
    profile = payload.get(field)
    if isinstance(profile, dict) and profile:
        return {str(key): str(value).upper() for key, value in sorted(profile.items())}
    precision_profile = payload.get("precision_profile") or {}
    result: dict[str, str] = {}
    for module_path, decision in precision_profile.items():
        if isinstance(decision, dict):
            key = "requested_precision" if field == "requested_precision_profile" else "realized_precision"
            result[str(module_path)] = str(decision.get(key, "FP16")).upper()
        else:
            result[str(module_path)] = str(decision).upper()
    return {key: result[key] for key in sorted(result)}


def _identity_hash(payload: dict[str, Any]) -> str:
    phenotype = _phenotype_dict(payload)
    metadata = phenotype.get("metadata") or {}
    return canonical_json_hash(
        {
            "pruned_unit_ids": sorted(phenotype.get("pruned_unit_ids") or []),
            "requested_precision_profile": _profile_from_phenotype(phenotype, "requested_precision_profile"),
            "realized_precision_profile": _profile_from_phenotype(phenotype, "realized_precision_profile"),
            "group_keep_map_by_scope": metadata.get("group_keep_map_by_scope", {}),
            "group_prune_map_by_scope": metadata.get("group_prune_map_by_scope", {}),
            "stage1_legalized_group_profile": metadata.get("stage1_legalized_group_profile", {}),
        }
    )


def _archive_hash(payload: dict[str, Any]) -> str:
    return canonical_json_hash(_phenotype_dict(payload))


def _unit_universe(run_dir: Path) -> list[str]:
    domains_path = run_dir / "local_pruning_domains.json"
    if not domains_path.is_file():
        return []
    units: list[str] = []
    seen: set[str] = set()
    for domain in _load_json(domains_path):
        for unit_id in domain.get("ordered_unit_ids", []):
            text = str(unit_id)
            if text not in seen:
                units.append(text)
                seen.add(text)
    return units


def _metrics_from_generation_csv(round_dir: Path, candidate_hash_value: str) -> dict[str, Any] | None:
    for path in sorted(round_dir.glob("generation_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("candidate_hash")) == str(candidate_hash_value):
                    return dict(row)
    return None


def _metrics_from_proxy_archive(run_dir: Path, phenotype: dict[str, Any]) -> dict[str, Any] | None:
    archive = run_dir / "archives" / "proxy_archive.jsonl"
    if not archive.is_file():
        return None
    target = _identity_hash(phenotype)
    target_archive = _archive_hash(phenotype)
    match: dict[str, Any] | None = None
    with archive.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("phenotype_archive_hash", "")) == target_archive:
                match = row
                continue
            archived = row.get("phenotype")
            if isinstance(archived, dict) and _identity_hash(archived) == target:
                match = row
    return match


def _proxy_archive_metric_index(run_dir: Path, phenotypes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    archive = run_dir / "archives" / "proxy_archive.jsonl"
    if not archive.is_file():
        return {}
    wanted = {_identity_hash(phenotype) for phenotype in phenotypes}
    wanted_archive = {
        _archive_hash(phenotype): _identity_hash(phenotype)
        for phenotype in phenotypes
    }
    matches: dict[str, dict[str, Any]] = {}
    with archive.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            archive_hash = str(row.get("phenotype_archive_hash", ""))
            if archive_hash in wanted_archive:
                matches[wanted_archive[archive_hash]] = row
                continue
            archived = row.get("phenotype")
            if not isinstance(archived, dict):
                continue
            identity = _identity_hash(archived)
            if identity in wanted:
                matches[identity] = row
    return matches


def _metric_value(metrics: dict[str, Any] | None, names: tuple[str, ...]) -> float | None:
    if metrics is None:
        return None
    for name in names:
        if name not in metrics or metrics[name] in ("", None):
            continue
        try:
            return float(metrics[name])
        except (TypeError, ValueError):
            return None
    return None


def _keep_and_prune_masks(phenotype: dict[str, Any], universe: list[str]) -> tuple[dict[str, int], dict[str, int], str]:
    pruned = {str(value) for value in phenotype.get("pruned_unit_ids", [])}
    if universe:
        ids = list(universe)
        status = "complete_from_local_pruning_domains"
    else:
        ids = sorted(pruned)
        status = "limited_to_pruned_units_no_unit_universe" if pruned else "empty_no_unit_universe"
    keep_mask = {unit_id: (0 if unit_id in pruned else 1) for unit_id in ids}
    prune_mask = {unit_id: (1 if unit_id in pruned else 0) for unit_id in ids}
    return keep_mask, prune_mask, status


def _physical_candidate_hash(phenotype: dict[str, Any]) -> str:
    metadata = phenotype.get("metadata") or {}
    return canonical_json_hash(
        {
            "version": "repaired-physical-candidate-v1",
            "pruned_unit_ids": sorted(phenotype.get("pruned_unit_ids", [])),
            "group_keep_map_by_scope": metadata.get("group_keep_map_by_scope", {}),
            "group_prune_map_by_scope": metadata.get("group_prune_map_by_scope", {}),
            "stage1_legalized_group_profile": metadata.get("stage1_legalized_group_profile", {}),
        }
    )


def build_repaired_topk_manifest(run_dir: str | Path, *, round_index: int = 0) -> dict[str, Any]:
    root = Path(run_dir)
    round_dir = root / f"round_{round_index:03d}"
    topk_path = round_dir / "stage1_topk.json"
    rows = _load_json(topk_path)
    universe = _unit_universe(root)
    sorted_rows = sorted(rows, key=lambda item: (float(item.get("F1", float("inf"))), str(item.get("candidate_hash", ""))))
    phenotypes = [_phenotype_dict(dict(row.get("phenotype") or {})) for row in sorted_rows]
    proxy_metrics = _proxy_archive_metric_index(root, phenotypes)
    candidates: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows):
        phenotype = phenotypes[rank]
        metadata = dict(phenotype.get("metadata") or {})
        candidate_hash_value = str(row.get("candidate_hash", ""))
        csv_metrics = _metrics_from_generation_csv(round_dir, candidate_hash_value)
        archive_metrics = csv_metrics or proxy_metrics.get(_identity_hash(phenotype))
        keep_mask, prune_mask, mask_status = _keep_and_prune_masks(phenotype, universe)
        metric_source = "generation_csv" if csv_metrics is not None else ("proxy_archive_identity_match" if archive_metrics is not None else "missing_in_stage1_topk_artifact")
        record = {
            "candidate_rank": rank,
            "candidate_rank_1_based": rank + 1,
            "raw_genotype_hash": None,
            "raw_genotype_hash_status": "missing_raw_to_repaired_mapping_in_current_artifacts",
            "repaired_phenotype_hash": candidate_hash_value,
            "physical_candidate_hash": _physical_candidate_hash(phenotype),
            "repaired_F1": float(row.get("F1", float("inf"))),
            "metric_source": metric_source,
            "keep_mask": keep_mask,
            "keep_mask_semantics": "1=keep,0=prune",
            "prune_mask": prune_mask,
            "prune_mask_semantics": "1=prune,0=keep",
            "mask_universe_status": mask_status,
            "resolved_prune_unit_ids": list(phenotype.get("pruned_unit_ids", [])),
            "resolved_prune_indices": {},
            "group_keep_map": metadata.get("group_keep_map", {}),
            "group_prune_map": metadata.get("group_prune_map", {}),
            "group_keep_map_by_scope": metadata.get("group_keep_map_by_scope", {}),
            "group_prune_map_by_scope": metadata.get("group_prune_map_by_scope", {}),
            "requested_precision_group_profile": metadata.get("requested_group_profile", {}),
            "legalized_precision_group_profile": metadata.get("stage1_legalized_group_profile", {}),
            "precision_group_expansion": metadata.get("precision_group_expansion", {}),
            "candidate_config_path": str(round_dir / "stage2_candidate_configs" / f"candidate_{rank:02d}_{candidate_hash_value}.json"),
            "phenotype": phenotype,
        }
        for output_name, input_names in METRIC_FIELDS.items():
            record[output_name] = _metric_value(archive_metrics, input_names)
        candidates.append(record)
    hashes = [row["repaired_phenotype_hash"] for row in candidates]
    manifest = {
        "status": "ok",
        "round_index": int(round_index),
        "source_stage1_topk": str(topk_path),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "checks": {
            "repaired_phenotype_hash_unique": len(set(hashes)) == len(hashes),
            "top5_sorted_by_repaired_F1": [row["repaired_F1"] for row in candidates] == sorted(row["repaired_F1"] for row in candidates),
            "keep_prune_semantics": "M_u=1 keep, M_u=0 prune",
            "repair_only_1_to_0_evidence": "metadata repair_mode mask_preserving_monotonic_floor; raw masks unavailable in current artifacts",
        },
    }
    return manifest


def write_repaired_topk_manifest(run_dir: str | Path, *, round_index: int = 0) -> dict[str, Any]:
    root = Path(run_dir)
    round_dir = root / f"round_{round_index:03d}"
    manifest = build_repaired_topk_manifest(root, round_index=round_index)
    _write_json(round_dir / "repaired_top5_manifest.json", manifest)
    config_dir = round_dir / "stage2_candidate_configs"
    for row in manifest["candidates"]:
        _write_json(Path(row["candidate_config_path"]), row["phenotype"])
    return manifest
