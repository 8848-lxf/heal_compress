"""Source-of-truth assembly for the lidar_pyramid prune/quant ablation.

The module deliberately contains no TensorRT or evaluation side effects.  It
turns the already accepted GA/greedy winners into three exact phenotypes:

* ``prune_quant``: the original searched structure and precision profile;
* ``prune_only``: the same physical pruning selection with strict FP32 compute;
* ``quant_only``: the original all-keep structure with the searched profile.

Heavy deployment is handled by ``scripts/run_lidar_pyramid_prune_quant_ablation.py``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision
from ..hashing import canonical_json_hash


ABLATION_VARIANTS = ("prune_quant", "prune_only", "quant_only")
DEFAULT_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _resolve_artifact_dir(value: str | Path, *, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _greedy_identity(candidate: CandidateGenotype) -> str:
    payload = {
        "pruning_width_genes": dict(sorted(candidate.pruning_width_genes.items())),
        "precision_genes": dict(sorted(candidate.precision_genes.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def replay_greedy_budget_candidate(
    greedy_path: str | Path,
    *,
    target: float = 0.30,
    tolerance: float = 0.005,
) -> dict[str, Any]:
    """Replay and select the lowest-Taylor feasible state on the saved path.

    The historical runner captured the first state below the target, which can
    overshoot a discrete BOPS band.  Replaying logged adjacent actions is exact
    and avoids rerunning the search.  The saved phenotype in the selected
    proxy row remains the authoritative fixed-ranking expansion.
    """

    path = Path(greedy_path).resolve()
    payload = _read_json(path)
    steps = [dict(row) for row in payload.get("steps", [])]
    feasible = [
        row
        for row in steps
        if abs(float(row.get("bops_after", float("inf"))) - float(target))
        <= float(tolerance)
    ]
    if not feasible:
        raise RuntimeError(f"greedy_budget_band_unreachable:{target}:{tolerance}")
    selected = min(
        feasible,
        key=lambda row: (
            float(dict(row.get("metrics") or {}).get("L_joint_weight_taylor", row.get("loss_after", float("inf")))),
            abs(float(row.get("bops_after", float("inf"))) - float(target)),
            int(row.get("step_index", 10**9)),
        ),
    )
    selected_index = int(selected["step_index"])
    current = CandidateGenotype.from_dict(payload["initial_candidate"])
    for step in steps:
        if int(step["step_index"]) > selected_index:
            break
        kind = str(step["action_kind"])
        gene_id = str(step["action_gene_id"])
        if kind == "precision":
            genes = dict(current.precision_genes)
            if str(genes[gene_id]) != str(step["previous_value"]):
                raise RuntimeError(f"greedy_replay_previous_value_mismatch:{step['step_index']}:{gene_id}")
            genes[gene_id] = str(step["selected_value"])
            current = CandidateGenotype(
                pruning_genes=current.pruning_genes,
                pruning_width_genes=current.pruning_width_genes,
                precision_genes=genes,
                meta={"created_by": "greedy_replay_precision"},
            )
        elif kind == "domain_width":
            genes = dict(current.pruning_width_genes)
            if int(genes[gene_id]) != int(step["previous_value"]):
                raise RuntimeError(f"greedy_replay_previous_value_mismatch:{step['step_index']}:{gene_id}")
            genes[gene_id] = int(step["selected_value"])
            current = CandidateGenotype(
                pruning_genes=current.pruning_genes,
                pruning_width_genes=genes,
                precision_genes=current.precision_genes,
                meta={"created_by": "greedy_replay_width"},
            )
        else:
            raise RuntimeError(f"unsupported_greedy_replay_action:{kind}")
    replay_hash = _greedy_identity(current)
    if replay_hash != str(selected["candidate_hash"]):
        raise RuntimeError(
            f"greedy_replay_hash_mismatch:{replay_hash}:{selected['candidate_hash']}"
        )
    metrics = dict(selected.get("metrics") or {})
    phenotype_payload = metrics.get("phenotype")
    if not isinstance(phenotype_payload, Mapping):
        raise RuntimeError("greedy_replay_selected_phenotype_missing")
    phenotype = CandidatePhenotype.from_dict(phenotype_payload)
    actual = float(metrics.get("R_bops_vs_fp32", selected["bops_after"]))
    if abs(actual - float(target)) > float(tolerance):
        raise RuntimeError(f"greedy_replay_hard_band_failed:{actual}:{target}:{tolerance}")
    return {
        "target": float(target),
        "tolerance": float(tolerance),
        "actual_bops": actual,
        "abs_delta": abs(actual - float(target)),
        "selected_step_index": selected_index,
        "selection_policy": "minimum_joint_weight_taylor_within_hard_band",
        "replayed_genotype_hash": replay_hash,
        "logged_genotype_hash": str(selected["candidate_hash"]),
        "proxy_candidate_hash": str(metrics.get("candidate_hash", "")),
        "genotype": current.to_dict(),
        "phenotype": phenotype.to_dict(),
        "proxy_metrics": metrics,
        "feasible_path_states": [
            {
                "step_index": int(row["step_index"]),
                "actual_bops": float(row["bops_after"]),
                "abs_delta": abs(float(row["bops_after"]) - float(target)),
                "joint_weight_taylor": float(
                    dict(row.get("metrics") or {}).get("L_joint_weight_taylor", row["loss_after"])
                ),
                "genotype_hash": str(row["candidate_hash"]),
            }
            for row in feasible
        ],
        "source_path": str(path),
    }


def _require_source_artifact(path: Path) -> None:
    required = (
        "engine.plan",
        "phenotype.json",
        "deployment_manifest.json",
        "physical_validation.json",
        "physical_plan_validation.json",
        "engine_structure_validation.json",
        "precision_realization_validation.json",
        "merge_precision_realization.json",
        "production_qdq_boundary_audit.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing or (path / "engine.plan").stat().st_size <= 0:
        raise RuntimeError(f"source_candidate_artifact_incomplete:{path}:{missing}")
    manifest = _read_json(path / "deployment_manifest.json")
    expected = str(manifest.get("engine_hash", ""))
    if expected:
        digest = hashlib.sha256((path / "engine.plan").read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"source_candidate_engine_hash_mismatch:{path}:{expected}:{digest}")


def collect_authoritative_candidates(
    *,
    ga_root: str | Path,
    greedy_root: str | Path,
    repository_root: str | Path,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    """Load six GA and six greedy winners without trusting filenames alone."""

    repo = Path(repository_root).resolve()
    ga = Path(ga_root).resolve()
    greedy = Path(greedy_root).resolve()
    rows: list[dict[str, Any]] = []
    ga_rows = []
    for path in sorted(ga.glob("round_*/round_best_candidate.json")):
        row = _read_json(path)
        target = float(row["BOPS_target"])
        actual = float(row["R_BOPS_vs_FP32"])
        if abs(actual - target) > float(tolerance):
            raise RuntimeError(f"ga_source_outside_hard_band:{target}:{actual}")
        artifact = _resolve_artifact_dir(row["artifact_dir"], repository_root=repo)
        _require_source_artifact(artifact)
        ga_rows.append(
            {
                "method": "ga",
                "budget": target,
                "actual_bops": actual,
                "candidate_hash": str(row["candidate_hash"]),
                "source_artifact_dir": str(artifact),
                "source_engine_reusable": True,
                "phenotype": _read_json(artifact / "phenotype.json"),
                "proxy_metrics": row,
                "source_record": str(path.resolve()),
            }
        )
    greedy_results = _read_json(greedy / "greedy" / "full_validation_results.json")
    for row in greedy_results.get("candidates", []):
        target = float(list(row.get("budgets") or [])[0])
        if abs(target - 0.30) < 1.0e-9:
            continue
        actual = float(dict(row.get("proxy_metrics") or {})["R_bops_vs_fp32"])
        if abs(actual - target) > float(tolerance):
            raise RuntimeError(f"greedy_source_outside_hard_band:{target}:{actual}")
        artifact = _resolve_artifact_dir(row["artifact_dir"], repository_root=repo)
        _require_source_artifact(artifact)
        rows.append(
            {
                "method": "greedy",
                "budget": target,
                "actual_bops": actual,
                "candidate_hash": str(row["candidate_hash"]),
                "source_artifact_dir": str(artifact),
                "source_engine_reusable": True,
                "phenotype": _read_json(artifact / "phenotype.json"),
                "proxy_metrics": dict(row.get("proxy_metrics") or {}),
                "source_record": str((greedy / "greedy" / "full_validation_results.json").resolve()),
            }
        )
    replay = replay_greedy_budget_candidate(
        greedy / "greedy" / "greedy_path.json",
        target=0.30,
        tolerance=tolerance,
    )
    rows.append(
        {
            "method": "greedy",
            "budget": 0.30,
            "actual_bops": replay["actual_bops"],
            "candidate_hash": "",
            "source_artifact_dir": "",
            "source_engine_reusable": False,
            "phenotype": replay["phenotype"],
            "genotype": replay["genotype"],
            "proxy_metrics": replay["proxy_metrics"],
            "source_record": replay["source_path"],
            "greedy_replay": replay,
        }
    )
    rows.extend(ga_rows)
    expected = {(method, round(float(budget), 6)) for method in ("ga", "greedy") for budget in budgets}
    actual_keys = {(str(row["method"]), round(float(row["budget"]), 6)) for row in rows}
    if actual_keys != expected:
        raise RuntimeError(
            f"authoritative_candidate_budget_set_mismatch:missing={sorted(expected-actual_keys)}:extra={sorted(actual_keys-expected)}"
        )
    return sorted(rows, key=lambda row: (str(row["method"]), -float(row["budget"])))


def _fp32_group_profile(metadata: dict[str, Any]) -> dict[str, str]:
    groups = set(dict(metadata.get("requested_group_profile") or {}))
    groups.update(dict(metadata.get("stage1_legalized_group_profile") or {}))
    groups.update(dict(metadata.get("quantization_group_contracts") or {}))
    return {str(group): "FP32" for group in sorted(groups)}


def build_ablation_phenotype(
    source: CandidatePhenotype,
    variant: str,
) -> CandidatePhenotype:
    """Create an exact P+Q, P-only, or Q-only phenotype."""

    name = str(variant)
    if name not in ABLATION_VARIANTS:
        raise ValueError(f"unsupported_ablation_variant:{name}")
    if name == "prune_quant":
        return CandidatePhenotype.from_dict(source.to_dict())
    metadata = deepcopy(dict(source.metadata))
    metadata["ablation_variant"] = name
    metadata["ablation_source_phenotype_hash"] = canonical_json_hash(source.to_dict())
    if name == "prune_only":
        profile = {
            module: PrecisionDecision("FP32", "FP32", "")
            for module in sorted(source.precision_profile)
        }
        group_profile = _fp32_group_profile(metadata)
        metadata["requested_group_profile"] = group_profile
        metadata["stage1_legalized_group_profile"] = group_profile
        metadata["fallback_report"] = {}
        metadata["precision_fallback_report"] = {}
        for group_id, contract in dict(metadata.get("quantization_group_contracts") or {}).items():
            contract["requested_precision"] = "FP32"
            contract["legalized_precision"] = "FP32"
            contract["realized_precision"] = "FP32"
            metadata["quantization_group_contracts"][group_id] = contract
        return CandidatePhenotype(
            pruned_unit_ids=list(source.pruned_unit_ids),
            precision_profile=profile,
            pruning_policy_version=source.pruning_policy_version,
            precision_policy_version=source.precision_policy_version,
            metadata=metadata,
        )
    domain_contracts = dict(metadata.get("domains") or {})
    original_width_profile = {
        str(domain_id): int(contract["original_width"])
        for domain_id, contract in domain_contracts.items()
        if isinstance(contract, dict) and contract.get("original_width") is not None
    }
    for key in (
        "domain_width_profile",
        "domain_width_expansion_hash",
        "domains",
        "group_keep_map",
        "group_prune_map",
        "group_keep_map_by_scope",
        "group_prune_map_by_scope",
        "pruned_unit_ids",
        "resolved_prune_unit_ids",
        "resolved_prune_indices",
        "resolved_prune_indices_by_scope",
    ):
        metadata.pop(key, None)
    # Unified Transformer candidates must still carry a complete legal all-keep
    # width profile.  Removing the profile is only valid for legacy CNN spaces
    # where physical identity is represented solely by ``pruned_unit_ids``.
    if original_width_profile:
        metadata["domain_width_profile"] = original_width_profile
    metadata["physical_structure_policy"] = "original_all_keep"
    return CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile=source.precision_profile,
        pruning_policy_version=source.pruning_policy_version,
        precision_policy_version=source.precision_policy_version,
        metadata=metadata,
    )


def ablation_config_signature(phenotype: CandidatePhenotype) -> str:
    """Hash only semantics that can change the physical/QDQ/engine result."""

    metadata = dict(phenotype.metadata)
    return canonical_json_hash(
        {
            "pruned_unit_ids": sorted(phenotype.pruned_unit_ids),
            "group_keep_map_by_scope": metadata.get("group_keep_map_by_scope", {}),
            "group_prune_map_by_scope": metadata.get("group_prune_map_by_scope", {}),
            "precision_profile": phenotype.realized_precision_profile,
            "quantization_group_contracts": metadata.get("quantization_group_contracts", {}),
            "pruning_policy_version": phenotype.pruning_policy_version,
            "precision_policy_version": phenotype.precision_policy_version,
        }
    )
