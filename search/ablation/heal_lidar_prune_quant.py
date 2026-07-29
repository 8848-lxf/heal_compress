"""Family-generic P/Q ablation planning for HEAL LiDAR baselines.

This module is deliberately side-effect free apart from reading accepted search
artifacts.  It does not build engines or evaluate them.  The resulting matrix
is the contract consumed by a build phase followed by an evaluation-only phase:

* ``prune_quant`` reuses the already accepted joint engine, but is freshly
  evaluated under the ablation protocol;
* ``prune_only`` keeps the exact source pruning phenotype and requests FP32 for
  every weighted module;
* ``quant_only`` restores the original all-keep structure and keeps the exact
  source precision/merge contract.

Build ownership is assigned by the deployment-relevant phenotype signature, so
the same P-only or Q-only configuration is never built twice merely because it
is selected by more than one budget.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..candidate import CandidatePhenotype
from ..hashing import canonical_json_hash
from .lidar_pyramid_prune_quant import (
    ABLATION_VARIANTS,
    DEFAULT_BUDGETS,
    ablation_config_signature,
    build_ablation_phenotype,
)


SUPPORTED_FAMILIES = (
    "heal_lidar_attfusion",
    "heal_lidar_fcooper",
    "heal_lidar_disco",
)
HEAL_LIDAR_AUXILIARY_PRECISION_KEY = "heal_lidar_auxiliary_precision"


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _require_family_run(run_root: Path, *, family_id: str) -> dict[str, Any]:
    report_path = run_root / "context_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"heal_lidar_ablation_context_report_missing:{report_path}")
    report = _read_json(report_path)
    if str(report.get("family_id", "")) != str(family_id):
        raise RuntimeError(
            f"heal_lidar_ablation_family_mismatch:{report.get('family_id')}:{family_id}"
        )
    if int(report.get("fixed_k", 0)) <= 0:
        raise RuntimeError("heal_lidar_ablation_fixed_k_missing")
    return report


def accepted_candidate_artifacts(
    artifact_dir: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Resolve and verify an accepted family-native Stage-2 artifact bundle."""

    repo = Path(repository_root).resolve()
    root = _resolve_path(artifact_dir, repository_root=repo)
    paths = {
        "phenotype": root / "phenotype.json",
        "engine": root / "deployment/candidate.plan",
        "precision_acceptance": root / "deployment/precision_realization_acceptance.json",
        "stage2_result": root / "candidate_stage2_result.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"heal_lidar_source_artifact_incomplete:{root}:{missing}")
    if paths["engine"].stat().st_size <= 0:
        raise RuntimeError(f"heal_lidar_source_engine_empty:{paths['engine']}")
    precision = _read_json(paths["precision_acceptance"])
    if not bool(precision.get("passed", False)):
        raise RuntimeError(f"heal_lidar_source_precision_not_accepted:{root}")
    stage2 = _read_json(paths["stage2_result"])
    if str(stage2.get("status", "")) != "ok":
        raise RuntimeError(f"heal_lidar_source_stage2_not_ok:{root}")
    engine_sha256 = _sha256(paths["engine"])
    expected_hash = str(stage2.get("engine_sha256", ""))
    if expected_hash and expected_hash != engine_sha256:
        raise RuntimeError(
            f"heal_lidar_source_engine_hash_mismatch:{expected_hash}:{engine_sha256}"
        )
    phenotype = _read_json(paths["phenotype"])
    return {
        "artifact_dir": str(root),
        "phenotype_path": str(paths["phenotype"]),
        "phenotype": phenotype,
        "phenotype_hash": canonical_json_hash(phenotype),
        "engine_path": str(paths["engine"]),
        "engine_sha256": engine_sha256,
        "precision_acceptance_path": str(paths["precision_acceptance"]),
        "stage2_result_path": str(paths["stage2_result"]),
        "stage2_result": stage2,
    }


def collect_authoritative_family_candidates(
    *,
    family_id: str,
    ga_root: str | Path,
    greedy_root: str | Path,
    repository_root: str | Path,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
    tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    """Collect six accepted GA and six accepted greedy source phenotypes.

    GA source identity comes from each round winner's Stage-2 directory.  Greedy
    identity comes from ``greedy/full_validation_results.json``.  Full-validation
    directories are intentionally not treated as deployment sources because they
    contain evaluation-only outputs rather than the ONNX/QDQ/engine provenance.
    """

    if family_id not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported_heal_lidar_ablation_family:{family_id}")
    repo = Path(repository_root).resolve()
    ga = Path(ga_root).resolve()
    greedy = Path(greedy_root).resolve()
    ga_context = _require_family_run(ga, family_id=family_id)
    greedy_context = _require_family_run(greedy, family_id=family_id)
    if int(ga_context["fixed_k"]) != int(greedy_context["fixed_k"]):
        raise RuntimeError("heal_lidar_ablation_fixed_k_mismatch_between_methods")
    if str(ga_context.get("checkpoint_hash", "")) != str(
        greedy_context.get("checkpoint_hash", "")
    ):
        raise RuntimeError("heal_lidar_ablation_checkpoint_mismatch_between_methods")

    rows: list[dict[str, Any]] = []
    for record_path in sorted(ga.glob("round_*/round_best_candidate.json")):
        record = _read_json(record_path)
        budget = float(record["BOPS_target"])
        actual = float(record["R_BOPS_vs_FP32"])
        if abs(actual - budget) > float(tolerance):
            raise RuntimeError(f"heal_lidar_ga_source_outside_hard_band:{budget}:{actual}")
        accepted = accepted_candidate_artifacts(
            record["artifact_dir"], repository_root=repo
        )
        rows.append(
            {
                "family_id": family_id,
                "method": "ga",
                "budget": budget,
                "actual_bops": actual,
                "candidate_hash": str(record["candidate_hash"]),
                "source_record": str(record_path.resolve()),
                **accepted,
            }
        )

    greedy_results_path = greedy / "greedy/full_validation_results.json"
    greedy_results = _read_json(greedy_results_path)
    for record in list(greedy_results.get("candidates") or []):
        selected_budgets = list(record.get("budgets") or [])
        if len(selected_budgets) != 1:
            raise RuntimeError(
                f"heal_lidar_greedy_budget_identity_ambiguous:{record.get('candidate_hash')}:{selected_budgets}"
            )
        budget = float(selected_budgets[0])
        proxy = dict(record.get("proxy_metrics") or {})
        actual = float(proxy["R_bops_vs_fp32"])
        if abs(actual - budget) > float(tolerance):
            raise RuntimeError(
                f"heal_lidar_greedy_source_outside_hard_band:{budget}:{actual}"
            )
        accepted = accepted_candidate_artifacts(
            record["artifact_dir"], repository_root=repo
        )
        rows.append(
            {
                "family_id": family_id,
                "method": "greedy",
                "budget": budget,
                "actual_bops": actual,
                "candidate_hash": str(record["candidate_hash"]),
                "source_record": str(greedy_results_path.resolve()),
                **accepted,
            }
        )

    expected = {
        (method, round(float(budget), 6))
        for method in ("ga", "greedy")
        for budget in budgets
    }
    actual_keys = {
        (str(row["method"]), round(float(row["budget"]), 6)) for row in rows
    }
    if actual_keys != expected:
        raise RuntimeError(
            "heal_lidar_ablation_budget_set_mismatch:"
            f"missing={sorted(expected-actual_keys)}:extra={sorted(actual_keys-expected)}"
        )
    return sorted(
        rows,
        key=lambda row: (
            0 if row["method"] == "ga" else 1,
            -float(row["budget"]),
        ),
    )


def collect_formal_family_candidates(
    *,
    family_id: str,
    formal_root: str | Path,
    repository_root: str | Path,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
    tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    """Collect accepted Greedy/GA engines from the unified formal runner.

    Current formal runs place both methods in one ``formal_ga_results.json``.
    Resource metrics are intentionally recomputed from each serialized
    phenotype during the build phase; the target stored here is only a
    provisional in-band value and is never reported as independently exact.
    """

    if family_id not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported_heal_lidar_ablation_family:{family_id}")
    repo = Path(repository_root).resolve()
    root = Path(formal_root).resolve()
    _require_family_run(root, family_id=family_id)
    result_path = root / "reports/formal_ga_results.json"
    if not result_path.is_file():
        raise RuntimeError(f"heal_lidar_formal_results_missing:{result_path}")
    formal = _read_json(result_path)
    requested = tuple(float(value) for value in budgets)
    labels = {f"{int(round(value * 100)):03d}": value for value in requested}
    missing = sorted(set(labels) - set(dict(formal.get("results") or {})))
    if missing:
        raise RuntimeError(f"heal_lidar_formal_budget_missing:{missing}")

    rows: list[dict[str, Any]] = []
    for method, key in (("ga", "final_winner"), ("greedy", "greedy_anchor")):
        for label, budget in labels.items():
            payload = dict(formal["results"][label][key])
            if str(payload.get("status", "")) != "ok":
                raise RuntimeError(
                    f"heal_lidar_formal_candidate_not_ok:{method}:{label}:"
                    f"{payload.get('status')}"
                )
            if payload.get("requested_realized_exact") is not True:
                raise RuntimeError(
                    f"heal_lidar_formal_candidate_precision_not_exact:{method}:{label}"
                )
            metadata = dict(payload.get("metadata") or {})
            raw = dict(metadata.get("raw") or {})
            artifact_value = (
                raw.get("source_artifact_dir")
                or metadata.get("source_artifact_dir")
            )
            if not artifact_value:
                engine_value = raw.get("engine_path") or metadata.get("engine_path")
                if not engine_value:
                    raise RuntimeError(
                        f"heal_lidar_formal_artifact_missing:{method}:{label}"
                    )
                engine_path = _resolve_path(engine_value, repository_root=repo)
                artifact_value = (
                    engine_path.parent.parent
                    if engine_path.parent.name == "deployment"
                    else engine_path.parent
                )
            accepted = accepted_candidate_artifacts(
                artifact_value, repository_root=repo
            )
            candidate_hash = str(payload.get("complete_phenotype_hash", ""))
            if not candidate_hash:
                raise RuntimeError(
                    f"heal_lidar_formal_candidate_hash_missing:{method}:{label}"
                )
            rows.append(
                {
                    "family_id": family_id,
                    "method": method,
                    "budget": float(budget),
                    "actual_bops": float(budget),
                    "actual_bops_source": "pending_exact_phenotype_recompute",
                    "candidate_hash": candidate_hash,
                    "source_record": str(result_path.resolve()),
                    **accepted,
                }
            )
    expected = {
        (method, round(float(budget), 6))
        for method in ("ga", "greedy")
        for budget in requested
    }
    actual = {
        (str(row["method"]), round(float(row["budget"]), 6)) for row in rows
    }
    if actual != expected:
        raise RuntimeError(
            "heal_lidar_formal_budget_set_mismatch:"
            f"missing={sorted(expected-actual)}:extra={sorted(actual-expected)}"
        )
    if any(abs(float(row["actual_bops"]) - float(row["budget"])) > tolerance for row in rows):
        raise RuntimeError("heal_lidar_formal_provisional_bops_out_of_band")
    return sorted(
        rows,
        key=lambda row: (0 if row["method"] == "ga" else 1, -float(row["budget"])),
    )


def validate_family_ablation_derivation(
    source: CandidatePhenotype,
    derived: CandidatePhenotype,
    *,
    variant: str,
) -> None:
    """Fail closed if the P/Q counterfactual changes the wrong axis."""

    if variant == "prune_quant":
        if derived.to_dict() != source.to_dict():
            raise RuntimeError("prune_quant_phenotype_not_exact_source")
        if str(
            derived.metadata.get(HEAL_LIDAR_AUXILIARY_PRECISION_KEY, "FP16")
        ).upper() != "FP16":
            raise RuntimeError("prune_quant_auxiliary_precision_not_fp16")
        return
    if variant == "prune_only":
        if list(derived.pruned_unit_ids) != list(source.pruned_unit_ids):
            raise RuntimeError("prune_only_pruned_units_changed")
        source_metadata = dict(source.metadata)
        derived_metadata = dict(derived.metadata)
        for key in ("group_keep_map_by_scope", "group_prune_map_by_scope"):
            if derived_metadata.get(key, {}) != source_metadata.get(key, {}):
                raise RuntimeError(f"prune_only_physical_map_changed:{key}")
        if any(
            decision.realized_precision.upper() != "FP32"
            or decision.requested_precision.upper() != "FP32"
            for decision in derived.precision_profile.values()
        ):
            raise RuntimeError("prune_only_precision_not_strict_fp32")
        if str(
            derived.metadata.get(HEAL_LIDAR_AUXILIARY_PRECISION_KEY, "")
        ).upper() != "FP32":
            raise RuntimeError("prune_only_auxiliary_precision_not_fp32")
        return
    if variant == "quant_only":
        if derived.pruned_unit_ids:
            raise RuntimeError("quant_only_not_all_keep")
        if derived.realized_precision_profile != source.realized_precision_profile:
            raise RuntimeError("quant_only_precision_profile_changed")
        if str(
            derived.metadata.get(HEAL_LIDAR_AUXILIARY_PRECISION_KEY, "FP16")
        ).upper() != "FP16":
            raise RuntimeError("quant_only_auxiliary_precision_not_fp16")
        for key in ("group_keep_map_by_scope", "group_prune_map_by_scope"):
            if key in derived.metadata:
                raise RuntimeError(f"quant_only_contains_physical_map:{key}")
        return
    raise ValueError(f"unsupported_ablation_variant:{variant}")


def build_family_ablation_matrix(
    sources: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build logical rows and deterministic unique-engine build ownership."""

    rows: list[dict[str, Any]] = []
    for raw_source in sources:
        source = dict(raw_source)
        phenotype = CandidatePhenotype.from_dict(source["phenotype"])
        source_id = (
            f"{source['family_id']}__{source['method']}__bops_{float(source['budget']):.2f}"
        )
        for variant in ABLATION_VARIANTS:
            derived = build_ablation_phenotype(phenotype, variant)
            if variant == "prune_only":
                metadata = dict(derived.metadata)
                metadata[HEAL_LIDAR_AUXILIARY_PRECISION_KEY] = "FP32"
                derived = CandidatePhenotype(
                    pruned_unit_ids=list(derived.pruned_unit_ids),
                    precision_profile=dict(derived.precision_profile),
                    pruning_policy_version=derived.pruning_policy_version,
                    precision_policy_version=derived.precision_policy_version,
                    metadata=metadata,
                )
            validate_family_ablation_derivation(
                phenotype, derived, variant=variant
            )
            auxiliary_precision = str(
                derived.metadata.get(HEAL_LIDAR_AUXILIARY_PRECISION_KEY, "FP16")
            ).upper()
            signature = canonical_json_hash(
                {
                    "phenotype_signature": ablation_config_signature(derived),
                    "heal_lidar_auxiliary_precision": auxiliary_precision,
                }
            )
            row_id = f"{source_id}__{variant}"
            rows.append(
                {
                    "row_id": row_id,
                    "family_id": source["family_id"],
                    "method": source["method"],
                    "budget": float(source["budget"]),
                    "actual_bops": float(source["actual_bops"]),
                    "variant": variant,
                    "candidate_hash": source["candidate_hash"],
                    "source_artifact_dir": source["artifact_dir"],
                    "source_phenotype_hash": source["phenotype_hash"],
                    "phenotype": derived.to_dict(),
                    "phenotype_hash": canonical_json_hash(derived.to_dict()),
                    "deployment_config_signature": signature,
                    "heal_lidar_auxiliary_precision": auxiliary_precision,
                    "source_joint_engine_path": source["engine_path"],
                    "source_joint_engine_sha256": source["engine_sha256"],
                    "engine_build_policy": (
                        "reuse_accepted_source_joint_engine"
                        if variant == "prune_quant"
                        else "fresh_build_once_per_deployment_signature"
                    ),
                    "requires_fresh_evaluation": True,
                }
            )

    owner_by_signature: dict[str, str] = {}
    for row in rows:
        if row["variant"] == "prune_quant":
            row["build_owner_row_id"] = None
            row["requires_engine_build"] = False
            row["engine_reuse_row_id"] = None
            continue
        signature = str(row["deployment_config_signature"])
        owner = owner_by_signature.setdefault(signature, str(row["row_id"]))
        row["build_owner_row_id"] = owner
        row["requires_engine_build"] = owner == row["row_id"]
        row["engine_reuse_row_id"] = None if owner == row["row_id"] else owner
    return rows


__all__ = [
    "SUPPORTED_FAMILIES",
    "accepted_candidate_artifacts",
    "build_family_ablation_matrix",
    "collect_authoritative_family_candidates",
    "collect_formal_family_candidates",
    "validate_family_ablation_derivation",
]
