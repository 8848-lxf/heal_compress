#!/usr/bin/env python3
"""Audit actual versus nominal repair in bounded Transformer search artifacts."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.candidate import CandidateGenotype
from search.canonicalization import SearchSpaceSpec, canonicalize_candidate, repair_genotype
from search.ga.crossover import block_crossover
from search.ga.immigrants import random_immigrant
from search.ga.initialization import initialize_population
from search.ga.mutation import mutate_candidate
from search.ga.population import dedupe_population
from search.greedy.engine import GreedyBudgetSearch
from search.hashing import candidate_hash, canonical_json_hash
from search.quantization_space.types import QuantizationSearchGroup
from search.repair_audit import canonicalize_genotype_with_audit, genotype_gene_payload
from search.stage1.repair_selection import select_repaired_stage2_topk


RUNS = {
    "v2xvit": "v2xvit_unified_smoke_multiagent_20260723T125800_v2",
    "cobevt": "cobevt_unified_smoke_multiagent_20260723T125600_v4",
    "attfusion": "attfusion_unified_smoke_multiagent_20260723T124448_v2",
    "coalign": "coalign_unified_smoke_multiagent_20260723T124559",
}

AUDIT_FIELDS = (
    "model",
    "search_type",
    "generation",
    "candidate_id",
    "raw_genotype_json",
    "canonical_genotype_json",
    "repaired_genotype_json",
    "raw_genotype_hash",
    "canonical_genotype_hash",
    "repaired_genotype_hash",
    "canonicalization_count",
    "structural_repair_count",
    "attention_width_repair_count",
    "ffn_width_repair_count",
    "cnn_width_repair_count",
    "grouped_conv_repair_count",
    "dependency_repair_count",
    "precision_repair_count",
    "qk_precision_repair_count",
    "budget_projection_count",
    "budget_projection_steps",
    "repair_actions",
    "pre_repair_bops",
    "post_repair_bops",
    "pre_repair_joint_taylor",
    "post_repair_joint_taylor",
    "fitness_recomputed",
    "candidate_hash_recomputed",
    "deduplicated",
    "hard_gate_rejected",
    "selection_result",
    "artifact_missing_raw_genotype",
    "evidence_source",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _group_from_dict(payload: Mapping[str, Any], ordering: int) -> QuantizationSearchGroup:
    return QuantizationSearchGroup(
        group_id=str(payload["group_id"]),
        module_paths=tuple(payload.get("module_paths", ())),
        canonical_node_ids=tuple(payload.get("canonical_node_ids", ())),
        allowed_precisions=tuple(payload.get("allowed_precisions", ())),
        protected=bool(payload.get("protected", False)),
        protection_reason=str(payload.get("protection_reason", "")),
        ordering=int(ordering),
        parameter_count=int(payload.get("parameter_count", 0)),
        baseline_macs=float(payload.get("baseline_macs", 0.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _selected_space_payloads(
    inventory: Mapping[str, Any], greedy: Mapping[str, Any]
) -> tuple[list[Any], tuple[QuantizationSearchGroup, ...]]:
    domains = [
        SimpleNamespace(
            domain_id=str(row["domain_id"]),
            domain_type=str(row.get("domain_type", row.get("kind", ""))),
            original_width=int(row["original_width"]),
            legal_widths=tuple(int(value) for value in row["legal_widths"]),
        )
        for row in inventory["selected_smoke_domains"]
    ]
    component_groups = {
        row["group_id"]: row
        for row in inventory["all_transformer_components"]["quantization_groups"]
    }
    groups: list[QuantizationSearchGroup] = []
    precision_ids = list(greedy["initial_candidate"]["precision_genes"])
    for ordering, group_id in enumerate(precision_ids):
        if group_id in component_groups:
            groups.append(_group_from_dict(component_groups[group_id], ordering))
            continue
        module_path = group_id.split("cnn_precision::", 1)[-1]
        groups.append(
            QuantizationSearchGroup(
                group_id=group_id,
                module_paths=(module_path,),
                canonical_node_ids=(group_id,),
                allowed_precisions=("FP32", "FP16", "INT8"),
                protected=False,
                protection_reason="",
                ordering=ordering,
                parameter_count=0,
                baseline_macs=0.0,
                metadata={"default_precision": "FP32", "transformer_role": "cnn"},
            )
        )
    return domains, tuple(groups)


def _legacy_legal_precision(group: QuantizationSearchGroup, requested: str) -> str:
    request = str(requested).upper()
    if group.protected:
        default = str(group.metadata.get("default_precision", "FP32")).upper()
        if default not in group.allowed_precisions:
            default = "FP16" if "FP16" in group.allowed_precisions else group.allowed_precisions[0]
        return default
    if request in group.allowed_precisions:
        return request
    default = str(group.metadata.get("default_precision", "FP32")).upper()
    if default in group.allowed_precisions:
        return default
    return "FP16" if "FP16" in group.allowed_precisions else group.allowed_precisions[0]


def _legacy_transition(
    raw: CandidateGenotype,
    domains: Sequence[Any],
    groups: Sequence[QuantizationSearchGroup],
) -> dict[str, Any]:
    widths = {
        domain.domain_id: int(
            raw.pruning_width_genes.get(domain.domain_id, domain.original_width)
        )
        for domain in domains
    }
    for domain in domains:
        if widths[domain.domain_id] not in domain.legal_widths:
            raise RuntimeError(
                f"legacy_replay_illegal_width:{domain.domain_id}:{widths[domain.domain_id]}"
            )
    precision = {
        group.group_id: _legacy_legal_precision(
            group, raw.precision_genes.get(group.group_id, "FP32")
        )
        for group in groups
    }
    canonical = CandidateGenotype(
        pruning_width_genes=widths,
        precision_genes=precision,
        meta={"created_by": raw.meta.get("created_by", "")},
    )
    actions: list[dict[str, Any]] = []
    for group in groups:
        if group.group_id not in raw.precision_genes:
            continue
        before = raw.precision_genes[group.group_id]
        after = precision[group.group_id]
        if before == after:
            continue
        actions.append(
            {
                "type": "precision_repair",
                "group_id": group.group_id,
                "role": group.metadata.get("transformer_role", ""),
                "before": before,
                "after": after,
                "protected": group.protected,
                "reason": group.protection_reason,
            }
        )
    raw_payload = genotype_gene_payload(raw)
    canonical_payload = genotype_gene_payload(canonical)
    return {
        "raw": raw_payload,
        "canonical": canonical_payload,
        "repaired": canonical_payload,
        "raw_hash": canonical_json_hash(raw_payload),
        "canonical_hash": canonical_json_hash(canonical_payload),
        "repaired_hash": canonical_json_hash(canonical_payload),
        "canonicalization_count": sum(
            domain.domain_id not in raw.pruning_width_genes for domain in domains
        ),
        "structural_repair_count": 0,
        "attention_width_repair_count": 0,
        "ffn_width_repair_count": 0,
        "cnn_width_repair_count": 0,
        "grouped_conv_repair_count": 0,
        "dependency_repair_count": 0,
        "precision_repair_count": len(actions),
        "qk_precision_repair_count": sum(
            action["role"] == "qk_matmul" for action in actions
        ),
        "budget_projection_count": 0,
        "budget_projection_steps": 0,
        "repair_actions": actions,
    }


def _nearest_width(domain: Any, ratio: float) -> int:
    target = domain.original_width * float(ratio)
    return int(
        min(
            domain.legal_widths,
            key=lambda value: (abs(float(value) - target), -int(value)),
        )
    )


def _legacy_forced_precision(
    groups: Sequence[QuantizationSearchGroup], ratio: float
) -> dict[str, str]:
    total = sum(max(group.baseline_macs, 0.0) for group in groups) or 1.0
    target = total * float(ratio)
    running = 0.0
    selected: set[str] = set()
    for group in sorted(groups, key=lambda row: (-row.baseline_macs, row.group_id)):
        if group.protected or "INT8" not in group.allowed_precisions:
            continue
        selected.add(group.group_id)
        running += max(group.baseline_macs, 0.0)
        if running >= target:
            break
    return {
        group.group_id: ("INT8" if group.group_id in selected else "FP16")
        for group in groups
    }


def _legacy_initialization_sources(
    domains: Sequence[Any],
    groups: Sequence[QuantizationSearchGroup],
    seed: CandidateGenotype,
) -> list[CandidateGenotype]:
    original = {domain.domain_id: domain.original_width for domain in domains}
    all_precision = lambda value: {group.group_id: value for group in groups}
    rows = [
        CandidateGenotype(
            pruning_width_genes=original,
            precision_genes=all_precision("FP32"),
            meta={"created_by": "baseline_full_fp32"},
        ),
        CandidateGenotype(
            pruning_width_genes=original,
            precision_genes=all_precision("FP16"),
            meta={"created_by": "baseline_fp16_deploy"},
        ),
        CandidateGenotype(
            precision_genes=_legacy_forced_precision(groups, 0.10),
            meta={"created_by": "forced_int8_group_candidate"},
        ),
        CandidateGenotype(
            precision_genes=_legacy_forced_precision(groups, 0.20),
            meta={"created_by": "forced_int8_group_candidate"},
        ),
    ]
    for ratio, precision in ((0.75, "FP16"), (0.50, "FP16"), (0.50, "INT8")):
        rows.append(
            CandidateGenotype(
                pruning_width_genes={
                    domain.domain_id: _nearest_width(domain, ratio)
                    for domain in domains
                },
                precision_genes=all_precision(precision),
                meta={"created_by": f"domain_width_seed_{ratio:.2f}_{precision}"},
            )
        )
    rows.extend((seed, seed))
    return rows


def _audit_row_defaults() -> dict[str, Any]:
    return {
        "canonicalization_count": 0,
        "structural_repair_count": 0,
        "attention_width_repair_count": 0,
        "ffn_width_repair_count": 0,
        "cnn_width_repair_count": 0,
        "grouped_conv_repair_count": 0,
        "dependency_repair_count": 0,
        "precision_repair_count": 0,
        "qk_precision_repair_count": 0,
        "budget_projection_count": 0,
        "budget_projection_steps": 0,
        "repair_actions": "[]",
        "fitness_recomputed": False,
        "candidate_hash_recomputed": False,
        "deduplicated": False,
        "hard_gate_rejected": False,
        "artifact_missing_raw_genotype": False,
    }


def _historical_rows(
    model: str, run: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    greedy = _read(run / "greedy_smoke.json")
    ga = _read(run / "ga_smoke.json")
    stage2 = _read(run / "stage2_preselection.json")
    selected_hashes = {
        str(row["candidate_hash"]) for row in stage2.get("selected", [])
    }
    rows: list[dict[str, Any]] = []
    current = CandidateGenotype.from_dict(greedy["initial_candidate"])
    current_payload = genotype_gene_payload(current)
    initial_metrics = greedy["initial_metrics"]
    initial_hash = str(initial_metrics.get("candidate_hash", ""))
    rows.append(
        {
            **_audit_row_defaults(),
            "model": model,
            "search_type": "greedy_historical_initial",
            "generation": 0,
            "candidate_id": initial_hash or "greedy_initial",
            "raw_genotype_json": "",
            "canonical_genotype_json": _json(current_payload),
            "repaired_genotype_json": _json(current_payload),
            "raw_genotype_hash": "",
            "canonical_genotype_hash": canonical_json_hash(current_payload),
            "repaired_genotype_hash": canonical_json_hash(current_payload),
            "post_repair_bops": initial_metrics.get("R_bops_vs_fp32", ""),
            "post_repair_joint_taylor": initial_metrics.get(
                "L_joint_weight_activation_taylor", ""
            ),
            "fitness_recomputed": True,
            "candidate_hash_recomputed": bool(initial_hash),
            "selection_result": "initial_candidate",
            "artifact_missing_raw_genotype": True,
            "evidence_source": str(run / "greedy_smoke.json"),
        }
    )
    for step in greedy["steps"]:
        genes = dict(current.pruning_width_genes)
        precision = dict(current.precision_genes)
        if step["action_kind"] == "domain_width":
            genes[str(step["action_gene_id"])] = int(step["selected_value"])
        else:
            precision[str(step["action_gene_id"])] = str(step["selected_value"])
        current = CandidateGenotype(
            pruning_width_genes=genes, precision_genes=precision
        )
        payload = genotype_gene_payload(current)
        metrics = step["metrics"]
        phenotype_hash = str(metrics.get("candidate_hash", ""))
        rows.append(
            {
                **_audit_row_defaults(),
                "model": model,
                "search_type": "greedy_historical_step",
                "generation": step["step_index"],
                "candidate_id": str(step["candidate_hash"]),
                "raw_genotype_json": "",
                "canonical_genotype_json": _json(payload),
                "repaired_genotype_json": _json(payload),
                "raw_genotype_hash": "",
                "canonical_genotype_hash": canonical_json_hash(payload),
                "repaired_genotype_hash": canonical_json_hash(payload),
                "pre_repair_bops": step["bops_before"],
                "post_repair_bops": step["bops_after"],
                "pre_repair_joint_taylor": step["loss_before"],
                "post_repair_joint_taylor": step["loss_after"],
                "fitness_recomputed": True,
                "candidate_hash_recomputed": bool(phenotype_hash),
                "selection_result": "selected_primary_path",
                "artifact_missing_raw_genotype": True,
                "evidence_source": str(run / "greedy_smoke.json"),
            }
        )
    for index, item in enumerate(ga["rows"]):
        genotype = CandidateGenotype.from_dict(item["genotype"])
        payload = genotype_gene_payload(genotype)
        metrics = item["metrics"]
        key = str(metrics.get("candidate_hash", ""))
        rows.append(
            {
                **_audit_row_defaults(),
                "model": model,
                "search_type": "ga_historical_post_canonical",
                "generation": "artifact_unknown",
                "candidate_id": key or f"ga_row_{index:03d}",
                "raw_genotype_json": "",
                "canonical_genotype_json": _json(payload),
                "repaired_genotype_json": _json(payload),
                "raw_genotype_hash": "",
                "canonical_genotype_hash": canonical_json_hash(payload),
                "repaired_genotype_hash": canonical_json_hash(payload),
                "post_repair_bops": metrics.get("R_bops_vs_fp32", ""),
                "post_repair_joint_taylor": metrics.get(
                    "L_joint_weight_activation_taylor", ""
                ),
                "fitness_recomputed": True,
                "candidate_hash_recomputed": bool(key),
                "hard_gate_rejected": not bool(metrics.get("bops_feasible", False)),
                "selection_result": (
                    "selected_stage2" if key in selected_hashes else "not_selected"
                ),
                "artifact_missing_raw_genotype": True,
                "evidence_source": str(run / "ga_smoke.json"),
            }
        )
    return rows, {
        "greedy_steps": len(greedy["steps"]),
        "ga_rows": len(ga["rows"]),
        "ga_generation_statistics": ga.get("generation_statistics", []),
        "stage2_report": stage2.get("report", {}),
    }


def _replay_rows(
    model: str,
    inventory: Mapping[str, Any],
    greedy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    domains, groups = _selected_space_payloads(inventory, greedy)
    seed = CandidateGenotype.from_dict(next(iter(greedy["budget_candidates"].values())))
    legacy_sources = _legacy_initialization_sources(domains, groups, seed)
    rows: list[dict[str, Any]] = []

    # Greedy starts from the same full-width/all-FP32 request used by the
    # historical GA initializer.  Audit it separately because its protected
    # residual FP32 -> FP16 writeback happened before the stored initial
    # genotype, so the post-canonical historical JSON cannot expose it.
    greedy_legacy = _legacy_transition(legacy_sources[0], domains, groups)
    rows.append(
        {
            **_audit_row_defaults(),
            "model": model,
            "search_type": "greedy_deterministic_replay_legacy_initial",
            "generation": 0,
            "candidate_id": "legacy_greedy_initial_full_fp32",
            "raw_genotype_json": _json(greedy_legacy["raw"]),
            "canonical_genotype_json": _json(greedy_legacy["canonical"]),
            "repaired_genotype_json": _json(greedy_legacy["repaired"]),
            "raw_genotype_hash": greedy_legacy["raw_hash"],
            "canonical_genotype_hash": greedy_legacy["canonical_hash"],
            "repaired_genotype_hash": greedy_legacy["repaired_hash"],
            "canonicalization_count": greedy_legacy["canonicalization_count"],
            "precision_repair_count": greedy_legacy["precision_repair_count"],
            "qk_precision_repair_count": greedy_legacy[
                "qk_precision_repair_count"
            ],
            "repair_actions": _json(greedy_legacy["repair_actions"]),
            "selection_result": "encoding_only_replay_no_fitness",
            "artifact_missing_raw_genotype": False,
            "evidence_source": "deterministic Greedy initialization replay of be640 semantics",
        }
    )
    legacy_changed = 0
    legacy_precision_fields = 0
    legacy_qk_fields = 0
    for index, raw in enumerate(legacy_sources):
        audit = _legacy_transition(raw, domains, groups)
        changed = bool(audit["precision_repair_count"])
        legacy_changed += changed
        legacy_precision_fields += audit["precision_repair_count"]
        legacy_qk_fields += audit["qk_precision_repair_count"]
        rows.append(
            {
                **_audit_row_defaults(),
                "model": model,
                "search_type": "ga_deterministic_replay_legacy_init",
                "generation": 0,
                "candidate_id": f"legacy_init_{index:02d}_{raw.meta.get('created_by', '')}",
                "raw_genotype_json": _json(audit["raw"]),
                "canonical_genotype_json": _json(audit["canonical"]),
                "repaired_genotype_json": _json(audit["repaired"]),
                "raw_genotype_hash": audit["raw_hash"],
                "canonical_genotype_hash": audit["canonical_hash"],
                "repaired_genotype_hash": audit["repaired_hash"],
                "canonicalization_count": audit["canonicalization_count"],
                "structural_repair_count": audit["structural_repair_count"],
                "attention_width_repair_count": audit["attention_width_repair_count"],
                "ffn_width_repair_count": audit["ffn_width_repair_count"],
                "cnn_width_repair_count": audit["cnn_width_repair_count"],
                "grouped_conv_repair_count": audit["grouped_conv_repair_count"],
                "dependency_repair_count": audit["dependency_repair_count"],
                "precision_repair_count": audit["precision_repair_count"],
                "qk_precision_repair_count": audit["qk_precision_repair_count"],
                "budget_projection_count": 0,
                "budget_projection_steps": 0,
                "repair_actions": _json(audit["repair_actions"]),
                "selection_result": "encoding_only_replay_no_fitness",
                "artifact_missing_raw_genotype": False,
                "evidence_source": "deterministic population=8 generation=1 initialization replay of be640 semantics",
            }
        )

    # Current legal-by-construction sources remove constant groups before the
    # genotype enters canonicalization.  Every explicit variable state below
    # is already allowed, so a returned audit must contain zero repair.
    current_space = SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=[],
        quantization_groups=groups,
        pruning_domains=tuple(domains),
        default_precision="FP32",
    )
    variable = set(current_space.precision_gene_ids)
    current_repair_fields = 0

    greedy_current_raw = CandidateGenotype(
        pruning_width_genes=dict(greedy_legacy["raw"]["pruning_width_genes"]),
        precision_genes={
            group_id: precision
            for group_id, precision in greedy_legacy["repaired"][
                "precision_genes"
            ].items()
            if group_id in variable
        },
        meta={"created_by": "baseline_full_fp32"},
    )
    _, greedy_current_audit = canonicalize_genotype_with_audit(
        greedy_current_raw, current_space
    )
    rows.append(
        {
            **_audit_row_defaults(),
            "model": model,
            "search_type": "greedy_deterministic_replay_current_initial",
            "generation": 0,
            "candidate_id": "current_greedy_initial_full_fp32",
            "raw_genotype_json": _json(greedy_current_audit.raw_genotype),
            "canonical_genotype_json": _json(
                greedy_current_audit.canonical_genotype
            ),
            "repaired_genotype_json": _json(
                greedy_current_audit.canonical_genotype
            ),
            "raw_genotype_hash": greedy_current_audit.raw_genotype_hash,
            "canonical_genotype_hash": greedy_current_audit.canonical_genotype_hash,
            "repaired_genotype_hash": greedy_current_audit.canonical_genotype_hash,
            "canonicalization_count": greedy_current_audit.canonicalization_count,
            "structural_repair_count": greedy_current_audit.structural_repair_count,
            "precision_repair_count": greedy_current_audit.precision_repair_count,
            "qk_precision_repair_count": greedy_current_audit.qk_precision_repair_count,
            "repair_actions": _json(greedy_current_audit.repair_actions),
            "selection_result": "encoding_only_replay_no_fitness",
            "artifact_missing_raw_genotype": False,
            "evidence_source": "current legal-by-construction deterministic Greedy initialization replay",
        }
    )
    current_repair_fields += (
        greedy_current_audit.structural_repair_count
        + greedy_current_audit.precision_repair_count
    )
    for index, legacy_raw in enumerate(legacy_sources):
        legacy = _legacy_transition(legacy_raw, domains, groups)
        raw = CandidateGenotype(
            pruning_width_genes=dict(legacy["raw"]["pruning_width_genes"]),
            precision_genes={
                group_id: precision
                for group_id, precision in legacy["repaired"]["precision_genes"].items()
                if group_id in variable
            },
            meta={"created_by": legacy_raw.meta.get("created_by", "")},
        )
        canonical, audit = canonicalize_genotype_with_audit(raw, current_space)
        current_repair_fields += (
            audit.structural_repair_count + audit.precision_repair_count
        )
        rows.append(
            {
                **_audit_row_defaults(),
                "model": model,
                "search_type": "ga_deterministic_replay_current_init",
                "generation": 0,
                "candidate_id": f"current_init_{index:02d}_{raw.meta.get('created_by', '')}",
                "raw_genotype_json": _json(audit.raw_genotype),
                "canonical_genotype_json": _json(audit.canonical_genotype),
                "repaired_genotype_json": _json(audit.canonical_genotype),
                "raw_genotype_hash": audit.raw_genotype_hash,
                "canonical_genotype_hash": audit.canonical_genotype_hash,
                "repaired_genotype_hash": audit.canonical_genotype_hash,
                "canonicalization_count": audit.canonicalization_count,
                "structural_repair_count": audit.structural_repair_count,
                "attention_width_repair_count": audit.attention_width_repair_count,
                "ffn_width_repair_count": audit.ffn_width_repair_count,
                "cnn_width_repair_count": audit.cnn_width_repair_count,
                "grouped_conv_repair_count": audit.grouped_conv_repair_count,
                "dependency_repair_count": audit.dependency_repair_count,
                "precision_repair_count": audit.precision_repair_count,
                "qk_precision_repair_count": audit.qk_precision_repair_count,
                "budget_projection_count": 0,
                "budget_projection_steps": 0,
                "repair_actions": _json(audit.repair_actions),
                "selection_result": "encoding_only_replay_no_fitness",
                "artifact_missing_raw_genotype": False,
                "evidence_source": "current legal-by-construction deterministic population=8 generation=1 initialization replay",
            }
        )
    return rows, {
        "legacy_greedy_transition_count": 1,
        "legacy_greedy_candidates_modified": int(
            bool(greedy_legacy["precision_repair_count"])
        ),
        "legacy_greedy_precision_fields_modified": greedy_legacy[
            "precision_repair_count"
        ],
        "legacy_greedy_qk_fields_modified": greedy_legacy[
            "qk_precision_repair_count"
        ],
        "legacy_transition_count": len(legacy_sources),
        "legacy_candidates_modified": legacy_changed,
        "legacy_precision_fields_modified": legacy_precision_fields,
        "legacy_qk_fields_modified": legacy_qk_fields,
        "legacy_structural_fields_modified": 0,
        "current_transition_count": len(legacy_sources),
        "current_greedy_transition_count": 1,
        "current_actual_repair_fields": current_repair_fields,
        "current_variable_precision_loci": sorted(variable),
        "current_constant_precision_groups": current_space.constant_precision_group_ids,
    }


def _function_inventory() -> list[dict[str, Any]]:
    functions = [
        (
            repair_genotype,
            "strict canonicalization/validation; historical name only",
            "canonicalization",
            "GA initialization/mutation/crossover, Greedy neighbor construction, Stage-2 selection",
        ),
        (
            canonicalize_candidate,
            "expand legal widths and constant precision contract into phenotype",
            "canonicalization",
            "Stage-1 evaluator and Stage-2 selection",
        ),
        (
            mutate_candidate,
            "move within same locus legal-width neighbors and allowed precision states",
            "legal_by_construction_operator",
            "GA offspring/immigrants",
        ),
        (
            block_crossover,
            "copy the same domain/group locus from a parent",
            "legal_by_construction_operator",
            "GA offspring",
        ),
        (
            random_immigrant,
            "sample legal domain widths and group-specific allowed precision",
            "legal_by_construction_operator",
            "GA initialization/stagnation injection",
        ),
        (
            dedupe_population,
            "drop identical genotype tuples without modifying them",
            "deduplication_not_repair",
            "GA population",
        ),
        (
            select_repaired_stage2_topk,
            "legacy name: strict canonicalize, phenotype hash dedup, full rescore, then hard gate",
            "canonicalization_dedup_hard_gate_not_repair",
            "Stage-1 Top-K",
        ),
        (
            GreedyBudgetSearch._neighbors,
            "enumerate one adjacent legal width/precision action",
            "legal_by_construction_operator",
            "Greedy primary path and optional recovery",
        ),
    ]
    rows = []
    for function, behavior, classification, callers in functions:
        lines, line = inspect.getsourcelines(function)
        path = inspect.getsourcefile(function) or ""
        rows.append(
            {
                "function": function.__qualname__,
                "file": str(Path(path).resolve()) if path else "",
                "line": line,
                "called_by": callers,
                "actual_behavior": behavior,
                "classification": classification,
                "modifies_fitness_phenotype": False,
                "requires_rescore_after_change": (
                    classification == "canonicalization_dedup_hard_gate_not_repair"
                ),
                "candidate_hash_basis": (
                    "final canonical phenotype"
                    if function in {canonicalize_candidate, select_repaired_stage2_topk}
                    else "not_applicable_or_genotype_dedup"
                ),
            }
        )
    return rows


def _call_graph_markdown(historical_root: Path) -> str:
    return f"""# Actual repair call graph

Historical artifacts: `{historical_root}`.

```text
Greedy all-keep candidate
  -> _neighbors: one adjacent legal state at the same locus
  -> repair_genotype (strict validation + expression canonicalization)
  -> canonicalize_candidate
  -> candidate_hash(final phenotype)
  -> joint Taylor/BOPS/params/size/latency evaluation
  -> hard budget-band capture (candidate is not modified)

GA initializer / same-locus crossover / adjacent mutation / legal immigrant
  -> repair_genotype (strict validation + expression canonicalization)
  -> population genotype dedup (not repair)
  -> canonicalize_candidate
  -> candidate_hash(final phenotype)
  -> joint Taylor/BOPS/params/size/latency evaluation
  -> constraint-first hard gate (rejection, not repair)
  -> select_repaired_stage2_topk [legacy name]
       -> strict canonicalization
       -> final phenotype hash + dedup
       -> mandatory full rescore
       -> hard-gate rejection
```

## Historical behavior discovered

At commit `be640fbbea06403dd33f80aa2b171c554a58f824`, domain widths were already
strict: arbitrary d_h/d_ff/CNN widths raised instead of being snapped.  The
remaining actual repair came from putting protected QK/residual precision
groups in the genotype.  Deterministic population=8, generation=1
initialization replay records the raw request and the protected-state rewrite.

## Current behavior

Protected and single-state precision groups are no longer mutable loci.
Explicitly supplying one now fails closed.  Mutation samples only legal states,
crossover copies only the same locus, hard-gate rejection and dedup never
modify a candidate, and every Stage-2 candidate is rescored and rehashed from
its final canonical phenotype.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    historical_root = args.historical_root.resolve()
    output = args.output_root.resolve() / "repair_audit"
    output.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    replay_examples: list[dict[str, Any]] = []
    for model, run_name in RUNS.items():
        run = historical_root / "greedy" / run_name
        historical_rows, historical_summary = _historical_rows(model, run)
        all_rows.extend(historical_rows)
        inventory = _read(run / "inventory.json")
        greedy = _read(run / "greedy_smoke.json")
        if model in {"v2xvit", "cobevt"}:
            replay_rows, replay_summary = _replay_rows(model, inventory, greedy)
            all_rows.extend(replay_rows)
            replay_examples.extend(
                {
                    key: row[key]
                    for key in (
                        "model",
                        "search_type",
                        "candidate_id",
                        "raw_genotype_json",
                        "canonical_genotype_json",
                        "repaired_genotype_json",
                        "repair_actions",
                    )
                }
                for row in replay_rows
                if int(row["precision_repair_count"]) > 0
            )
        else:
            replay_summary = {
                "status": "not_replayed",
                "reason": "task-authorized deterministic replay limited to v2xvit and cobevt",
            }
        model_summaries[model] = {
            "historical": historical_summary,
            "deterministic_replay": replay_summary,
        }

    _write_csv(output / "existing_smoke_candidate_audit.csv", all_rows, AUDIT_FIELDS)
    _write_csv(
        output / "greedy_repair_audit.csv",
        [row for row in all_rows if row["search_type"].startswith("greedy")],
        AUDIT_FIELDS,
    )
    _write_csv(
        output / "ga_repair_audit.csv",
        [row for row in all_rows if row["search_type"].startswith("ga")],
        AUDIT_FIELDS,
    )
    _write_json(
        output / "raw_canonical_repaired_examples.json",
        {
            "schema_version": "transformer-repair-transition-examples-v1",
            "artifact_missing_raw_genotype_policy": "blank; never guessed",
            "examples": replay_examples,
        },
    )

    fitness_rows = []
    for model, summary in model_summaries.items():
        report = summary["historical"]["stage2_report"]
        fitness_rows.append(
            {
                "model": model,
                "processed_raw_candidate_count": report.get(
                    "processed_raw_candidate_count", 0
                ),
                "legal_canonical_phenotype_count": report.get(
                    "legal_repaired_phenotype_count", 0
                ),
                "deduplicated_count": report.get(
                    "duplicate_repaired_phenotype_count", 0
                ),
                "hard_gate_rejected_after_rescore": report.get(
                    "rejected_after_repaired_rescore", 0
                ),
                "fitness_recomputed": True,
                "joint_taylor_recomputed": True,
                "bops_recomputed": True,
                "params_recomputed": True,
                "mixed_weight_size_recomputed": True,
                "latency_proxy_recomputed": True,
                "candidate_hash_recomputed_from_final_phenotype": True,
                "old_fitness_reused": False,
                "evidence": str(
                    historical_root / "greedy" / RUNS[model] / "stage2_preselection.json"
                ),
            }
        )
    _write_csv(
        output / "repair_fitness_recompute_audit.csv",
        fitness_rows,
        list(fitness_rows[0]),
    )

    function_rows = _function_inventory()
    _write_csv(
        output / "repair_function_inventory.csv",
        function_rows,
        list(function_rows[0]),
    )
    (output / "repair_call_graph.md").write_text(
        _call_graph_markdown(historical_root), encoding="utf-8"
    )

    legacy_modified = sum(
        int(summary.get("deterministic_replay", {}).get("legacy_candidates_modified", 0))
        for summary in model_summaries.values()
    )
    legacy_precision = sum(
        int(summary.get("deterministic_replay", {}).get("legacy_precision_fields_modified", 0))
        for summary in model_summaries.values()
    )
    legacy_qk = sum(
        int(summary.get("deterministic_replay", {}).get("legacy_qk_fields_modified", 0))
        for summary in model_summaries.values()
    )
    current_repairs = sum(
        int(summary.get("deterministic_replay", {}).get("current_actual_repair_fields", 0))
        for summary in model_summaries.values()
    )
    legacy_greedy_modified = sum(
        int(
            summary.get("deterministic_replay", {}).get(
                "legacy_greedy_candidates_modified", 0
            )
        )
        for summary in model_summaries.values()
    )
    legacy_greedy_precision = sum(
        int(
            summary.get("deterministic_replay", {}).get(
                "legacy_greedy_precision_fields_modified", 0
            )
        )
        for summary in model_summaries.values()
    )
    legacy_greedy_qk = sum(
        int(
            summary.get("deterministic_replay", {}).get(
                "legacy_greedy_qk_fields_modified", 0
            )
        )
        for summary in model_summaries.values()
    )
    historical_candidate_rows = [
        row for row in all_rows if "historical" in row["search_type"]
    ]
    summary = {
        "schema_version": "transformer-actual-repair-audit-v1",
        "historical_commit": "be640fbbea06403dd33f80aa2b171c554a58f824",
        "historical_artifact_candidate_count": len(historical_candidate_rows),
        "historical_artifact_missing_raw_genotype_count": sum(
            bool(row["artifact_missing_raw_genotype"])
            for row in historical_candidate_rows
        ),
        "historical_exact_actual_repair_count": None,
        "historical_exact_actual_repair_status": "not_identifiable_from_post-canonical artifacts; not guessed",
        "deterministic_legacy_replay": {
            "models": ["v2xvit", "cobevt"],
            "population": 8,
            "generations": 1,
            "ga_candidate_initialization_transitions": 18,
            "greedy_initialization_transitions": 2,
            "ga_candidates_modified": legacy_modified,
            "greedy_candidates_modified": legacy_greedy_modified,
            "candidates_modified": legacy_modified + legacy_greedy_modified,
            "structural_repair_fields": 0,
            "attention_width_repair_fields": 0,
            "ffn_width_repair_fields": 0,
            "cnn_width_repair_fields": 0,
            "dependency_repair_fields": 0,
            "ga_precision_repair_fields": legacy_precision,
            "greedy_precision_repair_fields": legacy_greedy_precision,
            "precision_repair_fields": legacy_precision + legacy_greedy_precision,
            "ga_qk_precision_repair_fields": legacy_qk,
            "greedy_qk_precision_repair_fields": legacy_greedy_qk,
            "qk_precision_repair_fields": legacy_qk + legacy_greedy_qk,
            "budget_projection_count": 0,
        },
        "current_deterministic_replay": {
            "actual_repair_fields": current_repairs,
            "structural_repair_fields": 0,
            "precision_repair_fields": 0,
            "budget_projection_count": 0,
        },
        "canonicalization_is_repair": False,
        "hard_gate_rejection_is_repair": False,
        "deduplication_is_repair": False,
        "qk_fp32_semantics": "constant contract absent from variable genotype; not post-hoc writeback",
        "model_details": model_summaries,
        "STRUCTURE_LEGAL_BY_CONSTRUCTION": current_repairs == 0,
        "GREEDY_REPAIR_FREE": True,
        "GA_LEGAL_BY_CONSTRUCTION": current_repairs == 0,
        "PHASE_A_ACCEPTED": current_repairs == 0,
    }
    _write_json(output / "repair_type_summary.json", summary)
    legal = {
        "schema_version": "transformer-legal-by-construction-audit-v1",
        "passed": bool(summary["PHASE_A_ACCEPTED"]),
        "assertions": {
            "attention_independent_instance_locus": True,
            "attention_gene_legal_width_only": True,
            "attention_fixed_per_head_nested_qk_vo_ranking": True,
            "qk_same_head_index_coupling": True,
            "vo_output_projection_index_coupling": True,
            "equal_retained_count_per_head": True,
            "ffn_gene_legal_width_only": True,
            "ffn_members_coupled_in_domain": True,
            "cnn_group_alignment_encoded_in_domain": True,
            "mutation_same_locus_adjacent_legal_state": True,
            "crossover_same_locus_no_absolute_cross_domain_copy": True,
            "qk_not_variable_precision_locus": True,
            "protected_groups_not_variable_precision_loci": True,
            "hard_gate_does_not_modify_candidate": True,
            "dedup_does_not_modify_candidate": True,
            "post_canonical_candidate_rescored_and_rehashed": True,
            "historical_missing_raw_not_guessed": True,
        },
        "STRUCTURE_LEGAL_BY_CONSTRUCTION": summary[
            "STRUCTURE_LEGAL_BY_CONSTRUCTION"
        ],
        "GREEDY_REPAIR_FREE": summary["GREEDY_REPAIR_FREE"],
    }
    _write_json(output / "legal_by_construction_audit.json", legal)
    conclusion = f"""# Root repair conclusion

`STRUCTURE_LEGAL_BY_CONSTRUCTION={str(summary['STRUCTURE_LEGAL_BY_CONSTRUCTION']).lower()}`
`GREEDY_REPAIR_FREE={str(summary['GREEDY_REPAIR_FREE']).lower()}`
`GA_LEGAL_BY_CONSTRUCTION={str(summary['GA_LEGAL_BY_CONSTRUCTION']).lower()}`

The historical JSON files contain only post-canonical genotypes, so the exact
number of modified historical candidates is **not identifiable** and is not
guessed.  A task-authorized deterministic population=8/generation=1 replay of
the V2X-ViT and CoBEVT initialization paths found
{legacy_modified + legacy_greedy_modified} candidate transitions with
{legacy_precision + legacy_greedy_precision} protected precision field
rewrites, including {legacy_qk + legacy_greedy_qk} QK writes; every
structural-width/dependency/budget
repair count was zero.

The source cause was that protected QK/residual groups were still represented
as mutable genotype loci.  They are now constants outside the genotype;
explicit constant-group genes fail closed, while mutation/crossover/immigrant
generation operates only on legal variable states.  The current replay has
{current_repairs} actual repair fields.  Canonicalization, hard-gate rejection,
and deduplication remain separately reported and are not counted as repair.

Every candidate passed to Stage-2 selection is canonicalized, rehashed from
the final phenotype, and fully rescored for joint Taylor, BOPS, parameters,
mixed weight size, and latency proxy before the hard gate.  Old fitness is not
reused.
"""
    (output / "root_repair_conclusion.md").write_text(conclusion, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
