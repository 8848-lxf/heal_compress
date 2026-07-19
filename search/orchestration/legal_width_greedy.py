"""Six-budget legal-width greedy Stage-1 orchestration."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..admission.bops_band import BopsBandPolicy
from ..canonicalization import canonicalize_legal_width_candidate
from ..encoding.legal_width_genotype import LegalWidthGenotype
from ..greedy.joint_budget_search import GreedySearchState, run_targeted_greedy
from ..greedy.legal_actions import enumerate_legal_actions
from ..hashing import candidate_hash
from ..proxy.joint_loss_scale import (
    calibrate_joint_loss_scale,
    write_joint_loss_scale,
)
from ..stage1.topk_selector import ProxyCandidateRecord


_PRECISION_BITS = {"FP32": 32, "FP16": 16, "INT8": 8}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _budget_label(target: float) -> str:
    return f"budget_{int(round(float(target) * 100.0)):03d}"


def _initial_genotype(space: Any) -> LegalWidthGenotype:
    inventory = space.legal_width_inventory
    widths = {
        domain.domain_id: len(domain.legal_keep_widths) - 1
        for domain in inventory.domains
    }
    precisions: dict[str, str] = {}
    for group_id, actions in sorted(space.precision_action_space.items()):
        normalized = tuple(dict.fromkeys(str(value).upper() for value in actions))
        unknown = sorted(set(normalized) - set(_PRECISION_BITS))
        if unknown or not normalized:
            raise RuntimeError(
                f"greedy_initial_precision_actions_invalid:{group_id}:{unknown}"
            )
        precisions[group_id] = max(
            normalized, key=lambda value: (_PRECISION_BITS[value], value)
        )
    genotype = LegalWidthGenotype(
        widths,
        precisions,
        {
            "seed_family": "all_keep_highest_deployable_precision",
            "normal_candidate_repair_invoked": False,
        },
    )
    genotype.validate(inventory, space.precision_action_space)
    return genotype


def _phenotype_hash(metrics: Mapping[str, Any], genotype: LegalWidthGenotype) -> str:
    metadata = dict(dict(metrics.get("phenotype", {}) or {}).get("metadata", {}) or {})
    return str(
        metadata.get("phenotype_hash")
        or metrics.get("candidate_hash")
        or genotype.genotype_hash
    )


def _path_row(
    state: GreedySearchState,
    *,
    target: float,
    sequence: int,
) -> dict[str, Any]:
    return {
        "target_bops": float(target),
        "path_sequence": int(sequence),
        "genotype_hash": state.genotype_hash,
        "phenotype_hash": _phenotype_hash(state.metrics, state.genotype),
        "parent_hash": state.parent_hash,
        "action_id": state.action_id,
        "normal_candidate_repair_invoked": False,
        "genotype": state.genotype.to_dict(),
        **dict(state.metrics),
    }


def _compact_trace_payload(result: Any) -> dict[str, Any]:
    """Keep greedy lineage without duplicating every evaluated genotype.

    Endpoints and accepted paths remain fully auditable.  The evaluated-state
    collection is represented by its deterministic hash and a small boundary
    sample; full traces are still available through ``trace_detail: full``
    for targeted debugging.
    """

    state_hashes = [state.genotype_hash for state in result.evaluated_states]
    digest = hashlib.sha256("\n".join(state_hashes).encode("utf-8")).hexdigest()
    return {
        "trace_schema": "greedy-compact-v1",
        "target": float(result.target),
        "status": result.status,
        "terminal": result.terminal.to_dict() if result.terminal else None,
        "admission_mode": result.admission_mode,
        "accepted_path_states": [
            state.to_dict() for state in result.accepted_path_states
        ],
        "expansion_count": result.expansion_count,
        "failure_reason": result.failure_reason,
        "evaluated_state_count": result.evaluated_state_count,
        "evaluated_state_digest": digest,
        "evaluated_state_sample": {
            "first": state_hashes[:3],
            "last": state_hashes[-3:],
        },
        "rejection_counts": dict(result.rejection_counts),
        "bops_funnel": dict(result.bops_funnel),
        "nearest_misses": [dict(row) for row in result.nearest_misses],
    }


def _configured_anchor_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in config.get("anchor_scale_rows", ()) or ()]
    raw_path = str(config.get("anchor_scale_rows_path", "")).strip()
    if not raw_path:
        return rows
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"greedy_anchor_scale_rows_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    source: Sequence[Mapping[str, Any]]
    if isinstance(payload, list):
        source = payload
    else:
        source = payload.get("rows", payload.get("anchors", ()))
    rows.extend(dict(row) for row in source)
    return rows


def load_greedy_endpoints(path: str | Path | None) -> list[dict[str, Any]]:
    """Load terminal greedy endpoint artifacts without loading path states."""

    if path is None or not str(path).strip():
        return []
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"greedy_endpoint_manifest_missing:{source}")
    paths = (
        sorted(source.glob("budget_*_endpoint.json"))
        if source.is_dir()
        else [source]
    )
    endpoints = []
    for item in paths:
        payload = json.loads(item.read_text(encoding="utf-8"))
        rows = (
            list(payload.get("endpoints", []))
            if isinstance(payload, dict) and "endpoints" in payload
            else [payload]
        )
        endpoints.extend(
            dict(row)
            for row in rows
            if str(row.get("status", "ok")) != "infeasible"
            and row.get("phenotype")
        )
    return endpoints


def run_six_budget_greedy(
    context: Any,
    proxy: Any,
    run_dir: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run proxy-only greedy searches and freeze one shared linear scale."""

    space = context.search_space
    if str(space.structure_gene_type) != "legal_keep_width":
        raise RuntimeError("six_budget_greedy_requires_legal_width_space")
    targets = sorted({float(value) for value in config.get("targets", ())})
    if not targets or any(not 0.0 < value <= 1.0 for value in targets):
        raise ValueError("greedy_targets_must_be_nonempty_unit_interval")
    primary_tolerance = float(config.get("primary_tolerance", 0.005))
    expanded_tolerance = float(config.get("expanded_tolerance", 0.0075))
    frontier_size = int(config.get("frontier_size", 8))
    max_expansions = int(config.get("max_expansions", 4096))
    destination = Path(run_dir)
    greedy_dir = destination / "greedy"
    greedy_dir.mkdir(parents=True, exist_ok=True)
    initial = _initial_genotype(space)
    metric_memo: dict[str, dict[str, Any]] = {}

    def evaluate(genotype: LegalWidthGenotype) -> Mapping[str, Any]:
        cached = metric_memo.get(genotype.genotype_hash)
        if cached is not None:
            return cached
        metrics = dict(proxy.evaluate(genotype, generation=0, outer_round=0))
        metrics.setdefault(
            "R_BOPS",
            metrics.get("R_bops_vs_fp32", metrics.get("R_bops")),
        )
        metrics["normal_candidate_repair_invoked"] = False
        metric_memo[genotype.genotype_hash] = metrics
        return metrics

    def evaluate_batch(
        genotypes: Sequence[LegalWidthGenotype],
    ) -> Sequence[Mapping[str, Any]]:
        missing = [
            genotype
            for genotype in genotypes
            if genotype.genotype_hash not in metric_memo
        ]
        if missing:
            result = proxy.evaluate_batch(missing, generation=0, outer_round=0)
            rows = list(result.metrics)
            if len(rows) != len(missing):
                raise RuntimeError(
                    f"greedy_proxy_batch_count_mismatch:{len(rows)}!={len(missing)}"
                )
            for genotype, source in zip(missing, rows):
                metrics = dict(source)
                metrics.setdefault(
                    "R_BOPS",
                    metrics.get("R_bops_vs_fp32", metrics.get("R_bops")),
                )
                metrics["normal_candidate_repair_invoked"] = False
                metric_memo[genotype.genotype_hash] = metrics
        return [metric_memo[genotype.genotype_hash] for genotype in genotypes]

    def successors(genotype: LegalWidthGenotype):
        return enumerate_legal_actions(
            genotype,
            inventory=space.legal_width_inventory,
            precision_actions=space.precision_action_space,
        )

    budget_results = []
    endpoint_rows: list[dict[str, Any]] = []
    endpoint_records: list[ProxyCandidateRecord] = []
    path_states: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        adjacent = tuple(
            targets[row]
            for row in (index - 1, index + 1)
            if 0 <= row < len(targets)
        )
        result = run_targeted_greedy(
            initial_genotype=initial,
            evaluate=evaluate,
            evaluate_batch=(
                evaluate_batch if callable(getattr(proxy, "evaluate_batch", None)) else None
            ),
            enumerate_successors=successors,
            policy=BopsBandPolicy(
                target=target,
                primary_tolerance=primary_tolerance,
                expanded_tolerance=expanded_tolerance,
                adjacent_targets=adjacent,
            ),
            frontier_size=frontier_size,
            max_expansions=max_expansions,
        )
        label = _budget_label(target)
        trace_detail = str(config.get("trace_detail", "compact")).lower()
        if trace_detail not in {"compact", "full"}:
            raise ValueError("greedy_trace_detail_must_be_compact_or_full")
        trace_payload = (
            result.to_dict()
            if trace_detail == "full"
            else _compact_trace_payload(result)
        )
        _write_json(greedy_dir / f"{label}_trace.json", trace_payload)
        budget_results.append(
            {
                "target_bops": target,
                "status": result.status,
                "admission_mode": result.admission_mode,
                "evaluated_state_count": result.evaluated_state_count,
                "expansion_count": result.expansion_count,
                "failure_reason": result.failure_reason,
                "bops_funnel": dict(result.bops_funnel),
            }
        )
        for sequence, state in enumerate(result.accepted_path_states):
            path_states.append(
                _path_row(state, target=target, sequence=sequence)
            )
        if result.terminal is None:
            _write_json(
                greedy_dir / f"{label}_endpoint.json",
                {
                    "target_bops": target,
                    "status": "infeasible",
                    "failure_reason": result.failure_reason,
                    "nearest_misses": [dict(row) for row in result.nearest_misses],
                },
            )
            continue
        state = result.terminal
        phenotype = canonicalize_legal_width_candidate(state.genotype, space)
        identity = str(
            state.metrics.get("candidate_hash")
            or candidate_hash(phenotype, space)
        )
        endpoint = {
            "target_bops": target,
            "actual_bops": float(state.metrics["R_BOPS"]),
            "signed_bops_error": float(state.metrics["R_BOPS"]) - target,
            "admission_mode": result.admission_mode,
            "candidate_hash": identity,
            "genotype_hash": state.genotype_hash,
            "phenotype_hash": str(
                phenotype.metadata.get("phenotype_hash", identity)
            ),
            "normal_candidate_repair_invoked": False,
            "genotype": state.genotype.to_dict(),
            "phenotype": phenotype.to_dict(),
            "metrics": dict(state.metrics),
        }
        endpoint_rows.append(endpoint)
        endpoint_records.append(
            ProxyCandidateRecord(
                candidate_hash=identity,
                genotype=state.genotype,
                phenotype=phenotype,
                F1=float(state.metrics.get("F1", state.metrics["L_joint_raw"])),
                metrics=dict(state.metrics),
            )
        )
        _write_json(greedy_dir / f"{label}_endpoint.json", endpoint)

    scale_rows = [
        {
            "phenotype_hash": row["phenotype_hash"],
            "L_joint_raw": row["L_joint_raw"],
        }
        for row in path_states
    ]
    scale_rows.extend(_configured_anchor_rows(config))
    scale = calibrate_joint_loss_scale(
        scale_rows,
        code_commit=str(getattr(context, "code_commit", "")),
    )
    scale_path = write_joint_loss_scale(destination / "joint_loss_scale.json", scale)
    summary = {
        "target_count": len(targets),
        "feasible_budget_count": len(endpoint_rows),
        "infeasible_budget_count": len(targets) - len(endpoint_rows),
        "targets": targets,
        "primary_tolerance": primary_tolerance,
        "expanded_tolerance": expanded_tolerance,
        "frontier_size": frontier_size,
        "max_expansions": max_expansions,
        "trace_detail": trace_detail,
        "unique_proxy_evaluations": len(metric_memo),
        "strict_fp32_search_group_count": sum(
            precision == "FP32" for precision in initial.precision_genes.values()
        ),
        "fixed_non_fp32_search_group_count": sum(
            precision != "FP32" for precision in initial.precision_genes.values()
        ),
        "normal_candidate_repair_invocation_count": 0,
        "budget_results": budget_results,
        "endpoints": endpoint_rows,
        "path_states": path_states,
        "joint_loss_scale": scale,
        "joint_loss_scale_path": str(scale_path.resolve()),
    }
    _write_json(greedy_dir / "greedy_summary.json", summary)
    return {**summary, "endpoint_records": endpoint_records}
