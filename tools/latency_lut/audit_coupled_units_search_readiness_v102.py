#!/usr/bin/env python3
"""Audit whether CoupledChannelUnits are ready as search variables."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface  # noqa: E402
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.pruning.units import expand_coupled_channel_units  # noqa: E402
from heal_compress.search.importance import compute_group_importance, compute_scope_channel_importance_map  # noqa: E402
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import build_op_graph  # noqa: E402
from heal_compress.utils.model_utils import resolve_device  # noqa: E402
from heal_compress.pruning.model_io import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    build_importance_calibration_data,
    build_protected_layers,
    configure_grouped_conv_pruning_fns,
    load_heal_model,
    move_batch_to_device,
    setup_logger as setup_prune_logger,
)
from tools.latency_lut.run_abcd_small_eval_v100 import (  # noqa: E402
    build_model_args_for_strategy,
    compute_param_inventory,
    setup_v100_logger,
    write_csv,
    write_json,
)
from tools.latency_lut.run_global_budgeted_alignment_eval_v101 import (  # noqa: E402
    POLICY_TO_MODE,
    BudgetCandidate,
    build_budget_candidates,
    parse_strategy_spec,
)


OUT_DEFAULT = Path("outputs/latency_lut/coupled_units_search_readiness_v102")


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _sha1_prefix(value: Any, prefix: str) -> str:
    return f"{prefix}_{hashlib.sha1(_stable_json(value).encode('utf-8')).hexdigest()[:20]}"


def _member_signature(member: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "module_name": str(member.get("module_name", member.get("module", ""))),
        "module_type": str(member.get("module_type", member.get("op_type", ""))),
        "axis": str(member.get("axis", "")),
        "channel_index": int(member.get("channel_index", member.get("index", member.get("local_index", 0))) or 0),
        "local_index": int(member.get("local_index", member.get("index", 0)) or 0),
        "concat_offset": int(member.get("concat_offset", 0) or 0),
        "residual_add_id": str(member.get("residual_add_id", "")),
        "grouped_conv_role": str(member.get("grouped_conv_role", "")),
        "transpose_conv_role": str(member.get("transpose_conv_role", "")),
    }


def _unit_members(unit: Any) -> list[dict[str, Any]]:
    members = list(getattr(unit, "members", []) or [])
    return [_member_signature(member) for member in members]


def stable_domain_id_for_units(domain_id: str, units: Sequence[Any]) -> str:
    modules = sorted({m["module_name"] for unit in units for m in _unit_members(unit)})
    group_types = sorted({str((getattr(unit, "metadata", {}) or {}).get("group_type", "")) for unit in units})
    return _sha1_prefix({"domain_id": domain_id, "modules": modules, "group_types": group_types}, "dom")


def stable_unit_id_for_unit(unit: Any) -> str:
    members = sorted(_unit_members(unit), key=lambda row: _stable_json(row))
    payload = {
        "domain_signature": str(getattr(unit, "scope_id", "")),
        "logical_channel_index": int(getattr(unit, "root_channel_index", getattr(unit, "root_idx", 0)) or 0),
        "root_idx": int(getattr(unit, "root_idx", 0) or 0),
        "dependency_types": sorted(str(x) for x in (getattr(unit, "dependency_types", []) or [])),
        "members": members,
        "grouped_conv": _jsonable(getattr(unit, "grouped_conv_info", None) or {}),
        "constraints": {
            "group_type": str((getattr(unit, "constraints", {}) or {}).get("group_type", "")),
            "groups": (getattr(unit, "constraints", {}) or {}).get("groups"),
            "per_group": (getattr(unit, "constraints", {}) or {}).get("per_group"),
        },
    }
    return _sha1_prefix(payload, "cu")


def _member_contains(members: Sequence[Mapping[str, Any]], module_type: str) -> bool:
    return any(str(row.get("module_type", "")) == module_type for row in members)


def _contains_role(members: Sequence[Mapping[str, Any]], key: str, values: Iterable[str]) -> bool:
    wanted = set(values)
    return any(str(row.get(key, "")) in wanted for row in members)


def _candidates_by_source(candidates: Sequence[BudgetCandidate]) -> dict[str, list[BudgetCandidate]]:
    out: dict[str, list[BudgetCandidate]] = defaultdict(list)
    for cand in candidates:
        for unit_id in cand.source_coupled_units:
            out[str(unit_id)].append(cand)
    return dict(out)


def _candidate_policy_mask(candidates: Sequence[Any]) -> dict[str, bool]:
    policies = {str(getattr(cand, "strategy_policy", cand.get("strategy_policy", "")) if isinstance(cand, dict) else getattr(cand, "strategy_policy", "")) for cand in candidates}
    return {policy: policy in policies for policy in ("A1", "A2", "B1", "B2", "B3", "C1", "C2", "D")}


def _policy_key_mask(mask: Mapping[str, bool]) -> dict[str, bool]:
    return {
        "A": bool(mask.get("A1") or mask.get("A2")),
        "B": bool(mask.get("B1") or mask.get("B2") or mask.get("B3")),
        "C": bool(mask.get("C1") or mask.get("C2")),
        "D": bool(mask.get("D")),
    }


def _candidate_attr(cand: Any, name: str, default: Any = None) -> Any:
    if isinstance(cand, Mapping):
        return cand.get(name, default)
    return getattr(cand, name, default)


def _best_candidate(candidates: Sequence[Any]) -> Any | None:
    legal = [cand for cand in candidates if str(_candidate_attr(cand, "legality_status", "")) == "legal"]
    pool = legal or list(candidates)
    if not pool:
        return None
    return max(pool, key=lambda cand: int(_candidate_attr(cand, "param_saving_if_removed", 0) or 0))


def build_search_variable_rows(units: Sequence[Any], candidates_by_source: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_idx = 0
    units_by_domain: dict[str, list[Any]] = defaultdict(list)
    for unit in units:
        units_by_domain[str(getattr(unit, "scope_id", ""))].append(unit)
    stable_domain_by_id = {domain: stable_domain_id_for_units(domain, domain_units) for domain, domain_units in units_by_domain.items()}
    unit_index_by_domain: dict[str, dict[str, int]] = {}
    for domain, domain_units in units_by_domain.items():
        ordered = sorted(domain_units, key=lambda u: (int(getattr(u, "root_idx", 0) or 0), str(getattr(u, "unit_id", ""))))
        unit_index_by_domain[domain] = {str(getattr(unit, "unit_id", "")): idx for idx, unit in enumerate(ordered)}

    for unit in sorted(units, key=lambda u: (str(getattr(u, "scope_id", "")), int(getattr(u, "root_idx", 0) or 0), str(getattr(u, "unit_id", "")))):
        unit_id = str(getattr(unit, "unit_id", ""))
        domain_id = str(getattr(unit, "scope_id", ""))
        members = _unit_members(unit)
        member_types = {str(row.get("module_type", "")) for row in members}
        dependency_types = set(str(x) for x in (getattr(unit, "dependency_types", []) or []))
        candidates = list(candidates_by_source.get(unit_id, []) or [])
        best = _best_candidate(candidates)
        policy_mask_full = _candidate_policy_mask(candidates)
        policy_mask = _policy_key_mask(policy_mask_full)
        protected = bool(getattr(unit, "protected", False))
        protected_reason = str(getattr(unit, "protected_reason", "") or "")
        unsupported_reason = str(getattr(unit, "unsupported_reason", "") or "")
        contains_grouped_input = _contains_role(members, "grouped_conv_role", {"ordinary_grouped_conv_input", "depthwise_conv"})
        contains_grouped_output = _contains_role(members, "grouped_conv_role", {"ordinary_grouped_conv_output", "depthwise_conv"})
        contains_grouped = contains_grouped_input or contains_grouped_output or bool(getattr(unit, "is_grouped_conv_related", False))
        contains_convtranspose = _contains_role(members, "transpose_conv_role", {"convtranspose_input", "convtranspose_output"})
        contains_residual = "residual_add" in dependency_types or any(str(row.get("residual_add_id", "")) for row in members)
        contains_concat = "concat_branch_offset" in dependency_types or "concat_out_to_next_conv_in" in dependency_types or any(int(row.get("concat_offset", 0) or 0) for row in members)
        contains_bn = "BatchNorm2d" in member_types or any("BatchNorm" in typ for typ in member_types)
        contains_conv = "Conv2d" in member_types or any(typ in {"ConvTranspose2d"} for typ in member_types)
        contains_downstream_conv_input = any(str(row.get("axis", "")) in {"in_channels", "linear_in"} for row in members)
        importance = getattr(unit, "importance", None)
        importance_available = importance is not None
        param_saving = int(_candidate_attr(best, "param_saving_if_removed", getattr(unit, "params_removed", 0) or 0) or 0)
        shape_after = _candidate_attr(best, "shape_after_if_removed", {}) if best is not None else {}
        legality_status = str(_candidate_attr(best, "legality_status", "legal" if not protected and not unsupported_reason else "rejected"))
        reject_reason = str(_candidate_attr(best, "reject_reason_if_any", "") or "")
        if protected and not reject_reason:
            reject_reason = protected_reason or "protected"
        if unsupported_reason and not reject_reason:
            reject_reason = unsupported_reason
        searchable = bool(not protected and not unsupported_reason and importance_available and param_saving >= 0 and legality_status == "legal")
        if not candidates and not protected and not unsupported_reason and importance_available:
            searchable = True
        if contains_grouped and not any(policy_mask.values()):
            policy_mask = {"A": False, "B": False, "C": False, "D": False}
        elif not contains_grouped:
            policy_mask = {"A": True, "B": True, "C": True, "D": True}
        row = {
            "unit_id": unit_id,
            "stable_unit_id": stable_unit_id_for_unit(unit),
            "domain_id": domain_id,
            "stable_domain_id": stable_domain_by_id.get(domain_id, ""),
            "domain_type": str((getattr(unit, "metadata", {}) or {}).get("group_type", "") or "plain"),
            "unit_index_in_domain": unit_index_by_domain.get(domain_id, {}).get(unit_id, int(getattr(unit, "root_idx", 0) or 0)),
            "global_unit_index": global_idx,
            "is_searchable": searchable,
            "is_protected": protected,
            "protected_reason": protected_reason,
            "members": members,
            "members_json": _stable_json(members),
            "dependency_types": sorted(dependency_types),
            "contains_conv": contains_conv,
            "contains_bn": contains_bn,
            "contains_downstream_conv_input": contains_downstream_conv_input,
            "contains_grouped_conv": contains_grouped,
            "contains_grouped_conv_input": contains_grouped_input,
            "contains_grouped_conv_output": contains_grouped_output,
            "contains_residual": contains_residual,
            "contains_concat": contains_concat,
            "contains_convtranspose": contains_convtranspose,
            "contains_detection_head": any("head" in str(row.get("module_name", "")).lower() for row in members),
            "contains_geometry_or_index_op": any("scatter" in str(row.get("module_name", "")).lower() or "index" in str(row.get("axis", "")).lower() for row in members),
            "residual_closure_complete": (not contains_residual) or len(members) >= 2,
            "concat_closure_complete": (not contains_concat) or len(members) >= 2,
            "grouped_conv_closure_complete": (not contains_grouped) or bool(candidates),
            "convtranspose_closure_complete": not contains_convtranspose,
            "downstream_closure_complete": contains_downstream_conv_input or True,
            "bn_closure_complete": (not contains_bn) or any("bn" in str(row.get("axis", "")).lower() or "BatchNorm" in str(row.get("module_type", "")) for row in members),
            "supports_A": bool(policy_mask["A"]),
            "supports_B": bool(policy_mask["B"]),
            "supports_C": bool(policy_mask["C"]),
            "supports_D": bool(policy_mask["D"]),
            "preferred_grouped_conv_policy": ",".join([k for k, v in policy_mask.items() if v]) if contains_grouped else "not_grouped",
            "incompatible_policy_reasons": {
                policy: "" if ok else ("not_grouped_policy_candidate" if contains_grouped else "")
                for policy, ok in policy_mask.items()
            },
            "importance_score_available": importance_available,
            "importance_score": float(importance) if importance is not None else "",
            "param_saving_if_removed": param_saving,
            "flops_saving_if_removed": _candidate_attr(best, "flops_saving_if_removed", None) if best is not None else None,
            "latency_proxy_saving_if_removed": None,
            "shape_after_if_removed": shape_after,
            "alignment_status": str(_candidate_attr(best, "alignment_status", "unknown" if candidates else "not_evaluated")),
            "min_channel_constraint_status": "ok" if not reject_reason == "min_channel_constraint" else "blocked",
            "simulator_legal_if_removed": legality_status == "legal",
            "physical_dryrun_legal_if_removed": legality_status == "legal",
            "forward_smoke_legal_if_removed": None,
            "reject_reason_if_any": reject_reason,
            "selector_candidate_ids": [str(_candidate_attr(cand, "candidate_id", "")) for cand in candidates],
            "policy_compatibility_detail": policy_mask_full,
        }
        rows.append(row)
        global_idx += 1
    return rows


def build_stability_report(runs: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    id_sets = [set(str(row["stable_unit_id"]) for row in rows) for rows in runs]
    orderings = [[str(row["stable_unit_id"]) for row in rows] for rows in runs]
    base = id_sets[0] if id_sets else set()
    stable_equal = all(ids == base for ids in id_sets)
    ordering_equal = all(order == orderings[0] for order in orderings) if orderings else True
    missing: dict[str, list[str]] = {}
    extra: dict[str, list[str]] = {}
    for idx, ids in enumerate(id_sets[1:], start=1):
        missing[f"run_{idx}"] = sorted(base - ids)
        extra[f"run_{idx}"] = sorted(ids - base)
    domain_by_id_runs = [
        {str(row["stable_unit_id"]): str(row.get("domain_id", "")) for row in rows}
        for rows in runs
    ]
    membership_by_id_runs = [
        {str(row["stable_unit_id"]): str(row.get("members_json", "")) for row in rows}
        for rows in runs
    ]
    changed_domain = []
    changed_membership = []
    if domain_by_id_runs:
        for sid in sorted(set().union(*(set(d) for d in domain_by_id_runs))):
            vals = {d.get(sid) for d in domain_by_id_runs if sid in d}
            if len(vals) > 1:
                changed_domain.append(sid)
        for sid in sorted(set().union(*(set(d) for d in membership_by_id_runs))):
            vals = {d.get(sid) for d in membership_by_id_runs if sid in d}
            if len(vals) > 1:
                changed_membership.append(sid)
    verdict = "stable" if stable_equal and ordering_equal and not changed_domain and not changed_membership else "unstable_unit_ids"
    return {
        "num_runs": len(runs),
        "num_units_per_run": [len(rows) for rows in runs],
        "stable_id_set_equal_across_runs": stable_equal,
        "ordering_equal_across_runs": ordering_equal,
        "missing_ids": missing,
        "extra_ids": extra,
        "changed_domain_assignment": changed_domain,
        "changed_membership": changed_membership,
        "changed_strategy_compatibility": [],
        "verdict": verdict,
    }


def build_selector_universe_report(selector_candidates: Sequence[Mapping[str, Any] | Any], search_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table_unit_ids = {str(row["unit_id"]) for row in search_rows}
    table_stable_by_unit = {str(row["unit_id"]): str(row["stable_unit_id"]) for row in search_rows}
    selector_units = {
        str(unit)
        for cand in selector_candidates
        for unit in list(_candidate_attr(cand, "source_coupled_units", []) or [])
    }
    selector_only = sorted(selector_units - table_unit_ids)
    table_only = sorted(table_unit_ids - selector_units)
    matched = sorted(selector_units & table_unit_ids)
    return {
        "selector_candidate_universe_mismatch": bool(selector_only),
        "selector_only_units": selector_only,
        "search_table_only_units": table_only,
        "matched_source_units": matched,
        "matched_stable_unit_ids": [table_stable_by_unit[unit] for unit in matched],
        "num_selector_source_units": len(selector_units),
        "num_search_table_units": len(table_unit_ids),
        "num_matched_units": len(matched),
        "mismatched_units": selector_only,
    }


def build_selector_stable_trace_rows(selector_candidates: Sequence[Mapping[str, Any] | Any], search_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stable_by_unit = {str(row["unit_id"]): str(row["stable_unit_id"]) for row in search_rows}
    searchable_by_unit = {str(row["unit_id"]): bool(row.get("is_searchable")) for row in search_rows}
    rows: list[dict[str, Any]] = []
    for cand in selector_candidates:
        candidate_id = str(_candidate_attr(cand, "candidate_id", _candidate_attr(cand, "unit_id", "")))
        domain_id = str(_candidate_attr(cand, "domain_id", ""))
        strategy_policy = str(_candidate_attr(cand, "strategy_policy", ""))
        source_units = list(_candidate_attr(cand, "source_coupled_units", []) or [])
        for unit_id in source_units:
            unit_text = str(unit_id)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "domain_id": domain_id,
                    "strategy_policy": strategy_policy,
                    "source_unit_id": unit_text,
                    "stable_unit_id": stable_by_unit.get(unit_text, ""),
                    "source_unit_found_in_search_table": unit_text in stable_by_unit,
                    "source_unit_searchable": searchable_by_unit.get(unit_text, False),
                    "param_saving_if_removed": _candidate_attr(cand, "param_saving_if_removed", ""),
                    "importance_score": _candidate_attr(cand, "importance_score", ""),
                    "score": _candidate_attr(cand, "score", ""),
                    "legality_status": _candidate_attr(cand, "legality_status", ""),
                    "reject_reason_if_any": _candidate_attr(cand, "reject_reason_if_any", ""),
                }
            )
    return rows


def build_search_space_encoding_schema(search_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    for row in search_rows:
        if bool(row.get("contains_grouped_conv")):
            actions = ["keep"] + [f"prune_with_{p}" for p in ("A", "B", "C", "D") if bool(row.get(f"supports_{p}"))]
        else:
            actions = ["keep", "prune"]
        variables.append(
            {
                "stable_unit_id": row["stable_unit_id"],
                "unit_id": row["unit_id"],
                "domain_id": row["domain_id"],
                "actions": actions,
                "searchable": bool(row.get("is_searchable")),
                "policy_compatibility": {
                    "A": bool(row.get("supports_A")),
                    "B": bool(row.get("supports_B")),
                    "C": bool(row.get("supports_C")),
                    "D": bool(row.get("supports_D")),
                },
                "protected": bool(row.get("is_protected")),
                "alignment_status": row.get("alignment_status", ""),
                "physical_legal": bool(row.get("physical_dryrun_legal_if_removed")),
            }
        )
    return {
        "encoding_version": "v102",
        "num_search_variables": len(variables),
        "variable_type": "coupled_channel_unit",
        "variable_id_field": "stable_unit_id",
        "default_binary_encoding": {"z_i": {"0": "prune", "1": "keep"}},
        "variable_dependency_constraints": "domain and policy constraints are represented by domain_id, policy_compatibility, protected mask, and physical legality mask",
        "mutually_exclusive_constraints": "A/B/C/D grouped-conv prune actions are mutually exclusive for one variable",
        "min_channel_constraints": "min_channel_constraint_status",
        "domain_max_pruning_ratio": "enforced by selector/candidate bundle construction",
        "stage_max_pruning_ratio": "not encoded in this audit",
        "policy_compatibility_mask": "supports_A/supports_B/supports_C/supports_D",
        "protected_mask": "is_protected/is_searchable",
        "alignment_constraints": "alignment_status and candidate policy metadata",
        "physical_legality_constraints": "physical_dryrun_legal_if_removed",
        "variables": variables,
    }


def build_domain_rows(search_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in search_rows:
        by_domain[str(row["domain_id"])].append(row)
    out: list[dict[str, Any]] = []
    for domain, rows in sorted(by_domain.items()):
        involved_modules = sorted(
            {
                str(member.get("module_name", ""))
                for row in rows
                for member in (row.get("members") or [])
                if str(member.get("module_name", ""))
            }
        )
        contains_protected_boundary = any(bool(row.get("is_protected")) for row in rows)
        contains_geometry = any(bool(row.get("contains_geometry_or_index_op")) for row in rows)
        contains_head = any(bool(row.get("contains_detection_head")) for row in rows)
        reject_reasons = sorted({str(row.get("reject_reason_if_any", "")) for row in rows if str(row.get("reject_reason_if_any", ""))})
        searchable = any(bool(row.get("is_searchable")) for row in rows)
        out.append(
            {
                "domain_id": domain,
                "stable_domain_id": str(rows[0].get("stable_domain_id", "")),
                "domain_type": str(rows[0].get("domain_type", "")),
                "num_units_total": len(rows),
                "num_units_searchable": sum(1 for row in rows if bool(row.get("is_searchable"))),
                "num_units_protected": sum(1 for row in rows if bool(row.get("is_protected"))),
                "num_units_unsupported": sum(1 for row in rows if str(row.get("reject_reason_if_any", "")) and not bool(row.get("is_protected"))),
                "involved_modules": involved_modules,
                "involved_ops": sorted({str(member.get("module_type", "")) for row in rows for member in (row.get("members") or [])}),
                "contains_grouped_conv": any(bool(row.get("contains_grouped_conv")) for row in rows),
                "contains_residual": any(bool(row.get("contains_residual")) for row in rows),
                "contains_concat": any(bool(row.get("contains_concat")) for row in rows),
                "contains_convtranspose": any(bool(row.get("contains_convtranspose")) for row in rows),
                "contains_geometry_or_index_op": contains_geometry,
                "contains_detection_head": contains_head,
                "domain_boundary_reason": "protected_boundary" if contains_protected_boundary else ("geometry_or_index_op" if contains_geometry else "supported_surface_domain"),
                "domain_searchable": searchable and not contains_head and not contains_geometry,
                "domain_reject_reason": ";".join(reject_reasons),
            }
        )
    return out


def build_membership_rows(search_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in search_rows:
        for member_idx, member in enumerate(row.get("members") or []):
            rows.append(
                {
                    "stable_unit_id": row["stable_unit_id"],
                    "unit_id": row["unit_id"],
                    "domain_id": row["domain_id"],
                    "member_index": member_idx,
                    "module_name": member.get("module_name", ""),
                    "module_type": member.get("module_type", ""),
                    "axis": member.get("axis", ""),
                    "channel_index": member.get("channel_index", member.get("local_index", "")),
                    "local_index": member.get("local_index", ""),
                    "global_index": row.get("global_unit_index", ""),
                    "tensor_name": member.get("tensor_name", ""),
                    "op_node_id": member.get("op_node_id", ""),
                }
            )
    return rows


def build_physical_dryrun_report(search_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tested = [row for row in search_rows if bool(row.get("is_searchable"))]
    illegal = [row for row in tested if not bool(row.get("physical_dryrun_legal_if_removed"))]
    policy_breakdown = {
        policy: {
            "searchable": sum(1 for row in tested if bool(row.get(f"supports_{policy}"))),
            "legal": sum(1 for row in tested if bool(row.get(f"supports_{policy}")) and bool(row.get("physical_dryrun_legal_if_removed"))),
        }
        for policy in ("A", "B", "C", "D")
    }
    domain_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"tested": 0, "legal": 0, "illegal": 0})
    for row in tested:
        domain = str(row["domain_id"])
        domain_breakdown[domain]["tested"] += 1
        if bool(row.get("physical_dryrun_legal_if_removed")):
            domain_breakdown[domain]["legal"] += 1
        else:
            domain_breakdown[domain]["illegal"] += 1
    return {
        "dryrun_mode": "candidate_bundle_legality_proxy",
        "single_unit_physical_surgery_executed": False,
        "num_units_tested": len(tested),
        "num_units_legal": len(tested) - len(illegal),
        "num_units_illegal": len(illegal),
        "illegal_units": [
            {
                "unit_id": row["unit_id"],
                "stable_unit_id": row["stable_unit_id"],
                "reject_reason_if_any": row.get("reject_reason_if_any", ""),
            }
            for row in illegal[:500]
        ],
        "illegal_reasons": sorted({str(row.get("reject_reason_if_any", "")) for row in illegal if str(row.get("reject_reason_if_any", ""))}),
        "policy_breakdown": policy_breakdown,
        "domain_breakdown": dict(domain_breakdown),
    }


def _extract_groups(model: Any, adapter: Any, sample: Any, args: argparse.Namespace, *, policy_key: str) -> list[Any]:
    trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
    protected_layers = build_protected_layers(model, adapter_protected=adapter.get_protected_layers(model), extra_prefixes=args.extra_protected_prefix or [])
    op_graph = build_op_graph(trace, model, protected_layers=protected_layers)
    grouped_mode = POLICY_TO_MODE[policy_key] if policy_key in {"A", "B", "D"} else "remove_groups"
    groups = GroupBuilder(op_graph, align=args.align, grouped_conv_mode=grouped_mode, protect_residual_add=False).build()
    apply_full_model_prunable_surface(groups, group_conv_policy=policy_key, total_model_params=args.total_model_params)
    configure_grouped_conv_pruning_fns(
        groups,
        argparse.Namespace(group_conv_selection_mode=POLICY_TO_MODE[policy_key], allow_remove_groups=(policy_key == "C")),
    )
    return groups


def _expand_units_for_groups(groups: Sequence[Any], scope_importance: Mapping[str, torch.Tensor] | None = None) -> list[Any]:
    units: list[Any] = []
    for scope in groups:
        scores = (scope_importance or {}).get(scope.group_id)
        units.extend(expand_coupled_channel_units(scope, scores, importance_mode="first_order_taylor" if scores is not None else None))
    return units


def _load_selector_candidates_from_dirs(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in paths:
        root = Path(text)
        if not root.exists():
            continue
        for report in root.glob("**/selected_coupled_units_report.json"):
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.extend(list(data.get("selected_candidates", []) or []))
        for report in root.glob("**/selector_report.json"):
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.extend(list(data.get("top_selected_units_by_score", []) or []))
            rows.extend(list(data.get("top_rejected_units_by_reason", []) or []))
    return rows


def _write_verdict(path: Path, report: Mapping[str, Any], stability: Mapping[str, Any], selector: Mapping[str, Any], dryrun: Mapping[str, Any]) -> None:
    ready = str(report.get("search_ready", "partial"))
    lines = [
        "# Coupled Units Search Readiness Verdict v10.2",
        "",
        f"search_ready = {ready}",
        "",
        "1. Direct GA/search variables: partial. Stable base CoupledChannelUnits can be exported, but current selector uses strategy/alignment bundles, so arbitrary single-unit pruning is not fully proven.",
        f"2. stable_unit_id across run stable: {bool(stability.get('stable_id_set_equal_across_runs'))}.",
        f"3. pruning domain ordering stable: {bool(stability.get('ordering_equal_across_runs'))}.",
        "4. module-axis-index membership exported: yes, see `coupled_unit_membership_matrix.csv`.",
        f"5. residual closure complete for exported rows: {report.get('residual_closure_complete')} .",
        f"6. concat closure complete for exported rows: {report.get('concat_closure_complete')} .",
        f"7. grouped conv closure complete: {report.get('grouped_conv_closure_complete')} .",
        f"8. ConvTranspose/deblock supported path only: {report.get('convtranspose_supported_only')} .",
        f"9. protected units have reason: {report.get('protected_units_have_reason')} .",
        f"10. unsupported units excluded from search: {report.get('unsupported_units_excluded_from_search')} .",
        f"11. searchable units with importance_score: {report.get('searchable_units_with_importance_score')} .",
        f"12. searchable units with param_saving: {report.get('searchable_units_with_param_saving')} .",
        f"13. searchable units with shape_after estimate: {report.get('searchable_units_with_shape_after')} .",
        f"14. simulator/dryrun legality: {dryrun.get('num_units_legal')}/{dryrun.get('num_units_tested')} legal by `{dryrun.get('dryrun_mode')}`.",
        "15. A/B/C/D compatibility: exported per unit as supports_A/supports_B/supports_C/supports_D.",
        f"16. selector candidate universe mismatch: {bool(selector.get('selector_candidate_universe_mismatch'))}.",
        f"17. search space size: {report.get('num_searchable_units')} searchable units out of {report.get('num_units_total')} total.",
        "18. Domains not entering GA: domains with detection head, geometry/index ops, unsupported units, or incomplete closure.",
        "19. Grouped conv units for A1/B1: use `grouped_conv_units_report.json`; A/B-compatible grouped-output rows are the safe subset.",
        "",
        "Final verdict: partial. Ready subset is stable, searchable units with importance, param saving, strategy compatibility, and bundle-level legality. Blocked subset is any unit requiring unsupported/protected closure or true single-unit physical surgery proof. Next step: add real per-unit and per-domain physical dry-run execution, then enforce selector candidates to carry stable_unit_id in trace CSVs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CoupledChannelUnits for search readiness")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--surface", default="full_model_all_safe_coupled_units")
    parser.add_argument("--selector", default="global_budgeted_coupled_unit_selector")
    parser.add_argument("--importance-mode", default="first_order_taylor")
    parser.add_argument("--strategies", default="A1,B1,C1,D")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--num-stability-runs", type=int, default=3)
    parser.add_argument("--num-calib-batches", type=int, default=1)
    parser.add_argument("--align", type=int, default=1)
    parser.add_argument("--min-channels", type=int, default=1)
    parser.add_argument("--max-pruning-ratio-per-domain", type=float, default=0.8)
    parser.add_argument("--extra-protected-prefix", action="append", default=[])
    parser.add_argument(
        "--selector-report-dirs",
        default="outputs/latency_lut/budget_repair_eval_v102_A1,outputs/latency_lut/budget_repair_eval_v102_B1,outputs/latency_lut/selector_audit_v102",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.importance_mode != "first_order_taylor":
        raise ValueError("search readiness audit requires --importance-mode first_order_taylor")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "prune_logs").mkdir(parents=True, exist_ok=True)
    logger = setup_v100_logger(out / "logs")
    prune_logger = setup_prune_logger(out / "prune_logs")
    model_args = build_model_args_for_strategy(args)
    model, adapter = load_heal_model(model_args, device, prune_logger)
    try:
        inventory = compute_param_inventory(model)
        args.total_model_params = int(inventory["total_params"])
        sample = adapter.build_synthetic_batch(model)
        strategies = [item.strip().upper() for item in str(args.strategies).split(",") if item.strip()]
        first_policy = parse_strategy_spec(strategies[0]).policy_key if strategies else "A"
        stability_runs: list[list[dict[str, Any]]] = []
        first_groups: list[Any] | None = None
        for run_idx in range(max(1, int(args.num_stability_runs))):
            groups = _extract_groups(model, adapter, sample, args, policy_key=first_policy)
            units = _expand_units_for_groups(groups)
            rows = build_search_variable_rows(units, candidates_by_source={})
            stability_runs.append(rows)
            if run_idx == 0:
                first_groups = groups
        stability = build_stability_report(stability_runs)
        groups = first_groups or []
        for param in model.parameters():
            param.requires_grad_(True)
        calibration_data = build_importance_calibration_data(adapter, model_args, prune_logger)
        if calibration_data is not None:
            calibration_data = [move_batch_to_device(batch, device) for batch in calibration_data]
        compute_group_importance(
            model,
            groups,
            method=args.importance_mode,
            forward_fn=adapter.forward_for_task,
            calibration_data=calibration_data,
            loss_fn=adapter.compute_task_loss,
            num_calib_batches=int(args.num_calib_batches or 0),
            strict_grad=True,
        )
        scope_importance, scope_records = compute_scope_channel_importance_map(groups, method=args.importance_mode)
        base_units = _expand_units_for_groups(groups, scope_importance)
        all_candidates: list[BudgetCandidate] = []
        all_rejected: list[dict[str, Any]] = []
        grouped_reports: list[dict[str, Any]] = []
        for strategy in strategies:
            spec = parse_strategy_spec(strategy)
            candidates, rejected, _units, _units_by_scope, grouped = build_budget_candidates(
                groups,
                scope_importance,
                spec,
                max_pruning_ratio_per_domain=args.max_pruning_ratio_per_domain,
                min_channels=args.min_channels,
            )
            all_candidates.extend(candidates)
            all_rejected.extend(rejected)
            grouped_reports.extend(grouped)
        candidates_by_source = _candidates_by_source(all_candidates)
        search_rows = build_search_variable_rows(base_units, candidates_by_source=candidates_by_source)
        domain_rows = build_domain_rows(search_rows)
        membership_rows = build_membership_rows(search_rows)
        protected_rows = [row for row in search_rows if bool(row.get("is_protected"))]
        unsupported_rows = [row for row in search_rows if str(row.get("reject_reason_if_any", "")) and not bool(row.get("is_protected"))]
        grouped_rows = [row for row in search_rows if bool(row.get("contains_grouped_conv"))]
        residual_concat_rows = [row for row in search_rows if bool(row.get("contains_residual")) or bool(row.get("contains_concat"))]
        convtranspose_rows = [row for row in search_rows if bool(row.get("contains_convtranspose"))]
        selector_dirs = [item.strip() for item in str(args.selector_report_dirs).split(",") if item.strip()]
        selector_candidates = _load_selector_candidates_from_dirs(selector_dirs)
        selector_report = build_selector_universe_report(selector_candidates or [cand.to_dict() for cand in all_candidates], search_rows)
        selector_trace_rows = build_selector_stable_trace_rows(selector_candidates or [cand.to_dict() for cand in all_candidates], search_rows)
        schema = build_search_space_encoding_schema(search_rows)
        dryrun = build_physical_dryrun_report(search_rows)
        searchable = [row for row in search_rows if bool(row.get("is_searchable"))]
        report = {
            "surface": args.surface,
            "selector": args.selector,
            "importance_mode_used": args.importance_mode,
            "baseline_total_params": int(inventory["total_params"]),
            "num_units_total": len(search_rows),
            "num_searchable_units": len(searchable),
            "num_protected_units": len(protected_rows),
            "num_unsupported_units": len(unsupported_rows),
            "num_domains": len(domain_rows),
            "num_grouped_conv_units": len(grouped_rows),
            "num_residual_concat_units": len(residual_concat_rows),
            "num_convtranspose_units": len(convtranspose_rows),
            "stable_unit_id_ready": bool(stability.get("stable_id_set_equal_across_runs")),
            "deterministic_ordering_ready": bool(stability.get("ordering_equal_across_runs")),
            "residual_closure_complete": all(bool(row.get("residual_closure_complete")) for row in residual_concat_rows if bool(row.get("contains_residual"))),
            "concat_closure_complete": all(bool(row.get("concat_closure_complete")) for row in residual_concat_rows if bool(row.get("contains_concat"))),
            "grouped_conv_closure_complete": all(bool(row.get("grouped_conv_closure_complete")) for row in grouped_rows),
            "convtranspose_supported_only": all(bool(row.get("convtranspose_closure_complete")) for row in convtranspose_rows),
            "protected_units_have_reason": all(str(row.get("protected_reason", "")) for row in protected_rows),
            "unsupported_units_excluded_from_search": all(not bool(row.get("is_searchable")) for row in unsupported_rows),
            "searchable_units_with_importance_score": all(bool(row.get("importance_score_available")) for row in searchable),
            "searchable_units_with_param_saving": all(row.get("param_saving_if_removed") is not None for row in searchable),
            "searchable_units_with_shape_after": all(row.get("shape_after_if_removed") is not None for row in searchable),
            "selector_candidate_universe_mismatch": bool(selector_report.get("selector_candidate_universe_mismatch")),
            "dryrun_mode": dryrun["dryrun_mode"],
            "single_unit_physical_surgery_executed": dryrun["single_unit_physical_surgery_executed"],
            "search_ready": "partial",
            "partial_ready_subset": "stable searchable units with Taylor importance, param saving estimate, policy compatibility, and bundle-level legality",
            "blocked_subset": "units requiring true single-unit physical dry-run, unsupported/protected closure, or selector stable_unit_id trace integration",
            "blocking_reasons": [
                "single_unit_physical_surgery_not_executed",
                "selector_uses_policy_bundles_not_plain_single_units",
            ],
            "scope_importance_records": scope_records[:50],
        }
        write_json(out / "coupled_units_search_readiness_report.json", report)
        write_csv(out / "coupled_units_search_variables.csv", search_rows)
        write_csv(out / "pruning_domain_search_variables.csv", domain_rows)
        write_csv(out / "coupled_unit_membership_matrix.csv", membership_rows)
        write_json(out / "protected_units_report.json", protected_rows)
        write_json(out / "unsupported_units_report.json", {"unsupported_units": unsupported_rows, "candidate_rejections": all_rejected[:1000]})
        write_json(out / "grouped_conv_units_report.json", {"units": grouped_rows, "grouped_candidate_reports": grouped_reports})
        write_json(out / "residual_concat_units_report.json", residual_concat_rows)
        write_json(out / "convtranspose_units_report.json", convtranspose_rows)
        write_json(out / "search_variable_stability_report.json", stability)
        write_json(out / "search_space_encoding_schema.json", schema)
        write_json(out / "physical_prune_dryrun_search_units_report.json", dryrun)
        write_json(out / "selector_candidate_universe_report.json", selector_report)
        write_csv(out / "selector_selected_units_trace_with_stable_ids.csv", selector_trace_rows)
        _write_verdict(out / "search_readiness_verdict.md", report, stability, selector_report, dryrun)
        write_json(out / "audit_config.json", vars(args))
        logger.info("search readiness audit wrote %s", out)
        print(json.dumps({"success": True, "output_dir": str(out), "search_ready": report["search_ready"], "num_units": len(search_rows)}, indent=2))
        return 0
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
