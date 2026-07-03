from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _csv_score_source(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
            if row:
                return str(row.get("score_source") or row.get("importance_mode") or row.get("importance") or "")
    except Exception:
        return ""
    return ""


def _unit_members(unit: dict[str, Any]) -> list[dict[str, Any]]:
    members = unit.get("members")
    if isinstance(members, list):
        return [m for m in members if isinstance(m, dict)]
    local = unit.get("local_indices_by_item")
    converted = []
    if isinstance(local, dict):
        for key, values in local.items():
            module, _, direction = str(key).partition(":")
            axis = "out_channels" if direction == "out" else "in_channels"
            for idx in values or []:
                converted.append({"module": module, "axis": axis, "index": idx})
    return converted


def _has_indices(units: list[dict[str, Any]]) -> bool:
    if not units:
        return False
    for unit in units:
        if unit.get("root_channel_index", unit.get("root_idx")) is None:
            return False
        members = _unit_members(unit)
        if not members:
            return False
        if any(member.get("index") is None for member in members):
            return False
    return True


def _has_cross_layer_members(units: list[dict[str, Any]]) -> bool:
    return any(len({m.get("module") or m.get("node") for m in _unit_members(unit)}) > 1 for unit in units)


def _residual_members_grouped(units: list[dict[str, Any]]) -> bool:
    residual_units = [
        unit for unit in units
        if "residual_add" in unit.get("dependency_types", []) or (unit.get("constraints") or {}).get("has_residual")
    ]
    if not residual_units:
        return False
    return all(len(_unit_members(unit)) >= 2 for unit in residual_units)


def _concat_offset_mapping(units: list[dict[str, Any]]) -> bool:
    concat_units = [
        unit for unit in units
        if any("concat" in str(dep) for dep in unit.get("dependency_types", []))
        or (unit.get("constraints") or {}).get("has_concat")
    ]
    if not concat_units:
        return False
    return all(any(member.get("concat_offset") is not None for member in _unit_members(unit)) for unit in concat_units)


def _domains(cdir: Path) -> list[dict[str, Any]]:
    direct = _load_json(cdir / "root_node_local_domains.json", {})
    if isinstance(direct, dict) and isinstance(direct.get("domains"), list):
        return direct["domains"]
    dep = _load_json(cdir / "dependency_scopes.json", [])
    if isinstance(dep, list):
        return [
            {
                "domain_id": row.get("scope_id"),
                "root_node": row.get("root_module") or row.get("scope_id"),
                "unit_ids": [],
                "ranking_scope": "local_scope",
            }
            for row in dep
        ]
    return []


def _grouped_conv_report(cdir: Path) -> dict[str, Any]:
    rows = _load_json(cdir / "grouped_conv_selection_report.json", [])
    if not isinstance(rows, list):
        rows = []
    violations = []
    for row in rows:
        if not row.get("structure_legal", True):
            violations.append({"scope_id": row.get("scope_id"), "reason": "structure_legal_false"})
        if not row.get("per_group_kept_count_align8", True):
            violations.append({"scope_id": row.get("scope_id"), "reason": "per_group_keep_count_align_violation"})
        if row.get("groups_after") != row.get("groups_before"):
            violations.append({"scope_id": row.get("scope_id"), "reason": "groups_changed"})
    return {
        "num_grouped_conv_units": len(rows),
        "mode": rows[0].get("group_conv_selection_mode", "none") if rows else "none",
        "groups_preserved": not any(v.get("reason") == "groups_changed" for v in violations),
        "per_group_keep_count_equal": not any(v.get("reason") == "per_group_keep_count_align_violation" for v in violations),
        "group_keep_map_present": any(bool(row.get("group_keep_map")) for row in rows),
        "violations": violations,
    }


def audit_candidate_artifact(candidate_dir: str | Path) -> dict[str, Any]:
    cdir = Path(candidate_dir)
    units = _load_json(cdir / "coupled_channel_units.json", [])
    if isinstance(units, dict):
        units = units.get("units", [])
    if not isinstance(units, list):
        units = []
    domains = _domains(cdir)
    selection = _load_json(cdir / "domain_selection_summary.json", {})
    if not selection:
        selection = _load_json(cdir / "selection_summary.json", {})
    structure = _load_json(cdir / "structure_audit.json", {})
    cand = _load_json(cdir / "candidate.json", {})
    importance_actual = _csv_score_source(cdir / "unit_importance.csv") or _csv_score_source(cdir / "group_importance.csv") or str(selection.get("importance_mode") or selection.get("importance") or "")
    grouped = _grouped_conv_report(cdir)

    has_units = bool(units)
    have_indices = _has_indices(units)
    domains_by_root = bool(domains) and all(str(d.get("domain_id", "")).startswith("root_node::") or d.get("ranking_scope") == "root_node_local" for d in domains)
    unit_ids = {str(u.get("unit_id")) for u in units}
    assigned = [uid for d in domains for uid in (d.get("unit_ids") or [])]
    assigned_set = set(str(uid) for uid in assigned)
    exact_once = bool(unit_ids) and assigned_set == unit_ids and len(assigned) == len(assigned_set)
    ranking_actual = str(selection.get("selection_mode") or selection.get("ranking_scope") or "")
    ranking_declared = str(cand.get("selection_mode") or (cand.get("pruning_config") or {}).get("selection_mode") or "")
    uses_root = ranking_actual == "root_node_local_unit_ratio" or all(d.get("ranking_scope") == "root_node_local" for d in domains if d)
    uses_global = bool(selection.get("global_ranking")) or "global" in ranking_actual
    uses_stage = bool(selection.get("module_stage_based_domain")) or "stage" in ranking_actual or "module" in ranking_actual
    domain_records = selection.get("domains") if isinstance(selection.get("domains"), list) else []

    row = {
        "candidate_id": cdir.name,
        "has_coupled_channel_units": has_units,
        "num_coupled_channel_units": len(units),
        "coupled_channel_units_have_indices": have_indices,
        "coupled_channel_units_have_cross_layer_members": _has_cross_layer_members(units),
        "residual_members_grouped": _residual_members_grouped(units),
        "concat_members_have_offset_mapping": _concat_offset_mapping(units),
        "has_root_node_local_domains": bool(domains),
        "num_root_node_local_domains": len(domains),
        "domains_are_keyed_by_root_node": domains_by_root,
        "all_units_assigned_to_exactly_one_domain": exact_once,
        "ranking_scope_declared": ranking_declared,
        "ranking_scope_actual": ranking_actual,
        "uses_root_node_local_ranking": uses_root,
        "uses_global_ranking": uses_global,
        "uses_module_stage_domain": uses_stage,
        "cross_domain_score_comparison_detected": uses_global,
        "importance_declared": str(cand.get("importance") or (cand.get("pruning") or {}).get("importance") or ""),
        "importance_actual": importance_actual,
        "uses_first_order_taylor": "taylor" in importance_actual.lower() or "taylor" in str(cand.get("importance", "")).lower(),
        "domain_keep_ratio_applied": bool(domain_records),
        "domain_keep_ratio_records": domain_records,
        "grouped_conv_handling": grouped,
        "structure_audit": {
            "is_width_changed_subnet": structure.get("is_width_changed_subnet"),
            "num_changed_conv_layers": structure.get("num_changed_conv_layers"),
            "param_keep_ratio": structure.get("param_keep_ratio"),
        },
    }
    violations = []
    if not row["has_coupled_channel_units"]:
        violations.append("missing_coupled_channel_units")
    if not row["coupled_channel_units_have_indices"]:
        violations.append("coupled_channel_units_missing_indices")
    if not row["coupled_channel_units_have_cross_layer_members"]:
        violations.append("coupled_channel_units_missing_cross_layer_members")
    if not row["has_root_node_local_domains"]:
        violations.append("missing_root_node_local_domains")
    if not row["domains_are_keyed_by_root_node"]:
        violations.append("domains_not_keyed_by_root_node")
    if not row["all_units_assigned_to_exactly_one_domain"]:
        violations.append("units_not_assigned_to_exactly_one_domain")
    if not row["uses_root_node_local_ranking"]:
        violations.append("not_using_root_node_local_ranking")
    if row["uses_global_ranking"]:
        violations.append("uses_global_ranking")
    if row["uses_module_stage_domain"]:
        violations.append("uses_module_stage_domain")
    if not row["uses_first_order_taylor"]:
        violations.append("missing_first_order_taylor")
    if not row["domain_keep_ratio_applied"]:
        violations.append("domain_keep_ratio_not_recorded")
    if grouped["violations"]:
        violations.append("grouped_conv_violations")
    row["semantic_pass"] = not violations
    row["violations"] = violations
    return row


def audit_export_dir(export_dir: str | Path) -> dict[str, Any]:
    root = Path(export_dir)
    rows = [audit_candidate_artifact(path) for path in sorted(root.iterdir()) if path.is_dir() and (path / "candidate.json").is_file()]
    summary = {
        "num_candidates_checked": len(rows),
        "num_candidates_semantic_pass": sum(1 for r in rows if r["semantic_pass"]),
        "num_candidates_missing_coupled_channel_units": sum(1 for r in rows if not r["has_coupled_channel_units"]),
        "num_candidates_units_missing_indices": sum(1 for r in rows if not r["coupled_channel_units_have_indices"]),
        "num_candidates_missing_root_node_domains": sum(1 for r in rows if not r["has_root_node_local_domains"]),
        "num_candidates_not_using_root_node_local_ranking": sum(1 for r in rows if not r["uses_root_node_local_ranking"]),
        "num_candidates_using_global_ranking": sum(1 for r in rows if r["uses_global_ranking"]),
        "num_candidates_using_module_stage_domain": sum(1 for r in rows if r["uses_module_stage_domain"]),
        "num_candidates_missing_first_order_taylor": sum(1 for r in rows if not r["uses_first_order_taylor"]),
        "num_candidates_grouped_conv_violations": sum(1 for r in rows if r["grouped_conv_handling"]["violations"]),
        "root_node_local_pruner_semantics_pass": bool(rows) and all(r["semantic_pass"] for r in rows),
        "blocking_issues": sorted({v for r in rows for v in r["violations"]}),
    }
    return {"candidates": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--output-json", default="outputs/latency_lut/root_node_local_pruner_audit_v8.json")
    parser.add_argument("--output-md", default="outputs/latency_lut/root_node_local_pruner_audit_v8.md")
    args = parser.parse_args(argv)
    out = audit_export_dir(args.export_dir)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Root-node Local Pruner Audit v8\n\n```json\n" + json.dumps(out["summary"], indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
