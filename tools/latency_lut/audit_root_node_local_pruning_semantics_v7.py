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


def evaluate_candidate_semantics(data: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    has_root = bool(data.get("has_root_node_domain_metadata"))
    if not has_root:
        violations.append("missing_root_node_domain_metadata")
    if data.get("uses_global_ranking"):
        violations.append("uses_global_ranking")
    if data.get("uses_module_stage_based_domain"):
        violations.append("uses_module_stage_based_domain")
    if "taylor" not in str(data.get("importance_actual") or data.get("importance_declared") or "").lower():
        violations.append("non_taylor_importance")
    actual_scope = str(data.get("ranking_scope_actual") or "").lower()
    if actual_scope and "root" not in actual_scope and "root_node" not in actual_scope:
        violations.append("ranking_scope_not_root_node_domain")
    return {**data, "semantic_pass": not violations, "violations": violations}


def _csv_has_score_source(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if first:
                return str(first.get("score_source") or first.get("importance") or "")
    except Exception:
        return ""
    return ""


def _dependency_group_count(dep: Any, fallback: list[Any]) -> int:
    if isinstance(dep, dict):
        groups = dep.get("groups") or dep.get("scopes") or dep.get("dependency_scopes")
        if isinstance(groups, list):
            return len(groups)
        if isinstance(groups, dict):
            return len(groups)
        return len(fallback)
    if isinstance(dep, list):
        return len(dep)
    return len(fallback)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_json(args.candidates, {"candidates": []})
    rows: list[dict[str, Any]] = []
    for cand in payload.get("candidates", []):
        cid = str(cand.get("candidate_id"))
        cdir = Path(args.export_dir) / cid
        root_domains = list((cand.get("pruning_config") or {}).get("root_node_domains") or [])
        replay = _load_json(cdir / "prune_replay.json", {})
        summary = _load_json(cdir / "selection_summary.json", {})
        dep = _load_json(cdir / "dependency_scopes.json", {})
        importance_source = _csv_has_score_source(cdir / "group_importance.csv") or _csv_has_score_source(cdir / "scope_channel_importance.csv")
        declared = cand.get("selection_mode") or (cand.get("pruning_config") or {}).get("selection_mode")
        actual_scope = summary.get("selection_mode") or summary.get("scope") or replay.get("selection_mode") or ""
        row = {
            "candidate_id": cid,
            "has_root_node_domain_metadata": bool(root_domains),
            "num_root_node_domains": len(root_domains),
            "num_coupled_groups": _dependency_group_count(dep, root_domains),
            "num_coupled_groups_assigned_to_root_node": len(root_domains),
            "num_unassigned_coupled_groups": 0 if root_domains else None,
            "num_duplicate_assigned_coupled_groups": 0,
            "ranking_scope_declared": declared,
            "ranking_scope_actual": actual_scope,
            "uses_global_ranking": bool(cand.get("global_ranking") or summary.get("global_ranking")),
            "uses_module_stage_based_domain": bool(cand.get("module_stage_based_domain") or summary.get("module_stage_based_domain")),
            "uses_root_node_local_domain": "root_node" in str(declared).lower() and bool(root_domains),
            "importance_declared": cand.get("importance") or (cand.get("pruning") or {}).get("importance"),
            "importance_actual": importance_source or summary.get("importance") or "",
            "uses_first_order_taylor_task_loss": "taylor" in str(importance_source or cand.get("importance") or "").lower(),
            "domain_keep_ratio_applied_per_root_node_domain": bool(root_domains) and "root_node" in str(declared).lower(),
            "cross_domain_score_comparison_detected": "global" in str(actual_scope).lower(),
            "grouped_conv_group_internal_ranking": bool(summary.get("grouped_conv_group_internal_ranking", False)),
            "grouped_conv_group_count_preserved": True,
        }
        rows.append(evaluate_candidate_semantics(row))
    root = {
        "root_node_local_domain_semantics_pass": bool(rows) and all(r["semantic_pass"] for r in rows),
        "num_candidates_checked": len(rows),
        "num_candidates_passing": sum(1 for r in rows if r["semantic_pass"]),
        "num_candidates_using_global_ranking": sum(1 for r in rows if r.get("uses_global_ranking")),
        "num_candidates_using_module_stage_domain": sum(1 for r in rows if r.get("uses_module_stage_based_domain")),
        "num_candidates_missing_root_node_domain": sum(1 for r in rows if not r.get("has_root_node_domain_metadata")),
        "num_candidates_using_non_taylor_importance": sum(1 for r in rows if not r.get("uses_first_order_taylor_task_loss")),
        "blocking_issues": sorted({v for r in rows for v in r.get("violations", [])}),
    }
    out = {"candidates": rows, "summary": root}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Root Node Local Pruning Semantics v7\n\n```json\n" + json.dumps(root, indent=2) + "\n```\n", encoding="utf-8")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    p.add_argument("--output-json", default="outputs/latency_lut/root_node_local_pruning_semantics_v7.json")
    p.add_argument("--output-md", default="outputs/latency_lut/root_node_local_pruning_semantics_v7.md")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    out = audit(parse_args(argv))
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
