from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _op_key(op: dict[str, Any]) -> tuple[str, str, str, tuple[int, ...]]:
    module = str(op.get("module") or op.get("layer") or "")
    axis = str(op.get("axis") or op.get("direction") or "")
    name = str(op.get("op") or axis)
    idxs = op.get("idxs")
    if idxs is None:
        idxs = op.get("indices") or op.get("prune_indices") or []
    return (module, name, axis, tuple(int(i) for i in idxs))


def compare_ops_to_tp_oracle(
    *,
    candidate_id: str,
    domain_id: str,
    root_node: str,
    root_module: str,
    root_type: str,
    requested_keep_ratio: float,
    requested_prune_ratio: float,
    num_units: int,
    pruned_root_indices: list[int],
    tp_ops: list[dict[str, Any]],
    current_ops: list[dict[str, Any]],
    tp_group_build_success: bool,
    tp_check_pruning_group_pass: bool,
) -> dict[str, Any]:
    tp_keys = {_op_key(op) for op in tp_ops}
    cur_keys = {_op_key(op) for op in current_ops}
    missing = [op for op in tp_ops if _op_key(op) not in cur_keys]
    extra = [op for op in current_ops if _op_key(op) not in tp_keys]
    idx_mismatch = []
    axis_mismatch = []
    tp_by_module = {}
    for op in tp_ops:
        tp_by_module.setdefault(str(op.get("module") or op.get("layer") or ""), []).append(op)
    for op in current_ops:
        module = str(op.get("module") or op.get("layer") or "")
        for ref in tp_by_module.get(module, []):
            if str(ref.get("axis")) != str(op.get("axis") or op.get("direction")):
                axis_mismatch.append({"tp": ref, "current": op})
            elif tuple(ref.get("idxs") or []) != tuple(op.get("idxs") or op.get("indices") or op.get("prune_indices") or []):
                idx_mismatch.append({"tp": ref, "current": op})
    ok = bool(tp_group_build_success and tp_check_pruning_group_pass and not missing and not extra and not idx_mismatch and not axis_mismatch)
    reason = ""
    if extra:
        reason = "extra_current_ops"
    if missing:
        reason = "missing_current_ops" if not reason else reason + "+missing_current_ops"
    if idx_mismatch:
        reason = "idx_mismatch" if not reason else reason + "+idx_mismatch"
    if axis_mismatch:
        reason = "axis_mismatch" if not reason else reason + "+axis_mismatch"
    if not tp_group_build_success:
        reason = "tp_group_build_failed" if not reason else reason + "+tp_group_build_failed"
    return {
        "candidate_id": candidate_id,
        "domain_id": domain_id,
        "root_node": root_node,
        "root_module": root_module,
        "root_type": root_type,
        "requested_keep_ratio": float(requested_keep_ratio),
        "requested_prune_ratio": float(requested_prune_ratio),
        "num_units": int(num_units),
        "num_pruned_units": len(pruned_root_indices),
        "pruned_root_indices": pruned_root_indices,
        "tp_group_build_success": bool(tp_group_build_success),
        "tp_check_pruning_group_pass": bool(tp_check_pruning_group_pass),
        "tp_group_ops": tp_ops,
        "current_replay_ops": current_ops,
        "missing_in_current_vs_tp": missing,
        "extra_in_current_vs_tp": extra,
        "idx_mismatch_vs_tp": idx_mismatch,
        "axis_mismatch_vs_tp": axis_mismatch,
        "current_matches_tp_oracle": ok,
        "mismatch_reason": reason,
    }


def _parse_idx(unit_id: str) -> int | None:
    m = re.search(r"(?:idx|root_ch_)(\d+)$", unit_id)
    return int(m.group(1)) if m else None


def _current_ops_for_domain(root_module: str, replay_ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ops = []
    for op in replay_ops:
        layer = str(op.get("layer") or "")
        if op.get("axis") == "grouped_merge" or (root_module and (layer == root_module or layer.startswith(root_module.rsplit(".", 1)[0]))):
            ops.append({"module": layer, "op": op.get("axis") or op.get("direction"), "axis": op.get("axis") or op.get("direction"), "idxs": op.get("indices") or op.get("prune_indices") or []})
    return ops


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for cdir in sorted(Path(args.export_dir).iterdir()):
        if not cdir.is_dir():
            continue
        selection = _load_json(cdir / "domain_selection_summary.json", {})
        replay = _load_json(cdir / "prune_replay.json", {})
        replay_ops = replay.get("operations", replay if isinstance(replay, list) else [])
        for dom in selection.get("domains", []):
            pruned = [_parse_idx(uid) for uid in dom.get("pruned_unit_ids", [])]
            pruned = [i for i in pruned if i is not None]
            if not pruned:
                continue
            root_module = str(dom.get("root_module") or str(dom.get("root_node", "")).replace("group::", ""))
            current = _current_ops_for_domain(root_module, replay_ops)
            # This audit intentionally does not claim equivalence without a live
            # TP DepGraph operation listing; it flags extra current replay ops,
            # especially grouped_merge normalization, as outside TP oracle.
            tp_ops = [{"module": root_module, "op": "tp_oracle_unresolved", "axis": "out", "idxs": pruned}]
            rows.append(
                compare_ops_to_tp_oracle(
                    candidate_id=cdir.name,
                    domain_id=str(dom.get("domain_id", "")),
                    root_node=str(dom.get("root_node", "")),
                    root_module=root_module,
                    root_type="Conv2d" if "conv" in root_module.lower() else ("Linear" if "linear" in root_module.lower() else "unknown"),
                    requested_keep_ratio=float(dom.get("requested_keep_ratio") or 1.0),
                    requested_prune_ratio=float(dom.get("requested_prune_ratio") or 0.0),
                    num_units=int(dom.get("num_units") or 0),
                    pruned_root_indices=pruned,
                    tp_ops=tp_ops,
                    current_ops=current,
                    tp_group_build_success=False,
                    tp_check_pruning_group_pass=False,
                )
            )
    summary = {
        "num_candidates": len([p for p in Path(args.export_dir).iterdir() if p.is_dir()]),
        "num_domains_checked": len(rows),
        "num_domains_tp_group_build_success": sum(1 for r in rows if r["tp_group_build_success"]),
        "num_domains_tp_check_pass": sum(1 for r in rows if r["tp_check_pruning_group_pass"]),
        "num_domains_current_matches_tp": sum(1 for r in rows if r["current_matches_tp_oracle"]),
        "num_domains_with_extra_current_ops": sum(1 for r in rows if r["extra_in_current_vs_tp"]),
        "num_domains_with_missing_current_ops": sum(1 for r in rows if r["missing_in_current_vs_tp"]),
        "num_domains_with_idx_mismatch": sum(1 for r in rows if r["idx_mismatch_vs_tp"]),
        "current_pruner_tp_equivalence_pass": bool(rows) and all(r["current_matches_tp_oracle"] for r in rows),
        "blocking_issues": ["tp_oracle_equivalence_not_proven", "pre_prune_group_alignment_normalization_outside_tp_oracle"],
    }
    out = {"domains": rows, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Current Pruner vs TP Oracle v8.3\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v81")
    p.add_argument("--output-json", default="outputs/latency_lut/current_pruner_vs_tp_oracle_v83.json")
    p.add_argument("--output-md", default="outputs/latency_lut/current_pruner_vs_tp_oracle_v83.md")
    args = p.parse_args(argv)
    print(json.dumps(audit(args)["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
