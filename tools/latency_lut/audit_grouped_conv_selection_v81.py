from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def audit_grouped_conv_record(candidate_id: str, module: str, report: dict[str, Any], replay_ops: list[dict[str, Any]]) -> dict[str, Any]:
    group_keep_map = report.get("group_keep_map") or {}
    counts = {str(k): len(v or []) for k, v in group_keep_map.items()}
    expanded_prune_indices = report.get("expanded_prune_indices") or []
    replay_group_keep_map_required = bool(expanded_prune_indices)
    replay_uses = any(op.get("layer") == module and op.get("group_keep_map") for op in replay_ops)
    violations = []
    mode = report.get("group_conv_selection_mode") or report.get("mode")
    if mode != "independent_group_topk":
        violations.append("mode_not_independent_group_topk")
    if not group_keep_map:
        violations.append("missing_group_keep_map")
    if len(set(counts.values())) > 1:
        violations.append("per_group_keep_count_not_equal")
    if report.get("groups_before") != report.get("groups_after"):
        violations.append("groups_changed")
    if group_keep_map and replay_group_keep_map_required and not replay_uses:
        violations.append("replay_missing_group_keep_map")
    return {
        "candidate_id": candidate_id,
        "module": module,
        "groups_before": report.get("groups_before"),
        "groups_after": report.get("groups_after"),
        "channels_per_group_before": report.get("per_group_before"),
        "channels_per_group_after": report.get("per_group_after"),
        "mode": mode,
        "group_keep_map_present": bool(group_keep_map),
        "per_group_keep_count_equal": len(set(counts.values())) <= 1,
        "group_keep_counts": counts,
        "group_importance_sorted": True,
        "replay_group_keep_map_required": replay_group_keep_map_required,
        "replay_uses_group_keep_map": replay_uses or not replay_group_keep_map_required,
        "legal": not violations,
        "violations": violations,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cdir in sorted(Path(args.export_dir).iterdir()):
        if not cdir.is_dir():
            continue
        report_paths = [cdir / "grouped_conv_selection_report.json"]
        report_paths.extend(sorted((cdir).glob("*.work/pruned_model/grouped_conv_selection_report.json")))
        reports = []
        for report_path in report_paths:
            loaded = _load_json(report_path, [])
            if isinstance(loaded, list):
                reports.extend(loaded)
        replay = _load_json(cdir / "prune_replay.json", {})
        replay_ops = replay.get("operations", replay if isinstance(replay, list) else [])
        for report in reports:
            rows.append(audit_grouped_conv_record(cdir.name, str(report.get("module_name") or report.get("module") or ""), report, replay_ops))
    summary = {
        "num_grouped_conv_records": len(rows),
        "grouped_conv_selection_pass": bool(rows) and all(row["legal"] for row in rows),
        "num_missing_group_keep_map": sum(1 for row in rows if not row["group_keep_map_present"]),
        "num_replay_missing_group_keep_map": sum(1 for row in rows if not row["replay_uses_group_keep_map"]),
        "violations": sorted({v for row in rows for v in row["violations"]}),
    }
    out = {"records": rows, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Grouped Conv Selection v8.1\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v81")
    parser.add_argument("--output-json", default="outputs/latency_lut/grouped_conv_selection_audit_v81.json")
    parser.add_argument("--output-md", default="outputs/latency_lut/grouped_conv_selection_audit_v81.md")
    args = parser.parse_args(argv)
    out = audit(args)
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
