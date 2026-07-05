from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.audit_grouped_conv_keep_distribution_v84 import summarize


def _load_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_comparison(tp_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]], tp_error: dict[str, Any] | None = None, current_error: dict[str, Any] | None = None) -> dict[str, Any]:
    tp_summary = summarize(tp_rows)
    current_summary = summarize(current_rows)
    tp_imbalance = any(not r.get("original_group_keep_count_equal_out", False) for r in tp_rows)
    current_balanced = bool(current_rows) and all(r.get("original_group_keep_count_equal_out", False) for r in current_rows)
    current_map_ok = bool(current_rows) and all(r.get("group_keep_map_present") and r.get("group_keep_map_matches_actual") for r in current_rows)
    current_safer = current_summary["deployment_friendly_grouped_conv_pass"] and not tp_summary["deployment_friendly_grouped_conv_pass"]
    same_behavior = (
        tp_summary["deployment_friendly_grouped_conv_pass"] == current_summary["deployment_friendly_grouped_conv_pass"]
        and tp_imbalance != current_balanced
    )
    return {
        "experiment": {
            "prune_ratio": 0.5,
            "importance": "l2_norm",
            "ranking_scope": "local",
            "protected_policy": "only_pfn_encoder_to_pointpillarscatter_boundary",
        },
        "tp_native": {
            "ran": bool(tp_rows) or not (tp_error or {}).get("not_run", False),
            "physical_prune_success": bool((tp_error or {}).get("physical_prune_success", False)),
            "forward_or_export_success": bool((tp_error or {}).get("forward_or_export_success", False)),
            **tp_summary,
            "main_failure_reasons": (tp_error or {}).get("failure_reasons", []),
        },
        "current_pruner": {
            "ran": bool(current_rows) or not (current_error or {}).get("not_run", False),
            "physical_prune_success": bool((current_error or {}).get("physical_prune_success", False)),
            "forward_or_export_success": bool((current_error or {}).get("forward_or_export_success", False)),
            **current_summary,
            "group_keep_map_required_and_present": current_map_ok,
            "main_failure_reasons": (current_error or {}).get("failure_reasons", []),
        },
        "direct_comparison": {
            "tp_native_allows_original_group_imbalance": tp_imbalance,
            "current_pruner_enforces_original_group_balance": current_balanced,
            "tp_native_equivalent_to_current_pruner_for_grouped_conv": bool(same_behavior),
            "current_pruner_extra_ops_not_in_tp": (current_error or {}).get("extra_ops_not_in_tp", []),
            "tp_native_ops_missing_in_current": (current_error or {}).get("tp_ops_missing_in_current", []),
            "current_pruner_safer_for_trt_grouped_conv": bool(current_safer),
        },
        "conclusion": (
            "current_pruner_strict_grouped_conv_is_more_deployment_friendly"
            if current_safer
            else "grouped_conv_policy_not_resolved"
        ),
        "next_fix": "use_tp_as_oracle_keep_current_policy_for_deployment_checks",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-audit-json", required=True)
    parser.add_argument("--current-audit-json", required=True)
    parser.add_argument("--tp-error-json", default="")
    parser.add_argument("--current-error-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    tp_data = _load_json(args.tp_audit_json, {})
    current_data = _load_json(args.current_audit_json, {})
    tp_rows = tp_data.get("records", tp_data if isinstance(tp_data, list) else [])
    current_rows = current_data.get("records", current_data if isinstance(current_data, list) else [])
    report = summarize_comparison(
        list(tp_rows or []),
        list(current_rows or []),
        _load_json(args.tp_error_json, {}) if args.tp_error_json else {},
        _load_json(args.current_error_json, {}) if args.current_error_json else {},
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# TP Native vs Current Pruner v8.4\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(report["direct_comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
