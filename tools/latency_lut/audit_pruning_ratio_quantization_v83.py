from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def feasible_grouped_keep_counts(*, channels_per_group: int, group_conv_align: int) -> list[int]:
    per = int(channels_per_group)
    align = max(1, int(group_conv_align))
    if per <= 0:
        return []
    counts = [per]
    v = per - (per % align if per % align else align)
    while v >= align:
        counts.append(v)
        v -= align
    return sorted(set(counts), reverse=True)


def _ceil_align(value: int, align: int, upper: int) -> int:
    if align <= 1 or value <= 0:
        return min(upper, max(0, value))
    return min(upper, int(math.ceil(value / align) * align))


def quantize_domain_keep(
    *,
    candidate_id: str,
    domain_id: str,
    root_node: str,
    num_units: int,
    target_keep_ratio: float,
    align: int,
    is_grouped_conv_domain: bool,
    groups: int = 0,
    channels_per_group: int = 0,
    group_conv_align: int = 8,
    actual_keep_units: int | None = None,
) -> dict[str, Any]:
    raw = float(num_units) * float(target_keep_ratio)
    rounded = int(round(raw))
    feasible_counts: list[int] = []
    feasible_ratios: list[float] = []
    if is_grouped_conv_domain and groups and channels_per_group:
        per_counts = feasible_grouped_keep_counts(channels_per_group=channels_per_group, group_conv_align=group_conv_align)
        feasible_counts = [int(groups) * c for c in per_counts]
        feasible_ratios = [c / max(1, int(num_units)) for c in feasible_counts]
        aligned = next((c for c in sorted(feasible_counts) if c >= rounded), int(num_units))
    else:
        aligned = _ceil_align(rounded, int(align), int(num_units))
    actual = int(actual_keep_units if actual_keep_units is not None else aligned)
    err = actual / max(1, int(num_units)) - float(target_keep_ratio)
    reason = "none"
    if actual != aligned:
        reason = "group_conv_align" if is_grouped_conv_domain else "align8"
    elif abs(err) > 1e-9:
        reason = "group_conv_align" if is_grouped_conv_domain else "align8"
    return {
        "candidate_id": candidate_id,
        "domain_id": domain_id,
        "root_node": root_node,
        "num_units": int(num_units),
        "target_keep_ratio": float(target_keep_ratio),
        "raw_target_keep_units": raw,
        "rounded_keep_units_before_align": rounded,
        "aligned_keep_units": int(aligned),
        "actual_keep_units": actual,
        "actual_prune_units": max(0, int(num_units) - actual),
        "align": int(align),
        "is_grouped_conv_domain": bool(is_grouped_conv_domain),
        "groups": int(groups or 0),
        "channels_per_group": int(channels_per_group or 0),
        "group_conv_align": int(group_conv_align),
        "feasible_keep_counts": feasible_counts,
        "feasible_keep_ratios": feasible_ratios,
        "ratio_error": err,
        "ratio_error_reason": reason,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cdir in sorted(Path(args.export_dir).iterdir()):
        if not cdir.is_dir():
            continue
        selection = _load_json(cdir / "domain_selection_summary.json", {})
        grouped_reports = []
        for p in [cdir / "grouped_conv_selection_report.json", *sorted(cdir.glob("*.work/pruned_model/grouped_conv_selection_report.json"))]:
            loaded = _load_json(p, [])
            if isinstance(loaded, list):
                grouped_reports.extend(loaded)
        grouped_by_scope = {r.get("scope_id"): r for r in grouped_reports}
        for dom in selection.get("domains", []):
            scope_id = str(dom.get("root_node") or dom.get("domain_id", "")).replace("root_node::", "")
            report = grouped_by_scope.get(scope_id) or {}
            groups = int(report.get("groups_before") or 0)
            per = int(report.get("per_group_before") or 0)
            rows.append(
                quantize_domain_keep(
                    candidate_id=cdir.name,
                    domain_id=str(dom.get("domain_id", "")),
                    root_node=str(dom.get("root_node", "")),
                    num_units=int(dom.get("num_units") or 0),
                    target_keep_ratio=float(dom.get("requested_keep_ratio") or 1.0),
                    align=int(dom.get("align") or 8),
                    is_grouped_conv_domain=bool(report),
                    groups=groups,
                    channels_per_group=per,
                    group_conv_align=int(report.get("group_conv_align") or 8),
                    actual_keep_units=int(dom.get("actual_num_keep") or 0),
                )
            )
    large = [r for r in rows if abs(float(r["ratio_error"])) > 0.05]
    grouped_coarse = [r["domain_id"] for r in rows if r["is_grouped_conv_domain"] and len(r["feasible_keep_ratios"]) <= 2]
    unreachable: dict[str, list[str]] = {}
    for r in rows:
        if r["is_grouped_conv_domain"] and r["target_keep_ratio"] not in r["feasible_keep_ratios"]:
            unreachable.setdefault(str(r["target_keep_ratio"]), []).append(r["domain_id"])
    summary = {
        "ratio_quantization_audit_pass": not large,
        "num_domains": len(rows),
        "num_domains_with_large_ratio_error": len(large),
        "max_ratio_error": max((abs(float(r["ratio_error"])) for r in rows), default=0.0),
        "grouped_conv_domains_with_coarse_feasible_set": sorted(set(grouped_coarse)),
        "target_keep_ratios_unreachable": {k: sorted(set(v)) for k, v in unreachable.items()},
        "recommendation": "grouped conv domains with per_group=16 and align=8 only support keep ratios 1.0/0.5; use ceil alignment, lower group_conv_align, or skip grouped conv for fine-grained light pruning.",
    }
    out = {"domains": rows, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Pruning Ratio Quantization v8.3\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v81")
    p.add_argument("--output-json", default="outputs/latency_lut/pruning_ratio_quantization_v83.json")
    p.add_argument("--output-md", default="outputs/latency_lut/pruning_ratio_quantization_v83.md")
    args = p.parse_args(argv)
    print(json.dumps(audit(args)["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
