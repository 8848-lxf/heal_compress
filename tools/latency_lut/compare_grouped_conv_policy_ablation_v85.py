from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VARIANTS = [
    "baseline",
    "tp_native_50_l2",
    "current_strict_50_l2",
    "current_relaxed_total_align8_50_l2",
]


def _speedup(baseline: float | None, value: float | None) -> float | None:
    if not baseline or not value or value <= 0:
        return None
    return baseline / value


def build_policy_ablation_summary(*, audits: dict[str, list[dict[str, Any]]], latency: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = latency.get("baseline", {}).get("latency_p50_ms")
    latency_out = {}
    for variant in VARIANTS:
        row = dict(latency.get(variant, {}))
        row.setdefault("engine_build_success", False)
        row.setdefault("latency_p50_ms", None)
        row["speedup_vs_baseline"] = _speedup(base, row.get("latency_p50_ms")) if variant != "baseline" else None
        latency_out[variant] = row
    tp_rows = audits.get("tp_native_50_l2", [])
    relaxed_rows = audits.get("current_relaxed_total_align8_50_l2", [])
    strict_rows = audits.get("current_strict_50_l2", [])
    relaxed_matches_tp = bool(relaxed_rows) and any(r.get("matches_tp_native_pattern") for r in relaxed_rows)
    return {
        "tp_native_original_group_imbalance": any(r.get("has_unequal_original_group_keep_counts") for r in tp_rows),
        "strict_policy_pass": bool(strict_rows) and all(r.get("valid_for_strict_policy") for r in strict_rows),
        "relaxed_policy_pass": bool(relaxed_rows) and all(r.get("valid_for_relaxed_total_align8_policy") for r in relaxed_rows),
        "relaxed_total_align8_matches_tp_native_pattern": relaxed_matches_tp,
        "latency_smoke": latency_out,
        "recommended_grouped_conv_policy": "strict_independent_group_topk_until_relaxed_trt_speedup_is_verified",
        "safe_to_use_for_GA": False,
    }


def _load(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", default="outputs/latency_lut/grouped_conv_policy_ablation_v85/reports/grouped_conv_keep_distribution_v85.json")
    parser.add_argument("--latency-json", default="outputs/latency_lut/grouped_conv_policy_ablation_v85/reports/latency_smoke_v85.json")
    parser.add_argument("--output-json", default="outputs/latency_lut/grouped_conv_policy_ablation_v85/reports/grouped_conv_policy_ablation_v85.json")
    parser.add_argument("--output-md", default="outputs/latency_lut/grouped_conv_policy_ablation_v85/reports/grouped_conv_policy_ablation_v85.md")
    args = parser.parse_args(argv)
    summary = build_policy_ablation_summary(audits=_load(args.audit_json, {}), latency=_load(args.latency_json, {}))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Grouped Conv Policy Ablation v8.5\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
