from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.fixed_width_boundary_registry_v84 import default_fixed_width_boundaries


def audit_policy_records(
    *,
    grouped_conv_records: list[dict[str, Any]],
    protection_records: list[dict[str, Any]],
    normalization_records: list[dict[str, Any]],
    fixed_width_boundaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hidden = [
        r for r in protection_records
        if r.get("protected") and not r.get("protected_reason")
    ]
    protected_reasons: dict[str, int] = {}
    for r in protection_records:
        if r.get("protected"):
            reason = str(r.get("protected_reason") or "hidden_protection")
            protected_reasons[reason] = protected_reasons.get(reason, 0) + 1
    channel_expansion = any(r.get("has_channel_expansion") or "channel_expansion" in r.get("violations", []) for r in grouped_conv_records)
    channel_expansion = channel_expansion or any(r.get("candidate_has_channel_expansion") for r in normalization_records)
    pre_norm = any(r.get("pre_prune_normalization_detected") for r in normalization_records)
    return {
        "num_scopes": len(protection_records),
        "num_protected_scopes": sum(1 for r in protection_records if r.get("protected")),
        "protected_reasons": protected_reasons,
        "hidden_protection_detected": bool(hidden),
        "fixed_width_boundaries": fixed_width_boundaries or default_fixed_width_boundaries(),
        "grouped_conv_feasible_sets": [r.get("feasible_set") for r in grouped_conv_records if r.get("feasible_set")],
        "channel_expansion_detected": bool(channel_expansion),
        "pre_prune_normalization_detected": bool(pre_norm),
        "all_ratio_deviation_explained": bool(not hidden and not channel_expansion and not pre_norm),
    }


def _load(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped-conv-json", default="")
    parser.add_argument("--protection-json", default="")
    parser.add_argument("--normalization-json", default="")
    parser.add_argument("--output-json", default="outputs/latency_lut/prune_policy_v84.json")
    parser.add_argument("--output-md", default="outputs/latency_lut/prune_policy_v84.md")
    args = parser.parse_args(argv)
    result = audit_policy_records(
        grouped_conv_records=_load(args.grouped_conv_json, []) if args.grouped_conv_json else [],
        protection_records=_load(args.protection_json, []) if args.protection_json else [],
        normalization_records=_load(args.normalization_json, []) if args.normalization_json else [],
        fixed_width_boundaries=default_fixed_width_boundaries(),
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Prune Policy v8.4\n\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
