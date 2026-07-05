from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning.grouped_conv_policy import audit_relaxed_group_total_align8
from tools.latency_lut.audit_grouped_conv_keep_distribution_v84 import audit_records


def audit_variant_records(variant: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        strict = audit_records(variant, [record], require_group_keep_map=variant == "current_strict_50_l2")[0]
        relaxed = audit_relaxed_group_total_align8(
            module=str(record.get("module") or record.get("module_name") or ""),
            groups_before=int(record.get("groups_before") or record.get("groups") or 1),
            groups_after=int(record.get("groups_after") or record.get("groups") or 1),
            c_in_before=int(record.get("C_in_before") or record.get("c_in_before") or record.get("C_out_before") or 0),
            c_out_before=int(record.get("C_out_before") or record.get("c_out_before") or 0),
            kept_out_indices=record.get("kept_out_indices") or record.get("expanded_keep_indices") or [],
            kept_in_indices=record.get("kept_in_indices") or record.get("expanded_keep_indices") or [],
        )
        rows.append(
            {
                "variant": variant,
                "module": strict["module"],
                "groups_before": strict["groups_before"],
                "groups_after": strict["groups_after"],
                "C_out_before": strict["C_out_before"],
                "C_out_after": strict["C_out_after"],
                "C_in_before": strict["C_in_before"],
                "C_in_after": strict["C_in_after"],
                "final_shape_group_divisible": strict["final_shape_group_divisible"],
                "total_C_out_align8": relaxed["total_C_out_align8"],
                "total_C_in_align8": relaxed["total_C_in_align8"],
                "original_group_keep_counts_out": strict["original_group_keep_counts_out"],
                "original_group_keep_count_equal_out": strict["original_group_keep_count_equal_out"],
                "original_group_keep_counts_in": strict["original_group_keep_counts_in"],
                "original_group_keep_count_equal_in": strict["original_group_keep_count_equal_in"],
                "per_group_keep_count_align8_out": strict["per_group_keep_count_align8_out"],
                "per_group_keep_count_align8_in": strict["per_group_keep_count_align8_in"],
                "has_unequal_original_group_keep_counts": not strict["original_group_keep_count_equal_out"],
                "has_channel_expansion": strict["has_channel_expansion"],
                "groups_changed": strict["changed_groups_count"],
                "matches_tp_native_pattern": not strict["original_group_keep_count_equal_out"] and relaxed["valid_relaxed_group_total_align8"],
                "valid_for_strict_policy": strict["valid_for_deployment_friendly_grouped_conv"],
                "valid_for_relaxed_total_align8_policy": relaxed["valid_relaxed_group_total_align8"],
                "violations": sorted(set(strict["violations"] + relaxed["violations"])),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8")) if Path(args.input_json).is_file() else []
    records = data.get("records", data) if isinstance(data, dict) else data
    rows = audit_variant_records(args.variant, list(records or []))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variant": args.variant, "records": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
