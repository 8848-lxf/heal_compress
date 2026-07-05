from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def _int_list(values: Iterable[Any] | None) -> list[int]:
    return sorted({int(v) for v in (values or [])})


def _counts_by_original_group(indices: Iterable[int], *, channels_before: int, groups_before: int) -> dict[str, int]:
    if groups_before <= 0 or channels_before <= 0 or channels_before % groups_before != 0:
        return {}
    per = channels_before // groups_before
    counts = {str(g): 0 for g in range(groups_before)}
    for idx in _int_list(indices):
        if 0 <= idx < channels_before:
            counts[str(idx // per)] += 1
    return counts


def _equal_nonempty(counts: dict[str, int]) -> bool:
    return bool(counts) and len(set(counts.values())) == 1


def _aligned_counts(counts: dict[str, int], align: int) -> bool:
    if not counts:
        return False
    return all(v > 0 and (align <= 1 or v % align == 0) for v in counts.values())


def _group_keep_map_counts(group_keep_map: dict[str, Any] | None) -> dict[str, int]:
    return {str(k): len(v or []) for k, v in (group_keep_map or {}).items()}


def audit_grouped_conv_distribution(
    *,
    pipeline: str,
    module: str,
    groups_before: int,
    groups_after: int,
    c_in_before: int,
    c_out_before: int,
    c_in_after: int,
    c_out_after: int,
    kept_out_indices: Iterable[int] | None,
    kept_in_indices: Iterable[int] | None,
    group_keep_map: dict[str, Any] | None,
    require_group_keep_map: bool = False,
    align: int = 8,
) -> dict[str, Any]:
    kept_out = _int_list(kept_out_indices)
    kept_in = _int_list(kept_in_indices)
    out_counts = _counts_by_original_group(kept_out, channels_before=c_out_before, groups_before=groups_before)
    in_counts = _counts_by_original_group(kept_in, channels_before=c_in_before, groups_before=groups_before)
    out_pruned = {k: (c_out_before // groups_before) - v for k, v in out_counts.items()} if groups_before else {}
    in_pruned = {k: (c_in_before // groups_before) - v for k, v in in_counts.items()} if groups_before else {}
    map_counts = _group_keep_map_counts(group_keep_map)

    groups_preserved = int(groups_before) == int(groups_after)
    has_expansion = int(c_in_after) > int(c_in_before) or int(c_out_after) > int(c_out_before)
    final_divisible = (
        groups_after > 0
        and c_in_after % groups_after == 0
        and c_out_after % groups_after == 0
    )
    equal_out = _equal_nonempty(out_counts)
    equal_in = _equal_nonempty(in_counts)
    align_out = _aligned_counts(out_counts, align)
    align_in = _aligned_counts(in_counts, align)
    map_matches = bool(map_counts) and (not out_counts or map_counts == out_counts)

    violations: list[str] = []
    if not groups_preserved:
        violations.append("groups_changed")
    if has_expansion:
        violations.append("channel_expansion")
    if not final_divisible:
        violations.append("final_shape_not_group_divisible")
    if not equal_out:
        violations.append("original_group_keep_count_unequal_out")
    if kept_in and not equal_in:
        violations.append("original_group_keep_count_unequal_in")
    if not align_out:
        violations.append("per_group_keep_count_not_align8_out")
    if kept_in and not align_in:
        violations.append("per_group_keep_count_not_align8_in")
    if require_group_keep_map and not group_keep_map:
        violations.append("missing_group_keep_map")
    if group_keep_map and not map_matches:
        violations.append("group_keep_map_mismatch")

    valid = not violations
    return {
        "pipeline": pipeline,
        "module": module,
        "groups_before": int(groups_before),
        "groups_after": int(groups_after),
        "C_in_before": int(c_in_before),
        "C_out_before": int(c_out_before),
        "C_in_after": int(c_in_after),
        "C_out_after": int(c_out_after),
        "in_per_group_before": int(c_in_before // groups_before) if groups_before else 0,
        "out_per_group_before": int(c_out_before // groups_before) if groups_before else 0,
        "in_per_group_after": int(c_in_after // groups_after) if groups_after and c_in_after % groups_after == 0 else 0,
        "out_per_group_after": int(c_out_after // groups_after) if groups_after and c_out_after % groups_after == 0 else 0,
        "original_group_keep_counts_out": out_counts,
        "original_group_prune_counts_out": out_pruned,
        "original_group_keep_counts_in": in_counts,
        "original_group_prune_counts_in": in_pruned,
        "original_group_keep_count_equal_out": equal_out,
        "original_group_keep_count_equal_in": equal_in if kept_in else False,
        "per_group_keep_count_align8_out": align_out,
        "per_group_keep_count_align8_in": align_in if kept_in else False,
        "final_shape_group_divisible": final_divisible,
        "groups_preserved": groups_preserved,
        "group_keep_map_present": bool(group_keep_map),
        "group_keep_map_matches_actual": map_matches,
        "has_channel_expansion": has_expansion,
        "changed_groups_count": not groups_preserved,
        "valid_for_deployment_friendly_grouped_conv": valid,
        "violations": violations,
    }


def _infer_record_from_report(pipeline: str, record: dict[str, Any], require_group_keep_map: bool) -> dict[str, Any]:
    groups = int(record.get("groups_before") or record.get("groups") or 0)
    per = int(record.get("per_group_before") or record.get("channels_per_group_before") or 0)
    kept = record.get("expanded_keep_indices") or record.get("kept_out_indices") or []
    if not kept and record.get("group_keep_map"):
        kept = [
            int(g) * per + int(local)
            for g, locals_ in record.get("group_keep_map", {}).items()
            for local in (locals_ or [])
        ]
    c_before = int(record.get("C_out_before") or record.get("c_out_before") or groups * per)
    c_after = int(record.get("C_out_after") or record.get("c_out_after") or len(kept))
    return audit_grouped_conv_distribution(
        pipeline=pipeline,
        module=str(record.get("module") or record.get("module_name") or ""),
        groups_before=groups,
        groups_after=int(record.get("groups_after") or groups),
        c_in_before=int(record.get("C_in_before") or record.get("c_in_before") or c_before),
        c_out_before=c_before,
        c_in_after=int(record.get("C_in_after") or record.get("c_in_after") or c_after),
        c_out_after=c_after,
        kept_out_indices=kept,
        kept_in_indices=record.get("kept_in_indices") or kept,
        group_keep_map=record.get("group_keep_map") or None,
        require_group_keep_map=require_group_keep_map,
    )


def audit_records(pipeline: str, records: list[dict[str, Any]], *, require_group_keep_map: bool = False) -> list[dict[str, Any]]:
    return [_infer_record_from_report(pipeline, r, require_group_keep_map) for r in records]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_grouped_convs_checked": len(rows),
        "num_grouped_convs_with_unequal_original_group_keep": sum(
            1 for r in rows if not r.get("original_group_keep_count_equal_out", False)
        ),
        "num_grouped_convs_with_non_align8_per_group_keep": sum(
            1 for r in rows if not r.get("per_group_keep_count_align8_out", False)
        ),
        "num_grouped_convs_groups_changed": sum(1 for r in rows if r.get("changed_groups_count")),
        "num_grouped_convs_channel_expansion": sum(1 for r in rows if r.get("has_channel_expansion")),
        "final_shape_group_divisible_pass": bool(rows) and all(r.get("final_shape_group_divisible") for r in rows),
        "deployment_friendly_grouped_conv_pass": bool(rows) and all(
            r.get("valid_for_deployment_friendly_grouped_conv") for r in rows
        ),
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def audit_file(input_json: Path, output_json: Path, output_md: Path, pipeline: str, require_group_keep_map: bool) -> dict[str, Any]:
    data = _load_json(input_json, [])
    records = data.get("records", data) if isinstance(data, dict) else data
    rows = audit_records(pipeline, list(records or []), require_group_keep_map=require_group_keep_map)
    out = {"records": rows, "summary": summarize(rows)}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text("# Grouped Conv Keep Distribution v8.4\n\n```json\n" + json.dumps(out["summary"], indent=2) + "\n```\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--pipeline", choices=["tp_native", "current_pruner"], required=True)
    parser.add_argument("--require-group-keep-map", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    out = audit_file(Path(args.input_json), Path(args.output_json), Path(args.output_md), args.pipeline, args.require_group_keep_map)
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
