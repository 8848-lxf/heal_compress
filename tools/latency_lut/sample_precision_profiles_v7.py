from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _unit_name(name: str) -> str:
    text = str(name).strip()
    if text.startswith("/"):
        text = text[1:]
    return text.replace("/", ".")


def _precision_name(name: Any) -> str:
    text = str(name).upper()
    if text in {"INT8", "TRT_INT8_QDQ"}:
        return "INT8_QDQ"
    if text in {"TRT_FP16"}:
        return "FP16"
    if text in {"TRT_FP32"}:
        return "FP32"
    return text


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _build_precision_groups(components: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    uf = UnionFind()
    normalized_components: list[dict[str, Any]] = []
    for comp in components:
        units = sorted({_unit_name(u) for u in (comp.get("producer_units") or []) + (comp.get("consumer_units") or []) if u})
        if len(units) < 2:
            continue
        head = units[0]
        for unit in units[1:]:
            uf.union(head, unit)
        normalized_components.append({**comp, "units": units})

    groups: dict[str, list[str]] = defaultdict(list)
    for unit in list(uf.parent):
        groups[uf.find(unit)].append(unit)
    groups = {gid: sorted(set(units)) for gid, units in groups.items()}

    comps_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comp in normalized_components:
        gid = uf.find(comp["units"][0])
        comps_by_group[gid].append(comp)
    return groups, comps_by_group


def _expand_profile(profile: dict[str, Any], groups: dict[str, list[str]], comps_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    raw_config = profile.get("precision_profile") or {}
    default_precision = _precision_name(raw_config.get("default", "FP16"))
    overrides = {_unit_name(k): _precision_name(v) for k, v in (raw_config.get("overrides") or {}).items()}

    unit_to_group = {unit: gid for gid, units in groups.items() for unit in units}
    group_assignment: dict[str, str] = {}
    loose_overrides: dict[str, str] = {}

    for unit, precision in overrides.items():
        gid = unit_to_group.get(unit)
        if not gid:
            loose_overrides[unit] = precision
            continue
        existing = group_assignment.get(gid)
        if existing and existing != precision:
            return None
        group_assignment[gid] = precision

    expanded_overrides = dict(loose_overrides)
    precision_groups = []
    for gid, precision in sorted(group_assignment.items()):
        units = groups[gid]
        for unit in units:
            expanded_overrides[unit] = precision
        comps = comps_by_group.get(gid, [])
        component_type = ",".join(sorted({str(c.get("component_type")) for c in comps}))
        regions = sorted({r for c in comps for r in (c.get("regions") or [])})
        int8_allowed = precision != "INT8_QDQ" or not any("plugin" in r or "geometry" in r for r in regions)
        precision_groups.append(
            {
                "group_id": gid,
                "component_type": component_type,
                "units": units,
                "region": ",".join(regions) if regions else "other",
                "assigned_precision": precision,
                "int8_allowed": int8_allowed,
                "int8_block_reason": "" if int8_allowed else "component_contains_non_int8_region",
            }
        )
        if not int8_allowed:
            return None

    violations = _find_violations(expanded_overrides, default_precision, groups, comps_by_group)
    if violations:
        return None

    precision_values = set(expanded_overrides.values()) | {default_precision}
    return {
        **profile,
        "precision_profile_id": str(profile.get("precision_profile_id", "profile")).replace("v6", "v7_constraint"),
        "precision_profile": {"default": default_precision, "overrides": dict(sorted(expanded_overrides.items()))},
        "precision_groups": precision_groups,
        "constraint_violations": [],
        "has_fp32": "FP32" in precision_values,
        "has_fp16": "FP16" in precision_values,
        "has_int8_qdq": "INT8_QDQ" in precision_values,
        "num_fp32_units": sum(1 for v in expanded_overrides.values() if v == "FP32"),
        "num_fp16_units": sum(1 for v in expanded_overrides.values() if v == "FP16"),
        "num_int8_qdq_units": sum(1 for v in expanded_overrides.values() if v == "INT8_QDQ"),
    }


def _find_violations(overrides: dict[str, str], default_precision: str, groups: dict[str, list[str]], comps_by_group: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for gid, units in groups.items():
        requested = {unit: overrides.get(unit, default_precision) for unit in units}
        if len(set(requested.values())) > 1:
            violations.append(
                {
                    "group_id": gid,
                    "component_type": ",".join(sorted({str(c.get("component_type")) for c in comps_by_group.get(gid, [])})),
                    "units": units,
                    "requested_precisions": requested,
                }
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/latency_lut/pruned_precision_profiles_v6.json")
    parser.add_argument("--constraint-graph", default="outputs/latency_lut/precision_constraint_graph_v7.json")
    parser.add_argument("--output", default="outputs/latency_lut/pruned_precision_profiles_v7.json")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_precision_profiles_v7_report.md")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.input).read_text(encoding="utf-8")) if Path(args.input).is_file() else {"profiles": []}
    graph = json.loads(Path(args.constraint_graph).read_text(encoding="utf-8")) if Path(args.constraint_graph).is_file() else {"components": []}
    groups, comps_by_group = _build_precision_groups(graph.get("components", []))

    profiles = []
    for profile in data.get("profiles", []):
        expanded = _expand_profile(profile, groups, comps_by_group)
        if expanded:
            profiles.append(expanded)

    # De-duplicate profile ids if multiple v6 patterns expand to the same group profile.
    seen: set[str] = set()
    deduped = []
    for idx, profile in enumerate(profiles):
        key = json.dumps(profile.get("precision_profile"), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        profile["precision_profile_id"] = f"v7_constraint_{idx:03d}"
        deduped.append(profile)

    payload = {
        "schema_version": 7,
        "profiles": deduped,
        "num_profiles": len(deduped),
        "num_with_int8_qdq": sum(1 for p in deduped if p.get("has_int8_qdq")),
        "num_with_fp32_fp16_int8_qdq": sum(1 for p in deduped if p.get("has_fp32") and p.get("has_fp16") and p.get("has_int8_qdq")),
        "constraint_violations": sum(len(p.get("constraint_violations") or []) for p in deduped),
        "precision_group_count": len(groups),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "profiles"}
    Path(args.report).write_text("# Pruned Precision Profiles v7\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
