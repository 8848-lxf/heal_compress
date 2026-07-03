from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


FLOAT_OPS = {"Conv", "Gemm", "MatMul"}
MERGE_OPS = {"Add", "Sum", "Concat", "Sub", "Mul", "Div", "Where", "GridSample"}


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _region(name: str) -> str:
    low = str(name).lower()
    if any(k in low for k in ("cls_head", "reg_head", "dir_head", "head", "single_head")):
        return "detection_head"
    if any(k in low for k in ("scatter", "warp", "grid", "geometry", "plugin")):
        return "geometry_or_plugin_or_scatter"
    if any(k in low for k in ("shrink", "neck", "downsample", "compression")):
        return "shrink_or_neck"
    if any(k in low for k in ("fusion", "pyramid", "fuse")):
        return "pyramid_fusion"
    if any(k in low for k in ("pillar", "pfn", "voxel", "encoder_m1")):
        return "pillar_or_voxel_encoder"
    if any(k in low for k in ("backbone", "resnet", "layer0", "layer1", "layer2", "layer3")):
        return "backbone"
    return "other"


def _producer_map(model: Any) -> dict[str, Any]:
    return {out: node for node in model.graph.node for out in node.output}


def _consumer_map(model: Any) -> dict[str, list[Any]]:
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return consumers


def _upstream_unit(tensor: str, producers: dict[str, Any], depth: int = 0) -> str | None:
    if depth > 20:
        return None
    node = producers.get(tensor)
    if node is None:
        return None
    if node.op_type in FLOAT_OPS:
        return node.name or node.output[0]
    for inp in node.input:
        got = _upstream_unit(inp, producers, depth + 1)
        if got:
            return got
    return None


def _downstream_units(tensor: str, consumers: dict[str, list[Any]], depth: int = 0) -> list[str]:
    if depth > 8:
        return []
    out: list[str] = []
    for node in consumers.get(tensor, []):
        if node.op_type in FLOAT_OPS:
            out.append(node.name or node.output[0])
        else:
            for output in node.output:
                out.extend(_downstream_units(output, consumers, depth + 1))
    return out


def build_precision_constraint_components(onnx_path: str | Path) -> list[dict[str, Any]]:
    model = onnx.load(str(onnx_path))
    producers = _producer_map(model)
    consumers = _consumer_map(model)
    components: list[dict[str, Any]] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type not in MERGE_OPS:
            continue
        producer_units = sorted({u for inp in node.input for u in [_upstream_unit(inp, producers)] if u})
        consumer_units = sorted({u for out in node.output for u in _downstream_units(out, consumers)})
        if len(producer_units) + len(consumer_units) < 2:
            continue
        ctype = "concat" if node.op_type == "Concat" else ("residual_add" if node.op_type in {"Add", "Sum"} else "merge")
        units = producer_units + consumer_units
        components.append(
            {
                "component_id": f"{Path(onnx_path).stem}:{idx}:{node.name or node.op_type}",
                "component_type": ctype,
                "onnx_nodes": [node.name or node.output[0]],
                "producer_units": producer_units,
                "consumer_units": consumer_units,
                "regions": sorted({_region(u) for u in units}),
                "must_share_precision": True,
                "allowed_precision_set": ["FP16", "FP32", "INT8_QDQ"],
                "reason": f"{node.op_type} inputs/consumers form a precision constraint component",
            }
        )
    return components


def _profile_precision(unit: str, profile: dict[str, Any]) -> str:
    default = str(profile.get("default", "FP16")).upper()
    default = "INT8_QDQ" if default in {"INT8", "TRT_INT8_QDQ"} else default
    best = default
    best_len = -1
    nunit = _norm(unit)
    for key, value in dict(profile.get("overrides") or {}).items():
        nkey = _norm(key)
        if nkey and (nkey in nunit or nunit in nkey) and len(nkey) > best_len:
            val = str(value).upper()
            best = "INT8_QDQ" if val in {"INT8", "TRT_INT8_QDQ"} else val
            best_len = len(nkey)
    return best


def _candidate_from_profile_id(profile: dict[str, Any]) -> str:
    return str(profile.get("candidate_id") or "")


def _profile_status(profile: dict[str, Any], result_dir: Path, failures: list[dict[str, Any]]) -> tuple[str, str]:
    cid = f"{profile.get('candidate_id')}__{profile.get('precision_profile_id')}"
    result = _load_json(result_dir / f"{cid}.result.json", {})
    if result.get("success"):
        return "success", ""
    for row in failures:
        if row.get("candidate_id") == cid:
            return str(row.get("failed_stage") or row.get("status") or "failed"), str(row.get("status") or "")
    return str(result.get("failed_stage") or result.get("status") or "unknown"), str(result.get("status") or "")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    profiles = _load_json(args.profiles, {"profiles": []}).get("profiles", [])
    failures = _load_json(args.failures, [])
    export_root = Path(args.export_dir)
    result_dir = Path(args.results_dir)
    by_candidate_components: dict[str, list[dict[str, Any]]] = {}
    all_components: list[dict[str, Any]] = []
    for onnx_path in sorted(export_root.glob("*/width_changed.onnx")):
        cid = onnx_path.parent.name
        comps = build_precision_constraint_components(onnx_path)
        by_candidate_components[cid] = comps
        all_components.extend(comps)
    profile_rows: list[dict[str, Any]] = []
    failed_with_violation = 0
    success_with_violation = 0
    violated_region = Counter()
    for profile in profiles:
        cid = _candidate_from_profile_id(profile)
        comps = by_candidate_components.get(cid, [])
        cfg = dict(profile.get("precision_profile") or {})
        status, status_name = _profile_status(profile, result_dir, failures)
        violated: list[dict[str, Any]] = []
        for comp in comps:
            units = comp.get("producer_units", []) + comp.get("consumer_units", [])
            precisions = {u: _profile_precision(u, cfg) for u in units}
            if len(set(precisions.values())) > 1:
                for region in comp.get("regions") or []:
                    violated_region[region] += 1
                violated.append(
                    {
                        "component_id": comp["component_id"],
                        "component_type": comp["component_type"],
                        "units": units,
                        "requested_precisions": precisions,
                        "observed_precisions": {},
                        "failure_stage": status,
                        "likely_related_to_engine_build_failure": status not in {"success", ""},
                    }
                )
        if violated and status != "success":
            failed_with_violation += 1
        if violated and status == "success":
            success_with_violation += 1
        profile_rows.append(
            {
                "candidate_id": cid,
                "precision_profile_id": profile.get("precision_profile_id"),
                "violated_components": violated,
                "num_residual_precision_violations": sum(1 for v in violated if v["component_type"] == "residual_add"),
                "num_concat_precision_violations": sum(1 for v in violated if v["component_type"] == "concat"),
                "num_merge_precision_violations": sum(1 for v in violated if v["component_type"] == "merge"),
                "status": status,
                "status_name": status_name,
            }
        )
    main_cause = failed_with_violation > 0 and failed_with_violation > success_with_violation
    summary = {
        "components": all_components,
        "profiles": profile_rows,
        "summary": {
            "residual_concat_precision_constraint_is_main_failure_cause": main_cause,
            "evidence": f"failed_with_violation={failed_with_violation}, success_with_violation={success_with_violation}",
            "num_failed_profiles_with_constraint_violation": failed_with_violation,
            "num_success_profiles_with_constraint_violation": success_with_violation,
            "dominant_violated_region": violated_region.most_common(1)[0][0] if violated_region else "",
            "recommended_fix": "sample precision by residual/concat precision components" if main_cause else "constraint violations are not proven as dominant cause",
        },
    }
    _write_json(args.output_json, summary)
    Path(args.output_md).write_text("# Precision Constraint Graph v7\n\n```json\n" + json.dumps(summary["summary"], indent=2) + "\n```\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    p.add_argument("--profiles", default="outputs/latency_lut/pruned_precision_profiles_v6.json")
    p.add_argument("--results-dir", default="outputs/latency_lut/pruned_route2_results_v6")
    p.add_argument("--failures", default="outputs/latency_lut/pruned_route2_full_engine_failures_v6.json")
    p.add_argument("--output-json", default="outputs/latency_lut/precision_constraint_graph_v7.json")
    p.add_argument("--output-md", default="outputs/latency_lut/precision_constraint_graph_v7.md")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = audit(parse_args(argv))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
