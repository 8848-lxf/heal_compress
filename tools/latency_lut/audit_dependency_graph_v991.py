#!/usr/bin/env python3
"""v9.9.1 dependency-audit taxonomy and multi-trace design artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.latency_lut.audit_dependency_graph_v98 import write_csv, write_json  # noqa: E402


V98_DEFAULT = Path("outputs/latency_lut/dependency_graph_audit_v98")
OUT_DEFAULT = Path("outputs/latency_lut/dependency_graph_audit_v991")

EVIDENCE_VALUES = {
    "proven_supported",
    "proven_unsupported",
    "coverage_insufficient",
    "resolver_missing",
    "audit_bug",
}
PRUNABILITY_VALUES = {
    "prunable_with_existing_resolver",
    "prunable_with_new_resolver",
    "not_channel_pruning_surface",
    "geometry_or_index_op_protected",
    "unknown_until_multitrace",
}


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _edge_src(edge: Any) -> str:
    if isinstance(edge, dict):
        return str(edge.get("src", ""))
    if isinstance(edge, (list, tuple)) and edge:
        return str(edge[0])
    return ""


def _edge_dst(edge: Any) -> str:
    if isinstance(edge, dict):
        return str(edge.get("dst", ""))
    if isinstance(edge, (list, tuple)) and len(edge) > 1:
        return str(edge[1])
    return ""


def _node(op_graph: dict[str, Any], name: str) -> dict[str, Any]:
    return dict((op_graph.get("nodes") or {}).get(name, {}) or {})


def _incoming(op_graph: dict[str, Any], node_name: str) -> list[str]:
    return [_edge_src(edge) for edge in op_graph.get("edges", []) if _edge_dst(edge) == node_name]


def _downstream(op_graph: dict[str, Any], node_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nodes = op_graph.get("nodes") or {}
    for edge in op_graph.get("edges", []):
        if _edge_src(edge) != node_name:
            continue
        dst = _edge_dst(edge)
        info = dict(nodes.get(dst, {}) or {})
        groups = int(info.get("groups") or 1)
        op_type = str(info.get("op_type") or info.get("raw_type") or "")
        rows.append(
            {
                "module_name": dst,
                "op_type": op_type,
                "raw_type": str(info.get("raw_type") or ""),
                "groups": groups,
                "is_plain_conv": op_type == "Conv" and groups == 1,
                "is_grouped_conv": op_type == "Conv" and groups > 1,
                "is_convtranspose": op_type == "ConvTranspose2d",
                "is_grouped_convtranspose": op_type == "ConvTranspose2d" and groups > 1,
                "is_linear": op_type == "Linear",
            }
        )
    return rows


def _shape_channel(shape: Any) -> int | None:
    if not isinstance(shape, list):
        return None
    if len(shape) >= 4:
        return int(shape[1])
    if len(shape) == 2:
        return int(shape[1])
    return None


def _is_pfn_text(text: str) -> bool:
    low = text.lower()
    return "pfn" in low or "pillar_vfe" in low or "pillar" in low


def _is_scatter_text(text: str) -> bool:
    low = text.lower()
    return "scatter" in low


def _is_geometry_text(text: str) -> bool:
    low = text.lower()
    return any(key in low for key in ("scatter", "voxel", "grid_sample", "warp", "bev", "geometry"))


def _is_regular_channelwise(row: dict[str, str], info: dict[str, Any]) -> tuple[bool, bool]:
    node_type = row.get("node_type", "")
    input_shapes = info.get("input_shapes") or []
    if node_type == "concat":
        return bool(info.get("cat_dim") == 1), bool(info.get("cat_dim") == 1 and len(input_shapes) >= 2)
    sizes = [_shape_channel(shape) for shape in input_shapes]
    sizes = [size for size in sizes if size is not None]
    channelwise = len(sizes) >= 2 and len(set(sizes)) == 1 and int(sizes[0]) > 1
    return channelwise, channelwise


def _classify_v991_row(row: dict[str, str], *, op_graph: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    node_name = row.get("node_name", "")
    info = _node(op_graph, node_name)
    downstream = _downstream(op_graph, node_name)
    incoming = _incoming(op_graph, node_name)
    scope = " ".join(str(v) for v in info.get("module_scope", []) or [])
    evidence_text = " ".join([node_name, scope, *incoming, *(d["module_name"] for d in downstream), *(d["op_type"] for d in downstream)])
    dynamic_single_trace = trace.get("dynamic_branch_enumeration_enabled") is False
    categories: set[str] = set()

    is_channel_dim, is_regular = _is_regular_channelwise(row, info)
    if row.get("node_type") == "concat" and info.get("cat_dim") != 1:
        categories.add("non_channel_dim_concat")
    elif not is_regular:
        categories.add("non_channelwise_add_or_cat")

    if row.get("node_type") == "concat" and is_channel_dim:
        if str(row.get("concat_offset_recorded", "")).lower() != "true":
            categories.add("missing_cat_offset_proof")
        if str(row.get("downstream_conv_input_offset_recorded", "")).lower() != "true":
            categories.add("missing_downstream_input_offset_proof")

    if not row.get("matched_scope_id"):
        categories.add("missing_scope_or_unit_mapping")

    if any(d["is_linear"] for d in downstream):
        categories.add("linear_feature_dim_mapping_missing")
    if _is_pfn_text(evidence_text):
        categories.add("pfn_feature_dim_resolver_missing")
    if _is_scatter_text(evidence_text):
        categories.add("scatter_geometry_index_op_protected")
    elif _is_geometry_text(evidence_text):
        categories.add("protected_geometry_or_fixed_shape_contract")

    if any(d["is_grouped_convtranspose"] for d in downstream):
        categories.add("grouped_convtranspose_resolver_missing")
        categories.add("currently_rejected_for_safety")
        categories.add("theoretically_prunable_with_group_balanced_input_output_resolver")
    elif any(d["is_convtranspose"] for d in downstream):
        categories.add("convtranspose_resolver_required")

    if downstream and not any(d["is_plain_conv"] for d in downstream):
        categories.add("downstream_consumer_not_plain_conv")

    if dynamic_single_trace:
        categories.add("coverage_insufficient_single_trace")

    if not categories:
        categories.add("audit_bug")

    evidence_status = _evidence_status(categories)
    theoretical_prunability = _theoretical_prunability(categories)
    required_resolver = _required_resolver(categories)
    current_action = _current_action(categories, evidence_status)
    assert evidence_status in EVIDENCE_VALUES
    assert theoretical_prunability in PRUNABILITY_VALUES

    return {
        "node_name": node_name,
        "node_type": row.get("node_type", ""),
        "proof_pass": row.get("proof_pass", ""),
        "num_input_branches": row.get("num_input_branches", ""),
        "input_branches": row.get("input_branches", ""),
        "matched_scope_id": row.get("matched_scope_id", ""),
        "downstream_consumers": json.dumps(downstream, ensure_ascii=False),
        "is_channel_dim_op": bool(is_channel_dim),
        "is_regular_channelwise_residual_or_concat": bool(is_regular),
        "covered_by_current_single_trace": node_name in (op_graph.get("nodes") or {}),
        "dynamic_path_limitation": dynamic_single_trace,
        "failure_category": ";".join(sorted(categories)),
        "evidence_status": evidence_status,
        "theoretical_prunability": theoretical_prunability,
        "required_resolver": required_resolver,
        "current_action": current_action,
    }


def _evidence_status(categories: set[str]) -> str:
    non_coverage = categories - {"coverage_insufficient_single_trace"}
    if not non_coverage:
        return "coverage_insufficient"
    if "audit_bug" in categories:
        return "audit_bug"
    if "non_channel_dim_concat" in categories or "non_channelwise_add_or_cat" in categories or "scatter_geometry_index_op_protected" in categories:
        return "proven_unsupported"
    if any(
        cat in categories
        for cat in (
            "linear_feature_dim_mapping_missing",
            "pfn_feature_dim_resolver_missing",
            "grouped_convtranspose_resolver_missing",
            "missing_cat_offset_proof",
            "missing_downstream_input_offset_proof",
            "convtranspose_resolver_required",
        )
    ):
        return "resolver_missing"
    if "coverage_insufficient_single_trace" in categories:
        return "coverage_insufficient"
    return "proven_unsupported"


def _theoretical_prunability(categories: set[str]) -> str:
    non_coverage = categories - {"coverage_insufficient_single_trace"}
    if not non_coverage:
        return "unknown_until_multitrace"
    if "scatter_geometry_index_op_protected" in categories or "protected_geometry_or_fixed_shape_contract" in categories:
        return "geometry_or_index_op_protected"
    if "non_channel_dim_concat" in categories or "non_channelwise_add_or_cat" in categories:
        return "not_channel_pruning_surface"
    if "audit_bug" in categories:
        return "prunable_with_existing_resolver"
    return "prunable_with_new_resolver"


def _required_resolver(categories: set[str]) -> str:
    resolvers = []
    if "linear_feature_dim_mapping_missing" in categories:
        resolvers.append("channel_to_linear_feature_dim_mapping")
    if "pfn_feature_dim_resolver_missing" in categories:
        resolvers.append("pfn_feature_dim_resolver")
    if "missing_cat_offset_proof" in categories or "missing_downstream_input_offset_proof" in categories:
        resolvers.append("concat_offset_and_downstream_input_resolver")
    if "grouped_convtranspose_resolver_missing" in categories:
        resolvers.append("grouped_convtranspose_group_balanced_input_output_resolver")
    if "convtranspose_resolver_required" in categories:
        resolvers.append("convtranspose_deblock_resolver")
    if not resolvers and "coverage_insufficient_single_trace" in categories:
        resolvers.append("multi_trace_union_coverage")
    if not resolvers:
        resolvers.append("none")
    return ";".join(resolvers)


def _current_action(categories: set[str], evidence_status: str) -> str:
    non_coverage = categories - {"coverage_insufficient_single_trace"}
    if not non_coverage:
        return "do_not_mark_unsupported_due_to_single_trace"
    if "scatter_geometry_index_op_protected" in categories or "protected_geometry_or_fixed_shape_contract" in categories:
        return "keep_protected_geometry_or_index_contract"
    if "non_channel_dim_concat" in categories or "non_channelwise_add_or_cat" in categories:
        return "exclude_not_channel_pruning_surface"
    if "grouped_convtranspose_resolver_missing" in categories:
        return "currently_rejected_for_safety_until_grouped_convtranspose_resolver"
    if evidence_status == "resolver_missing":
        return "keep_protected_until_resolver_is_implemented"
    if evidence_status == "audit_bug":
        return "fix_audit_script"
    return "keep_protected"


def _fixability(row: dict[str, Any]) -> dict[str, Any]:
    cats = set(str(row["failure_category"]).split(";"))
    if row["evidence_status"] == "coverage_insufficient":
        fixability = "fixable_with_dynamic_path_enumeration"
        required = "multi-trace union graph with branch-triggering sample manifest"
        priority = "P2"
        protected = False
    elif "linear_feature_dim_mapping_missing" in cats:
        fixability = "fixable_with_downstream_consumer_resolver"
        required = "channel-to-linear-column mapping resolver"
        priority = "P1"
        protected = True
    elif "pfn_feature_dim_resolver_missing" in cats:
        fixability = "fixable_with_downstream_consumer_resolver"
        required = "PFN feature-dimension resolver"
        priority = "P1"
        protected = True
    elif "grouped_convtranspose_resolver_missing" in cats:
        fixability = "fixable_with_convtranspose_resolver"
        required = "grouped ConvTranspose group-balanced input/output resolver"
        priority = "P1"
        protected = True
    elif "missing_cat_offset_proof" in cats:
        fixability = "fixable_with_concat_offset_resolver"
        required = "concat branch offset and downstream local-index proof"
        priority = "P1"
        protected = True
    elif row["theoretical_prunability"] in {"not_channel_pruning_surface", "geometry_or_index_op_protected"}:
        fixability = "should_remain_protected"
        required = "exclude from channel-pruning surface"
        priority = "P3"
        protected = True
    elif row["evidence_status"] == "audit_bug":
        fixability = "already_supported_audit_bug"
        required = "fix audit recognition"
        priority = "P0"
        protected = False
    else:
        fixability = "should_remain_protected"
        required = "unsupported structure until a resolver is proven"
        priority = "P3"
        protected = True
    return {
        "node_name": row["node_name"],
        "node_type": row["node_type"],
        "failure_category": row["failure_category"],
        "fixability": fixability,
        "required_implementation": required,
        "expected_test_name": "tests/test_v991_audit_taxonomy.py",
        "priority": priority,
        "should_remain_protected_now": protected,
    }


def _write_multitrace_design(out_dir: Path) -> None:
    design = """# Multi-Trace Union Design v9.9.1

`dynamic_branch_enumeration_enabled=false` means the current evidence is a
single forward path. It is a coverage limitation, not proof that a structure is
unprunable.

Planned union trace flow:

1. Select branch-triggering validation samples from a manifest.
2. Trace each sample with the same module hooks and tensor producer/consumer edge schema.
3. Canonicalize module nodes by qualified module name and op nodes by stable op signature.
4. Union edges across traces and annotate each edge with `path_id` and sample metadata.
5. Build dependency closures from the union graph; only unsupported/protected ops are excluded.

Current status:

`multitrace_status = design_ready_but_sample_loader_not_implemented`
"""
    (out_dir / "multitrace_design.md").write_text(design, encoding="utf-8")
    manifest = {
        "multitrace_status": "design_ready_but_sample_loader_not_implemented",
        "required_samples": [
            {"sample_type": "1-agent sample", "selection_rule": "validation sample with one cooperative agent"},
            {"sample_type": "2-agent sample", "selection_rule": "validation sample with exactly two agents"},
            {"sample_type": "max-agent sample", "selection_rule": "validation sample with maximum configured agents"},
            {"sample_type": "low voxel sample", "selection_rule": "low occupied voxel count bucket"},
            {"sample_type": "high voxel sample", "selection_rule": "high occupied voxel count bucket"},
            {"sample_type": "different K / bucket sample", "selection_rule": "dataset/model-specific branch bucket if applicable"},
            {"sample_type": "known branch-triggering sample", "selection_rule": "manual manifest entry from validation set"},
        ],
    }
    write_json(out_dir / "multitrace_required_sample_manifest.json", manifest)
    stub = {
        "multitrace_status": "design_ready_but_sample_loader_not_implemented",
        "num_paths": 0,
        "nodes": [],
        "edges": [],
    }
    write_json(out_dir / "union_trace_graph.json", stub)
    write_json(out_dir / "union_dependency_graph.json", stub)
    write_json(
        out_dir / "union_trace_coverage_report.json",
        {
            "multitrace_status": "design_ready_but_sample_loader_not_implemented",
            "dynamic_branch_enumeration_enabled": False,
            "coverage_interpretation": "single_trace_is_coverage_insufficient_not_unsupported",
        },
    )


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# v9.9.1 Residual / Concat Taxonomy", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['node_name']} ({row['node_type']})",
                f"- failure_category: {row['failure_category']}",
                f"- evidence_status: {row['evidence_status']}",
                f"- theoretical_prunability: {row['theoretical_prunability']}",
                f"- required_resolver: {row['required_resolver']}",
                f"- current_action: {row['current_action']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_verdict(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    categories = ";".join(row["failure_category"] for row in rows)
    lines = [
        "# v9.9.1 Audit Taxonomy Verdict",
        "",
        "1. coverage insufficient / resolver missing / true unsupported 已通过 `evidence_status` 区分。",
        "2. single-trace 只标记 `coverage_insufficient_single_trace`，不再写成 `dynamic_path_not_covered` 或永久 unsupported。",
        f"3. Linear/PFN/scatter/grouped ConvTranspose 分类覆盖: {categories}",
        "4. multi-trace union 当前为设计和 schema stub，状态为 `design_ready_but_sample_loader_not_implemented`。",
    ]
    (out_dir / "v991_audit_taxonomy_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_v991_dependency_audit_reports(v98_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = _load_json(v98_dir / "trace_graph_coverage_report.json", {})
    proof = _read_csv(v98_dir / "residual_concat_full_model_proof.csv")
    op_graph = _load_json(v98_dir / "op_graph_path_0.json", {"nodes": {}, "edges": []})
    rows = [
        _classify_v991_row(row, op_graph=op_graph, trace=trace)
        for row in proof
        if str(row.get("proof_pass")) != "True"
    ]
    fields = [
        "node_name",
        "node_type",
        "proof_pass",
        "num_input_branches",
        "input_branches",
        "matched_scope_id",
        "downstream_consumers",
        "is_channel_dim_op",
        "is_regular_channelwise_residual_or_concat",
        "covered_by_current_single_trace",
        "dynamic_path_limitation",
        "failure_category",
        "evidence_status",
        "theoretical_prunability",
        "required_resolver",
        "current_action",
    ]
    write_csv(out_dir / "residual_concat_fail_root_cause_report.csv", rows, fields)
    matrix = [_fixability(row) for row in rows]
    write_csv(
        out_dir / "residual_concat_fixability_matrix.csv",
        matrix,
        [
            "node_name",
            "node_type",
            "failure_category",
            "fixability",
            "required_implementation",
            "expected_test_name",
            "priority",
            "should_remain_protected_now",
        ],
    )
    _write_md(out_dir / "residual_concat_fail_root_cause_report.md", rows)
    _write_multitrace_design(out_dir)
    _write_verdict(out_dir, rows)
    return {
        "output_dir": str(out_dir),
        "num_fail_rows": len(rows),
        "num_coverage_insufficient": sum(1 for row in rows if row["evidence_status"] == "coverage_insufficient"),
        "num_resolver_missing": sum(1 for row in rows if row["evidence_status"] == "resolver_missing"),
        "num_proven_unsupported": sum(1 for row in rows if row["evidence_status"] == "proven_unsupported"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v98-dir", default=str(V98_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_v991_dependency_audit_reports(Path(args.v98_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
