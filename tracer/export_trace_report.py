from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from .coupled_channel_groups import normalize_group_collection
from .dynamic_path_staticizer import detect_dynamic_paths, staticize_paths
from .graph_trace import trace_lidar_pyramid
from .module_inspector import inspect_model_modules, summarize_trace
from .utils import add_repo_parent_to_sys_path, empty_trace_summary, ensure_dir, save_json, save_markdown


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export formal lidar_pyramid trace report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--heal-root", "--heal_root", dest="heal_root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    return parser.parse_args(argv)


def build_trace_report(args: argparse.Namespace) -> dict[str, Any]:
    add_repo_parent_to_sys_path()
    output_dir = ensure_dir(args.output_dir)
    try:
        from heal_compress.tracer.dependency_graph import DependencyGraphBuilder
        from heal_compress.tracer.coupled_channel_group import CoupledChannelGroupBuilder

        trace_graph, model, sample_meta = trace_lidar_pyramid(
            config=args.config,
            checkpoint=args.checkpoint,
            heal_root=args.heal_root,
            device=args.device,
            split=args.split,
        )
        dep_builder = DependencyGraphBuilder(trace_graph, model).build()
        dependency_graph = dep_builder.to_dict()
        raw_groups = [group.to_dict() for group in CoupledChannelGroupBuilder(dependency_graph, model).build()]
        group_collection = normalize_group_collection(raw_groups)
        dynamic_paths = detect_dynamic_paths(trace_graph)
        staticized = staticize_paths(dynamic_paths)
        module_summary = inspect_model_modules(model)
        trace_summary = summarize_trace(trace_graph, dependency_graph)
        summary = {
            "success": True,
            **module_summary,
            **trace_summary,
            "num_dependency_groups": len(dependency_graph.get("edges", [])),
            "num_coupled_channel_groups": group_collection["num_coupled_channel_groups"],
            "dynamic_paths_detected": dynamic_paths,
            "staticized_paths": staticized,
            "duplicate_coupled_layers_removed": group_collection["duplicate_coupled_layers_removed"],
            "min_channel_constraints_detected": [],
            "dummy_input": sample_meta,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
        }
    except Exception as exc:
        trace_graph = {"nodes": {}, "edges": []}
        dependency_graph = {"nodes": {}, "edges": [], "warnings": []}
        group_collection = {"schema_version": 1, "groups": [], "num_coupled_channel_groups": 0, "duplicate_coupled_layers_removed": 0}
        summary = empty_trace_summary(str(exc))
        summary["traceback"] = traceback.format_exc()
        summary["config"] = str(args.config)
        summary["checkpoint"] = str(args.checkpoint)

    save_json(trace_graph, output_dir / "trace_graph.json")
    save_json(dependency_graph, output_dir / "dependency_graph.json")
    save_json(group_collection, output_dir / "coupled_channel_groups.json")
    save_json(summary, output_dir / "trace_summary.json")
    lines = [
        "# Trace Summary",
        "",
        f"- success: {summary.get('success')}",
        f"- total_modules: {summary.get('total_modules')}",
        f"- traced_modules: {summary.get('traced_modules')}",
        f"- conv_layers: {summary.get('conv_layers')}",
        f"- bn_layers: {summary.get('bn_layers')}",
        f"- residual_edges: {summary.get('residual_edges')}",
        f"- concat_edges: {summary.get('concat_edges')}",
        f"- fusion_edges: {summary.get('fusion_edges')}",
        f"- detection_head_edges: {summary.get('detection_head_edges')}",
        f"- num_dependency_groups: {summary.get('num_dependency_groups')}",
        f"- num_coupled_channel_groups: {summary.get('num_coupled_channel_groups')}",
        f"- duplicate_coupled_layers_removed: {summary.get('duplicate_coupled_layers_removed')}",
        f"- error: {summary.get('error')}",
    ]
    save_markdown(lines, output_dir / "trace_summary.md")
    return summary


def main(argv: list[str] | None = None) -> int:
    summary = build_trace_report(parse_args(argv))
    print(json.dumps({"success": summary.get("success"), "num_coupled_channel_groups": summary.get("num_coupled_channel_groups")}, indent=2))
    return 0 if summary.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
