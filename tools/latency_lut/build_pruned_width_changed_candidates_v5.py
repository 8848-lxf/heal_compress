from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.v5_pruned_mixed_common import load_json, safe_id, write_json


TRACE_REPORT = Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/tracer_reports/lidar_pyramid/coupled_channel_groups.json")
PROTECTED_PREFIXES = [
    "cls_head",
    "reg_head",
    "dir_head",
]
SAMPLING_GRID = [
    {"name": "local_taylor_keep97", "domain_keep_ratio": 0.97, "min_group_keep_ratio": 0.875, "num_candidates": 6},
    {"name": "local_taylor_keep875", "domain_keep_ratio": 0.875, "min_group_keep_ratio": 0.75, "num_candidates": 8},
    {"name": "local_taylor_keep75", "domain_keep_ratio": 0.75, "min_group_keep_ratio": 0.625, "num_candidates": 8},
    {"name": "local_taylor_keep625", "domain_keep_ratio": 0.625, "min_group_keep_ratio": 0.5, "num_candidates": 8},
]


def _load_root_domains(trace_report: Path) -> list[dict[str, Any]]:
    data = load_json(trace_report, {"groups": []})
    groups = list(data.get("groups") or [])
    domains: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group.get("group_id") or "")
        root = group_id.replace("group::", "", 1) if group_id.startswith("group::") else group_id
        domains.append(
            {
                "root_node": root,
                "scope_id": group_id,
                "num_channels": len(group.get("channel_indices") or []),
                "group_type": group.get("group_type"),
                "source_modules": group.get("source_modules") or [],
                "is_prunable": bool(group.get("is_prunable", True)) and not bool(group.get("is_protected", False)),
            }
        )
    return domains


def _candidate(grid: dict[str, Any], local_idx: int, root_domains: list[dict[str, Any]]) -> dict[str, Any]:
    keep = float(grid["domain_keep_ratio"])
    min_keep = float(grid["min_group_keep_ratio"])
    cid = f"v5_{grid['name']}_{local_idx:02d}"
    # The existing formal runner executes first-order Taylor with gradients
    # before physical pruning. Candidate generation does not fake group masks;
    # the actual group_keep_map/prune_replay are produced in the export step.
    return {
        "candidate_id": cid,
        "deploy_mode": "single_engine_maxK",
        "fixed_K": 29696,
        "selection_mode": "root_node_local_pruning_domain",
        "importance": "first_order_taylor_task_loss",
        "global_ranking": False,
        "module_stage_based_domain": False,
        "pruning": {
            "enabled": True,
            "source": "pruning_tool",
            "importance": "first_order_taylor",
            "scope": "local",
            "local_scope": "prune_domain",
            "target_keep_ratio": keep,
            "min_keep_ratio": min_keep,
            "align": 8,
            "respect_group_conv_alignment": True,
            "num_calib_batches": 1,
            "extra_protected_prefixes": PROTECTED_PREFIXES,
        },
        "pruning_config": {
            "selection_mode": "root_node_local_pruning_domain",
            "importance": "first_order_taylor_task_loss",
            "group_mask": {},
            "channel_keep_map": {},
            "root_node_domains": root_domains,
        },
        "precision_config": {"default": "FP16", "overrides": {}},
        "group_mask": {},
        "channel_keep_map": {},
        "root_node_domains": root_domains,
        "resolved_units": [],
        "is_shape_legal": None,
        "illegal_reason": "",
        "expected_changed_conv_layers": [],
        "candidate_status": "pending_physical_prune",
        "note": "Concrete group_mask/channel_keep_map are intentionally produced only by the formal physical pruning export; this file is a legal local-root pruning request, not a fake subnet.",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    trace = Path(args.trace_report)
    root_domains = _load_root_domains(trace)
    if not root_domains:
        raise SystemExit(f"root-node domains unavailable from trace report: {trace}")
    candidates: list[dict[str, Any]] = []
    for grid in SAMPLING_GRID:
        for idx in range(int(grid["num_candidates"])):
            cand = _candidate(grid, idx, root_domains)
            cand["candidate_id"] = f"{cand['candidate_id']}_{safe_id(root_domains[idx % len(root_domains)]['root_node'])[:48]}"
            candidates.append(cand)
    payload = {
        "schema_version": 5,
        "selection_mode": "root_node_local_pruning_domain",
        "importance": "first_order_taylor_task_loss",
        "global_ranking": False,
        "module_stage_based_domain": False,
        "num_using_global_ranking": 0,
        "num_using_module_stage_based_domain": 0,
        "num_grouped_conv_group_count_changed": 0,
        "root_node_domain_count": len(root_domains),
        "candidates": candidates,
    }
    write_json(args.output, payload)
    lines = [
        "# Pruned Width-Changed Candidates v5",
        "",
        f"- candidates: {len(candidates)}",
        f"- root_node_domains: {len(root_domains)}",
        "- selection_mode: root_node_local_pruning_domain",
        "- importance: first_order_taylor_task_loss",
        "- global_ranking: false",
        "- module_stage_based_domain: false",
        "",
        "Concrete `group_mask`, `channel_keep_map`, and `resolved_units` are generated by `export_pruned_width_changed_onnx_v5.py` from the formal pruning replay; they are not hand-written in this candidate request file.",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(candidates), "root_node_domains": len(root_domains)}, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-report", "--trace_report", dest="trace_report", default=str(TRACE_REPORT))
    parser.add_argument("--output", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_width_changed_candidates_v5_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
