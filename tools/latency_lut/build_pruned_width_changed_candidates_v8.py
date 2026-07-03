from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.build_pruned_width_changed_candidates_v5 import PROTECTED_PREFIXES, TRACE_REPORT, _load_root_domains
from tools.latency_lut.v5_pruned_mixed_common import safe_id, write_json


KEEP_RATIOS = [0.97, 0.875, 0.75, 0.625]


def build(args: argparse.Namespace) -> dict[str, Any]:
    root_domains = [d for d in _load_root_domains(Path(args.trace_report)) if d.get("is_prunable")]
    if not root_domains:
        raise SystemExit(f"root-node domains unavailable from trace report: {args.trace_report}")
    candidates: list[dict[str, Any]] = []
    for keep in KEEP_RATIOS:
        for local_idx in range(2):
            root = root_domains[(len(candidates) + local_idx) % len(root_domains)]
            cid = f"v8_root_local_keep{str(keep).replace('.', '')}_{local_idx:02d}_{safe_id(root['root_node'])[:48]}"
            candidates.append(
                {
                    "candidate_id": cid,
                    "deploy_mode": "single_engine_maxK",
                    "fixed_K": 29696,
                    "selection_mode": "root_node_local_unit_ratio",
                    "importance": "first_order_taylor_task_loss",
                    "global_ranking": False,
                    "module_stage_based_domain": False,
                    "pruning": {
                        "enabled": True,
                        "source": "pruning_tool",
                        "importance": "first_order_taylor",
                        "scope": "root_node_local_unit_ratio",
                        "local_scope": "root_node_local_unit_ratio",
                        "target_keep_ratio": keep,
                        "min_keep_ratio": 0.0,
                        "align": 8,
                        "group_conv_align": 8,
                        "group_conv_selection_mode": "independent_group_topk",
                        "respect_group_conv_alignment": True,
                        "num_calib_batches": 1,
                        "extra_protected_prefixes": PROTECTED_PREFIXES,
                    },
                    "pruning_config": {
                        "selection_mode": "root_node_local_unit_ratio",
                        "importance": "first_order_taylor_task_loss",
                        "root_node_domains": root_domains,
                    },
                    "precision_config": {"default": "FP16", "overrides": {}},
                }
            )
    payload = {
        "schema_version": 8,
        "selection_mode": "root_node_local_unit_ratio",
        "importance": "first_order_taylor_task_loss",
        "global_ranking": False,
        "module_stage_based_domain": False,
        "target_keep_ratios": KEEP_RATIOS,
        "candidates": candidates,
    }
    write_json(args.output, payload)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        "# Pruned Width-Changed Candidates v8\n\n"
        f"- candidates: {len(candidates)}\n"
        "- selection_mode: root_node_local_unit_ratio\n"
        "- importance: first_order_taylor_task_loss\n"
        "- candidates_per_keep_ratio: 2\n"
        "- group_conv_selection_mode: independent_group_topk\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidates": len(candidates), "target_keep_ratios": KEEP_RATIOS}, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-report", "--trace_report", dest="trace_report", default=str(TRACE_REPORT))
    parser.add_argument("--output", default="outputs/latency_lut/pruned_width_changed_candidates_v8.json")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_width_changed_candidates_v8_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
