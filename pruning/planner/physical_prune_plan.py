from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pruning.utils.constraints import check_plan_legality, legal_keep_count
from pruning.utils.report import save_json, save_markdown, status


def _load_groups(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("groups") or [])
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported coupled group schema: {path}")


def build_plan_from_coupled_groups(
    coupled_groups: dict[str, Any] | list[dict[str, Any]],
    *,
    target_prune_ratio: float,
    min_keep_ratio: float,
    importance: str,
) -> dict[str, Any]:
    groups = coupled_groups.get("groups", []) if isinstance(coupled_groups, dict) else coupled_groups
    rows: list[dict[str, Any]] = []
    seen_layers: set[str] = set()
    for group in groups:
        channels = [int(v) for v in group.get("channel_indices", [])]
        total = len(channels)
        keep_count, prune_count = legal_keep_count(total, target_prune_ratio, min_keep_ratio) if total else (0, 0)
        keep_indices = channels[:keep_count]
        prune_indices = channels[keep_count:]
        source_modules = []
        for layer in group.get("source_modules", []):
            if layer in seen_layers:
                continue
            seen_layers.add(layer)
            source_modules.append(layer)
        rows.append(
            {
                "group_id": str(group.get("group_id")),
                "group_type": str(group.get("group_type", "conv_block")),
                "source_modules": source_modules,
                "dependent_modules": list(group.get("dependent_modules") or []),
                "num_channels": total,
                "keep_count": keep_count,
                "prune_count": prune_count,
                "keep_indices": keep_indices,
                "prune_indices": prune_indices,
                "importance": importance,
                "is_prunable": bool(group.get("is_prunable", True)),
                "is_protected": bool(group.get("is_protected", False)),
                "physical_operations": [],
            }
        )
    legality = check_plan_legality(rows, min_keep_ratio=min_keep_ratio)
    return {
        "schema_version": 1,
        "target_prune_ratio": float(target_prune_ratio),
        "min_keep_ratio": float(min_keep_ratio),
        "importance": importance,
        "prune_plan": rows,
        "legality": legality,
        "supports_physical_prune": False,
        "physical_prune_note": "Plan schema generated from coupled-channel groups; concrete surgery operations are produced by the existing GeneralPruner runner during actual model pruning.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate formal physical prune plan from coupled channel groups.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace-report", "--trace_report", dest="trace_report", required=True)
    parser.add_argument("--importance", default="l1")
    parser.add_argument("--target-prune-ratio", "--target_prune_ratio", dest="target_prune_ratio", type=float, default=0.2)
    parser.add_argument("--min-keep-ratio", "--min_keep_ratio", dest="min_keep_ratio", type=float, default=0.5)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    return parser.parse_args(argv)


def generate_prune_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _load_groups(args.trace_report)
    plan = build_plan_from_coupled_groups(
        {"groups": groups},
        target_prune_ratio=float(args.target_prune_ratio),
        min_keep_ratio=float(args.min_keep_ratio),
        importance=str(args.importance),
    )
    plan.update(
        {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "trace_report": str(args.trace_report),
            "formal_tool": "pruning.planner.physical_prune_plan",
        }
    )
    save_json(plan, output_dir / "prune_plan.json")
    save_json(plan["legality"], output_dir / "legality_check_report.json")
    lines = [
        "# Formal Prune Plan",
        "",
        f"- target_prune_ratio: {plan['target_prune_ratio']}",
        f"- min_keep_ratio: {plan['min_keep_ratio']}",
        f"- groups: {len(plan['prune_plan'])}",
        f"- legal: {plan['legality']['legal']}",
        f"- supports_physical_prune: {plan['supports_physical_prune']}",
        f"- note: {plan['physical_prune_note']}",
    ]
    save_markdown(lines, output_dir / "prune_plan_summary.md")
    return plan


def main(argv: list[str] | None = None) -> int:
    plan = generate_prune_plan(parse_args(argv))
    print(json.dumps({"legal": plan["legality"]["legal"], "groups": len(plan["prune_plan"])}, indent=2))
    return 0 if plan["legality"]["legal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
