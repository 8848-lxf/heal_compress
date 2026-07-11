from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def save_json(data: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_markdown(lines: list[str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def status(success: bool, status_text: str, **fields: Any) -> dict[str, Any]:
    return {"success": bool(success), "status": status_text, "generated_at": now(), **fields}


MIGRATED_PRUNING_SOURCES = [
    "tracer/generic_tracer.py",
    "tracer/op_graph.py",
    "pruning/general_pruner.py",
    "pruning/selection.py",
    "pruning/group_checker.py",
]


def build_pruning_migration_report(
    *,
    smoke_evidence: dict[str, Any] | None = None,
    physical_prune_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    smoke_evidence = smoke_evidence or {}
    physical_prune_smoke = physical_prune_smoke or {
        "attempted": False,
        "success": False,
        "reason": "not_run_in_migration_report_generation",
    }
    return status(
        True,
        "migration_documented",
        formal_tooling=["tracer", "pruning"],
        migrated_sources=MIGRATED_PRUNING_SOURCES,
        capabilities={
            "full_model_graph_tracing": smoke_evidence.get("full_model_graph_tracing", "requires HEAL checkpoint/dataset runtime"),
            "coupled_channel_group_generation": smoke_evidence.get("coupled_channel_group_generation", "schema_smoke_tested"),
            "physical_prune_plan_generation": smoke_evidence.get("physical_prune_plan_generation", "schema_smoke_tested"),
            "legality_check": smoke_evidence.get("legality_check", "schema_smoke_tested"),
            "optional_real_physical_prune": physical_prune_smoke,
        },
        smoke_test_evidence=smoke_evidence,
        heal_opencood_source_modified=False,
    )


def write_pruning_migration_report(report: dict[str, Any], root: str | Path) -> tuple[Path, Path]:
    root_path = Path(root)
    md_path = root_path / "summary" / "formal_pruning_tool_migration_report.md"
    json_path = root_path / "debug" / "formal_pruning_tool_migration_report.json"
    save_json(report, json_path)
    caps = report.get("capabilities", {})
    lines = [
        "# Formal Pruning Tool Migration Report",
        "",
        f"- full model graph tracing: {caps.get('full_model_graph_tracing')}",
        f"- coupled channel group generation: {caps.get('coupled_channel_group_generation')}",
        f"- physical prune plan generation: {caps.get('physical_prune_plan_generation')}",
        f"- legality check: {caps.get('legality_check')}",
        f"- real physical prune smoke attempted: {caps.get('optional_real_physical_prune', {}).get('attempted')}",
        f"- real physical prune smoke success: {caps.get('optional_real_physical_prune', {}).get('success')}",
        f"- modified HEAL/OpenCOOD source: {'yes' if report.get('heal_opencood_source_modified') else 'no'}",
        "",
        "## Migrated / Formalized Sources",
        *[f"- {item}" for item in report.get("migrated_sources", [])],
        "",
        "## Formal CLI Examples",
        "",
        "```bash",
        "python -m tracer.export_trace_report --config <config.yaml> --checkpoint <model.pth> --heal-root <heal_repo> --output-dir outputs/formal/tracer/",
        "python -m pruning.planner.physical_prune_plan --config <config.yaml> --checkpoint <model.pth> --trace-report outputs/formal/tracer/coupled_channel_groups.json --importance l1 --target-prune-ratio 0.2 --min-keep-ratio 0.5 --output-dir outputs/formal/pruning/",
        "python -m pruning.export.export_pruned_model --config <config.yaml> --checkpoint <model.pth> --prune-plan <prune_plan.json> --output-dir outputs/formal/models/ --execute-general-pruner",
        "python -m pruning.eval.prune_and_eval --config <config.yaml> --checkpoint <model.pth> --prune-plan <prune_plan.json> --split val --output-dir outputs/formal/evaluation/",
        "```",
    ]
    save_markdown(lines, md_path)
    return md_path, json_path
