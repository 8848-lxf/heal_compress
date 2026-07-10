from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pruning.utils.report import save_json, save_markdown, status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute formal pruning export or record why export is skipped.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prune-plan", "--prune_plan", dest="prune_plan", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute-general-pruner", "--execute_general_pruner", dest="execute_general_pruner", action="store_true")
    parser.add_argument("--target-prune-ratio", "--target_prune_ratio", dest="target_prune_ratio", type=float, default=None)
    parser.add_argument("--extra-protected-prefix", "--extra_protected_prefix", dest="extra_protected_prefix", action="append", default=[])
    parser.add_argument("--importance-mode", "--importance_mode", dest="importance_mode", default="l1_norm")
    parser.add_argument("--selection-mode", "--selection_mode", dest="selection_mode", default="constrained_global")
    parser.add_argument("--align", type=int, default=16)
    parser.add_argument("--group-conv-align", "--group_conv_align", dest="group_conv_align", type=int, default=8)
    parser.add_argument("--group-conv-selection-mode", "--group_conv_selection_mode", dest="group_conv_selection_mode", default="independent_group_topk")
    parser.add_argument("--num-calib-batches", "--num_calib_batches", dest="num_calib_batches", type=int, default=None)
    parser.add_argument("--run-forward-check", "--run_forward_check", dest="run_forward_check", action="store_true")
    return parser.parse_args(argv)


def _run_existing_general_pruner(args: argparse.Namespace, plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    raise RuntimeError(
        "execute_general_pruner_legacy_path_removed: formal runtime code may not depend on test helpers; "
        "use heal_compress.pruning.formal_pruner.HEALStructuredPruner"
    )


def _importance_failure_status(importance_mode: str, error: str) -> str:
    text = str(error)
    if "gradient_missing" not in text and "Missing gradients" not in text:
        return "existing_general_pruner_failed"
    if str(importance_mode) == "first_order_taylor":
        return "taylor_importance_gradient_missing"
    if str(importance_mode) == "second_order_fisher":
        return "fisher_importance_gradient_missing"
    return "importance_gradient_missing"


def export_pruned_model(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.prune_plan).read_text(encoding="utf-8"))
    if args.execute_general_pruner:
        try:
            summary = _run_existing_general_pruner(args, plan, output_dir)
            report = status(
                True,
                "exported_with_existing_general_pruner",
                formal_tool="pruning.export.export_pruned_model",
                config=str(args.config),
                checkpoint=str(args.checkpoint),
                prune_plan=str(args.prune_plan),
                existing_general_pruner_summary=summary,
                pruned_checkpoint=str(summary.get("pruned_checkpoint") or output_dir / "pruned_model.pth"),
                actual_output_dir=str(summary.get("output_dir") or output_dir),
                real_physical_prune_attempted=True,
                real_physical_prune_success=Path(str(summary.get("pruned_checkpoint") or output_dir / "pruned_model.pth")).is_file(),
            )
        except Exception as exc:
            failure_status = _importance_failure_status(str(args.importance_mode), str(exc))
            report = status(
                False,
                failure_status,
                formal_tool="pruning.export.export_pruned_model",
                error=str(exc),
                real_physical_prune_attempted=True,
                real_physical_prune_success=False,
            )
    else:
        report = status(
            False,
            "skipped_requires_execute_general_pruner",
            formal_tool="pruning.export.export_pruned_model",
            config=str(args.config),
            checkpoint=str(args.checkpoint),
            prune_plan=str(args.prune_plan),
            supports_physical_prune=bool(plan.get("supports_physical_prune")),
            reason=(
                "The formal prune_plan schema does not contain concrete surgery operations. "
                "Run with --execute-general-pruner to use the existing validated HEAL pruner path."
            ),
            real_physical_prune_attempted=False,
            real_physical_prune_success=False,
        )
    save_json(report, output_dir / "export_pruned_model_report.json")
    save_markdown(
        [
            "# Export Pruned Model Report",
            "",
            f"- status: {report['status']}",
            f"- real_physical_prune_attempted: {report.get('real_physical_prune_attempted')}",
            f"- real_physical_prune_success: {report.get('real_physical_prune_success')}",
            f"- reason: {report.get('reason', report.get('error', ''))}",
        ],
        output_dir / "export_pruned_model_report.md",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    report = export_pruned_model(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "status": report.get("status")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
