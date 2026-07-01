from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
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
    return parser.parse_args(argv)


def _run_existing_general_pruner(args: argparse.Namespace, plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    tests_dir = Path(__file__).resolve().parents[2] / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    if "pytest" not in sys.modules:
        try:
            __import__("pytest")
        except ImportError:
            def _skip(reason: str = "") -> None:
                raise RuntimeError(f"pytest.skip called while pytest is unavailable: {reason}")

            sys.modules["pytest"] = SimpleNamespace(skip=_skip)
    from test_general_pruner import parse_args as parse_old_args, run_pruning

    ratio = float(args.target_prune_ratio if args.target_prune_ratio is not None else plan.get("target_prune_ratio", 0.2))
    old_args = parse_old_args(
        [
            "--checkpoint", str(args.checkpoint),
            "--model-config", str(args.config),
            "--device", str(args.device),
            "--prune-ratio", str(ratio),
            "--importance-mode", "l1_norm",
            "--selection-mode", "constrained_global",
            "--group-conv-selection-mode", "shared_local_mean",
            "--group-conv-align", "8",
            "--align", "16",
            "--protect-residual-add", "true",
            "--output-dir", str(output_dir),
            "--skip-forward-check",
        ]
    )
    return run_pruning(old_args)


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
                pruned_checkpoint=str(output_dir / "pruned_model.pth"),
                real_physical_prune_attempted=True,
                real_physical_prune_success=Path(output_dir / "pruned_model.pth").is_file(),
            )
        except Exception as exc:
            report = status(
                False,
                "existing_general_pruner_failed",
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
