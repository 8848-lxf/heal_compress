from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pruning.utils.report import save_json, save_markdown, status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pruned lidar_pyramid checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prune-plan", "--prune_plan", dest="prune_plan", required=True)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--pruned-checkpoint", "--pruned_checkpoint", dest="pruned_checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pruned = Path(args.pruned_checkpoint).expanduser() if args.pruned_checkpoint else Path(args.checkpoint).expanduser()
    if not pruned.is_file():
        report = status(False, "skipped_missing_pruned_checkpoint", pruned_checkpoint=str(pruned))
    else:
        report = status(
            False,
            "skipped_requires_full_dataset_gpu_eval",
            formal_tool="pruning.eval.prune_and_eval",
            pruned_checkpoint=str(pruned),
            config=str(args.config),
            split=str(args.split),
            runnable_command=(
                "python tests/test_prune_and_eval.py "
                f"--original-checkpoint {args.checkpoint} --pruned-checkpoint {pruned} "
                f"--model-config {args.config} --device {args.device} --output-dir {output_dir}"
            ),
        )
    save_json(report, output_dir / "prune_and_eval_report.json")
    save_markdown(
        [
            "# Prune And Eval Report",
            "",
            f"- status: {report['status']}",
            f"- success: {report['success']}",
            f"- runnable_command: {report.get('runnable_command', '')}",
        ],
        output_dir / "prune_and_eval_report.md",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    report = evaluate(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "status": report.get("status")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
