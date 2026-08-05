"""Command-line entrypoint for all supported HEAL model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_search_config
from .families import registered_families
from .runner import UnifiedSearchRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--list-families", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/unified_search"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--heal-root", type=Path)
    parser.add_argument("--tensorrt-root", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--baseline-engine", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--search-method", choices=("ga", "greedy"))
    parser.add_argument(
        "--bops-target",
        type=float,
        action="append",
        dest="bops_targets",
        help=(
            "Restrict the run to one or more BOPS retention targets. Repeat the "
            "option to request multiple targets."
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        choices=(1, 3, 5, 10),
        help=(
            "Override GA generations. One generation is an explicit pipeline "
            "smoke contract; three is an explicit comparison contract; formal "
            "release searches use five or ten."
        ),
    )
    parser.add_argument(
        "--activation-taylor",
        choices=("on", "off"),
        help="Enable or disable activation-quantization Taylor disturbance in Stage1.",
    )
    parser.add_argument(
        "--objective-calibration",
        choices=("raw", "huber-nnls"),
        help=(
            "Use the raw Taylor sum or fit/freeze robust non-negative task-loss "
            "calibration coefficients before Greedy and GA."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_families:
        print(
            json.dumps(
                [
                    {
                        "family_id": row.family_id,
                        "display_name": row.display_name,
                        "config_template": row.config_template,
                    }
                    for row in registered_families()
                ],
                indent=2,
            )
        )
        return 0
    if args.config is None:
        raise SystemExit("--config is required unless --list-families is used")
    overrides = {
        "model.checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "model.config": str(args.model_config) if args.model_config else None,
        "runtime.heal_root": str(args.heal_root) if args.heal_root else None,
        "runtime.tensorrt_root": str(args.tensorrt_root) if args.tensorrt_root else None,
        "proxy.quant_calibration_npz_manifest": (
            str(args.calibration_manifest) if args.calibration_manifest else None
        ),
        "stage2.evaluation_manifest": (
            str(args.evaluation_manifest) if args.evaluation_manifest else None
        ),
        "baselines.strict_fp32_engine": (
            str(args.baseline_engine) if args.baseline_engine else None
        ),
        "runtime.plugin_path": str(args.plugin) if args.plugin else None,
        "runtime.physical_gpu": args.physical_gpu,
        "search.method": args.search_method,
        "search.bops_targets": args.bops_targets,
        "search.generations_per_round": args.generations,
        "proxy.include_activation_taylor": (
            args.activation_taylor == "on" if args.activation_taylor else None
        ),
        "proxy.objective_calibration": args.objective_calibration,
    }
    config = load_search_config(
        args.config,
        overrides=overrides,
        allow_unresolved=bool(args.dry_run),
    )
    result = UnifiedSearchRunner(config, output_root=args.output_root).run(
        dry_run=bool(args.dry_run)
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
