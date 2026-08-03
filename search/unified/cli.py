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
    parser.add_argument("--greedy-anchor-manifest", type=Path)
    parser.add_argument("--search-method", choices=("ga", "greedy"))
    parser.add_argument(
        "--activation-taylor",
        choices=("on", "off"),
        help="Enable or disable activation-quantization Taylor disturbance in Stage1.",
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
        "stage2.greedy_anchor_accuracy_gate.anchor_manifest": (
            str(args.greedy_anchor_manifest) if args.greedy_anchor_manifest else None
        ),
        "search.method": args.search_method,
        "proxy.include_activation_taylor": (
            args.activation_taylor == "on" if args.activation_taylor else None
        ),
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
