from __future__ import annotations

import argparse
import json

from pruning.utils.report import build_pruning_migration_report, write_pruning_migration_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write formal pruning migration report.")
    parser.add_argument("--root", default="outputs/formalization")
    parser.add_argument("--physical-prune-attempted", "--physical_prune_attempted", dest="physical_prune_attempted", action="store_true")
    parser.add_argument("--physical-prune-success", "--physical_prune_success", dest="physical_prune_success", action="store_true")
    parser.add_argument("--physical-prune-reason", "--physical_prune_reason", dest="physical_prune_reason", default="not_run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_pruning_migration_report(
        smoke_evidence={
            "formal_import_check": "release_import_smoke",
            "coupled_channel_group_generation": "schema_smoke_tested",
            "physical_prune_plan_generation": "schema_smoke_tested",
            "legality_check": "schema_smoke_tested",
        },
        physical_prune_smoke={
            "attempted": bool(args.physical_prune_attempted),
            "success": bool(args.physical_prune_success),
            "reason": args.physical_prune_reason,
        },
    )
    md_path, json_path = write_pruning_migration_report(report, args.root)
    print(json.dumps({"success": True, "markdown_path": str(md_path), "json_path": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
