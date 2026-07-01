from __future__ import annotations

import argparse
import json
from pathlib import Path

from pruning.utils.constraints import check_plan_legality
from pruning.utils.report import save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check formal prune_plan.json legality.")
    parser.add_argument("--prune-plan", "--prune_plan", dest="prune_plan", required=True)
    parser.add_argument("--min-keep-ratio", "--min_keep_ratio", dest="min_keep_ratio", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = json.loads(Path(args.prune_plan).read_text(encoding="utf-8"))
    result = check_plan_legality(data.get("prune_plan", []), min_keep_ratio=float(args.min_keep_ratio))
    save_json(result, Path(args.prune_plan).with_name("legality_check_report.json"))
    print(json.dumps({"legal": result["legal"], "issues": len(result["issues"])}, indent=2))
    return 0 if result["legal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
