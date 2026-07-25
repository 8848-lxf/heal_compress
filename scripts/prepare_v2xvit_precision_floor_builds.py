#!/usr/bin/env python3
"""Freeze all-keep P16/P8 genotypes for deployment-floor build regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("P16-max", "P8-max-requested"), default=("P16-max", "P8-max-requested"))
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    root = args.output_root.resolve()
    floor = json.loads((root / "precision_floor/precision_only_floor.json").read_text(encoding="utf-8"))
    template = json.loads((root / "winner/v2xvit_greedy030_winner.json").read_text(encoding="utf-8"))
    outputs = []
    for profile in args.profiles:
        destination = root / "precision_floor" / f"build_{profile}{args.suffix}"
        (destination / "winner").mkdir(parents=True, exist_ok=False)
        (destination / "proxy").mkdir(parents=True, exist_ok=False)
        payload = dict(template)
        payload["genotype"] = floor["profiles"][profile]["genotype"]
        payload["candidate_hash"] = floor["profiles"][profile]["candidate_hash"]
        payload["precision_floor_profile"] = profile
        winner = destination / "winner/candidate.json"
        winner.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(root / "proxy/structural_gate_mapping.json", destination / "proxy/structural_gate_mapping.json")
        outputs.append({"profile": profile, "root": str(destination), "winner_file": str(winner)})
    manifest = root / "precision_floor" / f"build_inputs{args.suffix}.json"
    manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
