#!/usr/bin/env python3
"""Materialize auditable winner inputs for the Greedy-0.30 Stage-2 controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    selected = json.loads((root / "reports/stage2_candidate_screening.json").read_text(encoding="utf-8"))["selected"]
    template = json.loads((root / "winner/v2xvit_greedy030_winner.json").read_text(encoding="utf-8"))
    gate_source = root / "proxy/structural_gate_mapping.json"
    manifest = []
    for index, row in enumerate(selected):
        destination = root / "stage2" / f"{index:02d}_{row['candidate_hash'][:12]}"
        (destination / "winner").mkdir(parents=True, exist_ok=False)
        (destination / "proxy").mkdir(parents=True, exist_ok=False)
        payload = dict(template)
        genotype = dict(template["genotype"])
        genotype["pruning_width_genes"] = {str(key): int(value) for key, value in row["widths"].items()}
        genotype["precision_genes"] = {str(key): str(value) for key, value in row["precision"].items()}
        genotype["meta"] = {**dict(genotype.get("meta", {})), "stage2_selection_reason": row["selection_reason"]}
        payload["candidate_hash"] = row["candidate_hash"]
        payload["genotype"] = genotype
        payload["stage2_selection"] = row
        winner_path = destination / "winner/candidate.json"
        winner_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(gate_source, destination / "proxy/structural_gate_mapping.json")
        manifest.append({"index": index, "candidate_hash": row["candidate_hash"], "selection_reason": row["selection_reason"], "root": str(destination), "winner_file": str(winner_path)})
    (root / "stage2/candidate_inputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
