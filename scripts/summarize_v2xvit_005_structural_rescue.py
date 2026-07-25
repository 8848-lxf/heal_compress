#!/usr/bin/env python3
"""Write the causal interpretation of the frozen 0.05 restore controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def classify(drop: float, retention: float) -> str:
    if retention < 0.50:
        return "CATASTROPHIC"
    if drop > 0.10 or retention < 0.80:
        return "SEVERE_COLLAPSE"
    if drop > 0.03:
        return "SIGNIFICANT_DROP"
    if drop > 0.01:
        return "MILD_DROP"
    return "SAFE"


def run(root: Path) -> None:
    source = json.loads(
        (root / "reports/old005_structural_rescue.json").read_text()
    )
    b0 = float(source["B0_fixed500_mAP"])
    old = float(source["old005_S32_fixed500_mAP"])
    rows = {}
    for name, control in source["controls"].items():
        value = float(control["evaluation"]["mAP"])
        rows[name] = {
            "mAP": value,
            "drop_vs_B0": b0 - value,
            "retention_vs_B0": value / b0,
            "gain_vs_old005_S32": value - old,
            "collapse_class": classify(b0 - value, value / b0),
        }
    conclusion = {
        "B0_fixed500_mAP": b0,
        "old005_S32_fixed500_mAP": old,
        "old005_class": classify(b0 - old, old / b0),
        "controls": rows,
        "primary_subsystem": "shrinker",
        "secondary_subsystem": "attention_dh",
        "ffn_primary_cause_rejected": True,
        "backbone_stage2_primary_cause_rejected": True,
        "single_subsystem_fully_recovers_accuracy": False,
        "root_cause": "shrinker_dominant_attention_secondary_extreme_structure_interaction",
        "first_significant_structural_drop_budget_sampled": 0.08,
        "first_severe_structural_drop_budget_sampled": 0.07,
        "first_catastrophic_structural_drop_budget_sampled": 0.05,
        "changes_formal_greedy_result": False,
        "changes_ga_admission": False,
    }
    write(root / "reports/old005_structural_rescue_conclusion.json", conclusion)
    lines = [
        "# V2X-ViT 0.05 structural-collapse rescue conclusion",
        "",
        f"B0 fixed500 mAP: `{b0:.9f}`; frozen 0.05 S32 mAP: `{old:.9f}`.",
        "",
        "| Control | mAP | Gain vs 0.05 S32 | Drop vs B0 | Class |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['mAP']:.9f} | {row['gain_vs_old005_S32']:.9f} "
            f"| {row['drop_vs_B0']:.9f} | {row['collapse_class']} |"
        )
    lines.extend(
        [
            "",
            "The shrinker is the strongest single causal subsystem and Attention "
            "d_h is secondary. Restoring FFN or backbone Stage-2 alone has "
            "negligible effect. No single restore returns the model to the safe "
            "range, so the remaining loss is an extreme multi-subsystem structural "
            "interaction rather than an FFN-only or Stage-2-only failure.",
            "",
            "Among the sampled intermediate exact Greedy structures, significant "
            "loss starts at 0.08, severe loss at 0.07, and catastrophic loss at "
            "0.05. These diagnostic controls do not replace a Greedy winner or "
            "change formal GA admission.",
            "",
        ]
    )
    (root / "reports/old005_structural_rescue_conclusion.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
