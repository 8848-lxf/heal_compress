#!/usr/bin/env python3
"""Evaluate each admitted budget's Greedy anchor and final GA winner on fixed500."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    formal = json.loads((root / "reports/ga_formal_results.json").read_text())
    greedy_fixed = json.loads((root / "reports/six_budget_fixed500.json").read_text())["controls"]
    request = json.loads((root / "evaluation_fixed500/B0/evaluation_request.json").read_text())
    results: dict[str, Any] = {"B0": greedy_fixed["B0"], "budgets": {}}
    for label, budget_row in formal["budgets"].items():
        greedy = budget_row["greedy_anchor"]
        winner = budget_row["final_winner"]
        rows = {
            "Greedy": greedy_fixed[f"budget_{label}/JMIX-FRESH"],
        }
        if winner["complete_phenotype_hash"] == greedy["complete_phenotype_hash"]:
            rows["GA-final"] = {**rows["Greedy"], "reused_identical_greedy_engine": True}
        else:
            engine = Path(winner["metadata"]["engine_path"])
            destination = root / f"ga_final_fixed500/budget_{label}/{winner['complete_phenotype_hash']}"
            existing = destination / "evaluation.json"
            if existing.is_file():
                evaluation = json.loads(existing.read_text())
            else:
                evaluation = evaluate_v2xvit_engine_modelopt(
                    engine_path=engine,
                    model_config=request["model_config"], heal_root=request["heal_root"],
                    output_dir=destination, tensorrt_root=args.tensorrt_root,
                    plugin_path=request["plugin_path"], eval_manifest_path=args.fixed500_manifest,
                    physical_gpu_id=args.physical_gpu, fixed_k=int(request["fixed_k"]),
                    max_agents=int(request["max_agents"]), num_frames=500,
                    warmup_frames=200, latency_rounds=1, dataloader_num_workers=8,
                )
            if not (
                evaluation.get("status") == "ok"
                and int(evaluation.get("num_evaluated_frames", -1)) == 500
                and int(evaluation.get("num_skipped_frames", -1)) == 0
            ):
                raise RuntimeError(f"ga_final_fixed500_failed:budget_{label}")
            rows["GA-final"] = evaluation
        results["budgets"][label] = {
            "budget": float(budget_row["budget"]),
            "greedy_hash": greedy["complete_phenotype_hash"],
            "ga_hash": winner["complete_phenotype_hash"],
            "ga_improved_greedy_stage2": bool(budget_row["ga_improved_greedy"]),
            "controls": rows,
        }
        atomic(root / "reports/greedy_vs_ga_fixed500.json", results)
    output_rows = []
    for label, budget_row in results["budgets"].items():
        for control, metric in budget_row["controls"].items():
            output_rows.append({
                "budget": budget_row["budget"], "control": control,
                "candidate_hash": budget_row["greedy_hash"] if control == "Greedy" else budget_row["ga_hash"],
                "AP30": metric["AP@0.3"], "AP50": metric["AP@0.5"],
                "AP70": metric["AP@0.7"], "mAP": metric["mAP"],
                "evaluated": metric["num_evaluated_frames"],
                "skipped": metric["num_skipped_frames"],
            })
    fields = list(output_rows[0])
    with (root / "reports/greedy_vs_ga_fixed500.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=6)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path,
                        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
