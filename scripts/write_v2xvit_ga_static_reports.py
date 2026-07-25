#!/usr/bin/env python3
"""Freeze the Greedy-0.30 anchor and emit formal GA contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from search.ga.stage12_v3 import StrictGAConfig


ANCHOR_HASH = "5e84b1364a0ad2ac51fc861f021fdeffed61cc1741a15eae461c7e78a76fff4f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    if isinstance(value, str):
        path.write_text(value.rstrip() + "\n", encoding="utf-8")
    else:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    anchor = args.anchor_root.resolve()
    winner_path = anchor / "reports/greedy030_winner.json"
    acceptance_path = anchor / "reports/final_acceptance.json"
    fixed500_path = anchor / "reports/fixed500_metrics.json"
    latency_path = anchor / "reports/latency_results.json"
    for path in (winner_path, acceptance_path, fixed500_path, latency_path):
        if not path.is_file():
            raise RuntimeError(f"anchor_artifact_missing:{path}")
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    if winner.get("candidate_hash") != ANCHOR_HASH:
        raise RuntimeError("greedy030_anchor_hash_mismatch")
    config = StrictGAConfig(target_bops_retention=0.30)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], check=True, text=True, capture_output=True
    ).stdout.strip()
    write(root / "anchor/greedy030_anchor.json", {
        "schema_version": "h800-v2xvit-greedy030-frozen-anchor-v1",
        "source_root": str(anchor),
        "source_commit": "6edd7164a3b52e8d00d8cf1c1daef638821478ce",
        "candidate_hash": ANCHOR_HASH,
        "R_bops": 0.3048414445559227,
        "parameter_retention": 0.7375835647095631,
        "shrinker_width": 116,
        "fixed500_map": {"B0": 0.658670, "S32": 0.659162, "JMIX": 0.659257},
        "formal_p50_ms": {"B0": 15.9393, "S32": 15.2464, "JMIX": 11.7599},
        "artifact_hashes": {
            str(path.relative_to(anchor)): sha256(path)
            for path in (winner_path, acceptance_path, fixed500_path, latency_path)
        },
        "immutable": True,
    })
    write(root / "reports/greedy_ga_naming_audit.json", {
        "greedy_uses_stage2_top5": False,
        "ga_uses_stage2_top5": True,
        "greedy_exact_winner_count_per_budget": 1,
        "greedy_real_metrics_can_replace_winner": False,
        "legacy_read_only_artifact_terms": [
            "reports/stage2_candidate_screening.csv",
            "scripts/prepare_v2xvit_greedy030_stage2.py",
        ],
        "legacy_artifacts_mutated": False,
        "active_replacements": {
            "greedy exact build": "greedy_exact_winner_build",
            "greedy exact fixed500": "greedy_exact_winner_fixed500",
            "greedy exact latency": "greedy_exact_winner_latency",
            "four extra historical S32 candidates": "posthoc_budget_band_robustness_audit",
        },
        "reason_legacy_names_remain_readable": "immutable prior run provenance; no active code describes them as GA Stage-2",
    })
    stage1 = {
        "schema_version": "v2xvit-ga-stage1-contract-v3",
        "order": [
            "genotype_schema_validation", "structure_precision_legality",
            "canonicalization", "physical_phenotype_hash", "dedup",
            "exact_bops", "bops_hard_gate", "unified_taylor_proxy",
            "stage1_ranking", "new_stage2_candidate_selection",
        ],
        "objective": "J_struct_gate + J_WQ + J_AQ",
        "bops_tolerance_abs": 0.005,
        "repair": {"structure": False, "precision": False, "budget": False},
        "fixed_loci_removed": True,
        "tie_epsilon": "max(1e-12,1e-8*max(abs(J1),abs(J2)))",
        "tie_break": ["bops_deviation", "higher_parameter_retention", "higher_mixed_weight_retention", "phenotype_hash"],
        "over_pruning_reward": False,
    }
    write(root / "reports/ga_stage1_contract.json", stage1)
    write(root / "reports/ga_stage1_contract.md", """# GA Stage-1 contract

The formal order is schema validation → legality → canonicalization → physical
hash → dedup → exact BOPS → hard gate → `J_struct_gate + J_WQ + J_AQ` →
ranking → selection of at most five unseen real candidates. Illegal or
out-of-band offspring are rejected unchanged. Parameter count and mixed size
are never compression rewards; higher retention is used only at a strict
Taylor tie.
""")
    stage2 = {
        "schema_version": "v2xvit-ga-stage2-contract-v3",
        "greedy_stage2_forbidden": True,
        "new_candidate_quota_per_generation": 5,
        "flow": ["physical_materialization", "fresh_train200_if_int8", "strongly_typed_onnx", "tensorrt_build", "requested_realized_audit", "fixed50", "screening_latency", "accuracy_gate", "F_S2"],
        "accuracy_gate": "mAP(C) >= mAP(Greedy)-0.005",
        "A_0.005": "clip((m_G-m_C)/0.005,-1,1)",
        "F_S2": "0.2*A_0.005 + 0.8*p50(C)/p50(G)",
        "historical_elites_consume_new_quota": False,
        "engine_failure_fallback": False,
    }
    write(root / "reports/ga_stage2_contract.json", stage2)
    write(root / "reports/ga_stage2_contract.md", """# GA Stage-2 contract

Only GA generations may select up to five unseen physical phenotypes. Each is
materialized, freshly calibrated when INT8 is present, built strongly typed,
audited, evaluated on fixed50, and measured with screening latency. Candidates
below the Greedy mAP by more than 0.005 are ineligible. No precision fallback
or relaxed accuracy gate is allowed.
""")
    write(root / "reports/ga_v1_v2_v3_contract.json", {
        "V1": "exact Greedy anchor is injected, retained every generation, and used for final no-worse comparison",
        "V2": "eligible new real winners and elites enter the immediately following population",
        "V3": ["Greedy anchor", "global lowest F_S2", "global highest mAP", "global lowest p50"],
        "anchors_deduplicated_by_physical_phenotype_hash": True,
        "black_box_surrogate": False,
    })
    write(root / "reports/ga_gen10_configuration.json", {
        "branch": branch,
        "commit_at_contract_generation": commit,
        "population_size": config.population_size,
        "offspring_size": config.offspring_size,
        "generations": config.generations,
        "seeds": 1,
        "seed_ids": [0],
        "stage2_new_candidate_quota": config.stage2_new_candidate_quota,
        "generation_zero_counted": False,
        "formal_generation_indices": list(range(1, 11)),
        "early_stop_for_stagnation": False,
        "maximum_generation_index": 10,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
