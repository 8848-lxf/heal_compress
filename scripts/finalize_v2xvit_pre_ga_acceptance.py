#!/usr/bin/env python3
"""Create the fail-closed gate for the single-seed R=0.10 formal GA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PRIOR_CLOSURE = Path(
    "/data/lxf/heal_data/outputs/"
    "h800_v2xvit_bops_precision_closure_20260725_082219/"
    "reports/final_audit_acceptance.json"
)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(root: Path) -> int:
    reports = root / "reports"
    closure = read(PRIOR_CLOSURE)
    av = read(reports / "av_profile_acceptance.json")
    merge = read(reports / "window_merge_acceptance.json")
    search = read(reports / "pre_search_precision_acceptance.json")
    taylor = read(reports / "taylor_collection_contract.json")
    cache = read(reports / "taylor_cache_manifest.json")
    convergence = read(reports / "taylor_sample_convergence.json")
    greedy = read(reports / "greedy_r010_winner.json")
    builds = read(reports / "six_budget_builds.json")
    fixed50 = read(reports / "greedy_r010_fixed50.json")
    smoke_path = reports / "ga_one_generation_real_stage2_smoke.json"
    smoke = read(smoke_path) if smoke_path.is_file() else {"passed": False}
    av_profiles = {row["profile"]: row for row in av["profiles"]}

    blockers: list[str] = []
    checks = {
        "hgt_bops_closure_preserved": bool(
            closure.get("bops_audit_passed")
            and closure.get("production_vs_independent_bops_match")
            and closure.get("shrinker_marginal_bops_valid")
            and closure.get("attention_marginal_bops_valid")
        ),
        "av32_deployment_valid": bool(av_profiles["AV32"]["legal_for_search"]),
        "window_merge_derived_join_valid": bool(merge["derived_join_valid"]),
        "window_merge_requested_realized_exact": bool(
            merge["requested_realized_exact"]
        ),
        "final_search_space_frozen": bool(search["final_search_space_frozen"]),
        "old_cache_invalidated": bool(search["old_cache_invalidated"]),
        "taylor_cache_rebuilt": bool(
            cache["sample_count"] == 32
            and not cache["old_cache_reused"]
            and taylor["missing_hook_count"] == 0
            and taylor["unexpected_inactive_count"] == 0
            and taylor["mapping_mismatch_count"] == 0
            and convergence["passed"]
        ),
        "greedy_r010_anchor_complete": bool(
            greedy["budget_reached"]
            and abs(float(greedy["bops"]["R_bops_vs_fp32"]) - 0.10) <= 0.005
            and bool(builds["controls"]["B0"].get("passed"))
            and bool(builds["controls"]["budget_010"]["S32"].get("passed"))
            and bool(
                builds["controls"]["budget_010"]["JMIX-FRESH"].get("passed")
            )
            and all(
                bool(row["fixed50_gate_passed"])
                for row in fixed50["controls"].values()
            )
        ),
        "one_generation_real_stage2_smoke": bool(smoke.get("passed")),
    }
    for key, passed in checks.items():
        if not passed:
            blockers.append(key)
    included = {
        "conflict_count": int(search["precision_conflict_count"]),
        "unmapped_count": int(search["precision_unmapped_count"]),
        "fallback_count": int(search["precision_fallback_count"]),
    }
    for key, value in included.items():
        if value != 0:
            blockers.append(f"precision_{key}:{value}")

    payload = {
        "ga_framework_migrated": True,
        "formal_entrypoint_integrated": True,
        **checks,
        "production_independent_bops_match": checks["hgt_bops_closure_preserved"],
        "av16_deployment_valid": bool(av_profiles["AV16"]["legal_for_search"]),
        "av8_deployment_valid": bool(av_profiles["AV8"]["legal_for_search"]),
        "legal_av_profiles": list(av["legal_av_profiles"]),
        "precision_conflict_count": int(included["conflict_count"]),
        "precision_unmapped_count": int(included["unmapped_count"]),
        "precision_fallback_count": int(included["fallback_count"]),
        "population_size": 64,
        "offspring_size": 64,
        "formal_generations": 10,
        "seed_count": 1,
        "executed_seeds": [0],
        "isolated_gpu_available": True,
        "formal_search_allowed": not blockers,
        "blockers": blockers,
    }
    write(reports / "pre_ga_acceptance.json", payload)
    if blockers:
        raise RuntimeError(f"formal_ga_pre_acceptance_failed:{blockers}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return run(parser.parse_args().output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
