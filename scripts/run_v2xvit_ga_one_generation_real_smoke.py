#!/usr/bin/env python3
"""Exercise one real generation-1 Stage-2 worker before formal GA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_v2xvit_six_budget_exact_winners import frozen_domains
from scripts.run_v2xvit_ga_stage2_worker import domain_from_payload
from search.candidate import CandidatePhenotype


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    reports = root / "reports"
    exact = json.loads((reports / "greedy_r010_winner.json").read_text())
    spec = json.loads((reports / "final_search_space_spec.json").read_text())
    request = json.loads(args.request_json.read_text())
    phenotype = CandidatePhenotype.from_dict(exact["phenotype"])
    domains = tuple(domain_from_payload(row) for row in spec["domains"])
    domains = frozen_domains(domains, exact["phenotype"])
    qkv_paths: list[str] = []
    for group in spec["groups"]:
        role = str(group.get("metadata", {}).get("transformer_role", ""))
        if role in {"qk_projection", "fused_qkv_projection"}:
            qkv_paths.extend(str(path) for path in group.get("module_paths", ()))
    qkv_paths = list(dict.fromkeys(qkv_paths))
    if not qkv_paths:
        raise RuntimeError("ga_real_smoke_qkv_inventory_empty")

    smoke_root = root / "stage2/one_generation_real_smoke"
    complete_hash = str(exact["candidate_hash"])
    job = {
        "root": str(smoke_root),
        "label": "010",
        "seed": 0,
        "generation": 1,
        "genotype": exact["genotype"],
        "complete_phenotype_hash": complete_hash,
        "phenotype": phenotype.to_dict(),
        "domains": [domain.to_dict() for domain in domains],
        "qkv_paths": qkv_paths,
        "request": request,
        "fixed50_manifest": str(args.fixed50_manifest.resolve()),
        "plugin": str(args.plugin.resolve()),
        "tensorrt_root": str(args.tensorrt_root.resolve()),
        "physical_gpu": int(args.physical_gpu),
    }
    job_path = smoke_root / "generation_01/worker/request.json"
    write(job_path, job)
    log_path = smoke_root / "generation_01/worker/worker.log"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/run_v2xvit_ga_stage2_worker.py"),
                "--request",
                str(job_path),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = (
        smoke_root
        / f"ga/stage2_cache/budget_010/{complete_hash}/stage2_result.json"
    )
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    calibration_path = (
        smoke_root
        / f"ga/stage2_cache/budget_010/{complete_hash}/JMIX-FRESH/calibration_manifest.json"
    )
    calibration = json.loads(calibration_path.read_text()) if calibration_path.is_file() else {}
    greedy_fixed50 = json.loads((reports / "greedy_r010_fixed50.json").read_text())
    greedy_map = float(greedy_fixed50["controls"]["Greedy-JMIX"]["mAP"])
    observed_map = result.get("mAP")
    eligible = bool(
        observed_map is not None and float(observed_map) >= greedy_map - 0.005
    )
    passed = bool(
        completed.returncode == 0
        and result.get("status") == "ok"
        and result.get("requested_realized_exact")
        and int(result.get("evaluated", -1)) == 50
        and int(result.get("skipped", -1)) == 0
        and int(calibration.get("processed_frames", -1)) == 200
        and int(calibration.get("skipped_frames", -1)) == 0
        and eligible
    )
    payload = {
        "schema_version": "v2xvit-ga-one-generation-real-stage2-smoke-v1",
        "generation": 1,
        "generation_zero_counted": False,
        "candidate_role": "V1_exact_greedy_anchor_pipeline_replay",
        "candidate_hash": complete_hash,
        "subprocess_returncode": int(completed.returncode),
        "stage2_result": result,
        "fresh_train200_processed": int(calibration.get("processed_frames", 0)),
        "fresh_train200_skipped": int(calibration.get("skipped_frames", 0)),
        "greedy_anchor_map": greedy_map,
        "accuracy_gate": greedy_map - 0.005,
        "eligible": eligible,
        "real_feedback_hashes": [complete_hash] if eligible else [],
        "passed": passed,
        "formal_generation_consumed": False,
        "formal_stage2_cache_reused": False,
    }
    write(reports / "ga_one_generation_real_stage2_smoke.json", payload)
    if not passed:
        raise RuntimeError(f"ga_one_generation_real_stage2_smoke_failed:{payload}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--fixed50-manifest", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
