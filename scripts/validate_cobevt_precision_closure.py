#!/usr/bin/env python3
"""Rebuild one frozen CoBEVT Greedy phenotype through the deployment closure.

This diagnostic intentionally does not run Greedy or GA.  It reconstructs the
same deterministic search schema/Taylor ranking, verifies that the selected
historical genotype still has the same phenotype identity, then performs a
fresh physical -> train200 -> Q/DQ -> TensorRT build in a new artifact root.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.candidate import CandidateGenotype, CandidatePhenotype
from search.canonicalization import canonicalize_candidate
from search.ga.cnn_stage12_v3 import create_real_evaluator, write_json
from search.ga.stage12_v3 import phenotype_identity
from search.hashing import candidate_hash_payload
from search.ga.transformer_stage12_v3 import prepare_cobevt_search
from search.integration.runtime_environment import query_gpus


def _text(command: list[str]) -> str:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout


def _snapshot(*, physical_gpu: int) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "physical_gpu": int(physical_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpus": query_gpus(),
        "compute_processes": _text(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                "--format=csv,noheader",
            ]
        ).splitlines(),
        "pyramid_processes": _text(["pgrep", "-af", "pyramid|Pyramid"]).splitlines(),
    }


def run(args: argparse.Namespace) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            "cobevt_closure_cuda_visible_devices_mismatch:"
            f"observed={visible!r}:expected={args.physical_gpu}"
        )
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise RuntimeError(f"cobevt_closure_output_root_not_empty:{root}")
    for name in ("provenance", "reports", "proxy", "representative_anchor"):
        (root / name).mkdir(parents=True, exist_ok=True)

    # Match the formal CoBEVT runner exactly.  Domain-local rankings are part
    # of the physical phenotype and therefore RNG must be frozen before model
    # construction and Taylor/Fisher collection, not merely before GA.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    source_root = args.source_root.resolve()
    winners_path = source_root / "reports/greedy_exact_winners.json"
    winner_rows = json.loads(winners_path.read_text(encoding="utf-8"))
    winner = winner_rows[str(args.budget)]
    old_identity = dict(winner["identity"])
    old_candidate_dir = (
        source_root
        / f"ga/budget_{int(round(float(args.budget) * 100)):03d}"
        / "stage2_screening_cache"
        / old_identity["complete_phenotype_hash"]
    )
    old_physical_report_path = old_candidate_dir / "physical/unified_physical_report.json"
    old_physical_report = json.loads(
        old_physical_report_path.read_text(encoding="utf-8")
    )
    write_json(
        root / ("provenance/resume.json" if args.resume else "provenance/start.json"),
        {
            "branch": _text(["git", "-C", str(REPO), "branch", "--show-current"]).strip(),
            "head": _text(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip(),
            "source_root": str(source_root),
            "source_winner_manifest": str(winners_path),
            "source_candidate_dir": str(old_candidate_dir),
            "source_identity": old_identity,
            "source_physical_structure_hash": old_physical_report["structure_hash"],
            "source_state_dict_shape_hash": old_physical_report["state_dict_shape_hash"],
            "budget": float(args.budget),
            "taylor_samples": int(args.taylor_samples),
            "rng_seed": 0,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "resume": bool(args.resume),
            "gpu": _snapshot(physical_gpu=args.physical_gpu),
        },
    )

    status: dict[str, Any] = {
        "status": "failed",
        "budget": float(args.budget),
        "source_identity": old_identity,
        "source_physical_structure_hash": old_physical_report["structure_hash"],
        "source_state_dict_shape_hash": old_physical_report["state_dict_shape_hash"],
        "greedy_rerun": False,
        "formal_ga_run": False,
        "fresh_train200_required": True,
    }
    try:
        prepared = prepare_cobevt_search(
            output_root=root,
            physical_gpu=int(args.physical_gpu),
            plugin=args.plugin.resolve(),
            tensorrt_root=args.tensorrt_root.resolve(),
            taylor_samples=int(args.taylor_samples),
        )
        genotype = CandidateGenotype.from_dict(winner["genotype"])
        phenotype = canonicalize_candidate(genotype, prepared.space)
        current_identity = phenotype_identity(genotype, prepared.space)
        status["current_identity"] = current_identity
        identity_mismatches = {
            key: {"expected": old_identity[key], "observed": current_identity[key]}
            for key in old_identity
            if current_identity.get(key) != old_identity[key]
        }
        status["identity_mismatches"] = identity_mismatches
        old_phenotype = CandidatePhenotype.from_dict(
            json.loads((old_candidate_dir / "phenotype.json").read_text(encoding="utf-8"))
        )
        old_payload_in_current_space = candidate_hash_payload(
            old_phenotype, prepared.space
        )
        current_payload = candidate_hash_payload(phenotype, prepared.space)
        payload_mismatches = {
            key: {
                "historical_phenotype": old_payload_in_current_space.get(key),
                "current_phenotype": current_payload.get(key),
            }
            for key in sorted(set(old_payload_in_current_space) | set(current_payload))
            if old_payload_in_current_space.get(key) != current_payload.get(key)
        }
        semantic_identity_mismatches = {
            key: value
            for key, value in identity_mismatches.items()
            if key != "complete_phenotype_hash"
        }
        status.update(
            {
                "current_candidate_hash_payload": current_payload,
                "historical_phenotype_payload_in_current_space": (
                    old_payload_in_current_space
                ),
                "phenotype_payload_mismatches": payload_mismatches,
                "semantic_identity_mismatches": semantic_identity_mismatches,
                "complete_hash_rebound_for_current_provenance": bool(
                    not semantic_identity_mismatches
                    and not payload_mismatches
                    and current_identity["complete_phenotype_hash"]
                    != old_identity["complete_phenotype_hash"]
                ),
            }
        )
        ranking_only_rebind = bool(
            args.allow_historical_ranking_rebind
            and not semantic_identity_mismatches
            and set(payload_mismatches) == {"domain_width_expansion_hash"}
        )
        status["historical_coordinate_ranking_rebound"] = ranking_only_rebind
        status["historical_coordinate_ranking_reuse_forbidden"] = ranking_only_rebind
        if semantic_identity_mismatches or (payload_mismatches and not ranking_only_rebind):
            raise RuntimeError(
                "cobevt_historical_phenotype_semantic_drift:"
                f"identity={semantic_identity_mismatches}:payload={payload_mismatches}"
            )
        cache_manifest = json.loads(
            (
                root / "proxy/cobevt_formal_presearch_proxy_cache_manifest.json"
            ).read_text(encoding="utf-8")
        )
        if not cache_manifest.get("physical_ranking_frozen_across_resume", False):
            raise RuntimeError("cobevt_physical_ranking_cache_not_frozen")
        status["proxy_cache_manifest"] = cache_manifest

        evaluator = create_real_evaluator(
            prepared,
            output_root=root,
            # Build-only still constructs the evaluator config, whose protocol
            # validator requires positive values even though no AP run occurs.
            num_frames=50,
            warmup_frames=100,
            run_dir_name="representative_closure_runtime",
        )
        candidate_dir = root / "representative_anchor" / current_identity[
            "complete_phenotype_hash"
        ]
        result = evaluator.build_candidate_artifacts(
            phenotype,
            output_dir=candidate_dir,
            candidate_hash=current_identity["complete_phenotype_hash"],
        )
        current_physical = json.loads(
            (candidate_dir / "physical/unified_physical_report.json").read_text(
                encoding="utf-8"
            )
        )
        physical_exact = bool(
            current_physical["structure_hash"] == old_physical_report["structure_hash"]
            and current_physical["state_dict_shape_hash"]
            == old_physical_report["state_dict_shape_hash"]
        )
        if not physical_exact:
            raise RuntimeError(
                "cobevt_historical_physical_structure_drift:"
                f"expected={old_physical_report['structure_hash']}:"
                f"observed={current_physical['structure_hash']}"
            )
        calibration = json.loads(
            (candidate_dir / "calibration/calibration_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        precision = json.loads(
            (
                candidate_dir
                / "deployment/precision_realization_acceptance.json"
            ).read_text(encoding="utf-8")
        )
        entropy_build = dict(calibration.get("entropy_build") or {})
        requested_frames = int(
            entropy_build.get(
                "requested_frames",
                dict(entropy_build.get("dependencies") or {}).get(
                    "num_batches", entropy_build.get("calibration_sample_count", 0)
                ),
            )
        )
        processed_frames = int(
            entropy_build.get(
                "processed_frames", entropy_build.get("calibration_sample_count", 0)
            )
        )
        skipped_frames = int(
            entropy_build.get(
                "skipped_frames", max(requested_frames - processed_frames, 0)
            )
        )
        status.update(
            {
                "status": "ok",
                "candidate_dir": str(candidate_dir),
                "build_result": result,
                "physical_structure_exact": physical_exact,
                "physical_structure_hash": current_physical["structure_hash"],
                "state_dict_shape_hash": current_physical["state_dict_shape_hash"],
                "train200_requested": requested_frames,
                "train200_processed": processed_frames,
                "train200_skipped": skipped_frames,
                "precision_acceptance": bool(precision.get("passed")),
                "engine_sha256": result.get("engine_sha256", ""),
                "transformer_attention_fp32_acceptance": result.get(
                    "transformer_attention_fp32_acceptance"
                ),
                "transformer_functional_precision_acceptance": result.get(
                    "transformer_functional_precision_acceptance"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        status.update(
            {
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "failure_traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        status["gpu_after"] = _snapshot(physical_gpu=args.physical_gpu)
        write_json(root / "reports/representative_closure_validation.json", status)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--budget", default="0.1")
    result.add_argument("--physical-gpu", type=int, required=True)
    result.add_argument("--taylor-samples", type=int, default=32)
    result.add_argument("--allow-historical-ranking-rebind", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--plugin", type=Path, required=True)
    result.add_argument("--tensorrt-root", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
