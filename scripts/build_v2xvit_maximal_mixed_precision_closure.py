#!/usr/bin/env python3
"""Build the single representative V2X-ViT maximal-mixed closure engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.audit_heal_transformer_search_models import _load
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.hashing import candidate_hash
from search.model_family.deployment import build_physical_structure_snapshot_v2


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(
            f"maximal_mixed_gpu_visibility_mismatch:"
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}!={args.physical_gpu}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"maximal_mixed_requires_one_visible_gpu:{torch.cuda.device_count()}"
        )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(
        adapter, hypes, device
    )
    built = _build_full_space(model, adapter, hypes, batch)
    space = built["space"]
    groups = {group.group_id: group for group in space.quantization_groups}
    precision_genes = {}
    for group_id in space.precision_gene_ids:
        allowed = tuple(groups[group_id].allowed_precisions)
        precision_genes[group_id] = (
            "INT8" if "INT8" in allowed else "FP16" if "FP16" in allowed else "FP32"
        )
    genotype = CandidateGenotype(
        pruning_width_genes={
            domain.domain_id: int(domain.original_width)
            for domain in space.pruning_domains
        },
        precision_genes=precision_genes,
        meta={
            "purpose": "representative_maximal_mixed_precision_closure",
            "audit_only": True,
            "formal_search_candidate": False,
        },
    )
    phenotype = canonicalize_candidate(genotype, space)
    identity = candidate_hash(phenotype, space)
    snapshot = build_physical_structure_snapshot_v2(
        model, model_family="heal_lidar_v2xvit"
    )
    qkv_paths = tuple(
        path
        for spec in built["components"].attention_instances
        for path in spec.q_projection_paths + spec.k_projection_paths
    )
    bops = built["bops"].evaluate_breakdown(phenotype)
    manifest = {
        "schema_version": "v2xvit-maximal-mixed-precision-closure-v1",
        "audit_only": True,
        "formal_search_candidate": False,
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "candidate_hash": identity,
        "physical_hash": snapshot["snapshot_hash"],
        "bops_formula_version": bops.get("bops_formula_version"),
        "bops_retention": bops["R_bops_vs_fp32"],
        "precision_gene_count": len(space.precision_gene_ids),
        "precision_counts": {
            state: sum(value == state for value in precision_genes.values())
            for state in ("INT8", "FP16", "FP32")
        },
        "constant_precision_group_count": len(space.constant_precision_group_ids),
        "genotype": genotype.to_dict(),
        "phenotype": phenotype.to_dict(),
        "physical_snapshot": snapshot,
    }
    _write(output / "maximal_mixed_manifest.json", manifest)
    exported = _export_candidate(
        output,
        model,
        adapter,
        batch,
        hypes,
        phenotype,
        identity,
        snapshot["snapshot_hash"],
        build_engine=True,
        tensorrt_root=args.tensorrt_root,
        plugin=args.plugin,
        calibration_frames=200,
        qkv_paths=qkv_paths,
        fixed_k_override=args.fixed_k,
        physical_gpu_id=args.physical_gpu,
    )
    _write(output / "maximal_mixed_export_result.json", exported)
    functional = dict(
        (exported.get("engine") or {}).get("functional_precision") or {}
    )
    build = dict((exported.get("engine") or {}).get("build") or {})
    precision = dict(build.get("precision_realization_validation") or {})
    acceptance = {
        "schema_version": "v2xvit-maximal-mixed-closure-acceptance-v1",
        "passed": bool(
            exported.get("passed")
            and (exported.get("engine") or {}).get("passed")
            and precision.get("passed")
            and functional.get("passed")
            and int(functional.get("conflict_count", 1)) == 0
            and int(functional.get("fallback_count", 1)) == 0
            and int(functional.get("unmapped_count", 1)) == 0
        ),
        "weighted_requested_realized_exact": bool(precision.get("passed")),
        "functional_requested_realized_exact": bool(functional.get("passed")),
        "conflict_count": int(functional.get("conflict_count", 0))
        + len(precision.get("mismatches", [])),
        "fallback_count": int(functional.get("fallback_count", 0)),
        "unmapped_count": int(functional.get("unmapped_count", 0))
        + int(precision.get("unresolved_layer_count", 0)),
        "engine_path": str(output / "candidate.plan"),
        "candidate_hash": identity,
        "physical_hash": snapshot["snapshot_hash"],
    }
    _write(output / "maximal_mixed_closure_acceptance.json", acceptance)
    print(json.dumps(acceptance, sort_keys=True), flush=True)
    if not acceptance["passed"]:
        raise RuntimeError(f"maximal_mixed_precision_closure_failed:{acceptance}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    parser.add_argument("--fixed-k", type=int, default=27904)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
