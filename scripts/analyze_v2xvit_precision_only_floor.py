#!/usr/bin/env python3
"""Compute all-keep V2X-ViT precision-only requested BOPS floors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.audit_heal_transformer_search_models import _load
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.hashing import candidate_hash


def _state(group: object, requested: str) -> str:
    allowed = tuple(str(value) for value in group.allowed_precisions)
    if requested == "FP32":
        return "FP32"
    if requested == "FP16":
        return "FP16" if "FP16" in allowed else "FP32"
    if "INT8" in allowed:
        return "INT8"
    if "FP16" in allowed:
        return "FP16"
    return "FP32"


def run(args: argparse.Namespace) -> int:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"precision_floor_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    built = _build_full_space(model, adapter, hypes, batch)
    space = built["space"]
    original_widths = {domain.domain_id: int(domain.original_width) for domain in space.pruning_domains}
    rows = []
    payloads = {}
    for name, requested in (("P32", "FP32"), ("P16-max", "FP16"), ("P8-max-requested", "INT8")):
        genes = {group.group_id: _state(group, requested) for group in space.quantization_groups if group.group_id in space.precision_gene_ids}
        candidate = CandidateGenotype(pruning_width_genes=original_widths, precision_genes=genes, meta={"precision_floor_profile": name})
        phenotype = canonicalize_candidate(candidate, space)
        bops = built["bops"].evaluate_breakdown(phenotype)
        size = built["size"].evaluate_breakdown(phenotype)
        counts = {state: sum(value == state for value in genes.values()) for state in ("INT8", "FP16", "FP32")}
        record = {
            "profile": name,
            "candidate_hash": candidate_hash(phenotype, space),
            "requested_precision_counts": counts,
            "buildable_precision_counts": None,
            "BOPS": float(bops["bops_total"]),
            "R_BOPS": float(bops["R_bops_vs_fp32"]),
            "mixed_weight_size_bytes": float(size["size_bits_total"]) / 8.0,
            "mixed_weight_retention": float(size["R_size_vs_fp32"]),
            "theoretical_weight_compression": 1.0 / float(size["R_size_vs_fp32"]),
            "engine_build_status": "pending_step4",
            "requested_realized_exact": None,
            "fixed50_mAP": None,
            "latency_p50_ms": None,
            "genotype": candidate.to_dict(),
            "bops_breakdown": bops,
            "size_breakdown": size,
        }
        payloads[name] = record
        rows.append({key: value for key, value in record.items() if key not in ("genotype", "bops_breakdown", "size_breakdown")})
    requested_floor = payloads["P8-max-requested"]["R_BOPS"]
    report = {
        "schema_version": "v2xvit-precision-only-floor-v1",
        "model": "lidar_v2xvit",
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "target_bops_retention": 0.30,
        "target_band": [0.295, 0.305],
        "profiles": payloads,
        "R_BOPS_precision_only_floor_requested": requested_floor,
        "R_BOPS_precision_only_floor_buildable": None,
        "target030_reachable_by_requested_quantization_only": 0.295 <= requested_floor <= 0.305,
        "target030_requires_structural_pruning_from_requested_floor": requested_floor > 0.305,
        "target030_requires_structural_pruning_from_buildable_floor": None,
        "remaining_structural_bops_reduction_from_buildable_floor": None,
        "buildable_floor_pending_step4": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "precision_only_floor.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    with (args.output_root / "precision_only_floor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_root / "precision_only_floor.md").write_text(
        "# V2X-ViT precision-only BOPS floor\n\n"
        f"Requested P8 all-keep floor: `{requested_floor}`. Buildable floor remains pending the TensorRT bisection and is not inferred from the requested profile.\n",
        encoding="utf-8",
    )
    print(json.dumps({"requested_floor": requested_floor, "profiles": {key: value["R_BOPS"] for key, value in payloads.items()}}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    visible = str(args.physical_gpu)
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise RuntimeError(f"physical_gpu_visibility_mismatch:{os.environ.get('CUDA_VISIBLE_DEVICES')}!={visible}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
