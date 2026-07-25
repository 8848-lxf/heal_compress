#!/usr/bin/env python3
"""Audit all-keep V2X-ViT precision-only BOPS and mixed-weight floors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash
from search.quantization_space.v2xvit_deployment_closed import (
    deployment_close_v2xvit_quantization_groups,
)


def _precision(group: Any, maximum_bits: int, *, buildable_int8: set[str] | None) -> str:
    allowed = {str(value).upper() for value in group.allowed_precisions}
    if maximum_bits <= 8 and "INT8" in allowed:
        if buildable_int8 is None or group.group_id in buildable_int8:
            return "INT8"
    if maximum_bits <= 16 and "FP16" in allowed:
        return "FP16"
    if "FP32" not in allowed:
        raise RuntimeError(f"precision_floor_missing_fp32:{group.group_id}:{sorted(allowed)}")
    return "FP32"


def _candidate(space: Any, *, maximum_bits: int, buildable_int8: set[str] | None, label: str):
    groups = {group.group_id: group for group in space.quantization_groups}
    return repair_genotype(
        CandidateGenotype(
            pruning_width_genes={
                domain.domain_id: int(domain.original_width)
                for domain in space.pruning_domains
            },
            precision_genes={
                group_id: _precision(
                    groups[group_id], maximum_bits, buildable_int8=buildable_int8
                )
                for group_id in space.precision_gene_ids
            },
            meta={"created_by": label},
        ),
        space,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buildable-int8-loci", type=Path)
    parser.add_argument("--artifact-prefix", default="precision_only_floor")
    parser.add_argument("--deployment-closed", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"precision_floor_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    built = _build_full_space(model, adapter, hypes, batch)
    space = built["space"]
    buildable_loci: set[str] | None = None
    buildable_source = "deployment_closure_pending"
    if args.buildable_int8_loci:
        payload = json.loads(args.buildable_int8_loci.read_text(encoding="utf-8"))
        buildable_loci = {str(value) for value in payload["buildable_int8_loci"]}
        unknown = sorted(buildable_loci - set(space.precision_gene_ids))
        if unknown:
            raise RuntimeError(f"unknown_buildable_int8_loci:{unknown}")
        buildable_source = str(args.buildable_int8_loci.resolve())
    if args.deployment_closed:
        space = replace(
            space,
            quantization_groups=deployment_close_v2xvit_quantization_groups(
                space.quantization_groups,
                buildable_int8_group_ids=buildable_loci,
            ),
        )

    specifications = [
        ("P32", 32, set()),
        ("P16-max", 16, set()),
        ("P8-max-requested", 8, None),
    ]
    if buildable_loci is not None:
        specifications.append(("P8-max-buildable", 8, buildable_loci))
    rows = []
    genotypes = {}
    for name, bits, allowed_int8 in specifications:
        candidate = _candidate(
            space,
            maximum_bits=bits,
            buildable_int8=allowed_int8,
            label=f"precision_only_floor::{name}",
        )
        phenotype = canonicalize_candidate(candidate, space)
        bops = built["bops"].evaluate_breakdown(phenotype)
        size = built["size"].evaluate_breakdown(phenotype)
        counts = {
            value: sum(
                precision == value
                for precision in phenotype.realized_precision_profile.values()
            )
            for value in ("FP32", "FP16", "INT8")
        }
        row = {
            "profile": name,
            "candidate_hash": candidate_hash(phenotype, space),
            "R_BOPS": float(bops["R_bops_vs_fp32"]),
            "BOPS": float(bops["bops_total"]),
            "mixed_weight_retention": float(size["R_size_vs_fp32"]),
            "mixed_weight_compression": 1.0 / float(size["R_size_vs_fp32"]),
            "parameter_retention": float(size["R_parameter_retention"]),
            "parameter_count": float(size["parameter_count_after"]),
            "fp32_count": counts["FP32"],
            "fp16_count": counts["FP16"],
            "int8_count": counts["INT8"],
            "structure_all_keep": True,
            "engine_build_status": "pending_deployment_closure",
        }
        rows.append(row)
        genotypes[name] = candidate.to_dict()

    requested = next(row for row in rows if row["profile"] == "P8-max-requested")
    buildable = next((row for row in rows if row["profile"] == "P8-max-buildable"), None)
    report = {
        "schema_version": "v2xvit-precision-only-floor-v1",
        "model": "V2X-ViT",
        "checkpoint": str(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "config": str(MODEL_SPECS["v2xvit"]["config"]),
        "config_sha256": _sha256(MODEL_SPECS["v2xvit"]["config"]),
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "target_R_BOPS": 0.30,
        "legal_band": [0.295, 0.305],
        "R_BOPS_precision_only_floor_requested": requested["R_BOPS"],
        "R_BOPS_precision_only_floor_buildable": None if buildable is None else buildable["R_BOPS"],
        "target030_reached_by_requested_quantization_only": 0.295 <= requested["R_BOPS"] <= 0.305,
        "target030_reached_by_buildable_quantization_only": None if buildable is None else 0.295 <= buildable["R_BOPS"] <= 0.305,
        "buildable_loci_source": buildable_source,
        "rows": rows,
        "genotypes": genotypes,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = args.artifact_prefix
    _write_json(output / f"{stem}.json", report)
    with (output / f"{stem}.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# V2X-ViT precision-only BOPS floor",
        "",
        f"- requested INT8 floor: `{requested['R_BOPS']:.12f}`",
        f"- buildable INT8 floor: `{None if buildable is None else format(buildable['R_BOPS'], '.12f')}`",
        f"- target 0.30 reachable by requested quantization only: `{report['target030_reached_by_requested_quantization_only']}`",
        f"- target 0.30 reachable by buildable quantization only: `{report['target030_reached_by_buildable_quantization_only']}`",
        "",
        "Buildable status is deliberately unresolved until fresh strongly typed TensorRT closure completes.",
    ]
    (output / f"{stem}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "requested_floor": requested["R_BOPS"], "buildable_floor": None if buildable is None else buildable["R_BOPS"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
