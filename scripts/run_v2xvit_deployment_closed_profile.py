#!/usr/bin/env python3
"""Fresh materialize/export/build one deployment-closed V2X-ViT profile."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash
from search.model_family.calibration_manifest import load_v2xvit_train_manifest
from search.model_family.deployment import load_v2xvit_pruning_domains
from search.pruning_space.unified_physical_pruner import materialize_unified_widths
from search.quantization_space.v2xvit_deployment_closed import (
    deployment_close_v2xvit_quantization_groups,
)
from search.stage2.v2xvit_candidate_deployer import export_build_candidate


DEFAULT_TRAIN200 = REPO / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"
DEFAULT_TRT = Path(
    "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118"
)
DEFAULT_PLUGIN = Path(
    "/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/"
    "pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
)


def _load_genotype(args: argparse.Namespace) -> dict[str, Any]:
    if args.precision_floor_artifact:
        payload = json.loads(args.precision_floor_artifact.read_text(encoding="utf-8"))
        try:
            return dict(payload["genotypes"][args.profile_id])
        except KeyError as exc:
            raise RuntimeError(f"precision_floor_profile_missing:{args.profile_id}") from exc
    if args.stage2_selection:
        payload = json.loads(args.stage2_selection.read_text(encoding="utf-8"))
        rows = list(payload.get("selected") or payload.get("candidates") or payload)
        if args.candidate_index < 0 or args.candidate_index >= len(rows):
            raise RuntimeError(f"stage2_candidate_index_out_of_range:{args.candidate_index}:{len(rows)}")
        row = dict(rows[args.candidate_index])
        return dict(row.get("genotype") or row.get("candidate", {}).get("genotype") or {})
    if args.genotype_json:
        payload = json.loads(args.genotype_json.read_text(encoding="utf-8"))
        return dict(payload.get("genotype") or payload)
    raise RuntimeError("candidate_genotype_source_missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--precision-floor-artifact", type=Path)
    sources.add_argument("--stage2-selection", type=Path)
    sources.add_argument("--genotype-json", type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument(
        "--override-precision",
        choices=("candidate", "all_fp32", "all_fp16"),
        default="candidate",
    )
    parser.add_argument("--search-space", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train200-manifest", type=Path, default=DEFAULT_TRAIN200)
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--tensorrt-root", type=Path, default=DEFAULT_TRT)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--functional-contract", choices=("P32", "F3"), default="F3")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"v2xvit_deploy_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    built = _build_full_space(model, adapter, hypes, batch)
    space = replace(
        built["space"],
        quantization_groups=deployment_close_v2xvit_quantization_groups(
            built["space"].quantization_groups
        ),
    )
    if args.search_space:
        domains = tuple(load_v2xvit_pruning_domains(args.search_space))
        if {row.domain_id for row in domains} != {row.domain_id for row in space.pruning_domains}:
            raise RuntimeError("frozen_search_space_domain_identity_mismatch")
        space = replace(space, pruning_domains=domains)
    raw = CandidateGenotype.from_dict(_load_genotype(args))
    if args.override_precision != "candidate":
        groups = {group.group_id: group for group in space.quantization_groups}
        precision = {}
        for group_id in space.precision_gene_ids:
            allowed = {str(value).upper() for value in groups[group_id].allowed_precisions}
            if args.override_precision == "all_fp32":
                value = "FP32"
            else:
                value = "FP16" if "FP16" in allowed else "FP32"
            if value not in allowed:
                raise RuntimeError(
                    f"precision_override_not_legal:{group_id}:{value}:{sorted(allowed)}"
                )
            precision[group_id] = value
        raw = CandidateGenotype(
            pruning_width_genes=dict(raw.pruning_width_genes),
            precision_genes=precision,
            meta={**dict(raw.meta), "precision_override": args.override_precision},
        )
    candidate = repair_genotype(raw, space)
    phenotype = canonicalize_candidate(candidate, space)
    physical = materialize_unified_widths(
        model,
        built["cnn_units"],
        space.pruning_domains,
        candidate.pruning_width_genes,
        model_name="lidar_v2xvit",
    )
    if not physical.report.passed:
        raise RuntimeError(f"v2xvit_physical_materialization_failed:{physical.report.issues}")
    with torch.inference_mode():
        outputs = adapter.forward_for_task(physical.model, batch)
    if not outputs or not all(
        (not value.is_floating_point()) or bool(torch.isfinite(value).all().item())
        for value in outputs.values()
        if torch.is_tensor(value)
    ):
        raise RuntimeError("v2xvit_physical_forward_nonfinite")
    manifest = load_v2xvit_train_manifest(args.train200_manifest)
    identity = {
        "profile_id": str(args.profile_id),
        "candidate_hash": candidate_hash(phenotype, space),
        "genotype": candidate.to_dict(),
        "phenotype": phenotype.to_dict(),
        "config_path": str(MODEL_SPECS["v2xvit"]["config"]),
        "checkpoint_path": str(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "physical_gpu": int(args.physical_gpu),
        "fresh_export": True,
        "fresh_engine": not bool(args.export_only),
    }
    result = export_build_candidate(
        candidate_dir=args.output_dir,
        model=physical.model,
        adapter=adapter,
        hypes=hypes,
        real_batch=batch,
        module_precision_profile=phenotype.realized_precision_profile,
        candidate_identity=identity,
        train200_manifest=manifest,
        train200_manifest_path=args.train200_manifest,
        checkpoint_path=MODEL_SPECS["v2xvit"]["checkpoint"],
        physical_report=physical.report.to_dict(),
        fixed_k=int(args.fixed_k),
        physical_gpu=int(args.physical_gpu),
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin,
        build_engine=not bool(args.export_only),
        functional_contract=args.functional_contract,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] in {"ok", "exported"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
