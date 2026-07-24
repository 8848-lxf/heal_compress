#!/usr/bin/env python3
"""Build only B0, S32 and JMIX-FRESH engines for the new Greedy winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import _load
from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.run_v2xvit_greedy005_full import _baseline_candidate
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"weight_only_controls_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    identity = _build_full_space(model, adapter, hypes, batch)
    space = identity["space"]
    winner_payload = json.loads((args.output_root / "winner/v2xvit_greedy005_winner.json").read_text(encoding="utf-8"))
    winner = CandidateGenotype.from_dict(winner_payload["genotype"])
    components = identity["components"]
    qkv_paths = tuple(path for spec in components.attention_instances for path in (spec.q_projection_paths + spec.k_projection_paths))
    baseline = _baseline_candidate(space)
    controls = {
        "B0": baseline,
        "S32": CandidateGenotype(pruning_width_genes=dict(winner.pruning_width_genes), precision_genes={key: "FP32" for key in space.precision_gene_ids}, meta={"control": "S32"}),
        "JMIX-FRESH": winner,
    }
    reports: dict[str, Any] = {}
    for name, raw_candidate in controls.items():
        candidate = repair_genotype(raw_candidate, space)
        # ``repair_genotype`` is retained as the strict legalizer API, but a
        # control must never be silently moved to another width/precision
        # state.  The only tolerated difference is removal of legacy binary
        # pruning fields during canonicalization.
        if candidate.pruning_width_genes != raw_candidate.pruning_width_genes:
            raise RuntimeError(f"weight_only_control_width_repair:{name}")
        if candidate.precision_genes != raw_candidate.precision_genes:
            raise RuntimeError(f"weight_only_control_precision_repair:{name}")
        phenotype = canonicalize_candidate(candidate, space)
        candidate_id = candidate_hash(phenotype, space)
        physical = materialize_unified_widths(model, identity["cnn_units"], space.pruning_domains, candidate.pruning_width_genes, model_name="lidar_v2xvit")
        if not physical.report.passed or physical.report.requested_widths != physical.report.realized_widths:
            raise RuntimeError(f"weight_only_control_physical_mismatch:{name}:{physical.report.issues}")
        destination = args.output_root / "engines" / name
        destination.mkdir(parents=True, exist_ok=False)
        export = _export_candidate(destination, physical.model, adapter, batch, hypes, phenotype, candidate_id, physical.report.structure_hash, build_engine=True, tensorrt_root=args.tensorrt_root, plugin=args.plugin, calibration_frames=4, qkv_paths=qkv_paths, fixed_k_override=args.fixed_k, physical_gpu_id=args.physical_gpu)
        if not export.get("passed") or not export.get("engine", {}).get("passed"):
            raise RuntimeError(f"weight_only_control_engine_failed:{name}:{export.get('failure','')}")
        record = {"control": name, "candidate_hash": candidate_id, "structure_hash": physical.report.structure_hash, "state_dict_shape_hash": physical.report.state_dict_shape_hash, "precision_counts": {state: sum(value == state for value in phenotype.realized_precision_profile.values()) for state in ("FP32", "FP16", "INT8")}, "requested_widths": physical.report.requested_widths, "realized_widths": physical.report.realized_widths, "export": export, "diagnostic_control": False, "structural_repair_count": 0, "precision_repair_count": 0, "budget_projection_count": 0}
        write(destination / "control_report.json", record); reports[name] = record
        del physical; torch.cuda.empty_cache()
    write(args.output_root / "reports/control_engine_builds.json", {"controls": reports, "smoothquant_used": False, "top_k": 1})
    print(json.dumps({"status": "ok", "controls": list(reports)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--physical-gpu", type=int, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
