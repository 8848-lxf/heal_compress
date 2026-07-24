#!/usr/bin/env python3
"""Export/build S32, S16 and freshly calibrated JMIX diagnostic controls."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from scripts.run_v2xvit_greedy005_full import _formal_space
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate, _frozen_domain
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    random.seed(20260724); np.random.seed(20260724); torch.manual_seed(20260724); torch.cuda.manual_seed_all(20260724)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    identity = _build_full_space(model, adapter, hypes, batch)
    manifest = json.loads((source / "greedy/v2xvit_greedy_search_manifest.json").read_text(encoding="utf-8"))
    formal = _formal_space(model, adapter, hypes, batch, identity, str(manifest["calibration_manifest_hash"]))
    ranking = json.loads((source / "rankings/v2xvit_fixed_rankings.json").read_text(encoding="utf-8"))
    domains = tuple(_frozen_domain(row) for row in ranking["domains"])
    formal["space"] = replace(formal["space"], pruning_domains=domains)
    winner = json.loads((source / "greedy/v2xvit_greedy_winner_config.json").read_text(encoding="utf-8"))
    winner_genotype = CandidateGenotype.from_dict(winner["genotype"])
    qkv_paths = tuple(path for spec in formal["components"].attention_instances for path in (spec.q_projection_paths + spec.k_projection_paths))
    reports = {}
    for mode in args.controls:
        control_name = "JMIX-FRESH" if mode == "JMIX-FRESH" else mode
        control_dir = Path(json.loads((output / "controls/control_manifest.json").read_text(encoding="utf-8"))["controls"][control_name]["control_dir"])
        control = json.loads((control_dir / "control.json").read_text(encoding="utf-8"))
        widths = dict(control["genotype"]["pruning_width_genes"])
        if mode == "S32":
            precision = {gene: "FP32" for gene in formal["space"].precision_gene_ids}
        elif mode == "S16":
            precision = {gene: "FP16" for gene in formal["space"].precision_gene_ids}
        elif mode == "JMIX-FRESH":
            precision = dict(winner_genotype.precision_genes)
        else:
            raise RuntimeError(f"unknown_backend_control:{mode}")
        candidate = repair_genotype(CandidateGenotype(pruning_width_genes=widths, precision_genes=precision, meta={"created_by": "accuracy_attribution", "diagnostic_control": True}), formal["space"])
        phenotype = canonicalize_candidate(candidate, formal["space"])
        realized_hash = candidate_hash(phenotype, formal["space"])
        destination = output / "tensorrt" / mode.replace("-", "_")
        destination.mkdir(parents=True, exist_ok=False)
        physical = materialize_unified_widths(model, identity["cnn_units"], domains, candidate.pruning_width_genes, model_name="lidar_v2xvit")
        if not physical.report.passed or physical.report.requested_widths != physical.report.realized_widths:
            raise RuntimeError(f"backend_control_physical_failed:{mode}:{physical.report.issues}")
        saved = torch.load(control_dir / "physical_state_dict.pth", map_location="cpu")
        physical.model.load_state_dict(saved["model"], strict=True)
        export = _export_candidate(destination, physical.model, adapter, batch, hypes, phenotype, realized_hash, physical.report.structure_hash, build_engine=True, tensorrt_root=args.tensorrt_root, plugin=args.plugin, calibration_frames=args.calibration_frames, qkv_paths=qkv_paths, fixed_k_override=args.fixed_k)
        if not export.get("passed") or not export.get("engine", {}).get("passed"):
            raise RuntimeError(f"backend_control_export_failed:{mode}:{export.get('failure','')}")
        calibration = json.loads((destination / "calibration_manifest.json").read_text(encoding="utf-8"))
        if calibration.get("physical_structure_hash") != physical.report.structure_hash or not calibration.get("precision_map_hash"):
            raise RuntimeError(f"backend_control_calibration_binding_failed:{mode}")
        record = {
            "mode": mode,
            "diagnostic_control": True,
            "candidate_hash": realized_hash,
            "structure_hash": physical.report.structure_hash,
            "precision_counts": {state: sum(value == state for value in phenotype.realized_precision_profile.values()) for state in ("FP32", "FP16", "INT8")},
            "calibration_manifest_hash": calibration["manifest_hash"],
            "physical_structure_hash": calibration["physical_structure_hash"],
            "precision_map_hash": calibration["precision_map_hash"],
            "export": export,
        }
        write(destination / "control_backend_report.json", record)
        reports[mode] = record
        del physical
        torch.cuda.empty_cache()
    summary_path = output / "reports/phase2_backend_builds.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        prior.setdefault("controls", {}).update(reports)
        summary_path.unlink()
        write(summary_path, prior)
    else:
        write(summary_path, {"schema_version": "v2xvit-greedy005-backend-controls-v1", "diagnostic_control": True, "controls": reports, "smoothquant_used": False})
    print(json.dumps({"status": "ok", "controls": list(reports), "smoothquant_used": False}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controls", nargs="+", default=["S32", "S16", "JMIX-FRESH"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--calibration-frames", type=int, default=4)
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
