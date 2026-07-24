#!/usr/bin/env python3
"""Build strict and precision-only controls for the frozen Greedy winner."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_heal_transformer_search_models import _load
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.run_v2xvit_greedy005_full import _formal_space
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate, _frozen_domain
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    output = root / "structures" / "v2xvit_greedy005_controls"
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    identity = _build_full_space(model, adapter, hypes, batch)
    manifest = json.loads((root / "greedy" / "v2xvit_greedy_search_manifest.json").read_text(encoding="utf-8"))
    formal = _formal_space(model, adapter, hypes, batch, identity, str(manifest["calibration_manifest_hash"]))
    ranking = json.loads((root / "rankings" / "v2xvit_fixed_rankings.json").read_text(encoding="utf-8"))
    frozen_domains = tuple(_frozen_domain(row) for row in ranking["domains"])
    formal["space"] = replace(formal["space"], pruning_domains=frozen_domains)
    winner = json.loads((root / "greedy" / "v2xvit_greedy_winner_config.json").read_text(encoding="utf-8"))
    winner_genotype = CandidateGenotype.from_dict(winner["genotype"])
    groups = {group.group_id: group for group in formal["space"].quantization_groups}
    qkv_paths = tuple(path for spec in formal["components"].attention_instances for path in (spec.q_projection_paths + spec.k_projection_paths))
    reports = []
    for mode in ("strict_baseline", "precision_only"):
        widths = {domain.domain_id: int(domain.original_width) for domain in formal["space"].pruning_domains}
        precision = {group_id: "FP32" for group_id in formal["space"].precision_gene_ids}
        if mode == "precision_only":
            precision = dict(winner_genotype.precision_genes)
        candidate = repair_genotype(CandidateGenotype(pruning_width_genes=widths, precision_genes=precision, meta={"created_by": mode}), formal["space"])
        phenotype = canonicalize_candidate(candidate, formal["space"])
        identity_hash = candidate_hash(phenotype, formal["space"])
        candidate_dir = output / mode
        candidate_dir.mkdir(parents=True, exist_ok=False)
        physical = materialize_unified_widths(model, identity["cnn_units"], formal["space"].pruning_domains, candidate.pruning_width_genes, model_name="lidar_v2xvit")
        if not physical.report.passed:
            raise RuntimeError(f"control_physical_failed:{mode}:{physical.report.issues}")
        physical.model.eval()
        with torch.inference_mode():
            out = adapter.forward_for_task(physical.model, batch)
        finite = all((not x.is_floating_point()) or bool(torch.isfinite(x).all().item()) for x in _tensors(out))
        _write(candidate_dir / "requested_vs_realized.json", {"mode": mode, "candidate_hash": identity_hash, "requested_widths": physical.report.requested_widths, "realized_widths": physical.report.realized_widths, "exact": physical.report.requested_widths == physical.report.realized_widths, "structure_hash": physical.report.structure_hash, "finite_forward": finite})
        export = _export_candidate(candidate_dir, physical.model, adapter, batch, hypes, phenotype, identity_hash, physical.report.structure_hash, build_engine=True, tensorrt_root=args.tensorrt_root, plugin=args.plugin, calibration_frames=args.calibration_frames, qkv_paths=qkv_paths, fixed_k_override=args.fixed_k)
        reports.append({"mode": mode, "candidate_hash": identity_hash, "structure_hash": physical.report.structure_hash, "parameter_count": physical.report.physical_parameter_count, "finite_forward": finite, "export": export})
        del physical
        torch.cuda.empty_cache()
    _write(root / "reports" / "v2xvit_greedy005_controls_summary.json", {"schema_version": "v2xvit-greedy005-controls-v1", "reports": reports, "full1789_executed": False, "formal_ga_search_executed": False})
    print(json.dumps(reports, indent=2, sort_keys=True), flush=True)
    return 0


def _tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value, dict):
        for child in value.values(): yield from _tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value: yield from _tensors(child)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--calibration-frames", type=int, default=4)
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path, required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
