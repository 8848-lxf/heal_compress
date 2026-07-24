#!/usr/bin/env python3
"""Materialize diagnostic physical controls from the frozen Greedy winner."""

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
from scripts.run_v2xvit_greedy005_stage2 import _frozen_domain
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash as canonical_candidate_hash
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _candidate_with_widths(winner: CandidateGenotype, domains: tuple[Any, ...], winner_widths: dict[str, int], role: str) -> CandidateGenotype:
    widths: dict[str, int] = {}
    for domain in domains:
        domain_id = str(domain.domain_id)
        domain_type = str(domain.domain_type)
        use_winner = role == "full_winner" or (role == "cnn_only" and domain_type == "cnn_channel") or (role == "attention_only" and domain_type == "attention_dh") or (role == "ffn_only" and domain_type == "ffn_hidden")
        module_path = str(getattr(domain, "root_module_path", ""))
        family = str(getattr(domain, "family", ""))
        if role == "cnn_attention" and domain_type in {"cnn_channel", "attention_dh"}:
            use_winner = True
        if role == "cnn_ffn" and domain_type in {"cnn_channel", "ffn_hidden"}:
            use_winner = True
        if role == "attention_ffn" and domain_type in {"attention_dh", "ffn_hidden"}:
            use_winner = True
        if role.startswith("cnn_stage"):
            stage = role.removeprefix("cnn_stage")
            use_winner = domain_type == "cnn_channel" and (
                f"blocks.{stage}." in domain_id or (stage == "shrinker" and "shrinker_m1" in domain_id)
            )
        if role.startswith("full_restore_"):
            restore = role.removeprefix("full_restore_")
            use_winner = role == "full_winner" or not (
                domain_type == "cnn_channel" and (
                    f"blocks.{restore}." in domain_id or (restore == "shrinker" and "shrinker_m1" in domain_id)
                )
            )
        if role.startswith("attention_family_"):
            requested_family = role.removeprefix("attention_family_")
            use_winner = domain_type == "attention_dh" and family == requested_family
        if role.startswith("attention_layer"):
            layer = role.removeprefix("attention_layer")
            use_winner = domain_type == "attention_dh" and f"encoder.layers.{layer}." in module_path
        widths[domain_id] = int(winner_widths[domain_id] if use_winner else domain.original_width)
    return replace(winner, pruning_width_genes=widths)


def run(args: argparse.Namespace) -> int:
    root = args.source_root.resolve()
    out = args.output_root.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("accuracy_controls_require_visible_cuda0")
    torch.cuda.set_device(device)
    random.seed(20260724); np.random.seed(20260724); torch.manual_seed(20260724); torch.cuda.manual_seed_all(20260724)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch = __import__("scripts.smoke_transformer_unified_search", fromlist=["_multi_agent_validation_batch"])._multi_agent_validation_batch(adapter, hypes, device)[0]
    identity = _build_full_space(model, adapter, hypes, batch)
    manifest = json.loads((root / "greedy/v2xvit_greedy_search_manifest.json").read_text(encoding="utf-8"))
    formal = _formal_space(model, adapter, hypes, batch, identity, str(manifest["calibration_manifest_hash"]))
    frozen = json.loads((root / "rankings/v2xvit_fixed_rankings.json").read_text(encoding="utf-8"))
    domains = tuple(_frozen_domain(row) for row in frozen["domains"])
    formal["space"] = replace(formal["space"], pruning_domains=domains)
    winner_payload = json.loads((root / "greedy/v2xvit_greedy_winner_config.json").read_text(encoding="utf-8"))
    winner = CandidateGenotype.from_dict(winner_payload["genotype"])
    winner = repair_genotype(winner, formal["space"])
    winner_widths = dict(winner.pruning_width_genes)
    roles = {
        "S32": {"role": "full_winner", "precision_mode": "all_fp32", "diagnostic_control": True},
        "S16": {"role": "full_winner", "precision_mode": "all_fp16", "diagnostic_control": True},
        "JMIX-FRESH": {"role": "full_winner", "precision_mode": "winner_mixed", "diagnostic_control": True},
        "CNN-only": {"role": "cnn_only", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-only": {"role": "attention_only", "precision_mode": "all_fp32", "diagnostic_control": True},
        "FFN-only": {"role": "ffn_only", "precision_mode": "all_fp32", "diagnostic_control": True},
        "CNN+Attention": {"role": "cnn_attention", "precision_mode": "all_fp32", "diagnostic_control": True},
        "CNN+FFN": {"role": "cnn_ffn", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention+FFN": {"role": "attention_ffn", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Full-winner": {"role": "full_winner", "precision_mode": "winner_mixed", "diagnostic_control": True},
        "CNN-stage0": {"role": "cnn_stage0", "precision_mode": "all_fp32", "diagnostic_control": True},
        "CNN-stage1": {"role": "cnn_stage1", "precision_mode": "all_fp32", "diagnostic_control": True},
        "CNN-stage2": {"role": "cnn_stage2", "precision_mode": "all_fp32", "diagnostic_control": True},
        "CNN-shrinker": {"role": "cnn_stageshrinker", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Full-restore-shrinker": {"role": "full_restore_shrinker", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-agent-relation": {"role": "attention_family_v2xvit_agent_relation", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-window-w4": {"role": "attention_family_v2xvit_spatial_window_w4", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-window-w8": {"role": "attention_family_v2xvit_spatial_window_w8", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-window-w16": {"role": "attention_family_v2xvit_spatial_window_w16", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-layer0": {"role": "attention_layer0", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-layer1": {"role": "attention_layer1", "precision_mode": "all_fp32", "diagnostic_control": True},
        "Attention-layer2": {"role": "attention_layer2", "precision_mode": "all_fp32", "diagnostic_control": True},
    }
    requested_roles = set(args.roles or roles)
    unknown_roles = sorted(requested_roles - set(roles))
    if unknown_roles:
        raise RuntimeError(f"unknown_control_roles:{unknown_roles}")
    summary_path = out / "controls" / "control_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"schema_version": "v2xvit-greedy005-accuracy-controls-v1", "diagnostic_control": True, "controls": {}}
    for name, meta in roles.items():
        if name not in requested_roles or name in summary.get("controls", {}):
            continue
        control_dir = out / "controls" / name.replace("+", "_").replace("-", "_")
        control_dir.mkdir(parents=True, exist_ok=False)
        candidate = _candidate_with_widths(winner, domains, winner_widths, str(meta["role"]))
        candidate = repair_genotype(candidate, formal["space"])
        phenotype = canonicalize_candidate(candidate, formal["space"])
        materialized = materialize_unified_widths(model, identity["cnn_units"], domains, candidate.pruning_width_genes, model_name="lidar_v2xvit")
        if not materialized.report.passed:
            raise RuntimeError(f"control_materialization_failed:{name}:{materialized.report.issues}")
        state_path = control_dir / "physical_state_dict.pth"
        torch.save({"model": {key: value.detach().cpu() for key, value in materialized.model.state_dict().items()}, "structure_hash": materialized.report.structure_hash}, state_path)
        payload = {
            "schema_version": "v2xvit-greedy005-diagnostic-control-v1",
            "control_name": name,
            "diagnostic_control": True,
            "precision_mode": meta["precision_mode"],
            "candidate_hash": canonical_candidate_hash(phenotype, formal["space"]),
            "genotype": candidate.to_dict(),
            "phenotype": phenotype.to_dict(),
            "requested_widths": materialized.report.requested_widths,
            "realized_widths": materialized.report.realized_widths,
            "physical_report": materialized.report.to_dict(),
            "physical_state_dict": str(state_path),
            "winner_reference_hash": "44551dcb6358b38447662e376ad1731d61862343d56da4c054103c784029547b",
            "checkpoint_path": "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth",
            "calibration_manifest_policy": "fresh_structure_specific_required_for_JMIX-FRESH",
        }
        write(control_dir / "control.json", payload)
        summary["controls"][name] = {"control_dir": str(control_dir), "candidate_hash": payload["candidate_hash"], "structure_hash": materialized.report.structure_hash, "physical_parameter_count": materialized.report.physical_parameter_count, "precision_mode": meta["precision_mode"], "diagnostic_control": True}
        del materialized
        torch.cuda.empty_cache()
    if summary_path.is_file():
        summary_path.unlink()
    write(summary_path, summary)
    print(json.dumps({"status": "ok", "controls": list(summary["controls"]), "diagnostic_control": True}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--roles", nargs="*")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
