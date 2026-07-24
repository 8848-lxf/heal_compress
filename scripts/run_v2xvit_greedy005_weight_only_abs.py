#!/usr/bin/env python3
"""Run one V2X-ViT 0.05 Greedy path with conservative weight-only Taylor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.run_v2xvit_greedy005_full import (
    _batch_content_hash,
    _baseline_candidate,
    _build_full_space,
    _formal_space,
)
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.canonicalization import canonicalize_candidate
from search.hashing import candidate_hash, canonical_json_hash
from search.greedy.weight_only_abs import run_weight_only_abs_greedy
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy


def _type_coverage_forward(adapter: Any, model: torch.nn.Module, batch: Any) -> Any:
    """Collect one deterministic loss with both HGT modality branches active.

    V2XViTFusion normally creates an all-zero prior encoding, so HGT's
    ``ModuleList`` index is always zero in ordinary validation frames.  This
    temporary hook changes only the calibration forward's type channel and is
    removed before returning; no search-loop forward uses it.
    """

    handles = []
    for module in model.modules():
        q_linears = getattr(module, "q_linears", None)
        a_linears = getattr(module, "a_linears", None)
        if q_linears is None or a_linears is None or len(q_linears) < 2:
            continue

        def hook(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
            prior = args[2] if len(args) >= 3 else kwargs.get("prior_encoding")
            if not torch.is_tensor(prior):
                return None
            prior = prior.clone()
            if prior.ndim < 5 or prior.shape[-1] < 3:
                return None
            length = int(prior.shape[1])
            pattern = torch.arange(length, device=prior.device, dtype=prior.dtype) % 2
            prior[..., 2] = pattern.view(1, length, 1, 1)
            if len(args) >= 3:
                return (*args[:2], prior, *args[3:]), kwargs
            return args, {**kwargs, "prior_encoding": prior}

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        return adapter.forward_for_task(model, batch)
    finally:
        for handle in handles:
            handle.remove()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    fields = list(rows[0]) if rows else ["step"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"weight_only_greedy_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    batch_hash = _batch_content_hash(batch)
    calibration_hash = canonical_json_hash({
        "model": "lidar_v2xvit", "config_sha256": _sha256(MODEL_SPECS["v2xvit"]["config"]),
        "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]), "split": "validation",
        "dataset_indices": [dataset_index], "agent_count": agent_count, "sample_count": 1,
        "seed": int(args.seed), "batch_content_sha256": batch_hash,
        "hmsa_type_coverage": "type0_type1_pre_forward_hook_v1",
    })
    identity = _build_full_space(model, adapter, hypes, batch)
    formal = _formal_space(
        model,
        adapter,
        hypes,
        batch,
        identity,
        calibration_hash,
        fisher_forward_fn=lambda module, values: _type_coverage_forward(
            adapter, module, values
        ),
    )
    space = formal["space"]
    baseline = _baseline_candidate(space)
    baseline_phenotype = canonicalize_candidate(baseline, space)
    baseline_bops = formal["bops"].evaluate_breakdown(baseline_phenotype)
    if abs(float(baseline_bops["R_bops_vs_fp32"]) - 1.0) > 1.0e-12:
        raise RuntimeError("weight_only_baseline_bops_not_unity")
    proxy = JointWeightTaylorProxy(
        model,
        statistics=formal["fisher"],
        unit_to_parameter_slices=formal["slices"],
        strict=True,
    )
    result = run_weight_only_abs_greedy(
        space,
        weight_proxy=proxy,
        bops_evaluator=formal["bops"].evaluate_breakdown,
        size_evaluator=formal["size"].evaluate_breakdown,
        target=0.05,
        tolerance_abs=0.005,
        maximum_steps=10000,
    )
    write_csv(root / "greedy_trace.csv", result["trace"])
    winner = result["winner_candidate"]
    phenotype = result["winner_phenotype"]
    metrics = result["winner_metrics"]
    winner_payload = {
        "candidate_hash": candidate_hash(phenotype, space),
        "genotype": winner.to_dict(),
        "phenotype": phenotype.to_dict(),
        "metrics": metrics,
        "bops": result["winner_bops_breakdown"],
        "size": result["winner_size_breakdown"],
        "diagnostic_control": False,
    }
    write_json(root / "winner/v2xvit_greedy005_winner.json", winner_payload)
    write_json(root / "search/v2xvit_greedy005_search_manifest.json", {
        "model": "V2X-ViT", "target_bops_retention": 0.05, "tolerance_abs": 0.005,
        "seed": int(args.seed), "calibration_manifest_hash": calibration_hash,
        "calibration_batch_content_sha256": batch_hash, "dataset_index": dataset_index,
        "agent_count": agent_count, "domain_type_counts": {
            key: sum(domain.domain_type == key for domain in space.pruning_domains)
            for key in ("cnn_channel", "grouped_conv_channel", "attention_dh", "ffn_hidden")
        },
        "precision_locus_count": len(space.precision_gene_ids),
        "budget_band": [0.045, 0.055], "budget_reached": result["budget_reached"],
        "budget_unreachable": result["budget_unreachable"],
        "budget_band_candidate_count": result["budget_band_candidate_count"],
        "selected_step_count": result["selected_step_count"], "visited_action_count": result["visited_action_count"],
        "termination_reason": result["termination_reason"], "winner": winner_payload,
        "search_loop_runtime_audit": {
            key: result[key] for key in (
                "search_loop_forward_calls", "search_loop_backward_calls", "search_loop_physical_exports",
                "search_loop_onnx_exports", "search_loop_trt_builds",
            )
        },
    })
    write_json(root / "proxy_audit/taylor_reduction_audit.json", {
        "pruning_formula": "sum_elementwise(abs(g*(-w)) + 0.5*abs(h*w^2)) then parameter/coupled-group/layer/sample aggregation",
        "quantization_formula": "sum_retained_elementwise(abs(g*(Q_next(w)-Q_current(w))) + 0.5*abs(h*(Q_next(w)-Q_current(w))^2))",
        "elementwise_abs_before_reduction": True, "cross_parameter_signed_cancellation": False,
        "cross_sample_signed_cancellation": False, "negative_element_scores_allowed": False,
        "fisher_report": formal["fisher_report"], "ranking_report": formal["ranking_report"],
        "search_statistics_collected_before_loop": True,
    })
    write_json(root / "proxy_audit/activation_taylor_disable_audit.json", {
        "activation_taylor_weight": 0.0, "activation_taylor_used_for_fitness": False,
        "joint_taylor_used_for_fitness": False, "cross_residual_used_for_fitness": False,
        "activation_taylor_diagnostic": "disabled_in_search; deployment calibration remains enabled",
    })
    write_json(root / "reports/search_loop_runtime_audit.json", {
        key: result[key] for key in (
            "search_loop_forward_calls", "search_loop_backward_calls", "search_loop_physical_exports",
            "search_loop_onnx_exports", "search_loop_trt_builds",
        )
    })
    write_json(root / "reports/input_provenance.json", {
        "branch": _git(REPO, "branch", "--show-current"), "commit": _git(REPO, "rev-parse", "HEAD"),
        "repo": str(REPO), "model": "V2X-ViT", "checkpoint": str(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "config": str(MODEL_SPECS["v2xvit"]["config"]), "calibration_manifest_hash": calibration_hash,
        "calibration_type_coverage": "hmsa_type0_type1_pre_forward_hook_v1",
        "fixed500_manifest": str(args.fixed500_manifest), "old_winner_hash": "44551dcb6358b38447662e376ad1731d61862343d56da4c054103c784029547b",
        "old_winner_bops_retention": 0.05491842298159359, "gpu_visible_index": 0,
    })
    print(json.dumps({"status": "ok", "winner": winner_payload["candidate_hash"], "budget_reached": result["budget_reached"], "steps": result["selected_step_count"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
