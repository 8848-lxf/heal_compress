#!/usr/bin/env python3
"""Run V2X-ViT R_BOPS=0.30 Greedy with fixed conservative action Taylor."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import gzip
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

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
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.greedy.conservative_joint import (
    run_conservative_joint_greedy,
    select_stage2_budget_pool,
)
from search.hashing import candidate_hash, canonical_json_hash
from search.proxy.conservative_action_taylor import (
    build_activation_taylor_units,
    collect_streaming_activation_action_statistics,
    collect_structural_gate_statistics,
)
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
from search.quantization_space.v2xvit_deployment_closed import (
    deployment_close_v2xvit_quantization_groups,
)
from search.integration.data_provider import move_batch_to_device


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_json_gzip(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
        json.dump(payload, handle, sort_keys=True, default=str, separators=(",", ":"))
        handle.write("\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    fields = list(rows[0]) if rows else ["candidate_hash"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _physical_gpu_uuid(index: int) -> str:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(index)}",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if not output.startswith("GPU-"):
        raise RuntimeError(f"v2xvit_greedy_gpu_uuid_invalid:{index}:{output}")
    return output


def _load_fixed_proxy_batches(
    *,
    adapter: Any,
    hypes: Mapping[str, Any],
    manifest_path: Path,
    sample_count: int,
    device: torch.device,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Load a deterministic, multi-agent-first train subset for Stage-1 only."""

    from opencood.data_utils.datasets import build_dataset

    payload = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload.get("samples", [])]
    rows = [row for row in rows if int(row.get("record_len", 0)) >= 2] + [
        row for row in rows if int(row.get("record_len", 0)) < 2
    ]
    selected = rows[: int(sample_count)]
    if len(selected) != int(sample_count):
        raise RuntimeError(
            f"v2xvit_proxy_manifest_insufficient_samples:{len(selected)}<{sample_count}"
        )
    train_hypes = adapter._absolutize_dataset_paths(copy.deepcopy(dict(hypes)))
    dataset = build_dataset(train_hypes, visualize=False, train=True)
    batches: list[Any] = []
    evidence: list[dict[str, Any]] = []
    for row in selected:
        sample_seed = int(row["sample_seed"])
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))
        torch.manual_seed(sample_seed)
        item = dataset[int(row["dataset_index"])]
        batch = dataset.collate_batch_train([item])
        if batch is None:
            raise RuntimeError(
                f"v2xvit_proxy_manifest_batch_none:{row['dataset_index']}"
            )
        observed_k = int(batch["ego"]["inputs_m1"]["voxel_features"].shape[0])
        if observed_k != int(row["voxel_count"]):
            raise RuntimeError(
                f"v2xvit_proxy_manifest_voxel_mismatch:{row['dataset_index']}:"
                f"{observed_k}!={row['voxel_count']}"
            )
        device_batch = move_batch_to_device(batch, device)
        batches.append(device_batch)
        evidence.append(
            {
                "ordinal": int(row["ordinal"]),
                "dataset_index": int(row["dataset_index"]),
                "record_len": int(row["record_len"]),
                "voxel_count": observed_k,
                "sample_seed": sample_seed,
                "batch_content_sha256": _batch_content_hash(device_batch),
            }
        )
    return batches, evidence, payload


def _public_band_row(row: Mapping[str, Any], space: Any) -> dict[str, Any]:
    candidate = row["candidate"]
    phenotype = row["phenotype"]
    precision_counts = {
        precision: sum(
            value == precision
            for value in phenotype.realized_precision_profile.values()
        )
        for precision in ("FP32", "FP16", "INT8")
    }
    widths = dict(candidate.pruning_width_genes)
    return {
        **{
            key: value
            for key, value in row.items()
            if key not in {"candidate", "phenotype", "bops_breakdown", "risk"}
        },
        "genotype": candidate.to_dict(),
        "phenotype": phenotype.to_dict(),
        "precision_counts": precision_counts,
        "widths": widths,
        "bops_breakdown": row["bops_breakdown"],
        "canonical_candidate_hash": candidate_hash(phenotype, space),
    }


def _legacy_gate_audit(
    *,
    space: Any,
    baseline: CandidateGenotype,
    gate_proxy: Any,
    weight_proxy: JointWeightTaylorProxy,
) -> list[dict[str, Any]]:
    current = canonicalize_candidate(baseline, space)
    rows = []
    for domain in space.pruning_domains:
        lower = [
            int(width)
            for width in domain.legal_widths
            if int(width) < int(domain.original_width)
        ]
        if not lower:
            continue
        width = max(lower)
        genes = dict(baseline.pruning_width_genes)
        genes[domain.domain_id] = width
        successor = CandidateGenotype(
            pruning_width_genes=genes,
            precision_genes=dict(baseline.precision_genes),
            meta={"created_by": "gate_vs_legacy_audit"},
        )
        successor_phenotype = canonicalize_candidate(successor, space)
        gate = gate_proxy.pruning_action_breakdown(current, successor_phenotype)
        legacy = weight_proxy.pruning_action_breakdown(current, successor_phenotype)
        rows.append(
            {
                "domain_id": domain.domain_id,
                "domain_type": domain.domain_type,
                "module_path": domain.module_path,
                "original_width": domain.original_width,
                "next_width": width,
                "gate_score": gate["delta_J_struct"],
                "legacy_coupled_weight_score": legacy["delta_J_prune"],
                "gate_used_for_fitness": True,
                "legacy_used_for_fitness": False,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v2xvit_greedy030_requires_one_visible_gpu:{torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, adapter, hypes, _ = _load("v2xvit", device)
    proxy_batches, sample_evidence, proxy_manifest = _load_fixed_proxy_batches(
        adapter=adapter,
        hypes=hypes,
        manifest_path=args.proxy_manifest,
        sample_count=args.proxy_samples,
        device=device,
    )
    batch = proxy_batches[0]
    dataset_index = int(sample_evidence[0]["dataset_index"])
    agent_count = int(sample_evidence[0]["record_len"])
    calibration_hash = canonical_json_hash(
        {
            "model": "lidar_v2xvit",
            "purpose": "fixed_pre_search_weight_activation_gate_taylor",
            "config_sha256": _sha256(MODEL_SPECS["v2xvit"]["config"]),
            "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
            "split": "train",
            "source_manifest_hash": proxy_manifest["manifest_hash"],
            "dataset_indices": [row["dataset_index"] for row in sample_evidence],
            "agent_counts": [row["record_len"] for row in sample_evidence],
            "sample_count": len(proxy_batches),
            "seed": int(args.seed),
            "batch_content_sha256": [
                row["batch_content_sha256"] for row in sample_evidence
            ],
            "hmsa_type_coverage": "type0_type1_pre_forward_hook_v1",
        }
    )
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
        fisher_batches=proxy_batches,
    )
    closed_groups = deployment_close_v2xvit_quantization_groups(
        formal["space"].quantization_groups
    )
    space = replace(formal["space"], quantization_groups=closed_groups)
    formal["space"] = space
    baseline = _baseline_candidate(space)
    baseline_phenotype = canonicalize_candidate(baseline, space)
    baseline_bops = formal["bops"].evaluate_breakdown(baseline_phenotype)
    legal_start_retention = float(baseline_bops["R_bops_vs_fp32"])
    if not float(args.target) < legal_start_retention <= 1.0:
        raise RuntimeError(
            f"v2xvit_greedy030_deployment_start_invalid:{legal_start_retention}"
        )

    gate_proxy, gate_manifest = collect_structural_gate_statistics(
        model,
        space.pruning_domains,
        proxy_batches,
        forward_fn=lambda module, values: _type_coverage_forward(
            adapter, module, values
        ),
        loss_fn=adapter.compute_task_loss,
    )
    activation_units, gene_to_units, activation_mapping = (
        build_activation_taylor_units(
            model,
            space.quantization_groups,
            mutable_gene_ids=space.precision_gene_ids,
            transformer_precision_units=formal["components"].precision_units,
        )
    )
    precision_order = ("FP32", "FP16", "INT8")
    groups_by_id = {group.group_id: group for group in space.quantization_groups}
    precision_ladders = {
        gene_id: tuple(
            precision
            for precision in precision_order
            if precision in groups_by_id[gene_id].allowed_precisions
        )
        for gene_id in space.precision_gene_ids
    }
    activation_proxy, activation_manifest = collect_streaming_activation_action_statistics(
        model,
        proxy_batches,
        forward_fn=lambda module, values: _type_coverage_forward(
            adapter, module, values
        ),
        loss_fn=adapter.compute_task_loss,
        units=activation_units,
        gene_to_unit_ids=gene_to_units,
        precision_ladders=precision_ladders,
        calibration_manifest_hash=calibration_hash,
    )
    weight_proxy = JointWeightTaylorProxy(
        model,
        statistics=formal["fisher"],
        unit_to_parameter_slices=formal["slices"],
        strict=True,
    )
    legacy_gate_rows = _legacy_gate_audit(
        space=space,
        baseline=baseline,
        gate_proxy=gate_proxy,
        weight_proxy=weight_proxy,
    )

    result = run_conservative_joint_greedy(
        space,
        structural_proxy=gate_proxy,
        weight_proxy=weight_proxy,
        activation_proxy=activation_proxy,
        bops_evaluator=formal["bops"].evaluate_breakdown,
        size_evaluator=formal["size"].evaluate_breakdown,
        target=args.target,
        tolerance_abs=args.tolerance,
        maximum_steps=args.maximum_steps,
    )
    public_band = [_public_band_row(row, space) for row in result["budget_candidates"]]
    selected = select_stage2_budget_pool(public_band, target=args.target, maximum=5)
    winner_candidate = result["winner_candidate"]
    winner_phenotype = result["winner_phenotype"]
    winner = {
        "candidate_hash": candidate_hash(winner_phenotype, space),
        "genotype": winner_candidate.to_dict(),
        "phenotype": winner_phenotype.to_dict(),
        "metrics": result["winner_metrics"],
        "bops": result["winner_bops_breakdown"],
        "size": result["winner_size_breakdown"],
    }

    write_csv(root / "greedy_trace.csv", result["trace"])
    write_json_gzip(
        root / "search/greedy030_budget_candidates.json.gz",
        {"candidates": public_band},
    )
    write_json(root / "search/greedy030_winner.json", winner)
    write_json(root / "search/stage2_candidate_selection.json", {"candidates": selected})
    write_csv(
        root / "reports/stage2_candidate_screening.csv",
        [
            {
                "candidate_id": f"stage2_{index:02d}",
                "candidate_hash": row["candidate_hash"],
                "bops_retention": row["current_retention"],
                "total_taylor": row["cumulative_total_taylor"],
                "parameter_retention": row["R_parameter_retention"],
                "pruned_unit_count": row["pruned_unit_count"],
                "int8_count": row["int8_count"],
                "selection_reasons": ";".join(row["selection_reasons"]),
            }
            for index, row in enumerate(selected, start=1)
        ],
    )
    write_json(root / "proxy/activation_taylor_mapping.json", {"rows": activation_mapping})
    write_json(root / "proxy/activation_taylor_audit.json", activation_manifest)
    write_json(root / "proxy/structural_gate_mapping.json", gate_manifest)
    write_csv(root / "proxy/structural_gate_mapping.csv", gate_manifest["mapping_rows"])
    write_csv(root / "proxy/gate_vs_legacy_weight_score.csv", legacy_gate_rows)
    write_json(
        root / "proxy/interaction_diagnostic.json",
        {
            "J_joint_diagnostic": None,
            "J_cross_diagnostic": None,
            "joint_taylor_used_for_fitness": False,
            "cross_residual_used_for_fitness": False,
            "negative_interaction_refund_allowed": False,
        },
    )
    runtime_audit = {
        key: result[key]
        for key in (
            "search_loop_forward_calls",
            "search_loop_backward_calls",
            "search_loop_physical_exports",
            "search_loop_onnx_exports",
            "search_loop_trt_builds",
        )
    }
    write_json(root / "proxy/search_loop_runtime_audit.json", runtime_audit)
    write_json(
        root / "proxy/weight_activation_taylor_formula.json",
        {
            "weight_quantization": "sum(abs(g_W*delta_W)+0.5*abs(h_W*delta_W^2))",
            "activation_quantization": "sum(abs(g_A*delta_A)+0.5*abs(g_A^2*delta_A^2))",
            "precision_fitness": "J_WQ + J_AQ",
            "structure_fitness": "functional_gate_output_taylor_only",
            "elementwise_abs_before_reduction": True,
            "legacy_coupled_weight_taylor_used_for_fitness": False,
        },
    )
    write_json(
        root / "search/search_space.json",
        {
            "pruning_domains": [domain.to_dict() for domain in space.pruning_domains],
            "quantization_groups": [group.to_dict() for group in space.quantization_groups],
            "calibration_manifest_hash": calibration_hash,
            "trace_snapshot_hash": space.trace_snapshot_hash,
            "fixed_rankings": {
                "fisher": formal["fisher_report"],
                "atomic": formal["ranking_report"],
                "transformer": formal["transformer_ranking_report"],
            },
        },
    )
    manifest = {
        "schema_version": "v2xvit-greedy030-conservative-v1",
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "model": "V2X-ViT",
        "target_bops_retention": float(args.target),
        "tolerance_abs": float(args.tolerance),
        "legal_budget_band": [args.target - args.tolerance, args.target + args.tolerance],
        "strict_fp32_reference_R_BOPS": 1.0,
        "deployment_closed_start_R_BOPS": legal_start_retention,
        "seed": int(args.seed),
        "gpu_visible_index": 0,
        "physical_gpu_index": int(args.physical_gpu),
        "gpu_uuid": _physical_gpu_uuid(args.physical_gpu),
        "calibration_manifest_hash": calibration_hash,
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "proxy_manifest_path": str(args.proxy_manifest.resolve()),
        "proxy_manifest_hash": proxy_manifest["manifest_hash"],
        "proxy_sample_count": len(proxy_batches),
        "proxy_samples": sample_evidence,
        "budget_reached": result["budget_reached"],
        "budget_band_candidate_count": result["budget_band_candidate_count"],
        "selected_step_count": result["selected_step_count"],
        "visited_action_count": result["visited_action_count"],
        "termination_reason": result["termination_reason"],
        "winner": winner,
        "stage2_candidate_count": len(selected),
        "search_loop_runtime_audit": runtime_audit,
    }
    write_json(root / "search/greedy030_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "winner": winner["candidate_hash"],
                "bops_retention": winner["bops"]["R_bops_vs_fp32"],
                "stage2_candidates": len(selected),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fixed50-manifest", type=Path, required=True)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.30)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--maximum-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument(
        "--proxy-manifest",
        type=Path,
        default=REPO
        / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json",
    )
    parser.add_argument("--proxy-samples", type=int, default=8)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
