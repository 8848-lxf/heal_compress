
#!/usr/bin/env python3
"""Build a formal train32 Taylor cache and six-budget Greedy path for HEAL Transformer models."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.run_v2xvit_greedy005_full import (
    _baseline_candidate,
    _batch_content_hash,
    _build_full_space,
    _formal_space,
)
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.greedy.engine import GreedyBudgetSearch, GreedySearchConfig
from search.greedy.weight_only_abs import run_weight_only_abs_greedy
from search.hashing import candidate_hash, canonical_json_hash
from search.model_family.calibration_manifest import load_v2xvit_train_manifest
from search.proxy.conservative_gate_activation_taylor import (
    ActivationTaylorCache,
    FunctionalGateTaylorProxy,
    GateDomainScores,
    build_activation_units,
    collect_activation_taylor_cache_multi,
    collect_functional_gate_scores_multi,
    rerank_domains_by_gate_scores,
)
from search.proxy.fisher_proxy import collect_task_loss_fisher_statistics
from search.proxy.joint_weight_activation_taylor import (
    taylor_units_from_transformer_precision,
)
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy


BUDGETS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
TOLERANCE = 0.005
SEED = 0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, default=str)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


class FrozenTrainPrefix(Sequence[Any]):
    """Re-load deterministic train samples instead of retaining GPU batches."""

    def __init__(
        self,
        *,
        adapter: Any,
        hypes: Mapping[str, Any],
        device: torch.device,
        manifest: Mapping[str, Any],
        count: int,
    ) -> None:
        from opencood.data_utils.datasets import build_dataset

        self.adapter = adapter
        self.device = device
        self.rows = tuple(dict(row) for row in manifest["samples"][: int(count)])
        self.dataset = build_dataset(
            adapter._absolutize_dataset_paths(dict(hypes)),
            visualize=False,
            train=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int) -> Any:
        from search.integration.data_provider import move_batch_to_device

        if position < 0:
            position += len(self.rows)
        if position < 0 or position >= len(self.rows):
            raise IndexError(position)
        row = self.rows[position]
        seed = int(row["sample_seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        item = self.dataset[int(row["dataset_index"])]
        batch = self.dataset.collate_batch_train([item])
        if batch is None:
            raise RuntimeError(f"train32_empty_batch:{row['dataset_index']}")
        observed = int(batch["ego"]["inputs_m1"]["voxel_features"].shape[0])
        if observed != int(row["voxel_count"]):
            raise RuntimeError(
                f"train32_manifest_voxel_mismatch:{row['dataset_index']}:"
                f"{observed}!={row['voxel_count']}"
            )
        return move_batch_to_device(batch, self.device)


def _bind_model_specific_train_prefix(
    *,
    adapter: Any,
    hypes: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    count: int,
    model_id: str,
) -> dict[str, Any]:
    """Bind fixed sample identities to one model's actual preprocessing graph.

    Dataset indices and RNG seeds are inherited from the frozen V2X-ViT
    selection policy, but voxel counts are *not* portable across model
    configurations.  CoBEVT, for example, uses a different voxelisation
    contract for the same DAIR frame.  Persisting the observed count prevents
    a V2X-ViT input-shape manifest from being silently treated as CoBEVT
    provenance while retaining a common nested sample identity.
    """

    from opencood.data_utils.datasets import build_dataset

    requested = int(count)
    source_rows = [dict(row) for row in source_manifest["samples"][:requested]]
    if len(source_rows) != requested:
        raise RuntimeError(
            f"model_train_prefix_incomplete:{model_id}:{len(source_rows)}!={requested}"
        )
    dataset = build_dataset(
        adapter._absolutize_dataset_paths(dict(hypes)),
        visualize=False,
        train=True,
    )
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        seed = int(source["sample_seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        item = dataset[int(source["dataset_index"])]
        batch = dataset.collate_batch_train([item])
        if batch is None:
            raise RuntimeError(
                f"model_train_prefix_empty_batch:{model_id}:{source['dataset_index']}"
            )
        observed = int(batch["ego"]["inputs_m1"]["voxel_features"].shape[0])
        if observed <= 0:
            raise RuntimeError(
                f"model_train_prefix_nonpositive_voxels:{model_id}:"
                f"{source['dataset_index']}:{observed}"
            )
        rows.append(
            {
                **source,
                "source_voxel_count": int(source["voxel_count"]),
                "voxel_count": observed,
                "model_id": str(model_id),
            }
        )
    identity = {
        "schema_version": "heal_model_specific_train_prefix_v1",
        "model_id": str(model_id),
        "source_manifest_hash": str(source_manifest["manifest_hash"]),
        "sample_count": requested,
        "samples": rows,
    }
    identity["manifest_hash"] = canonical_json_hash(identity)
    return identity


def _prefix_gate_scores(
    scores: Mapping[str, GateDomainScores], count: int
) -> dict[str, GateDomainScores]:
    result = {}
    for domain_id, row in scores.items():
        samples = row.per_sample_unit_scores[:count]
        if len(samples) != count:
            raise RuntimeError(f"gate_prefix_sample_count:{domain_id}:{len(samples)}!={count}")
        units = sorted({unit for sample in samples for unit in sample})
        mean = {
            unit: sum(float(sample.get(unit, 0.0)) for sample in samples) / count
            for unit in units
        }
        result[domain_id] = replace(
            row,
            unit_scores=mean,
            sample_count=count,
            per_sample_unit_scores=tuple(samples),
        )
    return result


def _prefix_activation(cache: ActivationTaylorCache, count: int) -> ActivationTaylorCache:
    samples = cache.per_sample_transitions[:count]
    if len(samples) != count:
        raise RuntimeError(f"activation_prefix_sample_count:{len(samples)}!={count}")
    keys = sorted({key for sample in samples for key in sample})
    transitions = {
        key: sum(float(sample[key]) for sample in samples) / count for key in keys
    }
    return replace(
        cache,
        transitions=transitions,
        sample_count=count,
        per_sample_transitions=tuple(samples),
    )


def _rank(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: (float(values[key]), key))
    return {key: float(index) for index, key in enumerate(ordered)}


def _spearman(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return 1.0
    lrank, rrank = _rank(left), _rank(right)
    lvalues = np.asarray([lrank[key] for key in keys], dtype=np.float64)
    rvalues = np.asarray([rrank[key] for key in keys], dtype=np.float64)
    if lvalues.std() == 0.0 or rvalues.std() == 0.0:
        return 1.0 if np.array_equal(lvalues, rvalues) else 0.0
    return float(np.corrcoef(lvalues, rvalues)[0, 1])


def _action_scores(
    *,
    space: Any,
    baseline: CandidateGenotype,
    structure_proxy: FunctionalGateTaylorProxy,
    weight_proxy: JointWeightTaylorProxy,
    activation_cache: ActivationTaylorCache,
) -> dict[str, dict[str, float]]:
    engine = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=BUDGETS,
            bops_tolerance_abs=TOLERANCE,
            run_to_exhaustion=True,
            enable_budget_recovery=False,
        ),
    )
    current = canonicalize_candidate(baseline, space)
    rows: dict[str, dict[str, float]] = {}
    for successor, action in engine._neighbors(baseline):
        phenotype = canonicalize_candidate(successor, space)
        if action["kind"] == "domain_width":
            structure = structure_proxy.pruning_action_breakdown(current, phenotype)
            j_struct = float(structure["delta_J_prune"])
            j_wq = j_aq = 0.0
        else:
            j_struct = 0.0
            j_wq = float(
                weight_proxy.weight_quantization_action_breakdown(current, phenotype)[
                    "delta_J_WQ"
                ]
            )
            j_aq = float(
                activation_cache.action_breakdown(current, phenotype)["delta_J_AQ"]
            )
        key = f"{action['kind']}::{action['gene_id']}::{action.get('to_width', action.get('to_precision'))}"
        rows[key] = {
            "J_struct": j_struct,
            "J_WQ": j_wq,
            "J_AQ": j_aq,
            "J_total": j_struct + j_wq + j_aq,
        }
    return rows


def _candidate_from_trace(row: Mapping[str, Any]) -> CandidateGenotype:
    state = json.loads(str(row["state_after"]))
    return CandidateGenotype(
        pruning_width_genes=dict(state["widths"]),
        precision_genes=dict(state["precision"]),
        meta={"created_by": "six_budget_greedy_visited_candidate", "repair_count": 0},
    )


def _winner_compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    jl, jr = float(left["cumulative_proxy"]), float(right["cumulative_proxy"])
    equal = abs(jl - jr) <= max(1.0e-12, 1.0e-8 * max(abs(jl), abs(jr)))
    if not equal:
        return -1 if jl < jr else 1
    left_key = (
        float(left["budget_deviation"]),
        -float(left["R_parameter_retention"]),
        -float(left["mixed_weight_retention"]),
        str(left["candidate_hash"]),
    )
    right_key = (
        float(right["budget_deviation"]),
        -float(right["R_parameter_retention"]),
        -float(right["mixed_weight_retention"]),
        str(right["candidate_hash"]),
    )
    return (left_key > right_key) - (left_key < right_key)


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"six_budget_proxy_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    actual_uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()[int(args.physical_gpu)].strip()
    print(f"[train32] physical_gpu={args.physical_gpu} uuid={actual_uuid}", flush=True)

    model, adapter, hypes, _ = _load(args.model, device)
    representative, validation_index, agent_count = _multi_agent_validation_batch(
        adapter, hypes, device
    )
    train_manifest_path = (
        REPO / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"
    )
    source_train200 = load_v2xvit_train_manifest(train_manifest_path)
    if args.model == "v2xvit":
        train200 = source_train200
    else:
        train200 = _bind_model_specific_train_prefix(
            adapter=adapter,
            hypes=hypes,
            source_manifest=source_train200,
            count=32,
            model_id=args.model,
        )
        _write_json(root / "reports/model_specific_train32_manifest.json", train200)
    train32 = FrozenTrainPrefix(
        adapter=adapter,
        hypes=hypes,
        device=device,
        manifest=train200,
        count=32,
    )
    calibration_hash = canonical_json_hash(
        {
            "schema": f"{args.model}-formal-taylor-train32-v1",
            "train_selection_manifest_hash": source_train200["manifest_hash"],
            "model_specific_train_prefix_hash": train200["manifest_hash"],
            "sample_ordinals": [int(row["ordinal"]) for row in train200["samples"][:32]],
            "dataset_indices": [
                int(row["dataset_index"]) for row in train200["samples"][:32]
            ],
            "checkpoint_sha256": _sha256(MODEL_SPECS[args.model]["checkpoint"]),
            "config_sha256": _sha256(MODEL_SPECS[args.model]["config"]),
        }
    )
    identity = _build_full_space(model, adapter, hypes, representative)
    print("[train32] collect Fisher E[g^2] on 32 samples", flush=True)
    formal = _formal_space(
        model,
        adapter,
        hypes,
        representative,
        identity,
        calibration_hash,
        fisher_forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        fisher_batches=train32,
    )
    print("[train32] collect functional gate scores", flush=True)
    gate32, gate_mapping = collect_functional_gate_scores_multi(
        model,
        formal["space"].pruning_domains,
        forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        loss_fn=adapter.compute_task_loss,
        calibration_batches=train32,
    )
    reranked = rerank_domains_by_gate_scores(formal["space"].pruning_domains, gate32)
    space = replace(
        formal["space"],
        pruning_domains=reranked,
        pruning_unit_ids=[unit for domain in reranked for unit in domain.ordered_unit_ids],
    )
    baseline = _baseline_candidate(space)
    transformer_units = taylor_units_from_transformer_precision(
        model,
        formal["components"].precision_units,
        active_module_paths=formal["active_paths"],
    )
    activation_units, group_to_units = build_activation_units(
        model, space, transformer_units
    )
    print("[train32] collect activation Q/DQ Taylor", flush=True)
    activation32 = collect_activation_taylor_cache_multi(
        model,
        activation_units,
        group_to_units,
        forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        loss_fn=adapter.compute_task_loss,
        calibration_batches=train32,
    )
    print("[train32] collect 8/16 Fisher prefixes", flush=True)
    fisher_prefix = {32: formal["fisher"]}
    fisher_reports = {32: formal["fisher_report"]}
    for count in (8, 16):
        statistics, report = collect_task_loss_fisher_statistics(
            model,
            FrozenTrainPrefix(
                adapter=adapter,
                hypes=hypes,
                device=device,
                manifest=train200,
                count=count,
            ),
            forward_fn=lambda current_model, batch: _type_coverage_forward(
                adapter, current_model, batch
            ),
            loss_fn=adapter.compute_task_loss,
            calibration_manifest_hash=canonical_json_hash(
                {"parent": calibration_hash, "prefix": count}
            ),
        )
        fisher_prefix[count] = statistics
        fisher_reports[count] = report

    prefix_actions = {}
    for count in (8, 16, 32):
        prefix_actions[count] = _action_scores(
            space=space,
            baseline=baseline,
            structure_proxy=FunctionalGateTaylorProxy(
                _prefix_gate_scores(gate32, count)
            ),
            weight_proxy=JointWeightTaylorProxy(
                model,
                statistics=fisher_prefix[count],
                unit_to_parameter_slices=formal["slices"],
                strict=True,
            ),
            activation_cache=_prefix_activation(activation32, count),
        )
    convergence = {}
    for left, right in ((8, 16), (16, 32)):
        left_total = {key: value["J_total"] for key, value in prefix_actions[left].items()}
        right_total = {key: value["J_total"] for key, value in prefix_actions[right].items()}
        left_order = sorted(left_total, key=lambda key: (left_total[key], key))
        right_order = sorted(right_total, key=lambda key: (right_total[key], key))
        convergence[f"{left}_to_{right}"] = {
            "spearman": _spearman(left_total, right_total),
            "top10_overlap": len(set(left_order[:10]) & set(right_order[:10])) / 10.0,
            "top20_overlap": len(set(left_order[:20]) & set(right_order[:20])) / 20.0,
        }
    convergence_pass = bool(
        convergence["16_to_32"]["spearman"] >= 0.95
        and convergence["16_to_32"]["top10_overlap"] >= 0.80
    )
    convergence_payload = {
        "sample_counts": [8, 16, 32],
        "manifest_hash": calibration_hash,
        "prefixes_share_frozen_train32_order": True,
        "comparisons": convergence,
        "passed": convergence_pass,
        "taylor_sample_convergence_insufficient": not convergence_pass,
        "action_scores": prefix_actions,
    }
    _write_json(
        root / "reports/taylor_sample_convergence.json",
        convergence_payload,
    )
    _write_csv(root / "reports/taylor_prefix_convergence.csv", [
        {"prefix_pair": key, **value} for key, value in convergence.items()
    ])
    _write_json(root / "reports/fisher_statistics_audit.json", {
        "formula": "h=E[g^2]",
        "not_formula": "E[g]^2",
        "elementwise_abs_before_reduction": True,
        "sample_reports": fisher_reports,
        "all_finite": all(bool(row["all_fisher_finite"]) for row in fisher_reports.values()),
    })
    _write_json(root / "reports/structural_gate_mapping.json", {
        "mapping": gate_mapping,
        "domain_count": len(gate32),
        "sample_count": 32,
        "missing_mapping_count": 0,
        "tracer_owns_physical_closure": True,
        "legacy_weight_taylor_used_for_fitness": False,
    })
    _write_json(root / "reports/activation_qdq_mapping.json", {
        "mapping": list(activation32.mapping),
        "group_to_units": {key: list(value) for key, value in group_to_units.items()},
        "sample_count": activation32.sample_count,
        "missing_observer_count": 0,
        "activation_taylor_used_for_fitness": True,
    })
    if not activation32.mapping or any(
        not row.get("precision_group_id") for row in activation32.mapping
    ):
        raise RuntimeError("v2xvit_taylor_mapping_missing_or_unowned")
    mapped_unit_ids = {str(row["unit_id"]) for row in activation32.mapping}
    action_unit_ids = {
        str(unit_id)
        for unit_ids in group_to_units.values()
        for unit_id in unit_ids
    }
    if mapped_unit_ids != action_unit_ids:
        raise RuntimeError(
            "v2xvit_taylor_mapping_inventory_mismatch:"
            f"unused={sorted(mapped_unit_ids-action_unit_ids)}:"
            f"missing={sorted(action_unit_ids-mapped_unit_ids)}"
        )
    av_mapping = [
        row for row in activation32.mapping
        if "::av::" in str(row.get("unit_id", ""))
        or "av_matmul" in str(row.get("unit_id", ""))
    ]
    if av_mapping:
        raise RuntimeError(f"v2xvit_fixed_av32_has_mutable_aq_mapping:{av_mapping}")
    _write_json(root / "reports/taylor_collection_contract.json", {
        "sample_count": 32,
        "sample_first": True,
        "empirical_fisher": "mean(g^2)",
        "elementwise_abs_before_reduction": True,
        "manifest_hash": calibration_hash,
        "missing_hook_count": 0,
        "unexpected_inactive_count": 0,
        "mapping_mismatch_count": 0,
        "mapped_unit_count": len(mapped_unit_ids),
        "action_unit_count": len(action_unit_ids),
        "av_aq_boundary_count": len(av_mapping),
        "av_aq_boundary": "none_AV32_is_fixed_after_pruned_shape_closure",
        "av_profile_policy": "AV32_fixed_no_chromosome_locus",
        "derived_merge_has_independent_aq_gene": False,
        "precision_gene_permutation_invariant": True,
    })
    _write_json(root / "reports/taylor_cache_manifest.json", {
        "schema_version": f"{args.model}-h800-sixbudget-taylor-cache-v1",
        "search_space_schema_version": f"{args.model}-h800-sixbudget-search-space-v1",
        "precision_contract_schema_version": f"{args.model}-transformer-fixed-qk-softmax-av-v1",
        "manifest_hash": calibration_hash,
        "sample_count": 32,
        "gate_domain_count": len(gate32),
        "activation_mapping_count": len(activation32.mapping),
        "av_activation_mapping_count": len(av_mapping),
        "prefix_convergence_passed": convergence_pass,
        "old_cache_reused": False,
    })
    _write_json(root / "reports/taylor_formula_audit.json", {
        "J_total": "J_struct_gate + J_WQ + J_AQ",
        "J_struct_gate": "mean_samples sum_elements(abs(g_u*(-u))+0.5*abs(E_sample_local[g_u^2]*u^2))",
        "J_WQ": "mean_samples sum_retained(abs(g_W*delta_W)+0.5*abs(E[g_W^2]*delta_W^2))",
        "J_AQ": "mean_samples sum_qdq_inputs(abs(g_A*delta_A)+0.5*abs(g_A^2*delta_A^2))",
        "reduction": "per_sample_elementwise_abs_then_tensor_channel_token_sum_then_sample_mean",
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
        "legacy_weight_taylor_used_for_fitness": False,
    })

    # A non-converged formal Taylor cache must never silently seed Greedy or GA.
    # Persist the diagnostics above first so the failure remains auditable.
    if not convergence_pass:
        raise RuntimeError(
            "v2xvit_taylor_sample_convergence_insufficient:"
            f"{convergence['16_to_32']}"
        )

    print("[greedy] run complete six-budget trajectory from B0", flush=True)
    weight32 = JointWeightTaylorProxy(
        model,
        statistics=formal["fisher"],
        unit_to_parameter_slices=formal["slices"],
        strict=True,
    )
    result = run_weight_only_abs_greedy(
        space,
        weight_proxy=weight32,
        structure_proxy=FunctionalGateTaylorProxy(gate32),
        activation_cache=activation32,
        activation_taylor_weight=1.0,
        bops_evaluator=formal["bops"].evaluate_breakdown,
        size_evaluator=formal["size"].evaluate_breakdown,
        target=min(BUDGETS),
        tolerance_abs=TOLERANCE,
        maximum_steps=None,
        capture_targets=BUDGETS,
    )
    _write_csv(root / "greedy_trace.csv", result["trace"])
    _write_csv(root / "reports/greedy_r010_trajectory.csv", result["trace"])
    winners = {}
    summary = []
    for budget in BUDGETS:
        candidates = []
        for trace_row in result["trace"]:
            retention = float(trace_row["current_retention"])
            if abs(retention - budget) > TOLERANCE:
                continue
            genotype = _candidate_from_trace(trace_row)
            phenotype = canonicalize_candidate(genotype, space)
            size = formal["size"].evaluate_breakdown(phenotype)
            candidates.append({
                **dict(trace_row),
                "genotype": genotype,
                "phenotype": phenotype,
                "budget_deviation": abs(retention - budget),
                "R_parameter_retention": float(size["R_parameter_retention"]),
                "mixed_weight_retention": float(size["R_size_vs_fp32"]),
                "size": size,
            })
        ordered = sorted(candidates, key=functools.cmp_to_key(_winner_compare))
        if not ordered:
            winners[f"{budget:.2f}"] = {"budget_reached": False}
            summary.append({"budget": budget, "budget_reached": False})
            continue
        winner = ordered[0]
        genotype = winner["genotype"]
        phenotype = winner["phenotype"]
        bops = formal["bops"].evaluate_breakdown(phenotype)
        key = candidate_hash(phenotype, space)
        payload = {
            "budget": budget,
            "tolerance_abs": TOLERANCE,
            "budget_reached": True,
            "candidate_hash": key,
            "genotype": genotype.to_dict(),
            "phenotype": phenotype.to_dict(),
            "bops": bops,
            "size": winner["size"],
            "cumulative_J_total": float(winner["cumulative_proxy"]),
            "cumulative_J_struct": float(winner["cumulative_pruning_taylor"]),
            "cumulative_J_WQ": float(winner["cumulative_weight_quantization_taylor"]),
            "cumulative_J_AQ": float(winner["cumulative_activation_taylor"]),
            "budget_band_candidate_count": len(ordered),
            "greedy_exact_winner": True,
            "stage2_top5_used": False,
            "real_metrics_used_for_selection": False,
            "repair_counts": {"structural": 0, "precision": 0, "budget": 0},
        }
        winners[f"{budget:.2f}"] = payload
        _write_json(root / f"greedy/budget_{int(round(budget*100)):03d}/exact_winner.json", payload)
        summary.append({
            "budget": budget,
            "budget_reached": True,
            "candidate_hash": key,
            "R_bops": float(bops["R_bops_vs_fp32"]),
            "budget_deviation": abs(float(bops["R_bops_vs_fp32"]) - budget),
            "cumulative_J_total": payload["cumulative_J_total"],
            "R_parameter_retention": float(winner["size"]["R_parameter_retention"]),
            "mixed_weight_retention": float(winner["size"]["R_size_vs_fp32"]),
            "budget_band_candidate_count": len(ordered),
        })
    _write_json(root / "reports/six_budget_greedy_winners.json", winners)
    _write_csv(root / "reports/six_budget_greedy_summary.csv", summary)
    winner010 = winners.get("0.10")
    if not winner010 or not winner010.get("budget_reached"):
        raise RuntimeError(f"{args.model}_greedy_r010_budget_unreachable")
    _write_json(root / "reports/greedy_r010_winner.json", winner010)
    _write_csv(root / "reports/greedy_r010_budget_capture.csv", [
        row for row in result["trace"]
        if bool(row.get("selected"))
        and abs(float(row["current_retention"]) - 0.10) <= TOLERANCE
    ])
    _write_json(root / "proxy/search_loop_runtime_audit.json", {
        key: result[key]
        for key in (
            "search_loop_forward_calls", "search_loop_backward_calls",
            "search_loop_physical_exports", "search_loop_onnx_exports",
            "search_loop_trt_builds", "termination_reason", "capture_targets",
            "captured_targets",
        )
    })
    _write_json(root / "reports/input_provenance.json", {
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip(),
        "model": args.model,
        "checkpoint": str(MODEL_SPECS[args.model]["checkpoint"]),
        "checkpoint_sha256": _sha256(MODEL_SPECS[args.model]["checkpoint"]),
        "config_sha256": _sha256(MODEL_SPECS[args.model]["config"]),
        "train200_manifest": str(train_manifest_path),
        "train_selection_manifest_hash": source_train200["manifest_hash"],
        "model_specific_train_prefix_hash": train200["manifest_hash"],
        "taylor_manifest_hash": calibration_hash,
        "taylor_sample_count": 32,
        "validation_shape_trace_index": validation_index,
        "validation_shape_trace_agent_count": agent_count,
        "validation_shape_trace_batch_hash": _batch_content_hash(representative),
        "physical_gpu": int(args.physical_gpu),
        "gpu_uuid": actual_uuid,
    })
    print(json.dumps({
        "status": "ok",
        "captured_budgets": result["captured_targets"],
        "selected_steps": result["selected_step_count"],
        "convergence_pass": convergence_pass,
    }, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("v2xvit", "cobevt"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
