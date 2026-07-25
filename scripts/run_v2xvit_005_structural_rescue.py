#!/usr/bin/env python3
"""Diagnose the 0.05 structural collapse without changing formal winners.

The four restore controls are deliberately outside the BOPS hard gate.  The
0.08/0.07/0.06 controls are exact proxy winners reconstructed from the frozen
Greedy trace and are evaluated as structure-only (S32) candidates.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import _load
from scripts.run_v2xvit_greedy005_full import _build_full_space, _formal_space
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from scripts.run_v2xvit_six_budget_proxy import (
    FrozenTrainPrefix,
    _candidate_from_trace,
    _winner_compare,
)
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.hashing import candidate_hash
from search.model_family.calibration_manifest import load_v2xvit_train_manifest
from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt
from search.pruning_space.unified_physical_pruner import materialize_unified_widths
from search.proxy.conservative_gate_activation_taylor import (
    collect_functional_gate_scores_multi,
    rerank_domains_by_gate_scores,
)


INTERMEDIATE_TARGETS = (0.08, 0.07, 0.06)


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fp32_candidate(source: CandidateGenotype) -> CandidateGenotype:
    return CandidateGenotype(
        pruning_width_genes=dict(source.pruning_width_genes),
        precision_genes={key: "FP32" for key in source.precision_genes},
        meta={"created_by": "diagnostic_structure_only", "repair_count": 0},
    )


def candidate_with_widths(
    source: CandidateGenotype,
    changes: Mapping[str, int],
    *,
    control: str,
) -> CandidateGenotype:
    widths = dict(source.pruning_width_genes)
    unknown = sorted(set(changes) - set(widths))
    if unknown:
        raise RuntimeError(f"diagnostic_unknown_width_loci:{unknown}")
    widths.update({key: int(value) for key, value in changes.items()})
    return CandidateGenotype(
        pruning_width_genes=widths,
        precision_genes=dict(source.precision_genes),
        meta={
            "created_by": "optional_budget_band_rescue",
            "diagnostic_control": True,
            "control": control,
            "repair_count": 0,
        },
    )


def exact_intermediate_winners(
    trace_path: Path,
    *,
    space: Any,
    size_evaluator: Any,
) -> dict[float, dict[str, Any]]:
    """Replay the frozen selector without retaining the 1.4-GiB trace."""

    bands: dict[float, list[dict[str, Any]]] = {
        target: [] for target in INTERMEDIATE_TARGETS
    }
    with trace_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            retention = float(row["current_retention"])
            for target in INTERMEDIATE_TARGETS:
                if abs(retention - target) <= 0.005:
                    bands[target].append(dict(row))

    winners: dict[float, dict[str, Any]] = {}
    for target, rows in bands.items():
        if not rows:
            raise RuntimeError(f"diagnostic_intermediate_budget_unreachable:{target}")
        minimum = min(float(row["cumulative_proxy"]) for row in rows)
        equivalent = [
            row
            for row in rows
            if abs(float(row["cumulative_proxy"]) - minimum)
            <= max(1.0e-12, 1.0e-8 * max(abs(minimum), abs(float(row["cumulative_proxy"]))))
        ]
        prepared = []
        for row in equivalent:
            genotype = _candidate_from_trace(row)
            phenotype = canonicalize_candidate(genotype, space)
            size = size_evaluator(phenotype)
            key = candidate_hash(phenotype, space)
            prepared.append(
                {
                    **row,
                    "genotype": genotype,
                    "phenotype": phenotype,
                    "candidate_hash": key,
                    "budget_deviation": abs(float(row["current_retention"]) - target),
                    "R_parameter_retention": float(size["R_parameter_retention"]),
                    "mixed_weight_retention": float(size["R_size_vs_fp32"]),
                    "size": size,
                }
            )
        winner = sorted(prepared, key=functools.cmp_to_key(_winner_compare))[0]
        winners[target] = {
            "candidate": winner["genotype"],
            "phenotype": winner["phenotype"],
            "candidate_hash": winner["candidate_hash"],
            "R_bops": float(winner["current_retention"]),
            "cumulative_J_total": float(winner["cumulative_proxy"]),
            "band_candidate_count": len(rows),
            "taylor_equivalence_count": len(equivalent),
        }
    return winners


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    rescue = root / "optional_budget_band_rescue"
    report_path = root / "reports/old005_structural_rescue.json"
    if report_path.is_file():
        print(json.dumps({"status": "already_complete", "report": str(report_path)}))
        return 0
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"structural_rescue_requires_one_visible_gpu:{torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    request = json.loads(
        (root / "evaluation_fixed500/B0/evaluation_request.json").read_text()
    )
    payload005 = json.loads((root / "greedy/budget_005/exact_winner.json").read_text())
    payload010 = json.loads((root / "greedy/budget_010/exact_winner.json").read_text())
    source005 = CandidateGenotype.from_dict(payload005["genotype"])
    source010 = CandidateGenotype.from_dict(payload010["genotype"])

    model, adapter, hypes, _ = _load("v2xvit", device)
    representative, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    identity = _build_full_space(model, adapter, hypes, representative)
    calibration_hash = json.loads((root / "reports/input_provenance.json").read_text())[
        "taylor_manifest_hash"
    ]
    # Fisher is not used by this diagnostic.  A single sample is sufficient to
    # instantiate the identical schema; train32 below reconstructs the frozen
    # functional-gate channel ordering used by the formal Greedy trajectory.
    formal = _formal_space(
        model, adapter, hypes, representative, identity, calibration_hash
    )
    train_manifest_path = (
        REPO / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"
    )
    train200 = load_v2xvit_train_manifest(train_manifest_path)
    train32 = FrozenTrainPrefix(
        adapter=adapter, hypes=hypes, device=device, manifest=train200, count=32
    )
    gate32, gate_mapping = collect_functional_gate_scores_multi(
        model,
        formal["space"].pruning_domains,
        forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        loss_fn=adapter.compute_task_loss,
        calibration_batches=train32,
    )
    domains = rerank_domains_by_gate_scores(formal["space"].pruning_domains, gate32)
    space = replace(
        formal["space"],
        pruning_domains=domains,
        pruning_unit_ids=[unit for domain in domains for unit in domain.ordered_unit_ids],
    )
    # A hash match on both frozen endpoints is a fail-closed ranking audit.
    endpoint_hashes = {}
    for label, payload, genotype in (
        ("005", payload005, source005),
        ("010", payload010, source010),
    ):
        actual = candidate_hash(canonicalize_candidate(genotype, space), space)
        endpoint_hashes[label] = {
            "expected": payload["candidate_hash"], "actual": actual,
            "match": actual == payload["candidate_hash"],
        }
        if actual != payload["candidate_hash"]:
            raise RuntimeError(f"diagnostic_frozen_ranking_hash_drift:{label}")

    domain_by_id = {domain.domain_id: domain for domain in domains}
    restore_ffn = {
        key: int(domain_by_id[key].original_width)
        for key in source005.pruning_width_genes
        if key.startswith("ffn_hidden::")
    }
    restore_shrinker = {
        key: int(domain_by_id[key].original_width)
        for key in source005.pruning_width_genes
        if key.startswith("shrinker_m1.")
    }
    restore_attention = {
        key: int(source010.pruning_width_genes[key])
        for key in source005.pruning_width_genes
        if key.startswith("attention_dh::")
    }
    restore_stage2 = {
        key: int(domain_by_id[key].original_width)
        for key in source005.pruning_width_genes
        if key.startswith("backbone_m1.blocks.2.")
    }
    controls: dict[str, CandidateGenotype] = {
        "restore_ffn_256": candidate_with_widths(
            source005, restore_ffn, control="restore_ffn_256"
        ),
        "restore_shrinker": candidate_with_widths(
            source005, restore_shrinker, control="restore_shrinker"
        ),
        "restore_attention_to_010": candidate_with_widths(
            source005, restore_attention, control="restore_attention_to_010"
        ),
        "restore_backbone_stage2": candidate_with_widths(
            source005, restore_stage2, control="restore_backbone_stage2"
        ),
    }
    intermediate = exact_intermediate_winners(
        root / "greedy_trace.csv",
        space=space,
        size_evaluator=formal["size"].evaluate_breakdown,
    )
    for target, row in intermediate.items():
        controls[f"greedy_intermediate_{int(round(target * 100)):03d}"] = row[
            "candidate"
        ]

    qkv_paths = tuple(
        path
        for spec in formal["components"].attention_instances
        for path in (spec.q_projection_paths + spec.k_projection_paths)
    )
    results: dict[str, Any] = {}
    for name, genotype in controls.items():
        destination = rescue / "controls" / name
        destination.mkdir(parents=True, exist_ok=False)
        phenotype = canonicalize_candidate(genotype, space)
        identity_hash = candidate_hash(phenotype, space)
        bops_mixed = formal["bops"].evaluate_breakdown(phenotype)
        s32_genotype = fp32_candidate(genotype)
        s32_phenotype = canonicalize_candidate(s32_genotype, space)
        bops_s32 = formal["bops"].evaluate_breakdown(s32_phenotype)
        physical = materialize_unified_widths(
            model,
            identity["cnn_units"],
            domains,
            genotype.pruning_width_genes,
            model_name="lidar_v2xvit",
        )
        if not physical.report.passed:
            raise RuntimeError(f"diagnostic_physical_failed:{name}")
        export_dir = destination / "S32"
        export_dir.mkdir(parents=True, exist_ok=False)
        exported = _export_candidate(
            export_dir,
            physical.model,
            adapter,
            representative,
            hypes,
            s32_phenotype,
            f"diagnostic:{identity_hash}:S32",
            physical.report.structure_hash,
            build_engine=True,
            tensorrt_root=args.tensorrt_root,
            plugin=args.plugin,
            calibration_frames=0,
            qkv_paths=qkv_paths,
            fixed_k_override=int(request["fixed_k"]),
            physical_gpu_id=args.physical_gpu,
        )
        if not exported.get("passed"):
            raise RuntimeError(f"diagnostic_export_failed:{name}:{exported.get('failure')}")
        engine = export_dir / "candidate.plan"
        evaluation = evaluate_v2xvit_engine_modelopt(
            engine_path=engine,
            model_config=request["model_config"],
            heal_root=request["heal_root"],
            output_dir=destination / "fixed500",
            tensorrt_root=args.tensorrt_root,
            plugin_path=args.plugin,
            eval_manifest_path=args.fixed500_manifest,
            physical_gpu_id=args.physical_gpu,
            fixed_k=int(request["fixed_k"]),
            max_agents=int(request["max_agents"]),
            num_frames=500,
            warmup_frames=20,
            latency_rounds=1,
            dataloader_num_workers=8,
        )
        if not (
            evaluation.get("status") == "ok"
            and int(evaluation.get("num_evaluated_frames", -1)) == 500
            and int(evaluation.get("num_skipped_frames", -1)) == 0
        ):
            raise RuntimeError(f"diagnostic_fixed500_failed:{name}")
        results[name] = {
            "diagnostic_control": True,
            "candidate_hash": identity_hash,
            "physical_structure_hash": physical.report.structure_hash,
            "widths": dict(genotype.pruning_width_genes),
            "mixed_path_R_bops": float(bops_mixed["R_bops_vs_fp32"]),
            "S32_R_bops": float(bops_s32["R_bops_vs_fp32"]),
            "engine_sha256": sha256(engine),
            "evaluation": evaluation,
            "requested_realized_width_exact": True,
            "repair_count": 0,
        }
        atomic_write(destination / "result.json", results[name])
        print(
            json.dumps(
                {
                    "control": name,
                    "mAP": evaluation["mAP"],
                    "AP50": evaluation["AP@0.5"],
                    "S32_R_bops": bops_s32["R_bops_vs_fp32"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del physical
        torch.cuda.empty_cache()

    frozen_fixed500 = json.loads(
        (root / "reports/six_budget_fixed500.json").read_text()
    )["controls"]
    baseline = frozen_fixed500["B0"]
    old005 = frozen_fixed500["budget_005/S32"]
    summary = {
        "schema_version": "v2xvit-old005-structural-rescue-v1",
        "diagnostic_control": True,
        "changes_formal_greedy_winner": False,
        "changes_ga_admission": False,
        "endpoint_hash_audit": endpoint_hashes,
        "train32_gate_sample_count": 32,
        "gate_mapping_count": len(gate_mapping),
        "B0_fixed500_mAP": float(baseline["mAP"]),
        "old005_S32_fixed500_mAP": float(old005["mAP"]),
        "intermediate_winners": {
            f"{target:.2f}": {
                key: (value.to_dict() if isinstance(value, CandidateGenotype) else value)
                for key, value in row.items()
                if key != "phenotype"
            }
            for target, row in intermediate.items()
        },
        "controls": results,
    }
    atomic_write(report_path, summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=6)
    parser.add_argument("--fixed500-manifest", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
