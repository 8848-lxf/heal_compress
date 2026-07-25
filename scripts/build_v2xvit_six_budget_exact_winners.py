#!/usr/bin/env python3
"""Build B0 and every six-budget Greedy exact winner, with no substitution."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load
from scripts.run_v2xvit_greedy005_full import _build_full_space, _formal_space
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision
from search.canonicalization import canonicalize_candidate
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


BUDGET_LABELS = ("030", "025", "020", "015", "010", "005")


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def frozen_domains(
    domains: tuple[Any, ...], phenotype_payload: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Restore the exact nested coordinate sets serialized by Stage-1."""

    metadata = dict(phenotype_payload["metadata"])
    states = dict(metadata["domains"])
    restored = []
    for domain in domains:
        state = dict(states[str(domain.domain_id)])
        target = int(state["retained_width"])
        pruned = tuple(str(value) for value in state["pruned_unit_ids"])
        width_map = dict(domain.width_to_pruned_unit_ids)
        width_map[target] = pruned
        ranking_groups = dict(domain.ranking_groups)
        decoded = dict(state.get("decoded_width_state") or {})
        if domain.domain_type == "attention_dh" and target < int(domain.original_width):
            original = int(domain.original_width)
            qk_keep = tuple(tuple(int(value) for value in row) for row in decoded["qk_keep_by_head"])
            vo_keep = tuple(tuple(int(value) for value in row) for row in decoded["vo_keep_by_head"])
            ranking_groups["qk_low_to_high_by_head"] = tuple(
                tuple(sorted(set(range(original)) - set(keep))) + keep for keep in qk_keep
            )
            ranking_groups["vo_low_to_high_by_head"] = tuple(
                tuple(sorted(set(range(original)) - set(keep))) + keep for keep in vo_keep
            )
        elif domain.domain_type == "ffn_hidden" and target < int(domain.original_width):
            keep = tuple(int(value) for value in decoded["keep_indices"])
            ranking_groups["ffn_low_to_high"] = (
                tuple(sorted(set(range(int(domain.original_width))) - set(keep))) + keep
            )
        restored.append(
            replace(
                domain,
                width_to_pruned_unit_ids=width_map,
                ranking_groups=ranking_groups,
            )
        )
    return tuple(restored)


def fp32_phenotype(source: CandidatePhenotype, *, pruned: bool) -> CandidatePhenotype:
    return CandidatePhenotype(
        pruned_unit_ids=list(source.pruned_unit_ids) if pruned else [],
        precision_profile={
            path: PrecisionDecision("FP32", "FP32", "")
            for path in source.precision_profile
        },
        pruning_policy_version=source.pruning_policy_version,
        precision_policy_version=source.precision_policy_version,
        metadata={
            "control": "S32" if pruned else "B0",
            "source_domain_width_profile": dict(
                source.metadata.get("domain_width_profile") or {}
            ),
        },
    )


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"exact_build_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    identity = _build_full_space(model, adapter, hypes, batch)
    provenance = json.loads((root / "reports/input_provenance.json").read_text())
    formal = _formal_space(
        model,
        adapter,
        hypes,
        batch,
        identity,
        str(provenance["taylor_manifest_hash"]),
    )
    qkv_paths = tuple(
        path
        for spec in formal["components"].attention_instances
        for path in (spec.q_projection_paths + spec.k_projection_paths)
    )
    winner_payloads = {
        label: json.loads(
            (root / f"greedy/budget_{label}/exact_winner.json").read_text()
        )
        for label in BUDGET_LABELS
    }
    source_phenotype = CandidatePhenotype.from_dict(
        winner_payloads["030"]["phenotype"]
    )
    controls_root = root / "engines/greedy_exact_winners"
    reports: dict[str, Any] = {
        "schema_version": "v2xvit-six-budget-greedy-exact-build-v1",
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "controls": {},
    }

    b0_dir = controls_root / "B0"
    b0_dir.mkdir(parents=True, exist_ok=False)
    b0 = fp32_phenotype(source_phenotype, pruned=False)
    b0_hash = stable_hash({"control": "B0", "checkpoint": provenance["checkpoint_sha256"]})
    b0_export = _export_candidate(
        b0_dir,
        model,
        adapter,
        batch,
        hypes,
        b0,
        b0_hash,
        b0_hash,
        build_engine=True,
        tensorrt_root=args.tensorrt_root,
        plugin=args.plugin,
        calibration_frames=0,
        qkv_paths=qkv_paths,
        fixed_k_override=args.fixed_k,
        physical_gpu_id=args.physical_gpu,
    )
    reports["controls"]["B0"] = b0_export
    if not b0_export.get("passed"):
        write(root / "reports/six_budget_builds.json", reports)
        raise RuntimeError(f"B0_exact_build_failed:{b0_export.get('failure')}")

    for label in BUDGET_LABELS:
        payload = winner_payloads[label]
        candidate_hash = str(payload["candidate_hash"])
        candidate = CandidateGenotype.from_dict(payload["genotype"])
        source = CandidatePhenotype.from_dict(payload["phenotype"])
        domains = frozen_domains(tuple(formal["space"].pruning_domains), payload["phenotype"])
        physical = materialize_unified_widths(
            model,
            identity["cnn_units"],
            domains,
            candidate.pruning_width_genes,
            model_name="lidar_v2xvit",
        )
        physical_report = physical.report.to_dict()
        budget_root = controls_root / f"budget_{label}"
        budget_root.mkdir(parents=True, exist_ok=False)
        write(budget_root / "physical_report.json", physical_report)
        write(budget_root / "exact_winner.json", payload)
        if not physical.report.passed:
            reports["controls"][f"budget_{label}"] = {
                "status": "physical_failed",
                "physical_report": physical_report,
            }
            continue
        state_path = budget_root / "physical_state_dict.pth"
        torch.save(
            {
                "model": {
                    name: value.detach().cpu()
                    for name, value in physical.model.state_dict().items()
                },
                "structure_hash": physical.report.structure_hash,
            },
            state_path,
        )
        control_results = {"physical_report": physical_report}
        for control, phenotype in (
            ("S32", fp32_phenotype(source, pruned=True)),
            ("JMIX-FRESH", source),
        ):
            destination = budget_root / control
            destination.mkdir(parents=True, exist_ok=False)
            export = _export_candidate(
                destination,
                physical.model,
                adapter,
                batch,
                hypes,
                phenotype,
                f"{candidate_hash}:{control}",
                physical.report.structure_hash,
                build_engine=True,
                tensorrt_root=args.tensorrt_root,
                plugin=args.plugin,
                calibration_frames=(200 if control == "JMIX-FRESH" else 0),
                qkv_paths=qkv_paths,
                fixed_k_override=args.fixed_k,
                physical_gpu_id=args.physical_gpu,
            )
            control_results[control] = export
            print(
                json.dumps(
                    {
                        "budget": label,
                        "control": control,
                        "passed": export.get("passed"),
                        "failure": export.get("failure"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            torch.cuda.empty_cache()
        reports["controls"][f"budget_{label}"] = control_results
        write(root / f"reports/greedy_exact_build_budget_{label}.json", control_results)
        torch.cuda.empty_cache()

    reports["all_exact_winners_attempted"] = len(
        [key for key in reports["controls"] if key.startswith("budget_")]
    ) == 6
    reports["exact_winner_substitution_count"] = 0
    write(root / "reports/six_budget_builds.json", reports)
    print(json.dumps({"status": "complete", "budgets": BUDGET_LABELS}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=6)
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    parser.add_argument("--plugin", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
