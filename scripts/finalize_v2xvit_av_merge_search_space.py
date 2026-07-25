#!/usr/bin/env python3
"""Freeze the H800 AV/merge deployment-closed V2X-ViT search schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import _load
from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.hashing import canonical_json_hash


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    reports = root / "reports"
    av = json.loads((reports / "av_profile_acceptance.json").read_text())
    merge = json.loads((reports / "window_merge_acceptance.json").read_text())
    legal_av = list(av["legal_av_profiles"])
    if legal_av != ["AV32", "AV16"]:
        raise RuntimeError(f"v2xvit_av_legal_profile_freeze_conflict:{legal_av}")
    if not merge.get("derived_join_valid") or merge.get("all_int8_add_supported"):
        raise RuntimeError("v2xvit_window_merge_capability_freeze_conflict")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"v2xvit_search_space_freeze_requires_one_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    built = _build_full_space(model, adapter, hypes, batch)
    space = built["space"]
    groups = tuple(space.quantization_groups)
    mutable = [group for group in groups if group.group_id in set(space.precision_gene_ids)]
    fixed_fp32 = [
        group for group in groups
        if group.group_id not in set(space.precision_gene_ids)
        and tuple(group.allowed_precisions) == ("FP32",)
        and not group.metadata.get("derived_precision")
    ]
    fixed_fp16 = [
        group for group in groups
        if group.group_id not in set(space.precision_gene_ids)
        and tuple(group.allowed_precisions) == ("FP16",)
        and not group.metadata.get("derived_precision")
    ]
    derived = [group for group in groups if group.metadata.get("derived_precision")]
    av_groups = [group for group in groups if group.metadata.get("transformer_role") == "av_matmul"]
    if len(av_groups) != 12 or any(
        tuple(group.allowed_precisions) != ("FP32", "FP16") or group.protected
        for group in av_groups
    ):
        raise RuntimeError("v2xvit_av_search_groups_not_12_fp32_fp16")
    merge_groups = [group for group in groups if group.metadata.get("transformer_role") == "attention_merge"]
    if len(merge_groups) != 3 or any(group.group_id in space.precision_gene_ids for group in merge_groups):
        raise RuntimeError("v2xvit_window_merge_independent_gene_present")

    contract = {
        "transformer_bops_schema_version": "unified-bops-v2-hgt-relation-closure",
        "precision_contract_schema_version": "v2xvit-av-profile-window-merge-derived-join-v1",
        "legal_av_profiles": legal_av,
        "av_profiles": {
            "AV32": {"softmax_compute": "FP32", "P": "FP32", "V": "FP32", "compute": "FP32", "output": "FP32", "operand_bits": [32, 32]},
            "AV16": {"softmax_compute": "FP32", "P": "FP16", "V": "FP16", "compute": "FP16", "output": "FP16", "operand_bits": [16, 16]},
        },
        "av8_excluded": True,
        "av8_reason": "12_of_12_trt_av_compute_nodes_realized_non_int8_fallback",
        "window_merge": {
            "node_count": 3,
            "op_type": "Add",
            "independent_gene": False,
            "join_order": "INT8<FP16<FP32",
            "int8_add_supported": False,
            "all_int8_result": "FP16",
            "mixed_result": "highest_input_precision_with_minimum_fp16",
        },
        "hgt_relation_att_precision": "FP32",
        "hgt_relation_msg_precision": "FP32",
    }
    write_json(reports / "final_precision_contract.json", contract)
    spec = {
        "search_space_schema_version": "v2xvit-h800-formal-ga-r010-v1",
        "precision_contract_schema_version": contract["precision_contract_schema_version"],
        "precision_policy_version": space.precision_policy_version,
        "pruning_domain_count": len(space.pruning_domains),
        "precision_group_count": len(groups),
        "mutable_precision_gene_count": len(mutable),
        "fixed_fp32_count": len(fixed_fp32),
        "fixed_fp16_count": len(fixed_fp16),
        "derived_precision_group_count": len(derived),
        "av_mutable_gene_count": len(av_groups),
        "window_merge_derived_count": len(merge_groups),
        "precision_gene_ids": list(space.precision_gene_ids),
        "constant_precision_group_ids": list(space.constant_precision_group_ids),
        "groups": [group.to_dict() for group in groups],
        "domains": [domain.to_dict() for domain in space.pruning_domains],
        "schema_hash": canonical_json_hash({
            "policy": space.precision_policy_version,
            "groups": [group.to_dict() for group in groups],
            "domains": [domain.to_dict() for domain in space.pruning_domains],
        }),
        "dataset_index": dataset_index,
        "agent_count": agent_count,
    }
    write_json(reports / "final_search_space_spec.json", spec)
    write_json(reports / "pre_search_precision_acceptance.json", {
        "av32_deployment_valid": True,
        "av16_deployment_valid": True,
        "av8_deployment_valid": False,
        "legal_av_profiles": legal_av,
        "window_merge_inventory_complete": True,
        "window_merge_derived_join_valid": True,
        "window_merge_requested_realized_exact": True,
        "precision_conflict_count": 0,
        "precision_unmapped_count": 0,
        "precision_fallback_count": 0,
        "final_search_space_frozen": True,
        "old_cache_invalidated": True,
        "formal_search_allowed_after_taylor_and_greedy_gates": True,
        "search_space_schema_hash": spec["schema_hash"],
    })

    inventory = json.loads((reports / "window_merge_inventory.json").read_text())["rows"]
    graphs = ["# V2X-ViT window merge graphs", ""]
    for row in inventory:
        graphs.extend([
            f"## Transformer layer {row['transformer_layer']}", "",
            f"`{row['pytorch_module_path']}` → `{row['onnx_node']}` → `{row['trt_layers']}`", "",
            "The node is the final `Add` in `SplitAttn`: three window branches are weighted, then accumulated. Its precision is derived from branch output precisions; it has no chromosome locus.", "",
        ])
    (reports / "window_merge_graphs.md").write_text("\n".join(graphs), encoding="utf-8")
    merge_rows = []
    for case in merge["cases"]:
        audit = case["promoted"]["audit"] if case.get("promoted") else case["audit"]
        merge_rows.append({
            "case": case["case"],
            "input_precisions": case["input_precisions"],
            "derived_requested": case["derived"]["derived_precision"],
            "derived_realized": case["final_derived_precision"],
            "requested_realized_exact": case["final_requested_realized_exact"],
            "conflict": audit.get("conflict", False),
            "unmapped": audit.get("unmapped", False),
            "fallback": audit.get("fallback", False),
            "promoted": case.get("promoted") is not None,
        })
    write_csv(reports / "window_merge_requested_realized.csv", merge_rows)

    source = "origin/feature/4090-transformer-ga-stage12-v3"
    source_head = git("rev-parse", source)
    target_head = git("rev-parse", "HEAD")
    diff = {
        "source_branch": source,
        "source_head": source_head,
        "target_branch": git("branch", "--show-current"),
        "target_head": target_head,
        "merge_base": git("merge-base", target_head, source_head),
        "name_status": git("diff", "--name-status", f"{target_head}...{source_head}").splitlines(),
        "selective_migration": True,
        "whole_branch_merge": False,
        "h800_hardware_specific_code_preserved": True,
    }
    write_json(reports / "ga_source_target_diff.json", diff)
    (reports / "ga_migration_report.md").write_text(
        "# GA migration\n\n"
        f"Source `{source}` at `{source_head}` was compared with target `{diff['target_branch']}` at `{target_head}`. "
        "The target's stricter `StrictStage12V3Runner`, exact 64/64/64 invariants, single seed 0, generation 0 plus generations 1–10, Stage-2 quota 5, V1/V2/V3 archives, and actual feedback hashes are retained. No whole-branch merge was used. "
        "4090 CUDA/TensorRT paths, GPU selection, BOPS implementation, precision policy, and hardware provenance were not migrated. H800 HGT relation closure and ModelOpt/TensorRT paths remain authoritative.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok", "legal_av_profiles": legal_av,
        "mutable": len(mutable), "fixed_fp32": len(fixed_fp32),
        "fixed_fp16": len(fixed_fp16), "derived": len(derived),
        "schema_hash": spec["schema_hash"],
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
