#!/usr/bin/env python3
"""Serial TensorRT precision-contract bisection on the immutable old 0.05 structure."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.audit_heal_transformer_search_models import _load
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.hashing import candidate_hash
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


def _base_state(group: Any, value: str) -> str:
    if value in group.allowed_precisions:
        return value
    return "FP32"


def _selected(group: Any, predicate: Callable[[Any], bool]) -> str:
    return "INT8" if predicate(group) and "INT8" in group.allowed_precisions else _base_state(group, "FP16")


def _gpu_uuid(log: Path) -> str:
    if not log.is_file():
        return ""
    match = re.search(r"Selected Device UUID:\s*(GPU-[0-9a-f-]+)", log.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else ""


def _inventory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"qdq_nodes": None, "cast_nodes": None}
    import onnx
    graph = onnx.load(str(path), load_external_data=False).graph
    counts = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {"qdq_nodes": counts.get("QuantizeLinear", 0) + counts.get("DequantizeLinear", 0), "quantize_nodes": counts.get("QuantizeLinear", 0), "dequantize_nodes": counts.get("DequantizeLinear", 0), "cast_nodes": counts.get("Cast", 0), "op_counts": counts}


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"bisection_gpu_visibility_mismatch:{os.environ.get('CUDA_VISIBLE_DEVICES')}!={args.physical_gpu}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"bisection_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    built = _build_full_space(model, adapter, hypes, batch)
    space = built["space"]
    old = json.loads((args.old_root / "winner/v2xvit_greedy005_winner.json").read_text(encoding="utf-8"))
    old_candidate = CandidateGenotype.from_dict(old["genotype"])
    physical = materialize_unified_widths(model, built["cnn_units"], space.pruning_domains, old_candidate.pruning_width_genes, model_name="lidar_v2xvit")
    if not physical.report.passed or physical.report.requested_widths != physical.report.realized_widths:
        raise RuntimeError(f"old005_physical_materialization_failed:{physical.report.issues}")
    components = built["components"]
    qkv_paths = tuple(path for spec in components.attention_instances for path in spec.q_projection_paths + spec.k_projection_paths)
    groups = [group for group in space.quantization_groups if group.group_id in space.precision_gene_ids]

    def path(group: Any) -> str:
        return "|".join(group.module_paths)

    profiles: list[tuple[str, dict[str, str]]] = []
    profiles.append(("S32", {group.group_id: _base_state(group, "FP32") for group in groups}))
    profiles.append(("S16", {group.group_id: _base_state(group, "FP16") for group in groups}))
    predicates: list[tuple[str, Callable[[Any], bool]]] = [
        ("INT8-CNN-only", lambda group: group.group_id.startswith("cnn_precision::backbone")),
        ("INT8-shrinker-only", lambda group: "shrinker_m1" in path(group)),
        ("INT8-FFN-only", lambda group: str(group.metadata.get("transformer_role", "")).startswith("ffn")),
        ("INT8-Attention-QKV-O-only", lambda group: group.metadata.get("transformer_role") in {"qk_projection", "v_projection", "fused_qkv_projection", "output_projection"}),
        ("INT8-agent-relation", lambda group: ".0.0.fn" in path(group)),
        ("INT8-layer0-window", lambda group: "encoder.layers.0.0.layers.0.1.fn" in path(group)),
        ("INT8-layer1-window", lambda group: "encoder.layers.1.0.layers.0.1.fn" in path(group)),
        ("INT8-layer2-window", lambda group: "encoder.layers.2.0.layers.0.1.fn" in path(group)),
    ]
    profiles.extend((name, {group.group_id: _selected(group, predicate) for group in groups}) for name, predicate in predicates)
    old_profile = {group.group_id: old_candidate.precision_genes[group.group_id] for group in groups}
    profiles.append(("old77-INT8-requested", old_profile))
    if args.only:
        wanted = {value.strip() for value in str(args.only).split(",") if value.strip()}
        profiles = [row for row in profiles if row[0] in wanted]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for order, (name, precision) in enumerate(profiles, 1):
        destination = args.output_root / name
        if destination.exists():
            raise RuntimeError(f"bisection_destination_exists:{destination}")
        destination.mkdir(parents=True)
        candidate = CandidateGenotype(pruning_width_genes=dict(old_candidate.pruning_width_genes), precision_genes=precision, meta={"bisection_profile": name})
        phenotype = canonicalize_candidate(candidate, space)
        profile_hash = candidate_hash(phenotype, space)
        export = _export_candidate(
            destination, physical.model, adapter, batch, hypes, phenotype, profile_hash,
            physical.report.structure_hash, build_engine=True, tensorrt_root=args.tensorrt_root,
            plugin=args.plugin, calibration_frames=200, qkv_paths=qkv_paths,
            fixed_k_override=args.fixed_k, physical_gpu_id=args.physical_gpu,
        )
        qdq_path = destination / "physical_mixed_qdq.onnx"
        inventory = _inventory(qdq_path)
        build = export.get("engine", {}).get("build", {})
        build_section = build.get("build", {}) if isinstance(build, dict) else {}
        failure = str(export.get("failure", ""))
        log_path = destination / "engine_build/engine_build.log"
        row = {
            "order": order,
            "profile": name,
            "candidate_hash": profile_hash,
            "physical_hash": physical.report.structure_hash,
            "precision_map_hash": phenotype.metadata.get("precision_profile_hash", ""),
            "INT8_count": sum(value == "INT8" for value in precision.values()),
            "FP16_count": sum(value == "FP16" for value in precision.values()),
            "FP32_count": sum(value == "FP32" for value in precision.values()),
            "onnx_hash": export.get("onnx", {}).get("sha256", ""),
            "qdq_node_count": inventory.get("qdq_nodes"),
            "cast_count": inventory.get("cast_nodes"),
            "requested_realized_exact": bool(export.get("passed")),
            "engine_status": "ok" if export.get("passed") and export.get("engine", {}).get("passed") else "failed",
            "failure": failure,
            "trt_failure_reason": build_section.get("failure_reason", ""),
            "actual_gpu_uuid": _gpu_uuid(log_path),
            "engine_path": str(destination / "candidate.plan") if (destination / "candidate.plan").is_file() else "",
        }
        rows.append(row)
        (destination / "bisection_profile.json").write_text(json.dumps({"row": row, "genotype": candidate.to_dict(), "phenotype": phenotype.to_dict(), "export": export, "qdq_inventory": inventory}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (args.output_root / "build_bisection.json").write_text(json.dumps({"rows": rows, "physical_report": physical.report.to_dict()}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        with (args.output_root / "build_bisection.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(json.dumps({"profile": name, "status": row["engine_status"], "gpu_uuid": row["actual_gpu_uuid"], "failure": failure[:240]}, sort_keys=True), flush=True)
        torch.cuda.empty_cache()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--only", default="", help="comma-separated profile names for a focused rerun")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
