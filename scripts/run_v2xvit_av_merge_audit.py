#!/usr/bin/env python3
"""Real AV32/AV16/AV8 and window-merge deployment audit on one H800."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load
from scripts.run_v2xvit_greedy005_full import _baseline_candidate
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.canonicalization import canonicalize_candidate
from search.hashing import candidate_hash
from search.model_family.deployment import build_physical_structure_snapshot_v2
from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def _window_merge_inventory(profile_dir: Path) -> list[dict[str, Any]]:
    import onnx

    mapping = json.loads((profile_dir / "functional_precision_onnx_mapping.json").read_text())
    trt = json.loads((profile_dir / "functional_precision_trt_audit.json").read_text())
    model = onnx.load(str(profile_dir / "physical_mixed_qdq.onnx"), load_external_data=False)
    nodes = {str(node.name): node for node in model.graph.node}
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    trt_by_unit = {str(row["unit_id"]): row for row in trt.get("rows", ())}
    rows = []
    for source in mapping.get("rows", ()):
        if source.get("role") != "attention_merge":
            continue
        node = nodes[str(source["onnx_node"])]
        evidence = trt_by_unit.get(str(source["unit_id"]), {})
        rows.append({
            "canonical_id": source["unit_id"],
            "pytorch_module_path": str(source["unit_id"]).split("::", 2)[1],
            "onnx_node": source["onnx_node"],
            "trt_layers": json.dumps(evidence.get("trt_layer_names", [])),
            "transformer_layer": str(source["onnx_node"]).split("/layers.", 1)[1].split(".", 1)[0],
            "op_type": node.op_type,
            "semantic": "SplitAttn final weighted three-window branch Add",
            "input_count": len(node.input),
            "input_producers": json.dumps([
                str(producers.get(str(name)).name) if producers.get(str(name)) is not None else "graph_input"
                for name in node.input
            ]),
            "requested_input_dtype": json.dumps(source.get("input_types", [])),
            "requested_output_dtype": source.get("requested_output_precision"),
            "realized_input_dtype": json.dumps(evidence.get("trt_input_formats", [])),
            "realized_output_dtype": json.dumps(evidence.get("trt_output_formats", [])),
            "qdq_before": any(
                producers.get(str(name)) is not None
                and str(producers[str(name)].op_type) == "DequantizeLinear"
                for name in node.input
            ),
            "cast_or_reformat": any(
                producers.get(str(name)) is not None
                and str(producers[str(name)].op_type) == "Cast"
                for name in node.input
            ),
            "independent_precision_gene": False,
            "precision_owner": "derived_join_from_three_window_output_projections",
            "protection_evidence": "transformer_precision.py + canonicalize_candidate derived state",
        })
    if len(rows) != 3 or any(row["op_type"] != "Add" for row in rows):
        raise RuntimeError(f"v2xvit_window_merge_inventory_invalid:{rows}")
    return rows


def run(args: argparse.Namespace) -> int:
    if args.physical_gpu not in {2, 3}:
        raise RuntimeError(f"v2xvit_av_audit_gpu_not_allowed:{args.physical_gpu}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"v2xvit_av_audit_requires_one_visible_gpu:{torch.cuda.device_count()}")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, adapter, hypes, _ = _load("v2xvit", device)
    representative, dataset_index, agent_count = _multi_agent_validation_batch(
        adapter, hypes, device
    )
    built = _build_full_space(model, adapter, hypes, representative)
    baseline = _baseline_candidate(built["space"])
    phenotype = canonicalize_candidate(baseline, built["space"])
    base_hash = candidate_hash(phenotype, built["space"])
    structure = build_physical_structure_snapshot_v2(model)
    qkv_paths = tuple(
        path
        for spec in built["components"].attention_instances
        for path in spec.q_projection_paths + spec.k_projection_paths
    )
    write_json(root / "reports/av_profile_contracts.json", {
        "candidate_hash": base_hash,
        "physical_structure_hash": structure["snapshot_hash"],
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "profiles": ["AV32", "AV16", "AV8"],
        "same_physical_structure": True,
        "same_checkpoint": MODEL_SPECS["v2xvit"]["checkpoint"],
        "same_train200_manifest": str(REPO / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"),
    })
    results = []
    for profile in ("AV32", "AV16", "AV8"):
        destination = root / "av_audit" / profile
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / "audit_result.json"
        if result_path.is_file():
            row = json.loads(result_path.read_text())
            results.append(row)
            continue
        exported = _export_candidate(
            destination,
            model,
            adapter,
            representative,
            hypes,
            phenotype,
            base_hash,
            structure["snapshot_hash"],
            build_engine=True,
            tensorrt_root=args.tensorrt_root,
            plugin=args.plugin,
            calibration_frames=200 if profile == "AV8" else 0,
            qkv_paths=qkv_paths,
            fixed_k_override=args.fixed_k,
            physical_gpu_id=args.physical_gpu,
            av_profile=profile,
        )
        engine = destination / "candidate.plan"
        evaluation = None
        if exported.get("passed") and engine.is_file():
            evaluation = evaluate_v2xvit_engine_modelopt(
                engine_path=engine,
                model_config=MODEL_SPECS["v2xvit"]["config"],
                heal_root=args.heal_root,
                output_dir=destination / "fixed50",
                tensorrt_root=args.tensorrt_root,
                plugin_path=args.plugin,
                eval_manifest_path=args.fixed50_manifest,
                physical_gpu_id=args.physical_gpu,
                fixed_k=args.fixed_k,
                max_agents=2,
                num_frames=50,
                warmup_frames=20,
                latency_rounds=1,
                dataloader_num_workers=8,
            )
        av_audit_path = destination / "av_profile_trt_audit.json"
        av_audit = json.loads(av_audit_path.read_text()) if av_audit_path.is_file() else {}
        calibration_path = destination / "calibration_manifest.json"
        calibration = json.loads(calibration_path.read_text()) if calibration_path.is_file() else {}
        ok_eval = bool(
            evaluation
            and evaluation.get("status") == "ok"
            and int(evaluation.get("num_evaluated_frames", -1)) == 50
            and int(evaluation.get("num_skipped_frames", -1)) == 0
        )
        row = {
            "profile": profile,
            "build_supported": bool(exported.get("passed") and engine.is_file()),
            "requested_realized_exact": bool(av_audit.get("passed")),
            "conflict_count": int(av_audit.get("conflict_count", 12)),
            "unmapped_count": int(av_audit.get("unmapped_count", 12)),
            "fallback_count": int(av_audit.get("fallback_count", 12)),
            "train200_processed": int(calibration.get("processed_frames", 0)),
            "train200_skipped": int(calibration.get("skipped_frames", 0)),
            "fixed50_completed": ok_eval,
            "evaluated": int((evaluation or {}).get("num_evaluated_frames", 0)),
            "skipped": int((evaluation or {}).get("num_skipped_frames", 0)),
            "AP30": (evaluation or {}).get("AP@0.3"),
            "AP50": (evaluation or {}).get("AP@0.5"),
            "AP70": (evaluation or {}).get("AP@0.7"),
            "mAP": (evaluation or {}).get("mAP"),
            "p50_ms": (evaluation or {}).get("forward_p50_ms"),
            "engine_sha256": sha256(engine) if engine.is_file() else None,
            "failure": exported.get("failure"),
        }
        write_json(result_path, row)
        results.append(row)
        torch.cuda.empty_cache()

    av32 = next(row for row in results if row["profile"] == "AV32")
    for row in results:
        row["delta_map_vs_av32"] = (
            None if row["mAP"] is None or av32["mAP"] is None
            else float(row["mAP"]) - float(av32["mAP"])
        )
        row["speedup_vs_av32"] = (
            None if row["p50_ms"] is None or av32["p50_ms"] is None
            else float(av32["p50_ms"]) / float(row["p50_ms"])
        )
        accuracy_limit = 0.0 if row["profile"] == "AV32" else 0.005 if row["profile"] == "AV16" else 0.010
        row["accuracy_safe"] = bool(
            row["mAP"] is not None
            and av32["mAP"] is not None
            and float(row["mAP"]) >= float(av32["mAP"]) - accuracy_limit
        )
        row["legal_for_search"] = bool(
            row["build_supported"]
            and row["requested_realized_exact"]
            and row["fixed50_completed"]
            and row["conflict_count"] == 0
            and row["unmapped_count"] == 0
            and row["fallback_count"] == 0
            and row["engine_sha256"]
            and row["accuracy_safe"]
        )
    write_csv(root / "reports/av_profile_requested_realized.csv", results)
    write_csv(root / "reports/av_profile_fixed50.csv", results)
    write_json(root / "reports/av_profile_latency.json", {
        row["profile"]: {"p50_ms": row["p50_ms"], "speedup_vs_av32": row["speedup_vs_av32"]}
        for row in results
    })
    write_json(root / "reports/av_profile_acceptance.json", {
        "profiles": results,
        "legal_av_profiles": [row["profile"] for row in results if row["legal_for_search"]],
        "av32_required_gate": bool(av32["legal_for_search"]),
    })
    if not av32["legal_for_search"]:
        raise RuntimeError("v2xvit_av32_required_gate_failed")
    inventory = _window_merge_inventory(root / "av_audit/AV32")
    write_csv(root / "reports/window_merge_inventory.csv", inventory)
    write_json(root / "reports/window_merge_inventory.json", {
        "count": len(inventory), "rows": inventory
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--fixed50-manifest", type=Path, required=True)
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
