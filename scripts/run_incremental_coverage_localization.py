#!/usr/bin/env python3
"""Localize the E27-to-E67 explicit-Q/DQ coverage accuracy cliff.

Every point is built through the production all-keep evaluator with Legacy
exact-name activation scales.  The deterministic cumulative batches follow
the forward graph: backbone, pyramid stage 0/1/2, then deblocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_matched_coverage_baseline import (  # noqa: E402
    build_context,
    evaluation_gate,
    sha256_file,
    toolchain_manifest,
    write_json,
)


def _build_batches(profile_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    from search.baselines.original_engines import TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    legacy_order = [str(value) for value in profile["int8_module_paths"]]
    legacy = set(legacy_order)
    selected = set(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)
    if len(selected) != 27 or not selected <= legacy:
        raise RuntimeError(
            f"E27_control_not_subset_of_legacy67:e27={len(selected)}:legacy={len(legacy)}:"
            f"unexpected={sorted(selected - legacy)}"
        )
    missing = [name for name in legacy_order if name not in selected]
    categories = [
        ("backbone", lambda name: name.startswith("backbone_m1.")),
        ("pyramid_stage0", lambda name: name.startswith("pyramid_backbone.resnet.layer0.")),
        ("pyramid_stage1", lambda name: name.startswith("pyramid_backbone.resnet.layer1.")),
        ("pyramid_stage2", lambda name: name.startswith("pyramid_backbone.resnet.layer2.")),
        ("deblocks", lambda name: name.startswith("pyramid_backbone.deblocks.")),
    ]
    batches: list[dict[str, Any]] = [
        {
            "label": "E27_LS_production_control",
            "added_modules": [],
            "int8_modules": sorted(selected),
        }
    ]
    consumed: set[str] = set()
    for category, predicate in categories:
        additions = [name for name in missing if predicate(name)]
        if not additions:
            continue
        consumed.update(additions)
        selected.update(additions)
        batches.append(
            {
                "label": f"E{len(selected):02d}_LS_after_{category}",
                "category": category,
                "added_modules": additions,
                "int8_modules": sorted(selected),
            }
        )
    unresolved = sorted(set(missing) - consumed)
    if unresolved or selected != legacy or len(batches[-1]["int8_modules"]) != 67:
        raise RuntimeError(
            f"incremental_batch_partition_failed:unresolved={unresolved}:final={len(selected)}"
        )
    return batches, legacy_order


def _build_backbone_refinement(profile_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    broad, legacy_order = _build_batches(profile_path)
    selected = set(str(value) for value in broad[0]["int8_modules"])
    backbone = next(row for row in broad if row.get("category") == "backbone")
    batches: list[dict[str, Any]] = []
    # E27 and E32 are already production-built endpoints in the broad run.
    # Only E28--E31 are fresh-built here to avoid redundant artifacts.
    for addition_index, module_path in enumerate(backbone["added_modules"][:-1], start=1):
        selected.add(str(module_path))
        batches.append(
            {
                "label": f"E{len(selected):02d}_LS_backbone_refine_{addition_index:02d}",
                "category": "backbone_single_layer_refinement",
                "added_modules": [str(module_path)],
                "cumulative_backbone_additions": list(backbone["added_modules"][:addition_index]),
                "int8_modules": sorted(selected),
            }
        )
    if [len(row["int8_modules"]) for row in batches] != [28, 29, 30, 31]:
        raise RuntimeError("backbone_refinement_count_contract_failed")
    return batches, legacy_order


def _phenotype_for_modules(context: Any, modules: set[str], label: str) -> Any:
    from search.candidate import CandidatePhenotype
    from search.quantization_space.legalizer import legalize_group_precision_genes

    groups = context.search_space.quantization_groups
    known = {module for group in groups for module in group.module_paths}
    missing = sorted(modules - known)
    if missing:
        raise RuntimeError(f"incremental_profile_modules_missing:{missing}")
    requested: dict[str, str] = {}
    for group in groups:
        selected = bool(set(group.module_paths) & modules)
        if selected and (group.protected or "INT8" not in group.allowed_precisions):
            raise RuntimeError(f"incremental_profile_group_not_legal:{group.group_id}")
        requested[group.group_id] = "INT8" if selected else "FP16"
    legalization = legalize_group_precision_genes(
        requested,
        groups,
        default_precision=context.search_space.default_precision,
    )
    expanded = legalization.expand_to_module_profile()
    realized = {
        module
        for module, decision in expanded.items()
        if decision.realized_precision == "INT8"
    }
    if realized != modules:
        raise RuntimeError(
            f"incremental_profile_expansion_changed_layer_set:missing={sorted(modules-realized)}:"
            f"extra={sorted(realized-modules)}"
        )
    return CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile=expanded,
        metadata={
            **legalization.to_dict(),
            "baseline_precision": "incremental_legacy_exact_scale",
            "incremental_label": label,
        },
    )


def main(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == "broad":
        batches, legacy_order = _build_batches(args.profile)
    elif args.mode == "backbone-refinement":
        batches, legacy_order = _build_backbone_refinement(args.profile)
    else:
        broad, legacy_order = _build_batches(args.profile)
        batches = [
            {
                **broad[0],
                "label": (
                    "E27_ENT_production_control"
                    if args.activation_scales == "production-entropy"
                    else "E27_LS_production_control"
                ),
            }
        ]
    write_json(output / "toolchain_manifest.json", toolchain_manifest(args.gpu))
    write_json(
        output / "incremental_plan.json",
        {
            "profile": str(args.profile.resolve()),
            "profile_sha256": sha256_file(args.profile),
            "scale_source": (
                "fresh production TensorRT EntropyCalibration2 train200"
                if args.activation_scales == "production-entropy"
                else "Legacy exact tensor-name calibration cache"
            ),
            "manifest_frames": 200,
            "mode": args.mode,
            "endpoint_reuse_policy": (
                "E27_and_E32_read_from_broad_run_not_rebuilt"
                if args.mode == "backbone-refinement"
                else "none"
            ),
            "batches": batches,
            "legacy_graph_order": legacy_order,
        },
    )

    context = build_context(
        output / "context",
        args.gpu,
        "E67-ENT" if args.activation_scales == "production-entropy" else "E67-LS",
    )
    from search.integration.data_provider import load_split_frame_ids, write_eval_manifest
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator

    available = load_split_frame_ids(context.model_bundle.adapter, context.model_config, split="val")
    manifest = write_eval_manifest(
        output / "manifests/eval_200.json",
        num_frames=200,
        warmup_frames=10,
        available_frame_ids=available,
        reset_after_warmup=True,
    )
    context.eval_manifest_path = manifest.path
    context.eval_manifest_hash = manifest.manifest_hash

    summaries: list[dict[str, Any]] = []
    previous_map: float | None = None
    for batch in batches:
        label = str(batch["label"])
        destination = output / label
        destination.mkdir(parents=True, exist_ok=False)
        selected = set(str(value) for value in batch["int8_modules"])
        phenotype = _phenotype_for_modules(context, selected, label)
        write_json(destination / "phenotype.json", phenotype.to_dict())
        write_json(destination / "coverage_step.json", batch)
        evaluator = LidarPyramidRealEvaluator(
            context=context,
            run_dir=destination / "production",
            num_frames=200,
            warmup_frames=10,
            latency_rounds=1,
        )
        result = evaluator._deploy_and_evaluate(
            phenotype=phenotype,
            output_dir=destination / "artifacts",
            candidate_label=label,
            pruned_unit_ids=[],
            baseline_precision=None,
        )
        write_json(destination / "production_result.json", result)
        if str(result.get("status", "")) != "ok":
            raise RuntimeError(
                f"incremental_production_failed:{label}:{result.get('status')}:"
                f"{result.get('failure_reason', '')}"
            )
        gate = evaluation_gate(dict(result["evaluation"]), 200)
        artifacts = destination / "artifacts"
        precision = json.loads((artifacts / "precision_realization_validation.json").read_text())
        merge = json.loads((artifacts / "merge_precision_realization.json").read_text())
        boundary = json.loads((artifacts / "production_qdq_boundary_audit.json").read_text())
        expected_fp16 = 70 - len(selected)
        realization_ok = (
            bool(precision.get("passed"))
            and int(precision.get("requested_int8_count", -1)) == len(selected)
            and int(precision.get("realized_int8_count", -1)) == len(selected)
            and int(precision.get("realized_fp16_count", -1)) == expected_fp16
            and int(precision.get("unresolved_layer_count", -1)) == 0
        )
        gate.update(
            {
                "int8_count": len(selected),
                "fp16_count": expected_fp16,
                "unmapped_count": 0,
                "precision_realization_passed": realization_ok,
                "merge_realization_passed": bool(merge.get("passed")),
                "boundary_audit_passed": bool(boundary.get("passed")),
                "added_modules": list(batch.get("added_modules", [])),
                "delta_mAP_from_previous": (
                    None if previous_map is None else float(gate["mAP"] - previous_map)
                ),
            }
        )
        gate["passed"] = bool(
            gate["passed"]
            and gate["precision_realization_passed"]
            and gate["merge_realization_passed"]
            and gate["boundary_audit_passed"]
        )
        write_json(destination / "gate_200.json", gate)
        summaries.append({"label": label, **gate})
        write_json(output / "incremental_summary.json", {"status": "running", "steps": summaries})
        if not gate["passed"]:
            raise RuntimeError(f"incremental_gate_failed:{label}:{gate}")
        previous_map = float(gate["mAP"])
    write_json(
        output / "incremental_summary.json",
        {
            "status": "complete",
            "mode": args.mode,
            "scale_source": (
                "fresh production TensorRT EntropyCalibration2 train200"
                if args.activation_scales == "production-entropy"
                else "Legacy exact tensor-name calibration cache"
            ),
            "coverage_path": [f"{row['int8_count']} INT8 / {row['fp16_count']} FP16" for row in summaries],
            "steps": summaries,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--mode", choices=("broad", "backbone-refinement", "e27-only"), default="broad")
    parser.add_argument(
        "--activation-scales",
        choices=("legacy-exact", "production-entropy"),
        default="legacy-exact",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
