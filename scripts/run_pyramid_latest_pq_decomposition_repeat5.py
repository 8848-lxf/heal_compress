#!/usr/bin/env python3
"""Build and repeatedly evaluate latest Pyramid P-only/Q-only controls.

The latest six-budget Greedy/GA P+Q engines are immutable inputs.  This runner
loads each accepted source phenotype (rather than re-decoding width genes),
derives P-only and Q-only controls, deploys every unique control, and evaluates
the logical 24-control matrix on one H800 with matched strict-FP32 pre/post
replays.  Build and evaluation phases run in separate processes so the model
used for deployment cannot retain CUDA allocations during engine evaluation.
The current campaign default is three repetitions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.ablation.lidar_pyramid_prune_quant import (  # noqa: E402
    ablation_config_signature,
    build_ablation_phenotype,
)
from search.candidate import CandidatePhenotype  # noqa: E402
from search.hashing import canonical_json_hash  # noqa: E402
from search.integration.heal_lidar_family_fair_evaluation import (  # noqa: E402
    compact_result,
    evaluate_existing_family_engine,
    gpu_snapshot,
    load_resumable_family_evaluation,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from search.integration.lidar_pyramid_context import (  # noqa: E402
    build_lidar_pyramid_context,
)
from search.proxy.bops_proxy import BOPSProxy  # noqa: E402
from search.proxy.parameter_slice_resolver import (  # noqa: E402
    build_unit_parameter_slices,
)
from search.proxy.runtime_shape_profiler import (  # noqa: E402
    profile_runtime_layer_shapes,
)
from search.proxy.size_proxy import SizeProxy  # noqa: E402
from search.stage2.lidar_pyramid_real_evaluator import (  # noqa: E402
    LidarPyramidRealEvaluator,
)
from search.stage2.objective import Stage2ObjectiveConfig  # noqa: E402


BUDGET_LABELS = ("030", "025", "020", "015", "010", "005")
VARIANTS = ("prune_only", "quant_only")
METRICS = (
    "AP@0.3", "AP@0.5", "AP@0.7", "mAP", "forward_mean_ms",
    "forward_p50_ms", "forward_p90_ms", "forward_p99_ms",
    "postprocess_mean_ms", "total_mean_ms", "total_p50_ms",
)
REQUIRED_ACCEPTANCE = (
    "physical_validation.json",
    "physical_plan_validation.json",
    "engine_structure_validation.json",
    "precision_realization_validation.json",
    "merge_precision_realization.json",
    "production_qdq_boundary_audit.json",
)


def _artifact_from_result(payload: Mapping[str, Any]) -> Path:
    metadata = dict(payload.get("metadata") or {})
    raw = dict(metadata.get("raw") or {})
    value = (
        raw.get("source_artifact_dir")
        or metadata.get("source_artifact_dir")
        or metadata.get("artifact_dir")
    )
    if not value:
        raise RuntimeError("pyramid_pq_source_artifact_missing")
    result = Path(str(value)).resolve()
    if not (result / "phenotype.json").is_file():
        raise RuntimeError(f"pyramid_pq_source_phenotype_missing:{result}")
    return result


def _parse_budget_labels(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"pyramid_pq_budget_labels_invalid:{value}")
    invalid = sorted(set(labels) - set(BUDGET_LABELS))
    if invalid:
        raise ValueError(f"pyramid_pq_budget_labels_unknown:{invalid}")
    return labels


def _source_rows(
    search_root: Path, budget_labels: tuple[str, ...]
) -> list[dict[str, Any]]:
    formal_path = search_root / "reports/formal_ga_results.json"
    formal = read_json(formal_path)
    results = dict(formal.get("results") or {})
    missing = sorted(set(budget_labels) - set(results))
    if missing:
        raise RuntimeError(f"pyramid_pq_missing_completed_budgets:{missing}")
    rows: list[dict[str, Any]] = []
    for method, key in (("greedy", "greedy_anchor"), ("ga", "final_winner")):
        for label in budget_labels:
            payload = dict(results[label][key])
            source_hash = str(payload["complete_phenotype_hash"])
            artifact = _artifact_from_result(payload)
            phenotype_path = artifact / "phenotype.json"
            rows.append(
                {
                    "method": method,
                    "budget_label": label,
                    "budget": float(int(label) / 100.0),
                    "source_candidate_hash": source_hash,
                    "source_artifact_dir": str(artifact),
                    "source_phenotype_path": str(phenotype_path),
                    "source_phenotype_sha256": sha256_file(phenotype_path),
                    "source_phenotype_size_bytes": phenotype_path.stat().st_size,
                }
            )
    return rows


def _load_compact_source_phenotype(
    source: Mapping[str, Any],
    *,
    quantization_groups: Iterable[Any],
) -> tuple[CandidatePhenotype, dict[str, Any]]:
    """Load one accepted phenotype without derived contract-history bloat.

    Stage-2 appends graph-resolved merge observations to quantization group
    contracts.  Those observations are deployment outputs rather than
    candidate state, and recursively carrying them into later phenotypes made
    some historical files hundreds of megabytes.  Drop only that derived field
    and rebuild clean static contracts from the current SearchSpace.  The
    accepted structure and precision decisions remain unchanged.
    """

    path = Path(str(source["source_phenotype_path"])).resolve()
    query = (
        "{source_contract_count:(.metadata.quantization_group_contracts|length),"
        "source_merge_boundary_count:([.metadata.quantization_group_contracts[]?"
        ".merge_boundaries? // [] | length]|add // 0),"
        "phenotype:del(.metadata.quantization_group_contracts)}"
    )
    filtered = subprocess.run(
        ["jq", "-c", query, str(path)],
        check=True,
        capture_output=True,
    )
    envelope = json.loads(filtered.stdout)
    compact_source = CandidatePhenotype.from_dict(envelope["phenotype"])
    metadata = dict(compact_source.metadata)
    requested = {
        str(key): str(value).upper()
        for key, value in dict(metadata.get("requested_group_profile") or {}).items()
    }
    legalized = {
        str(key): str(value).upper()
        for key, value in dict(
            metadata.get("stage1_legalized_group_profile") or requested
        ).items()
    }
    expansion = {
        str(key): [str(item) for item in value]
        for key, value in dict(metadata.get("precision_group_expansion") or {}).items()
    }
    groups = {str(group.group_id): group for group in quantization_groups}
    expected = set(groups)
    for name, values in (
        ("requested_group_profile", requested),
        ("stage1_legalized_group_profile", legalized),
        ("precision_group_expansion", expansion),
    ):
        if set(values) != expected:
            raise RuntimeError(
                f"pyramid_pq_source_group_schema_mismatch:{name}:"
                f"missing={sorted(expected-set(values))}:extra={sorted(set(values)-expected)}"
            )
    clean_contracts: dict[str, dict[str, Any]] = {}
    module_precision_mismatches: list[dict[str, str]] = []
    for group_id, group in sorted(groups.items()):
        members = [str(value) for value in group.module_paths]
        if expansion[group_id] != members:
            raise RuntimeError(
                f"pyramid_pq_source_group_expansion_mismatch:{group_id}:"
                f"{expansion[group_id]}:{members}"
            )
        if legalized[group_id] not in set(group.allowed_precisions):
            raise RuntimeError(
                f"pyramid_pq_source_illegal_group_precision:{group_id}:"
                f"{legalized[group_id]}:{group.allowed_precisions}"
            )
        for module_path in members:
            decision = compact_source.precision_profile.get(module_path)
            if decision is None or (
                decision.requested_precision != requested[group_id]
                or decision.realized_precision != legalized[group_id]
            ):
                module_precision_mismatches.append(
                    {
                        "group_id": group_id,
                        "module_path": module_path,
                        "group_requested": requested[group_id],
                        "group_legalized": legalized[group_id],
                        "module_requested": (
                            decision.requested_precision if decision is not None else "MISSING"
                        ),
                        "module_realized": (
                            decision.realized_precision if decision is not None else "MISSING"
                        ),
                    }
                )
        clean_contracts[group_id] = {
            **dict(group.metadata),
            "member_layers": members,
            "requested_precision": requested[group_id],
            "legalized_precision": legalized[group_id],
            "realized_precision": legalized[group_id],
        }
    if module_precision_mismatches:
        raise RuntimeError(
            f"pyramid_pq_source_module_precision_mismatch:{module_precision_mismatches}"
        )
    metadata["quantization_group_contracts"] = clean_contracts
    phenotype = CandidatePhenotype(
        pruned_unit_ids=list(compact_source.pruned_unit_ids),
        precision_profile=compact_source.precision_profile,
        pruning_policy_version=compact_source.pruning_policy_version,
        precision_policy_version=compact_source.precision_policy_version,
        metadata=metadata,
    )
    source_semantics = {
        "pruned_unit_ids": list(compact_source.pruned_unit_ids),
        "precision_profile": compact_source.realized_precision_profile,
        "domain_width_profile": metadata.get("domain_width_profile", {}),
        "group_keep_map_by_scope": metadata.get("group_keep_map_by_scope", {}),
        "group_prune_map_by_scope": metadata.get("group_prune_map_by_scope", {}),
    }
    audit = {
        "source_candidate_hash": source["source_candidate_hash"],
        "source_phenotype_path": str(path),
        "source_phenotype_sha256": source["source_phenotype_sha256"],
        "source_size_bytes": int(source["source_phenotype_size_bytes"]),
        "filtered_json_size_bytes": len(filtered.stdout),
        "source_contract_count": int(envelope["source_contract_count"]),
        "source_merge_boundary_count": int(envelope["source_merge_boundary_count"]),
        "clean_contract_count": len(clean_contracts),
        "clean_merge_boundary_count": sum(
            len(dict(contract).get("merge_boundaries", []))
            for contract in clean_contracts.values()
        ),
        "candidate_semantics_hash": canonical_json_hash(source_semantics),
        "structure_and_precision_preserved": True,
        "derived_contract_history_removed": True,
    }
    return phenotype, audit


def _precision_counts(phenotype: CandidatePhenotype) -> dict[str, int]:
    values = [str(value).upper() for value in phenotype.realized_precision_profile.values()]
    return {
        "fp32_count": values.count("FP32"),
        "fp16_count": values.count("FP16"),
        "int8_count": values.count("INT8"),
    }


def _acceptance(artifact: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    failures: list[str] = []
    for name in REQUIRED_ACCEPTANCE:
        path = artifact / name
        if not path.is_file():
            failures.append(f"missing:{name}")
            continue
        payload = read_json(path)
        rows[name] = {"passed": payload.get("passed"), "sha256": sha256_file(path)}
        if not bool(payload.get("passed", False)):
            failures.append(f"failed:{name}")
    return {"passed": not failures, "failures": failures, "reports": rows}


def _physical_metrics(artifact: Path) -> dict[str, Any]:
    payload = read_json(artifact / "physical_hash.json")
    base = payload.get("parameter_count_base")
    pruned = payload.get("parameter_count_pruned")
    return {
        "parameter_count_base": base,
        "parameter_count_pruned": pruned,
        "parameter_reduction": payload.get("parameter_reduction"),
        "parameter_retention": (
            float(pruned) / float(base) if base and pruned is not None else None
        ),
        "parameter_compression_ratio": (
            float(base) / float(pruned) if base and pruned else None
        ),
    }


def _build(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    start = gpu_snapshot(args.physical_gpu)
    if int(start["memory_used_mib"]) > 256 or int(start["utilization_percent"]) > 5:
        raise RuntimeError(f"pyramid_pq_build_gpu_not_idle:{start}")
    budget_labels = _parse_budget_labels(args.budget_labels)
    expected_logical_rows = 4 * len(budget_labels)
    sources = _source_rows(args.search_root.resolve(), budget_labels)
    logical_gpu = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else args.physical_gpu
    context = build_lidar_pyramid_context(
        checkpoint_path=args.checkpoint,
        output_dir=root / "builder_context",
        model_config_path=args.model_config,
        heal_root=args.heal_root,
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin,
        gpu_id=str(logical_gpu),
        exclude_gpu_ids=[],
        tensorrt_env="modelopt",
        fisher_calibration_batches=8,
        quant_calibration_batches=200,
        quant_calibration_npz_manifest=args.calibration_manifest,
        quant_activation_calibration_backend="tensorrt_entropy_calibration2",
        quant_calibration_force_rebuild=True,
        num_frames=1789,
        warmup_frames=200,
        reset_after_warmup=True,
        default_precision="FP32",
        pruning_gene_type="legal_domain_width",
        grouped_allowed_channels_per_group=[4, 8, 16, 32, 64, 128, 256, 512],
    )
    runtime = profile_runtime_layer_shapes(
        context.model,
        context.trace_example_inputs,
        forward_fn=context.model_bundle.adapter.forward_for_task,
    )
    slices = build_unit_parameter_slices(context.model, context.atomic_prune_units)
    bops = BOPSProxy(
        context.model,
        unit_to_parameter_slices=slices,
        runtime_shapes=runtime.shapes,
        default_precision="FP32",
    )
    size = SizeProxy(
        context.model,
        unit_to_parameter_slices=slices,
        default_precision="FP32",
        include_constant_parameters_in_size=True,
    )
    evaluator = LidarPyramidRealEvaluator(
        context=context,
        run_dir=root / "builder_runtime",
        num_frames=1789,
        warmup_frames=200,
        latency_rounds=3,
        stage2_config=Stage2ObjectiveConfig(),
    )
    logical_rows: list[dict[str, Any]] = []
    built_by_signature: dict[str, dict[str, Any]] = {}
    compaction_rows: list[dict[str, Any]] = []
    for source in sources:
        source_phenotype, compaction = _load_compact_source_phenotype(
            source,
            quantization_groups=context.search_space.quantization_groups,
        )
        compaction_rows.append(compaction)
        write_json(
            root / "reports/historical_metadata_compaction_audit.json",
            {
                "schema_version": "pyramid-phenotype-derived-contract-compaction-v1",
                "historical_quantization_group_contract_bloat_detected": any(
                    int(row["source_merge_boundary_count"])
                    > int(row["clean_merge_boundary_count"])
                    for row in compaction_rows
                ),
                "candidate_state_changed": False,
                "rows": compaction_rows,
            },
        )
        for variant in VARIANTS:
            phenotype = build_ablation_phenotype(source_phenotype, variant)
            signature = ablation_config_signature(phenotype)
            control_hash = canonical_json_hash(
                {
                    "schema": "pyramid-latest-pq-control-v1",
                    "variant": variant,
                    "phenotype": phenotype.to_dict(),
                }
            )
            destination = root / "engines" / signature
            if signature in built_by_signature:
                deployment = dict(built_by_signature[signature])
                deduplicated_from = deployment["control_hash"]
            else:
                status_path = destination / "stage2_deployment.json"
                if status_path.is_file():
                    deployment = read_json(status_path)
                else:
                    deployment = evaluator.deploy_candidate(
                        phenotype,
                        output_dir=destination,
                        candidate_hash=control_hash,
                    )
                if str(deployment.get("status")) != "ok":
                    raise RuntimeError(
                        f"pyramid_pq_deployment_failed:{control_hash}:"
                        f"{deployment.get('failure_reason', deployment.get('status'))}"
                    )
                deployment["control_hash"] = control_hash
                built_by_signature[signature] = dict(deployment)
                deduplicated_from = ""
            artifact = Path(str(deployment["artifact_dir"])).resolve()
            engine = artifact / "engine.plan"
            if not engine.is_file():
                engine = Path(str(deployment.get("engine_path", ""))).resolve()
            if not engine.is_file():
                raise RuntimeError(f"pyramid_pq_engine_missing:{artifact}")
            acceptance = _acceptance(artifact)
            if not acceptance["passed"]:
                raise RuntimeError(
                    f"pyramid_pq_acceptance_failed:{artifact}:{acceptance['failures']}"
                )
            bops_metrics = bops.evaluate_breakdown(phenotype)
            size_metrics = size.evaluate_breakdown(phenotype)
            physical = _physical_metrics(artifact)
            row = {
                "family_id": "lidar_pyramid",
                "assigned_method": source["method"],
                "method": source["method"],
                "variant": variant,
                "budget": source["budget"],
                "budget_label": source["budget_label"],
                "item_id": (
                    f"{source['method']}_budget_{source['budget_label']}_{variant}_"
                    f"{control_hash[:12]}"
                ),
                "source_candidate_hash": source["source_candidate_hash"],
                "source_artifact_dir": source["source_artifact_dir"],
                "source_phenotype_sha256": source["source_phenotype_sha256"],
                "control_hash": control_hash,
                "config_signature": signature,
                "deduplicated_from": deduplicated_from,
                "artifact_dir": str(artifact),
                "engine_path": str(engine),
                "engine_sha256": sha256_file(engine),
                "engine_size_bytes": engine.stat().st_size,
                "acceptance": acceptance,
                "actual_bops": bops_metrics.get("R_bops_vs_fp32"),
                "mixed_weight_retention": size_metrics.get("R_size_vs_fp32"),
                "mixed_weight_compression_ratio": (
                    1.0 / float(size_metrics["R_size_vs_fp32"])
                    if float(size_metrics.get("R_size_vs_fp32", 0.0)) > 0.0
                    else None
                ),
                **physical,
                **_precision_counts(phenotype),
            }
            logical_rows.append(row)
            write_json(
                root / "reports/engine_inventory.json",
                {
                    "schema_version": "pyramid-latest-pq-control-engine-v1",
                    "source_search_root": str(args.search_root.resolve()),
                    "logical_row_count": len(logical_rows),
                    "expected_logical_row_count": expected_logical_rows,
                    "unique_engine_count": len(built_by_signature),
                    "gpu_start": start,
                    "rows": logical_rows,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "pyramid_pq_engine_ready",
                        "method": source["method"],
                        "budget": source["budget"],
                        "variant": variant,
                        "control_hash": control_hash,
                        "unique_engine_count": len(built_by_signature),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del source_phenotype
        gc.collect()
    if len(logical_rows) != expected_logical_rows:
        raise RuntimeError(
            f"pyramid_pq_inventory_count:{len(logical_rows)}:"
            f"{expected_logical_rows}"
        )
    write_json(root / "reports/build_complete.json", {
        "passed": True,
        "logical_row_count": len(logical_rows),
        "unique_engine_count": len(built_by_signature),
        "all_acceptance_passed": all(row["acceptance"]["passed"] for row in logical_rows),
    })
    del evaluator, bops, size, runtime, slices, context
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


def _baseline_source(args: argparse.Namespace) -> dict[str, Any]:
    engine = (
        args.search_root.resolve()
        / "generation_winner_validation_runtime/baselines/original_strict_fp32/engine.plan"
    )
    if not engine.is_file():
        raise RuntimeError(f"pyramid_pq_baseline_missing:{engine}")
    return {
        "family_id": "lidar_pyramid",
        "assigned_method": "baseline",
        "method": "baseline",
        "variant": "fp32",
        "budget": None,
        "actual_bops": 1.0,
        "item_id": "strict_fp32",
        "engine_path": str(engine),
        "engine_sha256": sha256_file(engine),
        "parameter_reduction": 0.0,
        "parameter_retention": 1.0,
        "parameter_compression_ratio": 1.0,
        "mixed_weight_retention": 1.0,
        "mixed_weight_compression_ratio": 1.0,
        "fp32_count": None,
        "fp16_count": None,
        "int8_count": None,
    }


def _evaluate_one(
    source: Mapping[str, Any],
    *,
    repeat: int,
    destination: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if destination.exists():
        return load_resumable_family_evaluation(
            source=source,
            gpu_id=args.physical_gpu,
            repeat_index=repeat,
            output_dir=destination,
            eval_manifest_path=args.eval_manifest,
            num_frames=1789,
            warmup_frames=200,
            latency_rounds=3,
            fixed_k=29696,
        )
    return evaluate_existing_family_engine(
        source=source,
        gpu_id=args.physical_gpu,
        repeat_index=repeat,
        output_dir=destination,
        model_config=args.model_config,
        heal_root=args.heal_root,
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin,
        eval_manifest_path=args.eval_manifest,
        num_frames=1789,
        warmup_frames=200,
        latency_rounds=3,
        fixed_k=29696,
        max_agents=2,
    )


def _aggregate(
    rows: list[dict[str, Any]], repeat_count: int
) -> list[dict[str, Any]]:
    baselines: dict[int, float] = {}
    for repeat in range(repeat_count):
        values = [
            float(row["forward_p50_ms"])
            for row in rows
            if row["assigned_method"] == "baseline" and row["repeat_index"] == repeat
        ]
        if len(values) != 2:
            raise RuntimeError(f"pyramid_pq_baseline_replay_count:{repeat}:{values}")
        baselines[repeat] = statistics.fmean(values)
    grouped: dict[tuple[str, float | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["assigned_method"]), row.get("budget"), str(row["variant"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, budget, variant), values in grouped.items():
        expected = 2 * repeat_count if method == "baseline" else repeat_count
        if len(values) != expected:
            raise RuntimeError(
                f"pyramid_pq_repeat_count:{method}:{budget}:{variant}:{len(values)}"
            )
        first = values[0]
        record = {
            key: first.get(key)
            for key in (
                "assigned_method", "variant", "budget", "actual_bops",
                "source_candidate_hash", "control_hash", "engine_sha256",
                "engine_size_bytes", "parameter_count_base",
                "parameter_count_pruned", "parameter_reduction",
                "parameter_retention", "parameter_compression_ratio",
                "mixed_weight_retention", "mixed_weight_compression_ratio",
                "fp32_count", "fp16_count", "int8_count",
            )
        }
        record["repeat_count"] = expected
        for metric in METRICS:
            data = [float(row[metric]) for row in values]
            record[f"{metric}_mean"] = statistics.fmean(data)
            record[f"{metric}_std"] = statistics.pstdev(data)
        if method != "baseline":
            speedups = [
                baselines[int(row["repeat_index"])] / float(row["forward_p50_ms"])
                for row in values
            ]
            record["speedup_vs_matched_b0_mean"] = statistics.fmean(speedups)
            record["speedup_vs_matched_b0_std"] = statistics.pstdev(speedups)
        output.append(record)
    return sorted(
        output,
        key=lambda row: (
            str(row["assigned_method"]),
            -(float(row.get("budget") or 1.0)),
            str(row["variant"]),
        ),
    )


def _evaluate(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    build_complete = read_json(root / "reports/build_complete.json")
    if not bool(build_complete.get("passed")):
        raise RuntimeError("pyramid_pq_build_not_complete")
    start = gpu_snapshot(args.physical_gpu)
    if int(start["memory_used_mib"]) > 256 or int(start["utilization_percent"]) > 5:
        raise RuntimeError(f"pyramid_pq_eval_gpu_not_idle:{start}")
    inventory = read_json(root / "reports/engine_inventory.json")
    candidates = [dict(row) for row in inventory["rows"]]
    expected_logical_rows = 4 * len(_parse_budget_labels(args.budget_labels))
    if len(candidates) != expected_logical_rows:
        raise RuntimeError(
            f"pyramid_pq_eval_inventory_count:{len(candidates)}:"
            f"{expected_logical_rows}"
        )
    baseline = _baseline_source(args)
    rows: list[dict[str, Any]] = []
    for repeat in range(int(args.repeat_count)):
        ordered: list[tuple[str, dict[str, Any]]] = [
            ("b0_pre", baseline),
            *[(str(row["item_id"]), row) for row in candidates],
            ("b0_post", baseline),
        ]
        for order, (name, source) in enumerate(ordered):
            item = dict(source)
            item["sequence_index"] = order
            if item["assigned_method"] == "baseline":
                item["item_id"] = name
            destination = root / f"repeat_{repeat:02d}/{order:02d}_{name}"
            result = _evaluate_one(
                item, repeat=repeat, destination=destination, args=args
            )
            rows.append({**item, **compact_result(result), "repeat_index": repeat})
            write_csv(root / "reports/repeat_results.csv", rows)
            write_json(root / "reports/progress.json", {
                "completed_items": len(rows),
                "expected_items": int(args.repeat_count) * (
                    2 + expected_logical_rows
                ),
                "last_repeat": repeat,
                "last_item": name,
            })
    aggregate = _aggregate(rows, int(args.repeat_count))
    write_csv(
        root / f"reports/repeat{int(args.repeat_count)}_mean_std.csv", aggregate
    )
    passed = all(
        int(row["num_evaluated_frames"]) == 1789
        and int(row["num_skipped_frames"]) == 0
        and math.isfinite(float(row["mAP"]))
        for row in rows
    )
    write_json(root / "reports/final_report.json", {
        "passed": passed,
        "logical_control_count": expected_logical_rows,
        "unique_engine_count": build_complete["unique_engine_count"],
        "repeat_count": int(args.repeat_count),
        "evaluation_frames": 1789,
        "result_count": len(rows),
        "all_evaluated_1789": all(int(row["num_evaluated_frames"]) == 1789 for row in rows),
        "all_skipped_zero": all(int(row["num_skipped_frames"]) == 0 for row in rows),
        "aggregate": aggregate,
        "prior_joint_repeat_report": str(args.prior_pq_report.resolve()),
        "gpu_end": gpu_snapshot(args.physical_gpu),
    })
    if not passed:
        raise RuntimeError("pyramid_pq_repeat_acceptance_failed")
    return 0


def _child_command(args: argparse.Namespace, phase: str) -> list[str]:
    values = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase", phase,
        "--search-root", str(args.search_root),
        "--output-root", str(args.output_root),
        "--physical-gpu", str(args.physical_gpu),
        "--repeat-count", str(args.repeat_count),
        "--eval-manifest", str(args.eval_manifest),
        "--prior-pq-report", str(args.prior_pq_report),
        "--checkpoint", str(args.checkpoint),
        "--model-config", str(args.model_config),
        "--heal-root", str(args.heal_root),
        "--tensorrt-root", str(args.tensorrt_root),
        "--plugin", str(args.plugin),
        "--calibration-manifest", str(args.calibration_manifest),
        "--budget-labels", str(args.budget_labels),
    ]
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "build", "evaluate"), default="all")
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument(
        "--budget-labels", default=",".join(BUDGET_LABELS),
        help="Comma-separated completed budget labels, for example 005 or 030,025.",
    )
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--prior-pq-report", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"),
    )
    parser.add_argument(
        "--model-config", type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"),
    )
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--calibration-manifest", type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200/manifest.json"),
    )
    args = parser.parse_args()
    if int(args.repeat_count) < 2:
        raise ValueError("pyramid_pq_repeat_count_must_be_at_least_two")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", str(args.physical_gpu)):
        raise RuntimeError("pyramid_pq_cuda_visible_devices_mismatch")
    if args.phase == "build":
        return _build(args)
    if args.phase == "evaluate":
        return _evaluate(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(args.output_root / "reports/provenance.json", {
        "schema_version": "pyramid-pq-decomposition-repeat-configurable-v2",
        "source_search_root": str(args.search_root.resolve()),
        "source_formal_results_sha256": sha256_file(
            args.search_root / "reports/formal_ga_results.json"
        ),
        "prior_joint_repeat_report": str(args.prior_pq_report.resolve()),
        "prior_joint_repeat_report_sha256": sha256_file(args.prior_pq_report),
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": gpu_snapshot(args.physical_gpu)["uuid"],
        "variants_built_and_evaluated": list(VARIANTS),
        "repeat_count": int(args.repeat_count),
        "budget_labels": list(_parse_budget_labels(args.budget_labels)),
        "evaluation_frames": 1789,
        "warmup_frames": 200,
        "source_phenotype_decode_policy": "load_accepted_source_phenotype_no_gene_redecode",
    })
    subprocess.run(_child_command(args, "build"), check=True)
    subprocess.run(_child_command(args, "evaluate"), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
