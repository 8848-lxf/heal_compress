#!/usr/bin/env python3
"""Repeat full-validation audit of Pyramid Greedy and GA winners.

The runner is evaluation-only: it locks the existing B0/Greedy/GA engine
SHA256 values, evaluates every candidate on the same 1789-frame manifest, and
uses a matched B0 pre/post replay in every repeat for forward-latency speedup.
The current campaign default is three repetitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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


BUDGET_LABELS = ("030", "025", "020", "015", "010", "005")
METRICS = (
    "AP@0.3", "AP@0.5", "AP@0.7", "mAP", "forward_mean_ms",
    "forward_p50_ms", "forward_p90_ms", "forward_p99_ms",
    "postprocess_mean_ms", "total_mean_ms", "total_p50_ms",
)


def _walk(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _resource_metrics(search_root: Path, label: str, candidate_hash: str) -> dict[str, Any]:
    paths = sorted(
        (search_root / f"ga/budget_{label}/seed_0").glob(
            "generation_*/generation_summary.json"
        )
    )
    matches: list[Mapping[str, Any]] = []
    for path in paths:
        for row in _walk(read_json(path)):
            observed = str(
                row.get("complete_phenotype_hash", row.get("candidate_hash", ""))
            )
            if observed == candidate_hash and "R_bops_vs_fp32" in row:
                matches.append(row)
    if not matches:
        return {}
    row = matches[-1]
    return {
        key: row.get(key)
        for key in (
            "R_bops_vs_fp32", "R_parameter_retention", "R_size_vs_fp32",
            "mixed_weight_retention", "parameter_count", "mixed_weight_size",
        )
        if row.get(key) is not None
    }


def _candidate_source(
    search_root: Path,
    *,
    method: str,
    label: str,
    payload: Mapping[str, Any],
    sequence_index: int,
) -> dict[str, Any]:
    candidate_hash = str(payload["complete_phenotype_hash"])
    metadata = dict(payload.get("metadata") or {})
    artifact = Path(
        str(metadata.get("source_artifact_dir") or metadata.get("artifact_dir") or "")
    ).resolve()
    engine = artifact / "engine.plan"
    if not engine.is_file():
        raise RuntimeError(f"pyramid_repeat_engine_missing:{method}:{label}:{engine}")
    actual_sha = sha256_file(engine)
    expected_sha = str(metadata.get("engine_hash") or "")
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"pyramid_repeat_engine_hash_mismatch:{candidate_hash}:{expected_sha}:{actual_sha}"
        )
    physical = read_json(artifact / "physical_hash.json")
    genotype = dict(payload.get("genotype") or {})
    precision = dict(genotype.get("precision_genes") or {})
    counts = {name: sum(value == name for value in precision.values()) for name in ("FP32", "FP16", "INT8")}
    resources = _resource_metrics(search_root, label, candidate_hash)
    parameter_base = physical.get("parameter_count_base")
    parameter_pruned = physical.get("parameter_count_pruned")
    return {
        "family_id": "lidar_pyramid",
        "assigned_method": method,
        "sequence_index": int(sequence_index),
        "item_id": f"{method}_budget_{label}_{candidate_hash[:12]}",
        "variant": "pq",
        "budget": float(int(label) / 100.0),
        "actual_bops": resources.get("R_bops_vs_fp32"),
        "engine_path": str(engine),
        "engine_sha256": actual_sha,
        "candidate_hash": candidate_hash,
        "parameter_count_base": parameter_base,
        "parameter_count_pruned": parameter_pruned,
        "parameter_reduction": physical.get("parameter_reduction"),
        "parameter_compression_ratio": (
            float(parameter_base) / float(parameter_pruned)
            if parameter_base and parameter_pruned else None
        ),
        "mixed_weight_retention": resources.get(
            "mixed_weight_retention", resources.get("R_size_vs_fp32")
        ),
        "mixed_weight_compression_ratio": (
            1.0 / float(resources.get("mixed_weight_retention", resources.get("R_size_vs_fp32")))
            if resources.get("mixed_weight_retention", resources.get("R_size_vs_fp32"))
            else None
        ),
        "fp32_count": counts["FP32"],
        "fp16_count": counts["FP16"],
        "int8_count": counts["INT8"],
        "search_result_status": payload.get("status"),
        "search_requested_realized_exact": payload.get("requested_realized_exact"),
    }


def _parse_budget_labels(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"pyramid_repeat_budget_labels_invalid:{value}")
    invalid = sorted(set(labels) - set(BUDGET_LABELS))
    if invalid:
        raise ValueError(f"pyramid_repeat_budget_labels_unknown:{invalid}")
    return labels


def _inventory(
    search_root: Path, budget_labels: tuple[str, ...]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    formal = read_json(search_root / "reports/formal_ga_results.json")
    missing = sorted(set(budget_labels) - set(formal.get("results", {})))
    if missing:
        raise RuntimeError(f"pyramid_repeat_missing_completed_budgets:{missing}")
    baseline_engine = (
        search_root
        / "generation_winner_validation_runtime/baselines/original_strict_fp32/engine.plan"
    )
    if not baseline_engine.is_file():
        raise RuntimeError(f"pyramid_repeat_baseline_engine_missing:{baseline_engine}")
    baseline = {
        "family_id": "lidar_pyramid",
        "assigned_method": "baseline",
        "sequence_index": 0,
        "item_id": "strict_fp32",
        "variant": "fp32",
        "budget": None,
        "actual_bops": 1.0,
        "engine_path": str(baseline_engine),
        "engine_sha256": sha256_file(baseline_engine),
        "candidate_hash": "strict_fp32",
        "parameter_count_base": 5464791,
        "parameter_count_pruned": 5464791,
        "parameter_reduction": 0.0,
        "parameter_compression_ratio": 1.0,
        "mixed_weight_retention": 1.0,
        "mixed_weight_compression_ratio": 1.0,
        "fp32_count": None,
        "fp16_count": None,
        "int8_count": None,
    }
    candidates: list[dict[str, Any]] = []
    sequence = 1
    for method, key in (("greedy", "greedy_anchor"), ("ga", "final_winner")):
        for label in budget_labels:
            payload = formal["results"][label][key]
            candidates.append(
                _candidate_source(
                    search_root,
                    method=method,
                    label=label,
                    payload=payload,
                    sequence_index=sequence,
                )
            )
            sequence += 1
    if len({row["engine_sha256"] for row in candidates}) != len(candidates):
        # Equal Greedy/GA phenotypes are valid; record rather than reject them.
        pass
    return baseline, candidates


def _evaluate(
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


def _aggregate(rows: list[dict[str, Any]], repeat_count: int) -> list[dict[str, Any]]:
    baselines: dict[int, float] = {}
    for repeat in range(repeat_count):
        values = [
            float(row["forward_p50_ms"])
            for row in rows
            if row["assigned_method"] == "baseline" and row["repeat_index"] == repeat
        ]
        if len(values) != 2:
            raise RuntimeError(f"pyramid_repeat_baseline_pre_post_count:{repeat}:{values}")
        baselines[repeat] = statistics.fmean(values)
    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["assigned_method"]), row.get("budget"))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, budget), values in groups.items():
        if method == "baseline":
            if len(values) != repeat_count * 2:
                raise RuntimeError("pyramid_repeat_baseline_total_count")
        elif sorted(int(row["repeat_index"]) for row in values) != list(range(repeat_count)):
            raise RuntimeError(f"pyramid_repeat_candidate_count:{method}:{budget}")
        first = values[0]
        result = {
            key: first.get(key)
            for key in (
                "assigned_method", "variant", "budget", "actual_bops",
                "candidate_hash", "engine_sha256", "parameter_count_base",
                "parameter_count_pruned", "parameter_reduction",
                "parameter_compression_ratio", "mixed_weight_retention",
                "mixed_weight_compression_ratio", "fp32_count", "fp16_count",
                "int8_count",
            )
        }
        result["repeat_count"] = repeat_count if method != "baseline" else repeat_count * 2
        for metric in METRICS:
            data = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = statistics.fmean(data)
            result[f"{metric}_std"] = statistics.pstdev(data)
        if method != "baseline":
            speedups = [
                baselines[int(row["repeat_index"])] / float(row["forward_p50_ms"])
                for row in values
            ]
            result["speedup_vs_matched_b0_mean"] = statistics.fmean(speedups)
            result["speedup_vs_matched_b0_std"] = statistics.pstdev(speedups)
        output.append(result)
    return sorted(output, key=lambda row: (row["assigned_method"], -(row.get("budget") or 1.0)))


def run(args: argparse.Namespace) -> int:
    if int(args.repeat_count) < 2:
        raise ValueError("pyramid_repeat_count_must_be_at_least_two")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    snapshot = gpu_snapshot(args.physical_gpu)
    if int(snapshot["memory_used_mib"]) > 256 or int(snapshot["utilization_percent"]) > 5:
        raise RuntimeError(f"pyramid_repeat_gpu_not_idle:{snapshot}")
    budget_labels = _parse_budget_labels(args.budget_labels)
    baseline, candidates = _inventory(args.search_root.resolve(), budget_labels)
    write_json(root / "engine_inventory.json", {"baseline": baseline, "candidates": candidates})
    write_json(root / "provenance.json", {
        "schema_version": "pyramid-greedy-ga-full1789-repeat-configurable-v2",
        "source_search_root": str(args.search_root.resolve()),
        "source_formal_results_sha256": sha256_file(
            args.search_root / "reports/formal_ga_results.json"
        ),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "eval_manifest_sha256": sha256_file(args.eval_manifest),
        "physical_gpu": args.physical_gpu,
        "gpu_start": snapshot,
        "repeat_count": int(args.repeat_count),
        "budget_labels": list(budget_labels),
        "evaluation_frames": 1789,
        "warmup_frames": 200,
        "latency_rounds": 3,
        "engine_build_invoked": False,
        "source_engines_modified": False,
    })
    rows: list[dict[str, Any]] = []
    for repeat in range(int(args.repeat_count)):
        ordered = [("b0_pre", baseline), *[(row["item_id"], row) for row in candidates], ("b0_post", baseline)]
        for order, (name, source) in enumerate(ordered):
            item = dict(source)
            item["sequence_index"] = order
            item["item_id"] = name if item["assigned_method"] == "baseline" else item["item_id"]
            destination = root / f"repeat_{repeat:02d}/{order:02d}_{name}"
            result = _evaluate(item, repeat=repeat, destination=destination, args=args)
            compact = {**item, **compact_result(result), "repeat_index": repeat}
            rows.append(compact)
            write_json(root / "reports/progress.json", {
                "completed_items": len(rows),
                "expected_items": int(args.repeat_count) * (
                    2 + 2 * len(budget_labels)
                ),
                "last_repeat": repeat, "last_item": name,
            })
            write_csv(root / "reports/repeat_results.csv", rows)
    aggregate = _aggregate(rows, int(args.repeat_count))
    write_csv(
        root / f"reports/repeat{int(args.repeat_count)}_mean_std.csv", aggregate
    )
    write_json(root / "reports/final_report.json", {
        "passed": all(
            int(row["num_evaluated_frames"]) == 1789
            and int(row["num_skipped_frames"]) == 0
            and math.isfinite(float(row["mAP"]))
            for row in rows
        ),
        "repeat_count": int(args.repeat_count),
        "evaluation_frames": 1789,
        "warmup_frames": 200,
        "engine_build_invoked": False,
        "result_count": len(rows),
        "aggregate": aggregate,
        "gpu_end": gpu_snapshot(args.physical_gpu),
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument(
        "--budget-labels", default=",".join(BUDGET_LABELS),
        help="Comma-separated completed budget labels, for example 005 or 030,025.",
    )
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"),
    )
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
