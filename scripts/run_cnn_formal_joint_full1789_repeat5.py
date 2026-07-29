#!/usr/bin/env python3
"""Repeat full-validation of formal-CNN Greedy/GA engines on one GPU.

This is an evaluation-only bridge for the current ``formal_ga_results.json``
layout.  A budget subset can be evaluated immediately, so a completed 0.05
campaign does not have to wait for the other five budgets. Every repeat is
serial on one physical GPU and brackets candidates with matched strict-FP32
pre/post replays.  Source engines are hash locked and never rebuilt here.

The current campaign default is three repetitions; the explicit repeat count
is recorded in every report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.ga.cnn_stage12_v3 import MODEL_SPECS  # noqa: E402
from search.ga.transformer_stage12_v3 import COBEVT_SPEC  # noqa: E402
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
FORMAL_MODEL_SPECS = {**MODEL_SPECS, "cobevt": COBEVT_SPEC}
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


def _parse_budget_labels(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"cnn_repeat_budget_labels_invalid:{value}")
    invalid = sorted(set(labels) - set(BUDGET_LABELS))
    if invalid:
        raise ValueError(f"cnn_repeat_budget_labels_unknown:{invalid}")
    return labels


def _artifact(payload: Mapping[str, Any]) -> Path:
    metadata = dict(payload.get("metadata") or {})
    raw = dict(metadata.get("raw") or {})
    value = raw.get("source_artifact_dir") or metadata.get("source_artifact_dir")
    if value:
        result = Path(str(value)).expanduser().resolve()
    else:
        engine = raw.get("engine_path") or metadata.get("engine_path")
        if not engine:
            raise RuntimeError("cnn_repeat_source_artifact_missing")
        path = Path(str(engine)).expanduser().resolve()
        result = path.parent.parent if path.parent.name == "deployment" else path.parent
    if not (result / "phenotype.json").is_file():
        raise RuntimeError(f"cnn_repeat_source_phenotype_missing:{result}")
    return result


def _engine(artifact: Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    metadata = dict(payload.get("metadata") or {})
    raw = dict(metadata.get("raw") or {})
    candidates = (
        artifact / "deployment/candidate.plan",
        artifact / "engine.plan",
        Path(str(raw.get("engine_path") or "")),
    )
    path = next((item.resolve() for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError(f"cnn_repeat_engine_missing:{artifact}")
    digest = sha256_file(path)
    expected = str(
        raw.get("engine_sha256")
        or metadata.get("engine_hash")
        or metadata.get("engine_sha256")
        or ""
    )
    if expected and expected != digest:
        raise RuntimeError(f"cnn_repeat_engine_hash_mismatch:{expected}:{digest}")
    return path, digest


def _exact_bops(search_root: Path, candidate_hash: str, target: float) -> tuple[float, str]:
    paths = [search_root / "reports/greedy_exact_winners.json"]
    paths.extend(sorted(search_root.glob("ga/budget_*/seed_0/generation_*/generation_summary.json")))
    values: list[float] = []
    for path in paths:
        if not path.is_file():
            continue
        for row in _walk(read_json(path)):
            observed = str(row.get("complete_phenotype_hash", row.get("candidate_hash", "")))
            if observed != candidate_hash:
                continue
            for key in ("R_bops_vs_fp32", "R_BOPS_vs_FP32", "bops_retention"):
                if row.get(key) is not None:
                    values.append(float(row[key]))
    unique = sorted({round(value, 15) for value in values})
    if len(unique) > 1:
        raise RuntimeError(f"cnn_repeat_candidate_bops_conflict:{candidate_hash}:{unique}")
    if unique:
        value = float(unique[0])
        if abs(value - target) > 0.005 + 1e-12:
            raise RuntimeError(f"cnn_repeat_candidate_out_of_band:{value}:{target}")
        return value, "exact_search_artifact"
    # The engine remains valid for evaluation, but never mislabel the target as
    # an independently recomputed exact resource value.
    return target, "target_placeholder_pending_exact_recompute"


def _candidate(
    search_root: Path,
    *,
    model: str,
    method: str,
    label: str,
    payload: Mapping[str, Any],
    sequence_index: int,
) -> dict[str, Any]:
    candidate_hash = str(payload["complete_phenotype_hash"])
    artifact = _artifact(payload)
    engine, engine_hash = _engine(artifact, payload)
    target = float(int(label) / 100.0)
    actual_bops, bops_source = _exact_bops(search_root, candidate_hash, target)
    genotype = dict(payload.get("genotype") or {})
    precision = dict(genotype.get("precision_genes") or {})
    counts = {
        name: sum(str(value).upper() == name for value in precision.values())
        for name in ("FP32", "FP16", "INT8")
    }
    stage2_path = artifact / "candidate_stage2_result.json"
    stage2 = read_json(stage2_path) if stage2_path.is_file() else {}
    before = stage2.get("physical_parameter_count_before")
    after = stage2.get("physical_parameter_count_after")
    reduction = stage2.get("physical_parameter_pruning_ratio")
    return {
        "family_id": FORMAL_MODEL_SPECS[model].family_id,
        "assigned_method": method,
        "sequence_index": int(sequence_index),
        "item_id": f"{method}_budget_{label}_{candidate_hash[:12]}",
        "variant": "prune_quant",
        "budget": target,
        "actual_bops": actual_bops,
        "actual_bops_source": bops_source,
        "engine_path": str(engine),
        "engine_sha256": engine_hash,
        "candidate_hash": candidate_hash,
        "source_artifact_dir": str(artifact),
        "parameter_count_base": before,
        "parameter_count_pruned": after,
        "parameter_reduction": reduction,
        "parameter_compression_ratio": (
            float(before) / float(after) if before and after else None
        ),
        "fp32_count": counts["FP32"],
        "fp16_count": counts["FP16"],
        "int8_count": counts["INT8"],
        "search_requested_realized_exact": payload.get("requested_realized_exact"),
    }


def _inventory(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = args.search_root.resolve()
    labels = _parse_budget_labels(args.budget_labels)
    formal = read_json(root / "reports/formal_ga_results.json")
    missing = sorted(set(labels) - set(formal.get("results", {})))
    if missing:
        raise RuntimeError(f"cnn_repeat_missing_completed_budgets:{missing}")
    spec = FORMAL_MODEL_SPECS[args.model]
    if spec.strict_fp32_engine is None or not spec.strict_fp32_engine.is_file():
        raise RuntimeError(f"cnn_repeat_strict_baseline_missing:{spec.strict_fp32_engine}")
    baseline = {
        "family_id": spec.family_id,
        "assigned_method": "baseline",
        "sequence_index": 0,
        "item_id": "strict_fp32",
        "variant": "fp32",
        "budget": None,
        "actual_bops": 1.0,
        "actual_bops_source": "definition",
        "engine_path": str(spec.strict_fp32_engine.resolve()),
        "engine_sha256": sha256_file(spec.strict_fp32_engine),
        "candidate_hash": "strict_fp32",
        "parameter_reduction": 0.0,
        "parameter_compression_ratio": 1.0,
        "fp32_count": None,
        "fp16_count": 0,
        "int8_count": 0,
    }
    rows: list[dict[str, Any]] = []
    sequence = 1
    for method, key in (("greedy", "greedy_anchor"), ("ga", "final_winner")):
        for label in labels:
            rows.append(
                _candidate(
                    root,
                    model=args.model,
                    method=method,
                    label=label,
                    payload=formal["results"][label][key],
                    sequence_index=sequence,
                )
            )
            sequence += 1
    return baseline, rows


def _evaluate(source: Mapping[str, Any], *, repeat: int, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = dict(
        source=source,
        gpu_id=args.physical_gpu,
        repeat_index=repeat,
        output_dir=output,
        eval_manifest_path=args.eval_manifest,
        num_frames=1789,
        warmup_frames=200,
        latency_rounds=3,
        fixed_k=29696,
    )
    if output.exists():
        return load_resumable_family_evaluation(**kwargs)
    return evaluate_existing_family_engine(
        **kwargs,
        model_config=FORMAL_MODEL_SPECS[args.model].config,
        heal_root=args.heal_root,
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin,
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
            raise RuntimeError(f"cnn_repeat_baseline_replay_count:{repeat}:{values}")
        baselines[repeat] = statistics.fmean(values)
    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["assigned_method"]), row.get("budget")), []).append(row)
    result: list[dict[str, Any]] = []
    for (method, budget), values in groups.items():
        expected = 2 * repeat_count if method == "baseline" else repeat_count
        if len(values) != expected:
            raise RuntimeError(f"cnn_repeat_count:{method}:{budget}:{len(values)}")
        first = values[0]
        item = {
            key: first.get(key)
            for key in (
                "assigned_method", "variant", "budget", "actual_bops",
                "actual_bops_source", "candidate_hash", "engine_sha256",
                "parameter_count_base", "parameter_count_pruned",
                "parameter_reduction", "parameter_compression_ratio",
                "fp32_count", "fp16_count", "int8_count",
            )
        }
        item["repeat_count"] = expected
        for metric in METRICS:
            data = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = statistics.fmean(data)
            item[f"{metric}_std"] = statistics.pstdev(data)
        if method != "baseline":
            speedups = [
                baselines[int(row["repeat_index"])] / float(row["forward_p50_ms"])
                for row in values
            ]
            item["speedup_vs_matched_b0_mean"] = statistics.fmean(speedups)
            item["speedup_vs_matched_b0_std"] = statistics.pstdev(speedups)
        result.append(item)
    return sorted(result, key=lambda row: (str(row["assigned_method"]), -(row.get("budget") or 1.0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("attfusion", "cobevt", "disco", "fcooper"),
        required=True,
    )
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--budget-labels", default=",".join(BUDGET_LABELS))
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path, required=True)
    args = parser.parse_args()
    if int(args.repeat_count) < 2:
        raise ValueError("cnn_repeat_count_must_be_at_least_two")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "reports").mkdir()
    snapshot = gpu_snapshot(args.physical_gpu)
    if int(snapshot["memory_used_mib"]) > 256 or int(snapshot["utilization_percent"]) > 5:
        raise RuntimeError(f"cnn_repeat_gpu_not_idle:{snapshot}")
    baseline, candidates = _inventory(args)
    labels = _parse_budget_labels(args.budget_labels)
    write_json(root / "engine_inventory.json", {"baseline": baseline, "candidates": candidates})
    write_json(root / "provenance.json", {
        "schema_version": "cnn-formal-joint-full1789-repeat-configurable-v2",
        "model": args.model,
        "source_search_root": str(args.search_root.resolve()),
        "source_formal_results_sha256": sha256_file(args.search_root / "reports/formal_ga_results.json"),
        "physical_gpu": args.physical_gpu,
        "gpu_start": snapshot,
        "budget_labels": list(labels),
        "same_gpu_serial": True,
        "repeat_count": int(args.repeat_count),
        "evaluation_frames": 1789,
        "warmup_frames": 200,
        "engine_build_invoked": False,
    })
    rows: list[dict[str, Any]] = []
    for repeat in range(int(args.repeat_count)):
        ordered = [("b0_pre", baseline), *[(row["item_id"], row) for row in candidates], ("b0_post", baseline)]
        for order, (name, source) in enumerate(ordered):
            item = dict(source)
            item["sequence_index"] = order
            if item["assigned_method"] == "baseline":
                item["item_id"] = name
            output = root / f"repeat_{repeat:02d}/{order:02d}_{name}"
            result = _evaluate(item, repeat=repeat, output=output, args=args)
            rows.append({**item, **compact_result(result), "repeat_index": repeat})
            write_csv(root / "reports/repeat_results.csv", rows)
            write_json(root / "reports/progress.json", {
                "completed_items": len(rows),
                "expected_items": int(args.repeat_count) * (2 + 2 * len(labels)),
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
        "same_gpu_serial": True,
        "repeat_count": int(args.repeat_count),
        "evaluation_frames": 1789,
        "result_count": len(rows),
        "aggregate": aggregate,
        "gpu_end": gpu_snapshot(args.physical_gpu),
    })
    if not passed:
        raise RuntimeError("cnn_repeat_acceptance_failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
