#!/usr/bin/env python3
"""Unify five-repeat P/Q ablations for Pyramid, F-Cooper, and DiscoNet.

The source evaluators intentionally use different schemas.  This script reads
only their compact JSON/CSV reports, validates the complete 3-model matrix and
the shared full-validation protocol, then writes small synchronized CSV/JSON/MD
summaries.  It never reads or copies ONNX, checkpoint, calibration, or engine
payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


MODELS = ("lidar_pyramid", "lidar_fcooper", "lidar_disco")
METHODS = ("ga", "greedy")
BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
VARIANTS = ("prune_quant", "prune_only", "quant_only")
REPEAT_COUNT = 5
EVALUATED_FRAMES = 1789

AP_METRICS = ("AP@0.3", "AP@0.5", "AP@0.7", "mAP")
LATENCY_METRICS = tuple(
    f"{component}_{stat}_ms"
    for component in ("forward", "postprocess", "total")
    for stat in ("mean", "p50", "p90", "p99")
)
ALL_AGGREGATED_METRICS = (*AP_METRICS, *LATENCY_METRICS)


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _none_or_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _none_or_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _raw_identity(row: Mapping[str, Any], *, family_schema: bool) -> tuple[Any, ...]:
    return (
        str(row["assigned_method"]),
        str(row["variant"]),
        _none_or_float(row.get("budget")),
        int(row["repeat_index"]),
    )


def _validate_raw_repeats(
    rows: Iterable[Mapping[str, Any]],
    *,
    model: str,
    family_schema: bool,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    expected_count = len(METHODS) * (1 + len(BUDGETS) * len(VARIANTS)) * REPEAT_COUNT
    if len(values) != expected_count:
        raise RuntimeError(f"unified_raw_repeat_count:{model}:{len(values)}!={expected_count}")
    identities = [_raw_identity(row, family_schema=family_schema) for row in values]
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"unified_duplicate_repeat_identity:{model}")
    expected = {
        (method, "fp32", None, repeat)
        for method in METHODS
        for repeat in range(REPEAT_COUNT)
    }
    expected.update(
        (method, variant, budget, repeat)
        for method in METHODS
        for budget in BUDGETS
        for variant in VARIANTS
        for repeat in range(REPEAT_COUNT)
    )
    normalized = {
        (method, variant, None if budget is None else round(float(budget), 2), repeat)
        for method, variant, budget, repeat in identities
    }
    if normalized != expected:
        raise RuntimeError(
            f"unified_repeat_matrix_mismatch:{model}:"
            f"missing={sorted(expected-normalized, key=str)}:"
            f"extra={sorted(normalized-expected, key=str)}"
        )

    frame_hashes = {str(row.get("frame_order_hash", "")) for row in values}
    if len(frame_hashes) != 1 or "" in frame_hashes:
        raise RuntimeError(f"unified_frame_order_mismatch:{model}:{sorted(frame_hashes)}")
    for row in values:
        evaluated_key = "num_evaluated_frames" if family_schema else "evaluated_frames"
        skipped_key = "num_skipped_frames" if family_schema else "skipped_frames"
        if int(row.get(evaluated_key, -1)) != EVALUATED_FRAMES:
            raise RuntimeError(
                f"unified_evaluated_frames:{model}:{row.get(evaluated_key)}"
            )
        if int(row.get(skipped_key, -1)) != 0:
            raise RuntimeError(f"unified_skipped_frames:{model}:{row.get(skipped_key)}")
    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = {}
    for row in values:
        budget = _none_or_float(row.get("budget"))
        key = (
            str(row["assigned_method"]),
            str(row["variant"]),
            None if budget is None else round(budget, 2),
        )
        grouped.setdefault(key, []).append(row)
    for key, group in grouped.items():
        if sorted(int(row["repeat_index"]) for row in group) != list(range(REPEAT_COUNT)):
            raise RuntimeError(f"unified_repeat_indices:{model}:{key}")
        if len({str(row["engine_sha256"]) for row in group}) != 1:
            raise RuntimeError(f"unified_engine_changed_across_repeats:{model}:{key}")
        if len({str(row["frame_order_hash"]) for row in group}) != 1:
            raise RuntimeError(f"unified_frame_order_changed_across_repeats:{model}:{key}")
    return {
        "raw_row_count": len(values),
        "frame_order_hash": next(iter(frame_hashes)),
        "evaluated_frames_per_repeat": EVALUATED_FRAMES,
        "skipped_frames_per_repeat": 0,
    }


def _manifest_protocol(
    manifest: Mapping[str, Any],
    *,
    model: str,
    family_schema: bool,
    source_root: Path,
) -> dict[str, Any]:
    protocol = dict(manifest.get("protocol") or {})
    if int(protocol.get("num_frames", -1)) != EVALUATED_FRAMES:
        raise RuntimeError(f"unified_manifest_num_frames:{model}:{protocol.get('num_frames')}")
    if int(protocol.get("warmup_frames", -1)) != 200:
        raise RuntimeError(
            f"unified_manifest_warmup_frames:{model}:{protocol.get('warmup_frames')}"
        )
    if int(protocol.get("dataloader_num_workers", -1)) != 8:
        raise RuntimeError(
            f"unified_manifest_dataloader_workers:{model}:"
            f"{protocol.get('dataloader_num_workers')}"
        )
    if not bool(protocol.get("cuda_postprocess", False)):
        raise RuntimeError(f"unified_manifest_cuda_postprocess:{model}")
    if int(protocol.get("fixed_k", -1)) != 29696:
        raise RuntimeError(f"unified_manifest_fixed_k:{model}:{protocol.get('fixed_k')}")
    if not family_schema and not bool(protocol.get("reset_after_warmup", False)):
        raise RuntimeError(f"unified_manifest_warmup_not_reset:{model}")
    manifest_hash = str(protocol.get("eval_manifest_hash", ""))
    if not manifest_hash:
        raise RuntimeError(f"unified_eval_manifest_hash_missing:{model}")
    manifest_value = protocol.get(
        "eval_manifest" if family_schema else "eval_manifest_path"
    )
    if not manifest_value:
        raise RuntimeError(f"unified_eval_manifest_path_missing:{model}")
    eval_manifest_path = Path(str(manifest_value)).expanduser()
    if not eval_manifest_path.is_absolute():
        eval_manifest_path = source_root / eval_manifest_path
    eval_manifest_path = eval_manifest_path.resolve()
    if not eval_manifest_path.is_file():
        raise RuntimeError(f"unified_eval_manifest_file_missing:{model}:{eval_manifest_path}")
    file_sha256 = _sha256(eval_manifest_path)
    if file_sha256 != str(protocol.get("eval_manifest_file_sha256", "")):
        raise RuntimeError(f"unified_eval_manifest_file_hash_mismatch:{model}")
    eval_manifest = _read_json(eval_manifest_path)
    if str(eval_manifest.get("manifest_hash", "")) != manifest_hash:
        raise RuntimeError(f"unified_eval_manifest_content_hash_mismatch:{model}")
    if not bool(eval_manifest.get("reset_after_warmup", False)):
        raise RuntimeError(f"unified_eval_manifest_content_not_reset:{model}")
    if len(list(eval_manifest.get("warmup_frame_ids") or [])) < 200:
        raise RuntimeError(f"unified_eval_manifest_warmup_ids_short:{model}")
    if len(list(eval_manifest.get("evaluation_frame_ids") or [])) < EVALUATED_FRAMES:
        raise RuntimeError(f"unified_eval_manifest_evaluation_ids_short:{model}")
    return {
        "eval_manifest_hash": manifest_hash,
        "eval_manifest_path": str(eval_manifest_path),
        "eval_manifest_file_sha256": file_sha256,
        "fixed_k": int(protocol["fixed_k"]),
        "num_frames": int(protocol["num_frames"]),
        "warmup_frames": int(protocol["warmup_frames"]),
        "latency_rounds": int(protocol["latency_rounds"]),
        "dataloader_num_workers": int(protocol["dataloader_num_workers"]),
        "cuda_postprocess": bool(protocol["cuda_postprocess"]),
    }


def _validate_aggregate_against_raw(
    aggregate_rows: Iterable[Mapping[str, Any]],
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    model: str,
) -> None:
    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = {}
    for raw in raw_rows:
        row = dict(raw)
        budget = _none_or_float(row.get("budget"))
        key = (
            str(row["assigned_method"]),
            str(row["variant"]),
            None if budget is None else round(budget, 2),
        )
        grouped.setdefault(key, []).append(row)
    for raw_aggregate in aggregate_rows:
        aggregate = dict(raw_aggregate)
        budget = _none_or_float(aggregate.get("budget"))
        key = (
            str(aggregate["assigned_method"]),
            str(aggregate["variant"]),
            None if budget is None else round(budget, 2),
        )
        values = grouped.get(key, [])
        if len(values) != REPEAT_COUNT:
            raise RuntimeError(f"unified_aggregate_raw_group_missing:{model}:{key}")
        if str(aggregate["engine_sha256"]) != str(values[0]["engine_sha256"]):
            raise RuntimeError(f"unified_aggregate_engine_mismatch:{model}:{key}")
        if str(aggregate["frame_order_hash"]) != str(values[0]["frame_order_hash"]):
            raise RuntimeError(f"unified_aggregate_frame_order_mismatch:{model}:{key}")
        for metric in (*ALL_AGGREGATED_METRICS, "speedup_vs_same_gpu_fp32"):
            samples = [float(row[metric]) for row in values]
            expected = {
                "mean": statistics.fmean(samples),
                "std": statistics.pstdev(samples),
            }
            for statistic, target in expected.items():
                source_key = f"{metric}_across_runs_{statistic}"
                realized = float(aggregate[source_key])
                if not math.isclose(realized, target, rel_tol=1.0e-10, abs_tol=1.0e-10):
                    raise RuntimeError(
                        f"unified_aggregate_numeric_mismatch:{model}:{key}:"
                        f"{source_key}:{realized}!={target}"
                    )


def _normalized_aggregate_row(
    row: Mapping[str, Any],
    *,
    model: str,
    manifest_hash: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model,
        "method": str(row["assigned_method"]),
        "budget": _none_or_float(row.get("budget")),
        "variant": str(row["variant"]),
        "source_actual_bops": _none_or_float(row.get("actual_bops")),
        "parameter_count_base": _none_or_int(row.get("parameter_count_base")),
        "parameter_count_effective": _none_or_int(row.get("parameter_count_pruned")),
        "parameter_pruning_rate": _none_or_float(row.get("parameter_reduction")),
        "int8_count": int(row["int8_count"]),
        "fp16_count": int(row["fp16_count"]),
        "fp32_count": int(row["fp32_count"]),
        "engine_sha256": str(row["engine_sha256"]),
        "frame_order_hash": str(row["frame_order_hash"]),
        "eval_manifest_hash": manifest_hash,
        "repeat_count": int(row["repeat_count"]),
        "evaluated_frames_per_repeat": EVALUATED_FRAMES,
        "evaluated_frames_total": EVALUATED_FRAMES * REPEAT_COUNT,
        "skipped_frames_per_repeat": 0,
        "skipped_frames_total": 0,
    }
    if result["variant"] == "fp32" and result["source_actual_bops"] is None:
        result["source_actual_bops"] = 1.0
    for metric in ALL_AGGREGATED_METRICS:
        for statistic in ("mean", "std"):
            source_key = f"{metric}_across_runs_{statistic}"
            if source_key not in row:
                raise RuntimeError(f"unified_aggregate_metric_missing:{model}:{source_key}")
            result[f"{metric}_five_run_{statistic}"] = float(row[source_key])
    for statistic in ("mean", "std"):
        source_key = f"speedup_vs_same_gpu_fp32_across_runs_{statistic}"
        if source_key not in row:
            raise RuntimeError(f"unified_speedup_missing:{model}:{source_key}")
        result[f"speedup_vs_same_gpu_fp32_five_run_{statistic}"] = float(
            row[source_key]
        )
    return result


def _fill_parameter_identity(rows: list[dict[str, Any]], *, model: str) -> None:
    bases = {
        int(row["parameter_count_base"])
        for row in rows
        if row["parameter_count_base"] is not None
    }
    if len(bases) != 1:
        raise RuntimeError(f"unified_parameter_base_ambiguous:{model}:{sorted(bases)}")
    base = next(iter(bases))
    for method in METHODS:
        baseline = next(
            row
            for row in rows
            if row["method"] == method and row["variant"] == "fp32"
        )
        baseline["parameter_count_base"] = base
        baseline["parameter_count_effective"] = base
        baseline["parameter_pruning_rate"] = 0.0
        for budget in BUDGETS:
            group = [
                row
                for row in rows
                if row["method"] == method
                and row["budget"] == budget
                and row["variant"] in VARIANTS
            ]
            p_only = next(row for row in group if row["variant"] == "prune_only")
            if p_only["parameter_count_effective"] is None:
                raise RuntimeError(
                    f"unified_prune_only_parameter_missing:{model}:{method}:{budget}"
                )
            pruned = int(p_only["parameter_count_effective"])
            rate = 1.0 - pruned / base
            for row in group:
                row["parameter_count_base"] = base
                if row["variant"] == "quant_only":
                    row["parameter_count_effective"] = base
                    row["parameter_pruning_rate"] = 0.0
                else:
                    row["parameter_count_effective"] = pruned
                    row["parameter_pruning_rate"] = rate


def _validate_aggregate_matrix(rows: list[dict[str, Any]], *, model: str) -> None:
    if len(rows) != len(METHODS) * (1 + len(BUDGETS) * len(VARIANTS)):
        raise RuntimeError(f"unified_aggregate_count:{model}:{len(rows)}")
    expected = {(method, None, "fp32") for method in METHODS}
    expected.update(
        (method, budget, variant)
        for method in METHODS
        for budget in BUDGETS
        for variant in VARIANTS
    )
    actual = {
        (
            row["method"],
            None if row["budget"] is None else round(float(row["budget"]), 2),
            row["variant"],
        )
        for row in rows
    }
    if actual != expected:
        raise RuntimeError(
            f"unified_aggregate_matrix:{model}:missing={sorted(expected-actual, key=str)}:"
            f"extra={sorted(actual-expected, key=str)}"
        )
    for row in rows:
        if row["repeat_count"] != REPEAT_COUNT:
            raise RuntimeError(f"unified_aggregate_repeat_count:{model}:{row['repeat_count']}")


def load_pyramid_root(root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = Path(root).resolve()
    report_path = directory / "split_gpu_five_repeat_report.json"
    repeat_path = directory / "split_gpu_repeat_results.csv"
    manifest_path = directory / "split_gpu_evaluation_manifest.json"
    report = _read_json(report_path)
    if not bool(report.get("passed", False)) or int(report.get("repeat_count", -1)) != REPEAT_COUNT:
        raise RuntimeError("unified_pyramid_report_not_accepted")
    with repeat_path.open("r", encoding="utf-8", newline="") as handle:
        repeat_rows = list(csv.DictReader(handle))
    raw_audit = _validate_raw_repeats(
        repeat_rows, model="lidar_pyramid", family_schema=False
    )
    protocol = _manifest_protocol(
        _read_json(manifest_path),
        model="lidar_pyramid",
        family_schema=False,
        source_root=directory,
    )
    aggregate_source_rows = list(report.get("five_repeat_mean_results") or [])
    _validate_aggregate_against_raw(
        aggregate_source_rows, repeat_rows, model="lidar_pyramid"
    )
    rows = [
        _normalized_aggregate_row(
            row,
            model="lidar_pyramid",
            manifest_hash=protocol["eval_manifest_hash"],
        )
        for row in aggregate_source_rows
    ]
    _validate_aggregate_matrix(rows, model="lidar_pyramid")
    _fill_parameter_identity(rows, model="lidar_pyramid")
    return rows, {
        "model": "lidar_pyramid",
        "root": str(directory),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "repeat_path": str(repeat_path),
        "repeat_sha256": _sha256(repeat_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "protocol": protocol,
        **raw_audit,
    }


def load_family_root(
    root: str | Path, *, model: str, family_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = Path(root).resolve()
    report_path = directory / "family_fair_evaluation_report.json"
    manifest_path = directory / "family_fair_evaluation_manifest.json"
    report = _read_json(report_path)
    if str(report.get("status", "")) != "complete":
        raise RuntimeError(f"unified_family_report_not_complete:{model}")
    if str(report.get("family_id", "")) != family_id:
        raise RuntimeError(
            f"unified_family_id_mismatch:{model}:{report.get('family_id')}:{family_id}"
        )
    if int(report.get("repeat_count", -1)) != REPEAT_COUNT:
        raise RuntimeError(f"unified_family_repeat_count:{model}")
    repeat_rows = list(report.get("repeat_results") or [])
    raw_audit = _validate_raw_repeats(repeat_rows, model=model, family_schema=True)
    manifest = _read_json(manifest_path)
    if str(manifest.get("family_id", "")) != family_id:
        raise RuntimeError(f"unified_family_manifest_identity:{model}")
    protocol = _manifest_protocol(
        manifest, model=model, family_schema=True, source_root=directory
    )
    aggregate_source_rows = list(report.get("five_repeat_mean_std") or [])
    _validate_aggregate_against_raw(aggregate_source_rows, repeat_rows, model=model)
    rows = [
        _normalized_aggregate_row(
            row, model=model, manifest_hash=protocol["eval_manifest_hash"]
        )
        for row in aggregate_source_rows
    ]
    _validate_aggregate_matrix(rows, model=model)
    _fill_parameter_identity(rows, model=model)
    return rows, {
        "model": model,
        "family_id": family_id,
        "root": str(directory),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "protocol": protocol,
        **raw_audit,
    }


def aggregate_three_models(
    *,
    pyramid_root: str | Path,
    fcooper_root: str | Path,
    disco_root: str | Path,
) -> dict[str, Any]:
    loaders = (
        load_pyramid_root(pyramid_root),
        load_family_root(
            fcooper_root,
            model="lidar_fcooper",
            family_id="heal_lidar_fcooper",
        ),
        load_family_root(
            disco_root, model="lidar_disco", family_id="heal_lidar_disco"
        ),
    )
    rows = [row for model_rows, _ in loaders for row in model_rows]
    sources = [source for _, source in loaders]
    if len(rows) != len(MODELS) * len(METHODS) * (
        1 + len(BUDGETS) * len(VARIANTS)
    ):
        raise RuntimeError(f"unified_three_model_row_count:{len(rows)}")
    manifest_hashes = {
        str(source["protocol"]["eval_manifest_hash"]) for source in sources
    }
    if len(manifest_hashes) != 1:
        raise RuntimeError(f"unified_eval_manifest_mismatch:{sorted(manifest_hashes)}")
    frame_hashes = {str(source["frame_order_hash"]) for source in sources}
    if len(frame_hashes) != 1:
        raise RuntimeError(f"unified_frame_order_mismatch_across_models:{sorted(frame_hashes)}")
    manifest_file_hashes = {
        str(source["protocol"]["eval_manifest_file_sha256"]) for source in sources
    }
    if len(manifest_file_hashes) != 1:
        raise RuntimeError(
            f"unified_eval_manifest_file_mismatch:{sorted(manifest_file_hashes)}"
        )
    rows.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            METHODS.index(row["method"]),
            0 if row["budget"] is None else 1,
            0.0 if row["budget"] is None else -float(row["budget"]),
            -1 if row["variant"] == "fp32" else VARIANTS.index(row["variant"]),
        )
    )
    return {
        "schema_version": "heal-three-model-pq-ablation-summary-v1",
        "status": "accepted",
        "model_count": len(MODELS),
        "row_count": len(rows),
        "repeat_count": REPEAT_COUNT,
        "evaluated_frames_per_repeat": EVALUATED_FRAMES,
        "skipped_frames_per_repeat": 0,
        "shared_eval_manifest_hash": next(iter(manifest_hashes)),
        "shared_eval_manifest_file_sha256": next(iter(manifest_file_hashes)),
        "shared_frame_order_hash": next(iter(frame_hashes)),
        "sources": sources,
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# HEAL three-model P/Q ablation summary",
        "",
        f"- Rows: {payload['row_count']}",
        f"- Repeats: {payload['repeat_count']}",
        f"- Frames/repeat: {payload['evaluated_frames_per_repeat']}",
        f"- Skips/repeat: {payload['skipped_frames_per_repeat']}",
        f"- Shared manifest: `{payload['shared_eval_manifest_hash']}`",
        f"- Shared frame order: `{payload['shared_frame_order_hash']}`",
        "",
        "| Model | Method | Budget | Variant | actual BOPS | Params | Prune | INT8/FP16/FP32 | AP30/50/70 | mAP | Forward mean/p50/p90/p99 ms | Post mean ms | Total mean ms | Speedup |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        budget = "-" if row["budget"] is None else f"{row['budget']:.2f}"
        lines.append(
            "| {model} | {method} | {budget} | {variant} | {bops:.6f} | {params} | "
            "{prune:.4f} | {i8}/{f16}/{f32} | {ap30:.6f}/{ap50:.6f}/{ap70:.6f} | "
            "{map:.6f} | {fmean:.4f}/{p50:.4f}/{p90:.4f}/{p99:.4f} | {post:.4f} | "
            "{total:.4f} | {speedup:.4f}x |".format(
                model=row["model"],
                method=row["method"],
                budget=budget,
                variant=row["variant"],
                bops=float(row["source_actual_bops"]),
                params=row["parameter_count_effective"],
                prune=float(row["parameter_pruning_rate"]),
                i8=row["int8_count"],
                f16=row["fp16_count"],
                f32=row["fp32_count"],
                ap30=row["AP@0.3_five_run_mean"],
                ap50=row["AP@0.5_five_run_mean"],
                ap70=row["AP@0.7_five_run_mean"],
                map=row["mAP_five_run_mean"],
                fmean=row["forward_mean_ms_five_run_mean"],
                p50=row["forward_p50_ms_five_run_mean"],
                p90=row["forward_p90_ms_five_run_mean"],
                p99=row["forward_p99_ms_five_run_mean"],
                post=row["postprocess_mean_ms_five_run_mean"],
                total=row["total_mean_ms_five_run_mean"],
                speedup=row["speedup_vs_same_gpu_fp32_five_run_mean"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any], output_dir: str | Path) -> None:
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"unified_output_dir_not_empty:{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rows = list(payload["rows"])
    _write_csv(destination / "three_model_pq_ablation_summary.csv", rows)
    (destination / "three_model_pq_ablation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / "three_model_pq_ablation_summary.md").write_text(
        _markdown(payload), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyramid-root", type=Path, required=True)
    parser.add_argument("--fcooper-root", type=Path, required=True)
    parser.add_argument("--disco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = aggregate_three_models(
        pyramid_root=args.pyramid_root,
        fcooper_root=args.fcooper_root,
        disco_root=args.disco_root,
    )
    write_outputs(payload, args.output_dir)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "row_count": payload["row_count"],
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
