#!/usr/bin/env python3
"""v10.7.1 real validation-500 reload eval for exported Stage512 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.latency_lut.save_reload_stage0_variants_v1061 import (  # noqa: E402
    load_model_object_artifact,
)
from tools.latency_lut.select_idle_gpu_for_latency import (  # noqa: E402
    collect_gpu_state,
    snapshot_to_dict,
    wait_for_idle_gpu,
)
from tools.latency_lut.stage0_reblock8_aligned_prune_v106 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    write_csv,
    write_json,
)


DEFAULT_STAGE512_MODELS_DIR = "outputs/latency_lut/stage512_pergroup8_prune_v107/models"
DEFAULT_OUTPUT_DIR = "outputs/latency_lut/stage512_pergroup8_prune_v107/real_val500_reload_eval"
BASELINE_VARIANT = "baseline"
PRUNED_VARIANT = "stage512_pergroup16_to8_prune_all_blocks"
EVAL_ENTRYPOINT = "test_prune_and_eval.evaluate_one_model"


def count_skip_reasons(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("success"):
            continue
        reason = str(row.get("skip_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _round_float(value: Any, ndigits: int = 6) -> float:
    try:
        return round(float(value), ndigits)
    except Exception:
        return 0.0


def build_latency_report_row(
    *,
    variant: str,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
    requested_frames: int,
) -> dict[str, Any]:
    p50 = _round_float(summary.get("forward_time_p50_ms"), 6)
    mean = _round_float(summary.get("forward_time_mean_ms"), 6)
    total_p50 = _round_float(summary.get("total_time_p50_ms"), 6)
    total_mean = _round_float(summary.get("total_time_mean_ms"), 6)
    baseline_p50 = _round_float(baseline.get("forward_latency_p50"), 6) if baseline else p50
    baseline_mean = _round_float(baseline.get("forward_latency_mean"), 6) if baseline else mean
    return {
        "variant": variant,
        "num_frames": int(requested_frames),
        "evaluated_frames": int(summary.get("actual_frames", 0) or 0),
        "skipped_frames": int(summary.get("missing_frames", 0) or 0),
        "skip_reason_counts": count_skip_reasons(rows),
        "forward_latency_p50": p50,
        "forward_latency_mean": mean,
        "forward_latency_p90": _percentile_from_rows(rows, "forward_time_ms", 0.90),
        "forward_latency_p95": _percentile_from_rows(rows, "forward_time_ms", 0.95),
        "total_latency_p50": total_p50,
        "total_latency_mean": total_mean,
        "speedup_p50_vs_baseline": baseline_p50 / p50 if p50 > 0 else 0.0,
        "speedup_mean_vs_baseline": baseline_mean / mean if mean > 0 else 0.0,
    }


def _percentile_from_rows(rows: Sequence[Mapping[str, Any]], key: str, pct: float) -> float:
    vals = sorted(float(row.get(key, 0.0) or 0.0) for row in rows if row.get("success"))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return round(vals[0], 6)
    rank = (len(vals) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return round(vals[lo], 6)
    frac = rank - lo
    return round(vals[lo] * (1.0 - frac) + vals[hi] * frac, 6)


def build_ap_report(
    *,
    variant: str,
    summary: Mapping[str, Any],
    baseline_ap30: float | None,
    requested_frames: int,
    failure_reason: str,
) -> dict[str, Any]:
    ap30 = _round_float(summary.get("AP_0_30"), 6)
    ap50 = _round_float(summary.get("AP_0_50"), 6)
    ap70 = _round_float(summary.get("AP_0_70"), 6)
    aps = [ap30, ap50, ap70]
    return {
        "variant": variant,
        "num_frames": int(requested_frames),
        "evaluated_frames": int(summary.get("actual_frames", 0) or 0),
        "AP@0.30": ap30,
        "AP@0.50": ap50,
        "AP@0.70": ap70,
        "mAP": sum(aps) / len(aps),
        "AP_drop_vs_baseline": round(float(baseline_ap30) - ap30, 6) if baseline_ap30 is not None else 0.0,
        "metric_helper_used": EVAL_ENTRYPOINT,
        "failure_reason": failure_reason or str(summary.get("first_failure", "") or ""),
    }


def _csv_ready_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        converted = dict(row)
        if isinstance(converted.get("skip_reason_counts"), dict):
            converted["skip_reason_counts"] = json.dumps(converted["skip_reason_counts"], ensure_ascii=False, sort_keys=True)
        out.append(converted)
    return out


def _select_device(args: argparse.Namespace, out_dir: Path) -> tuple[torch.device, int | None]:
    if args.auto_select_idle_gpu:
        selected, reason, attempts = wait_for_idle_gpu(
            max_utilization=args.max_gpu_utilization,
            max_memory_ratio=args.max_gpu_memory_ratio,
            wait_timeout_minutes=args.wait_timeout_minutes,
            poll_seconds=args.poll_seconds,
        )
        write_json(
            out_dir / "selected_gpu.json",
            {
                "success": True,
                "selected_gpu_index": selected.index,
                "selected_gpu": snapshot_to_dict(selected),
                "selected_gpu_reason": reason,
                "attempts": attempts,
            },
        )
        device = torch.device(f"cuda:{selected.index}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device, int(selected.index)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        return device, int(device.index or 0)
    return device, None


def _model_artifact_path(models_dir: Path, variant: str) -> Path:
    return models_dir / f"{variant}_model_object.pth"


def _load_eval_helpers():
    attempted = []
    try:
        attempted.append("test_prune_and_eval.HEALLiDARAdapter/build_dataset/evaluate_one_model")
        from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
        from heal_compress.pruning.eval.prune_and_eval import build_dataset, evaluate_one_model, setup_logger

        return HEALLiDARAdapter, build_dataset, evaluate_one_model, setup_logger, attempted
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to import real validation eval helpers: {exc}") from exc


def _smoke_real_batch(model: nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    from opencood.tools import train_utils

    try:
        batch = next(iter(loader))
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            output = model(batch["ego"])
        finite = True
        if torch.is_tensor(output):
            finite = bool(torch.isfinite(output).all().item()) if output.is_floating_point() else True
        elif isinstance(output, Mapping):
            tensors = [v for v in output.values() if torch.is_tensor(v) and v.is_floating_point()]
            finite = all(bool(torch.isfinite(v).all().item()) for v in tensors)
        return {"reload_forward_smoke_passed": True, "output_finite": finite, "failure_reason": ""}
    except Exception as exc:  # noqa: BLE001
        return {"reload_forward_smoke_passed": False, "output_finite": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _contamination_report(samples: Sequence[Mapping[str, Any]], selected_index: int | None) -> dict[str, Any]:
    current_pid = os.getpid()
    external: list[dict[str, Any]] = []
    for sample in samples:
        for process in sample.get("running_processes", []) or []:
            if int(process.get("pid", -1)) != current_pid:
                external.append(dict(process))
    return {
        "selected_gpu_index": selected_index,
        "current_pid": current_pid,
        "latency_contamination_risk": bool(external),
        "external_processes_seen": external,
    }


def _write_failure_outputs(out_dir: Path, *, attempted: Sequence[str], failure: str, tb: str) -> None:
    payload = {
        "success": False,
        "failure_reason": failure,
        "attempted_eval_entrypoints": list(attempted),
        "failure_traceback": tb,
        "missing_adapter_reason": failure,
        "next_required_patch": "Patch tools/latency_lut/real_val500_stage512_reload_eval_v1071.py to match the available HEAL/OpenCOOD eval API.",
    }
    write_json(out_dir / "failure_report.json", payload)
    write_json(out_dir / "real_val500_ap.json", [{"variant": BASELINE_VARIANT, "num_frames": 500, "evaluated_frames": 0, "AP@0.30": None, "AP@0.50": None, "mAP": None, "AP_drop_vs_baseline": None, "metric_helper_used": "", "failure_reason": failure}])
    write_json(out_dir / "real_val500_reload_report.json", [{"variant": BASELINE_VARIANT, "reload_success": False, "validation_dataloader_used": False, "synthetic_used": False, "failure_reason": failure}])
    write_csv(out_dir / "real_val500_latency.csv", [])
    _write_verdict(out_dir, latency_rows=[], ap_rows=[], reload_rows=[], failure=failure)


def _write_verdict(
    out_dir: Path,
    *,
    latency_rows: Sequence[Mapping[str, Any]],
    ap_rows: Sequence[Mapping[str, Any]],
    reload_rows: Sequence[Mapping[str, Any]],
    failure: str,
) -> None:
    baseline_lat = next((row for row in latency_rows if row.get("variant") == BASELINE_VARIANT), {})
    pruned_lat = next((row for row in latency_rows if row.get("variant") == PRUNED_VARIANT), {})
    baseline_ap = next((row for row in ap_rows if row.get("variant") == BASELINE_VARIANT), {})
    pruned_ap = next((row for row in ap_rows if row.get("variant") == PRUNED_VARIANT), {})
    used_real = all(bool(row.get("validation_dataloader_used")) for row in reload_rows) if reload_rows else False
    reloaded = all(bool(row.get("reload_success")) for row in reload_rows) if reload_rows else False
    lines = [
        "# v10.7.1 Real Val500 Stage512 Reload Eval Verdict",
        "",
        f"failure: {failure or 'none'}",
        f"- real validation dataloader used: {used_real}",
        f"- models loaded from exported model_object artifacts: {reloaded}",
        f"- evaluated frames baseline/pruned: {baseline_lat.get('evaluated_frames', '')}/{pruned_lat.get('evaluated_frames', '')}",
        f"- baseline AP@0.30: {baseline_ap.get('AP@0.30', '')}",
        f"- pruned AP@0.30: {pruned_ap.get('AP@0.30', '')}",
        f"- AP_drop_vs_baseline: {pruned_ap.get('AP_drop_vs_baseline', '')}",
        f"- baseline forward p50 ms: {baseline_lat.get('forward_latency_p50', '')}",
        f"- pruned forward p50 ms: {pruned_lat.get('forward_latency_p50', '')}",
        f"- p50 speedup vs baseline: {pruned_lat.get('speedup_p50_vs_baseline', '')}",
        "- synthetic v10.7 p50 speedup reference: 1.0659",
        "",
        "Interpretation:",
        "- If AP drops substantially, the Stage512 per-group8 candidate is only worth continuing with recovery/distillation experiments, not direct deployment.",
        "- If AP drop is acceptable and real-val p50 speedup remains close to the synthetic 1.0659 reference, this candidate can move to the next precision-recovery stage.",
        "- This verdict does not use synthetic batches as validation evidence.",
    ]
    (out_dir / "real_val500_stage512_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_val500(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args))
    attempted: list[str] = []
    gpu_samples: list[dict[str, Any]] = []
    selected_index: int | None = None
    failure = ""
    try:
        device, selected_index = _select_device(args, out_dir)
        if selected_index is not None:
            before = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_before.json", before)
            gpu_samples.append({"sample_reason": "before", **before})

        HEALLiDARAdapter, build_dataset, evaluate_one_model, setup_logger, attempted = _load_eval_helpers()
        logger = setup_logger(out_dir)
        adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
        dataset, loader = build_dataset(adapter, args.model_config, batch_size=1, num_workers=args.num_workers)

        models_dir = Path(args.models_dir)
        variants = [BASELINE_VARIANT, PRUNED_VARIANT]
        models: dict[str, nn.Module] = {}
        reload_rows: list[dict[str, Any]] = []
        for variant in variants:
            path = _model_artifact_path(models_dir, variant)
            reload_success = False
            smoke = {"reload_forward_smoke_passed": False, "failure_reason": "not_run"}
            model: nn.Module | None = None
            reason = ""
            try:
                model = load_model_object_artifact(path, device=device)
                reload_success = True
                smoke = _smoke_real_batch(model, loader, device)
                models[variant] = model
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"
            reload_rows.append(
                {
                    "variant": variant,
                    "model_object_path": str(path),
                    "reload_success": reload_success,
                    "reload_forward_smoke_passed": bool(smoke.get("reload_forward_smoke_passed", False)),
                    "validation_dataloader_used": True,
                    "synthetic_used": False,
                    "failure_reason": reason or str(smoke.get("failure_reason", "")),
                }
            )
        write_json(out_dir / "real_val500_reload_report.json", reload_rows)
        if len(models) != 2 or any(not row["reload_forward_smoke_passed"] for row in reload_rows):
            raise RuntimeError("model_object_reload_or_real_batch_smoke_failed")

        summaries: dict[str, dict[str, Any]] = {}
        per_frame_rows: dict[str, list[dict[str, Any]]] = {}
        for idx, variant in enumerate(variants):
            rows, summary = evaluate_one_model(
                model=models[variant],
                checkpoint=str(_model_artifact_path(models_dir, variant)),
                metadata={},
                model_type=variant,
                dataset=dataset,
                loader=loader,
                device=device,
                round_id=idx,
                max_frames=int(args.eval_frames),
                warmup_frames=int(args.warmup_frames),
                logger=logger,
            )
            summaries[variant] = summary
            per_frame_rows[variant] = rows
            write_csv(out_dir / f"{variant}_per_frame_latency.csv", rows)
            if selected_index is not None:
                sample = collect_gpu_state(selected_index)
                write_json(out_dir / f"gpu_state_after_{variant}.json", sample)
                gpu_samples.append({"sample_reason": f"after_{variant}", **sample})

        latency_rows: list[dict[str, Any]] = []
        baseline_row: dict[str, Any] | None = None
        for variant in variants:
            row = build_latency_report_row(
                variant=variant,
                summary=summaries[variant],
                rows=per_frame_rows[variant],
                baseline=baseline_row,
                requested_frames=int(args.eval_frames),
            )
            if variant == BASELINE_VARIANT:
                baseline_row = row
            latency_rows.append(row)
        write_csv(out_dir / "real_val500_latency.csv", _csv_ready_rows(latency_rows))

        ap_rows: list[dict[str, Any]] = []
        baseline_ap30: float | None = None
        for variant in variants:
            row = build_ap_report(
                variant=variant,
                summary=summaries[variant],
                baseline_ap30=baseline_ap30,
                requested_frames=int(args.eval_frames),
                failure_reason=str(summaries[variant].get("first_failure", "") or ""),
            )
            if variant == BASELINE_VARIANT:
                baseline_ap30 = float(row["AP@0.30"])
            ap_rows.append(row)
        write_json(out_dir / "real_val500_ap.json", ap_rows)

        if selected_index is not None:
            after = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_after.json", after)
            gpu_samples.append({"sample_reason": "after", **after})
        write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
        write_json(out_dir / "gpu_contamination_report.json", _contamination_report(gpu_samples, selected_index))
        write_json(
            out_dir / "failure_report.json",
            {
                "success": True,
                "failure_reason": "",
                "attempted_eval_entrypoints": attempted,
                "failure_traceback": "",
                "missing_adapter_reason": "",
                "next_required_patch": "",
            },
        )
        _write_verdict(out_dir, latency_rows=latency_rows, ap_rows=ap_rows, reload_rows=reload_rows, failure="")
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        _write_failure_outputs(out_dir, attempted=attempted, failure=failure, tb=traceback.format_exc())
        if selected_index is not None:
            try:
                after = collect_gpu_state(selected_index)
                write_json(out_dir / "gpu_state_after.json", after)
                gpu_samples.append({"sample_reason": "after_failure", **after})
                write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
                write_json(out_dir / "gpu_contamination_report.json", _contamination_report(gpu_samples, selected_index))
            except Exception:
                pass
        print(json.dumps({"success": False, "failure": failure, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.7.1 real validation 500-frame reload eval")
    parser.add_argument("--models-dir", default=DEFAULT_STAGE512_MODELS_DIR)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-frames", type=int, default=500)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rc = run_real_val500(args)
    if rc == 0:
        print(json.dumps({"success": True, "output_dir": args.output_dir}, indent=2, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
