#!/usr/bin/env python3
"""v10.6.1 save/reload evaluation for v10.6 Stage0 variants."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import sys
import time
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

from tools.latency_lut.select_idle_gpu_for_latency import (  # noqa: E402
    collect_gpu_state,
    snapshot_to_dict,
    wait_for_idle_gpu,
)
from tools.latency_lut.stage0_reblock8_aligned_prune_v106 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    _build_variants,
    _output_shapes,
    _outputs_finite,
    find_stage0_blocks,
    percentile,
    write_csv,
    write_json,
)


STAGE0_VARIANTS = (
    "stage0_reblock16_no_prune",
    "stage0_reblock8_no_prune",
    "stage0_reblock8_prune_pergroup8_all_blocks",
)


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def artifact_paths_for_variant(output_dir: Path, variant: str) -> dict[str, Path]:
    models_dir = output_dir / "models"
    return {
        "model_object": models_dir / f"{variant}_model_object.pth",
        "state_dict_manifest": models_dir / f"{variant}_state_dict_with_manifest.pth",
    }


def build_model_artifact_manifest(
    *,
    variant: str,
    modified_blocks: Sequence[Mapping[str, Any]],
    parameter_count_before: int,
    parameter_count_after: int,
) -> dict[str, Any]:
    if variant == "stage0_reblock16_no_prune":
        groups_new = 16
        hidden_width_new = 128
        per_group_new = 8
        hidden_keep_indices: list[int] = []
        keep_subgroups = []
    elif variant == "stage0_reblock8_no_prune":
        groups_new = 8
        hidden_width_new = 128
        per_group_new = 16
        hidden_keep_indices = []
        keep_subgroups = []
    else:
        groups_new = 8
        hidden_width_new = 64
        per_group_new = 8
        hidden_keep_indices = list(modified_blocks[0].get("hidden_keep_indices", [])) if modified_blocks else []
        keep_subgroups = list(modified_blocks[0].get("keep_subgroups_per_new_group", [0, 1])) if modified_blocks else [0, 1]
    return {
        "variant": variant,
        "modified_blocks": list(modified_blocks),
        "groups_old": 32,
        "groups_new": groups_new,
        "hidden_width_old": 128,
        "hidden_width_new": hidden_width_new,
        "per_group_old": 4,
        "per_group_new": per_group_new,
        "hidden_keep_indices": hidden_keep_indices,
        "keep_subgroups_per_new_group": keep_subgroups,
        "parameter_count_before": int(parameter_count_before),
        "parameter_count_after": int(parameter_count_after),
        "requires_architecture_patch": True,
    }


def save_model_artifacts(
    *,
    model: nn.Module,
    variant: str,
    models_dir: Path,
    model_config: str,
    checkpoint_source: str,
    manifest: Mapping[str, Any],
    architecture_changed: bool,
    notes: str = "physical Stage0 reblock/pruned model object",
) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_object": models_dir / f"{variant}_model_object.pth",
        "state_dict_manifest": models_dir / f"{variant}_state_dict_with_manifest.pth",
    }
    model_cpu = copy.deepcopy(model).cpu().eval()
    torch.save(
        {
            "variant": variant,
            "model_object": model_cpu,
            "config": model_config,
            "checkpoint_source": checkpoint_source,
            "architecture_changed": bool(architecture_changed),
            "notes": notes,
        },
        paths["model_object"],
    )
    torch.save(
        {
            "variant": variant,
            "state_dict": model_cpu.state_dict(),
            "config": model_config,
            "checkpoint_source": checkpoint_source,
            "architecture_manifest": dict(manifest),
            "requires_architecture_patch": True,
        },
        paths["state_dict_manifest"],
    )
    return paths


def _torch_load(path: Path, *, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_object_artifact(path: Path, *, device: torch.device) -> nn.Module:
    payload = _torch_load(path, map_location=device)
    model = payload["model_object"]
    if not isinstance(model, nn.Module):
        raise TypeError("model_object artifact does not contain an nn.Module")
    return model.to(device).eval()


def smoke_reloaded_model(model: nn.Module, sample: Any, *, forward_fn=None) -> dict[str, Any]:
    try:
        with torch.no_grad():
            output = forward_fn(model, sample) if forward_fn is not None else model(sample)
        return {
            "reload_forward_smoke_passed": True,
            "output_shapes": _output_shapes(output),
            "output_finite": _outputs_finite(output),
            "failure_reason": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "reload_forward_smoke_passed": False,
            "output_shapes": [],
            "output_finite": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _frame_latency_stats(model: nn.Module, samples: Sequence[Any], *, forward_fn, device: torch.device, warmup: int) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "mean": 0.0, "p90": 0.0, "p95": 0.0}
    with torch.no_grad():
        warm_sample = samples[0]
        for _ in range(max(0, int(warmup))):
            forward_fn(model, warm_sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times: list[float] = []
        for sample in samples:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward_fn(model, sample)
                end.record()
                torch.cuda.synchronize(device)
                times.append(float(start.elapsed_time(end)))
            else:
                t0 = _now_ms()
                forward_fn(model, sample)
                times.append(_now_ms() - t0)
    vals = [float(v) for v in times]
    return {
        "p50": statistics.median(vals),
        "mean": statistics.mean(vals),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
    }


def _eval_500(model: nn.Module, samples: Sequence[Any], *, forward_fn) -> dict[str, Any]:
    all_passed = True
    all_finite = True
    failure = ""
    for sample in samples:
        try:
            with torch.no_grad():
                output = forward_fn(model, sample)
            all_finite = all_finite and _outputs_finite(output)
        except Exception as exc:  # noqa: BLE001
            all_passed = False
            all_finite = False
            failure = f"{type(exc).__name__}: {exc}"
            break
    return {
        "num_frames": len(samples),
        "AP@0.30": None,
        "AP_drop_vs_baseline": None,
        "AP_unavailable_reason": "stable 500-frame AP eval helper is not available in this script; used fixed synthetic forward validation",
        "forward_all_passed": all_passed,
        "output_finite": all_finite,
        "failure_reason": failure,
    }


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


def _artifact_report_row(
    *,
    variant: str,
    model_path: Path | None,
    state_path: Path | None,
    reloaded: bool,
    smoke: Mapping[str, Any],
    architecture_changed: bool,
    parameter_count: int,
    modified_blocks: Sequence[Any],
    failure_reason: str = "",
) -> dict[str, Any]:
    file_exists = True if model_path is None else model_path.is_file()
    if state_path is not None:
        file_exists = file_exists and state_path.is_file()
    return {
        "variant": variant,
        "model_object_path": str(model_path or ""),
        "state_dict_manifest_path": str(state_path or ""),
        "file_exists": file_exists,
        "reload_model_object_success": bool(reloaded),
        "reload_forward_smoke_passed": bool(smoke.get("reload_forward_smoke_passed", False)),
        "architecture_changed": bool(architecture_changed),
        "parameter_count": int(parameter_count),
        "modified_blocks": list(modified_blocks),
        "failure_reason": failure_reason or str(smoke.get("failure_reason", "")),
    }


def write_microbench_scope_explanation(output_dir: Path) -> None:
    text = """# v10.6 Microbenchmark Scope Explanation

1. conv2-only:
   This only tests an isolated grouped Conv2d layer. It does not include the real bottleneck block or the rest of HEAL forward.
   Shapes:
   - baseline: C=128, groups=32, per_group=4
   - reblock16: C=128, groups=16, per_group=8
   - reblock8: C=128, groups=8, per_group=16
   - reblock8_pruned: C=64, groups=8, per_group=8

2. bottleneck block:
   This tests a local bottleneck block including conv1/bn1/conv2/bn2/conv3 and local residual behavior. It is closer to the real model than conv2-only, but it is still not a complete HEAL forward.

3. whole-model forward:
   Whole-model forward latency is the deciding measurement for real acceleration. v10.6 in-memory variants showed that local microbench speedups did not translate into a reusable exported model artifact.
"""
    (output_dir / "microbench_scope_explanation.md").write_text(text, encoding="utf-8")


def run_stage0_save_reload(args: argparse.Namespace, out_dir: Path, device: torch.device, selected_index: int | None) -> dict[str, Any]:
    from heal_compress.utils.model_utils import resolve_device
    from heal_compress.pruning.model_io import load_heal_model, setup_logger

    logger = setup_logger(out_dir)
    args.device = str(device)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, adapter = load_heal_model(args, device, logger)
    model.eval()
    parameter_count_before = count_parameters(model)
    block_rows = find_stage0_blocks(model)
    eligible_blocks = [row for row in block_rows if row.get("eligible")]
    if not eligible_blocks:
        raise RuntimeError("stage0_blocks_not_found")

    variants, transform_report = _build_variants(model, eligible_blocks, device)
    write_json(out_dir / "stage0_reblock8_aligned_prune_transform_report_v1061.json", transform_report)
    models_dir = out_dir / "models"
    artifact_paths: dict[str, dict[str, Path]] = {}
    for variant in STAGE0_VARIANTS:
        variant_model = variants[variant]
        modified = transform_report["variant_transforms"][variant].get("blocks_pruned") or transform_report["variant_transforms"][variant].get("modules_reblocked", [])
        manifest = build_model_artifact_manifest(
            variant=variant,
            modified_blocks=modified,
            parameter_count_before=parameter_count_before,
            parameter_count_after=count_parameters(variant_model),
        )
        artifact_paths[variant] = save_model_artifacts(
            model=variant_model,
            variant=variant,
            models_dir=models_dir,
            model_config=args.model_config,
            checkpoint_source=args.checkpoint,
            manifest=manifest,
            architecture_changed=True,
        )

    reload_dir = out_dir / "reload_eval_500"
    reload_dir.mkdir(parents=True, exist_ok=True)
    samples = [adapter.build_synthetic_batch(model) for _ in range(int(args.eval_frames))]
    reloaded_models: dict[str, nn.Module] = {"baseline": model}
    artifact_rows: list[dict[str, Any]] = []
    baseline_smoke = smoke_reloaded_model(model, samples[0], forward_fn=adapter.forward_for_task)
    artifact_rows.append(
        _artifact_report_row(
            variant="baseline",
            model_path=None,
            state_path=None,
            reloaded=True,
            smoke=baseline_smoke,
            architecture_changed=False,
            parameter_count=count_parameters(model),
            modified_blocks=[],
        )
    )
    for variant in STAGE0_VARIANTS:
        paths = artifact_paths[variant]
        failure = ""
        reloaded = False
        smoke: dict[str, Any] = {"reload_forward_smoke_passed": False, "failure_reason": "not_run"}
        try:
            reloaded_model = load_model_object_artifact(paths["model_object"], device=device)
            reloaded_models[variant] = reloaded_model
            reloaded = True
            smoke = smoke_reloaded_model(reloaded_model, samples[0], forward_fn=adapter.forward_for_task)
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        modified = transform_report["variant_transforms"][variant].get("blocks_pruned") or transform_report["variant_transforms"][variant].get("modules_reblocked", [])
        artifact_rows.append(
            _artifact_report_row(
                variant=variant,
                model_path=paths["model_object"],
                state_path=paths["state_dict_manifest"],
                reloaded=reloaded,
                smoke=smoke,
                architecture_changed=True,
                parameter_count=count_parameters(reloaded_models[variant]) if variant in reloaded_models else 0,
                modified_blocks=modified,
                failure_reason=failure,
            )
        )
    write_json(reload_dir / "saved_model_artifact_report.json", artifact_rows)

    eval_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    raw_stats: dict[str, dict[str, float]] = {}
    for variant, variant_model in reloaded_models.items():
        eval_row = {"variant": variant, **_eval_500(variant_model, samples, forward_fn=adapter.forward_for_task)}
        eval_rows.append(eval_row)
        raw_stats[variant] = _frame_latency_stats(
            variant_model,
            samples,
            forward_fn=adapter.forward_for_task,
            device=device,
            warmup=args.latency_warmup,
        )
        if selected_index is not None:
            write_json(reload_dir / f"gpu_state_after_{variant}.json", collect_gpu_state(selected_index))
    baseline_stats = raw_stats["baseline"]
    for variant, stats in raw_stats.items():
        latency_rows.append(
            {
                "variant": variant,
                "num_frames": int(args.eval_frames),
                "forward_latency_p50": stats["p50"],
                "forward_latency_mean": stats["mean"],
                "forward_latency_p90": stats["p90"],
                "forward_latency_p95": stats["p95"],
                "total_latency_p50": "",
                "total_latency_mean": "",
                "speedup_p50_vs_baseline": baseline_stats["p50"] / stats["p50"] if stats["p50"] > 0 else 0.0,
                "speedup_mean_vs_baseline": baseline_stats["mean"] / stats["mean"] if stats["mean"] > 0 else 0.0,
                "notes": "500 fixed synthetic forward frames; total latency not available",
            }
        )
    write_csv(reload_dir / "reload_latency_500.csv", latency_rows)
    write_json(reload_dir / "reload_eval_500.json", eval_rows)
    return {"artifact_rows": artifact_rows, "latency_rows": latency_rows, "eval_rows": eval_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.6.1 save/reload Stage0 variant evaluation")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-frames", type=int, default=500)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-repeat", type=int, default=300)
    parser.add_argument("--save-model-artifacts", action="store_true")
    parser.add_argument("--output-dir", default="outputs/latency_lut/stage0_reblock8_aligned_prune_v106")
    return parser.parse_args(argv)


def _contamination_report(samples: Sequence[Mapping[str, Any]], selected_index: int | None) -> dict[str, Any]:
    current_pid = os.getpid()
    external: list[dict[str, Any]] = []
    for sample in samples:
        for process in sample.get("running_processes", []) or []:
            if int(process.get("pid", -1)) != current_pid:
                external.append(dict(process))
    return {"selected_gpu_index": selected_index, "current_pid": current_pid, "latency_contamination_risk": bool(external), "external_processes_seen": external}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_microbench_scope_explanation(out_dir)
    write_json(out_dir / "run_config_v1061.json", vars(args))
    gpu_samples: list[dict[str, Any]] = []
    selected_index: int | None = None
    failure = ""
    try:
        device, selected_index = _select_device(args, out_dir)
        if selected_index is not None:
            before = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_before_v1061.json", before)
            gpu_samples.append({"sample_reason": "before_v1061", **before})
        run_stage0_save_reload(args, out_dir, device, selected_index)
        if selected_index is not None:
            after = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_after_v1061.json", after)
            gpu_samples.append({"sample_reason": "after_v1061", **after})
        write_json(out_dir / "gpu_state_during_samples_v1061.json", gpu_samples)
        write_json(out_dir / "gpu_contamination_report_v1061.json", _contamination_report(gpu_samples, selected_index))
        write_json(out_dir / "failure_report_v1061.json", {"success": True, "failure_reason": "", "traceback": ""})
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        write_json(out_dir / "failure_report_v1061.json", {"success": False, "failure_reason": failure, "traceback": traceback.format_exc()})
    print(json.dumps({"success": not failure, "failure": failure, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
    return 0 if not failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
