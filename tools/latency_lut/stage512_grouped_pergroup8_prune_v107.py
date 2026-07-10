#!/usr/bin/env python3
"""v10.7 Stage512 grouped Conv2d per-group 16->8 pruning experiment."""

from __future__ import annotations

import argparse
import copy
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
    _eval_500,
    _frame_latency_stats,
    count_parameters,
    load_model_object_artifact,
    save_model_artifacts,
    smoke_reloaded_model,
)
from tools.latency_lut.select_idle_gpu_for_latency import (  # noqa: E402
    collect_gpu_state,
    snapshot_to_dict,
    wait_for_idle_gpu,
)
from tools.latency_lut.stage0_reblock8_aligned_prune_v106 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    _bn_shape,
    _conv_shape,
    _new_conv_like,
    _outputs_finite,
    _prune_batchnorm,
    _prune_conv_in,
    _prune_conv_out,
    percentile,
    write_csv,
    write_json,
)


BASELINE_VARIANT = "baseline"
PRUNED_VARIANT = "stage512_pergroup16_to8_prune_all_blocks"


def build_stage512_keep_indices() -> list[int]:
    keep: list[int] = []
    for group_id in range(32):
        base = group_id * 16
        keep.extend(range(base, base + 8))
    return keep


def _prune_stage512_grouped_conv2(conv: nn.Conv2d, keep_indices: Sequence[int]) -> nn.Conv2d:
    keep = [int(v) for v in keep_indices]
    if int(conv.groups) != 32 or int(conv.in_channels) != 512 or int(conv.out_channels) != 512:
        raise ValueError("conv2 must have C_in=C_out=512 and groups=32")
    if len(keep) != 256:
        raise ValueError("Stage512 keep_indices must contain 256 channels")
    new_conv = _new_conv_like(conv, in_channels=256, out_channels=256, groups=32, bias=conv.bias is not None)
    with torch.no_grad():
        for group_id in range(32):
            group_keep = [idx for idx in keep if group_id * 16 <= idx < (group_id + 1) * 16]
            if len(group_keep) != 8:
                raise ValueError("each Stage512 group must keep exactly 8 channels")
            local_in = torch.as_tensor([idx - group_id * 16 for idx in group_keep], dtype=torch.long, device=conv.weight.device)
            for local_out, old_oc in enumerate(group_keep):
                new_oc = group_id * 8 + local_out
                new_conv.weight[new_oc].copy_(conv.weight[old_oc].index_select(0, local_in))
                if conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias[new_oc].copy_(conv.bias[old_oc])
    return new_conv


def prune_stage512_block_hidden_width(block: nn.Module) -> dict[str, Any]:
    required = ("conv1", "bn1", "conv2", "bn2", "conv3")
    missing = [name for name in required if not hasattr(block, name)]
    if missing:
        raise ValueError(f"block is missing required modules: {missing}")
    conv1, bn1, conv2, bn2, conv3 = block.conv1, block.bn1, block.conv2, block.bn2, block.conv3
    if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d) or not isinstance(conv3, nn.Conv2d):
        raise TypeError("conv1/conv2/conv3 must be Conv2d")
    if not isinstance(bn1, nn.modules.batchnorm._BatchNorm) or not isinstance(bn2, nn.modules.batchnorm._BatchNorm):
        raise TypeError("bn1/bn2 must be BatchNorm")
    if int(conv2.groups) != 32 or int(conv2.in_channels) != 512 or int(conv2.out_channels) != 512:
        raise ValueError("conv2 must be Stage512 grouped conv: C_in=C_out=512, groups=32")
    if int(conv1.out_channels) != 512 or int(bn1.num_features) != 512 or int(bn2.num_features) != 512 or int(conv3.in_channels) != 512:
        raise ValueError("block hidden width must be 512 before pruning")

    keep = build_stage512_keep_indices()
    old_conv3_out = int(conv3.out_channels)
    block.conv1 = _prune_conv_out(conv1, keep)
    block.bn1 = _prune_batchnorm(bn1, keep)
    block.conv2 = _prune_stage512_grouped_conv2(conv2, keep)
    block.bn2 = _prune_batchnorm(bn2, keep)
    block.conv3 = _prune_conv_in(conv3, keep)
    return {
        "hidden_width_before": 512,
        "hidden_width_after": 256,
        "groups_before": 32,
        "groups_after": 32,
        "per_group_before": 16,
        "per_group_after": 8,
        "hidden_keep_indices": keep,
        "keep_strategy": "keep_first8_per_group",
        "conv3_out_channels_unchanged": int(block.conv3.out_channels) == old_conv3_out,
    }


def _block_shapes(block: nn.Module) -> dict[str, Any]:
    return {
        "conv1": _conv_shape(block.conv1),
        "bn1": _bn_shape(block.bn1),
        "conv2": _conv_shape(block.conv2),
        "bn2": _bn_shape(block.bn2),
        "conv3": _conv_shape(block.conv3),
    }


def find_stage512_blocks(model: nn.Module) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for conv_name, module in modules.items():
        if not (
            isinstance(module, nn.Conv2d)
            and int(module.groups) == 32
            and int(module.in_channels) == 512
            and int(module.out_channels) == 512
        ):
            continue
        block_name, _, leaf = conv_name.rpartition(".")
        row: dict[str, Any] = {
            "block_name": block_name,
            "conv2_module_name": conv_name,
            "eligible": False,
            "reject_reason": "",
            "before_shapes": {},
        }
        if leaf != "conv2" or not block_name or block_name not in modules:
            row["reject_reason"] = "matching conv is not a bottleneck conv2"
            rows.append(row)
            continue
        block = modules[block_name]
        try:
            if not all(hasattr(block, attr) for attr in ("conv1", "bn1", "conv2", "bn2", "conv3")):
                raise ValueError("parent block missing conv1/bn1/conv2/bn2/conv3")
            row["before_shapes"] = _block_shapes(block)
            checks = [
                (int(block.conv1.out_channels) == 512, "conv1 out_channels != 512"),
                (int(block.bn1.num_features) == 512, "bn1 num_features != 512"),
                (int(block.bn2.num_features) == 512, "bn2 num_features != 512"),
                (int(block.conv3.in_channels) == 512, "conv3 in_channels != 512"),
            ]
            failures = [reason for ok, reason in checks if not ok]
            if failures:
                row["reject_reason"] = "; ".join(failures)
            else:
                row["eligible"] = True
        except Exception as exc:  # noqa: BLE001
            row["reject_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _prune_stage512_blocks(model: nn.Module, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("eligible"):
            continue
        block_name = str(row["block_name"])
        block = model.get_submodule(block_name)
        before = _block_shapes(block)
        report = prune_stage512_block_hidden_width(block)
        after = _block_shapes(block)
        reports.append({"block_name": block_name, "conv2_module_name": row["conv2_module_name"], "before_shapes": before, "after_shapes": after, **report})
    return reports


def build_stage512_manifest(
    *,
    eligible_blocks: Sequence[Mapping[str, Any]],
    rejected_blocks: Sequence[Mapping[str, Any]],
    modified_blocks: Sequence[Mapping[str, Any]],
    parameter_count_before: int,
    parameter_count_after: int,
) -> dict[str, Any]:
    return {
        "variant": PRUNED_VARIANT,
        "modified_blocks": list(modified_blocks),
        "eligible_blocks": list(eligible_blocks),
        "rejected_blocks": list(rejected_blocks),
        "groups_before": 32,
        "groups_after": 32,
        "hidden_width_before": 512,
        "hidden_width_after": 256,
        "per_group_before": 16,
        "per_group_after": 8,
        "hidden_keep_indices": build_stage512_keep_indices(),
        "keep_strategy": "keep_first8_per_group",
        "parameter_count_before": int(parameter_count_before),
        "parameter_count_after": int(parameter_count_after),
        "requires_architecture_patch": True,
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
    model_path: Path,
    state_path: Path,
    reload_success: bool,
    smoke: Mapping[str, Any],
    parameter_count: int,
    modified_blocks: Sequence[Any],
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "model_object_path": str(model_path),
        "state_dict_manifest_path": str(state_path),
        "reload_success": bool(reload_success),
        "reload_forward_smoke_passed": bool(smoke.get("reload_forward_smoke_passed", False)),
        "parameter_count": int(parameter_count),
        "modified_blocks": list(modified_blocks),
        "failure_reason": failure_reason or str(smoke.get("failure_reason", "")),
    }


def _write_verdict(out_dir: Path, *, block_rows: Sequence[Mapping[str, Any]], latency_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], artifact_rows: Sequence[Mapping[str, Any]], failure: str) -> None:
    eligible = [row for row in block_rows if row.get("eligible")]
    baseline = next((row for row in latency_rows if row.get("variant") == BASELINE_VARIANT), {})
    pruned = next((row for row in latency_rows if row.get("variant") == PRUNED_VARIANT), {})
    pruned_eval = next((row for row in eval_rows if row.get("variant") == PRUNED_VARIANT), {})
    pruned_artifact = next((row for row in artifact_rows if row.get("variant") == PRUNED_VARIANT), {})
    lines = [
        "# Stage512 Per-Group 16 to 8 Prune v10.7 Verdict",
        "",
        f"failure: {failure or 'none'}",
        f"- found_512_groups32_pergroup16_blocks: {len(eligible)}",
        f"- pruned_to_256_groups32_pergroup8: {bool(eligible) and bool(pruned_artifact.get('reload_forward_smoke_passed', False))}",
        f"- model_exported: {bool(pruned_artifact.get('model_object_path'))}",
        f"- model_reloaded: {bool(pruned_artifact.get('reload_success', False))}",
        f"- reload_forward_smoke_passed: {bool(pruned_artifact.get('reload_forward_smoke_passed', False))}",
        f"- 500_frame_validation_passed: {bool(pruned_eval.get('forward_all_passed', False)) and bool(pruned_eval.get('output_finite', False))}",
        f"- forward_p50_speedup_vs_baseline: {pruned.get('speedup_p50_vs_baseline', '')}",
        f"- total_latency_speedup_vs_baseline: unavailable (total pipeline latency helper not available)",
        f"- AP_drop: unavailable; reason={pruned_eval.get('AP_unavailable_reason', '')}",
        f"- baseline_forward_p50_ms: {baseline.get('forward_latency_p50', '')}",
        f"- pruned_forward_p50_ms: {pruned.get('forward_latency_p50', '')}",
        "",
        "Conclusion:",
        "- This experiment answers whether original 512/groups32/per_group16 bottleneck hidden widths can be physically pruned to per_group8 and reloaded as a model artifact.",
        "- Accuracy was not recovered; if AP is unavailable or drops materially, any speed result is only a latency candidate.",
        "- If forward speedup is <=1, Stage512 per-group8 pruning should not be promoted without additional bottleneck analysis.",
        "- If forward speedup is >1, the next step is precision recovery or distillation before considering this in a formal decoder.",
    ]
    (out_dir / "reload_eval_500" / "stage512_pergroup8_prune_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage512_experiment(args: argparse.Namespace, out_dir: Path, device: torch.device, selected_index: int | None) -> None:
    from heal_compress.utils.model_utils import resolve_device
    from heal_compress.pruning.model_io import load_heal_model, setup_logger

    logger = setup_logger(out_dir)
    args.device = str(device)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    baseline, adapter = load_heal_model(args, device, logger)
    baseline.eval()
    parameter_count_before = count_parameters(baseline)
    block_rows = find_stage512_blocks(baseline)
    eligible = [row for row in block_rows if row.get("eligible")]
    rejected = [row for row in block_rows if not row.get("eligible")]
    if not eligible:
        raise RuntimeError("stage512_blocks_not_found")

    pruned = copy.deepcopy(baseline).to(device).eval()
    modified = _prune_stage512_blocks(pruned, eligible)
    parameter_count_after = count_parameters(pruned)
    block_rows_after: list[dict[str, Any]] = []
    modified_by_name = {row["block_name"]: row for row in modified}
    for row in block_rows:
        out_row = dict(row)
        if row.get("block_name") in modified_by_name:
            out_row["after_shapes"] = modified_by_name[row["block_name"]]["after_shapes"]
        block_rows_after.append(out_row)

    reload_dir = out_dir / "reload_eval_500"
    models_dir = out_dir / "models"
    reload_dir.mkdir(parents=True, exist_ok=True)
    write_json(reload_dir / "stage512_blocks_report.json", block_rows_after)
    write_json(out_dir / "stage512_blocks_report.json", block_rows_after)

    baseline_manifest = {
        "variant": BASELINE_VARIANT,
        "modified_blocks": [],
        "eligible_blocks": eligible,
        "rejected_blocks": rejected,
        "parameter_count_before": parameter_count_before,
        "parameter_count_after": parameter_count_before,
        "requires_architecture_patch": False,
    }
    pruned_manifest = build_stage512_manifest(
        eligible_blocks=eligible,
        rejected_blocks=rejected,
        modified_blocks=modified,
        parameter_count_before=parameter_count_before,
        parameter_count_after=parameter_count_after,
    )
    baseline_paths = save_model_artifacts(
        model=baseline,
        variant=BASELINE_VARIANT,
        models_dir=models_dir,
        model_config=args.model_config,
        checkpoint_source=args.checkpoint,
        manifest=baseline_manifest,
        architecture_changed=False,
        notes="baseline model object for Stage512 reload comparison",
    )
    pruned_paths = save_model_artifacts(
        model=pruned,
        variant=PRUNED_VARIANT,
        models_dir=models_dir,
        model_config=args.model_config,
        checkpoint_source=args.checkpoint,
        manifest=pruned_manifest,
        architecture_changed=True,
        notes="physical Stage512 per-group16-to8 pruned model object",
    )

    samples = [adapter.build_synthetic_batch(baseline) for _ in range(int(args.eval_frames))]
    artifact_rows: list[dict[str, Any]] = []
    reloaded_models: dict[str, nn.Module] = {}
    for variant, paths, blocks in [
        (BASELINE_VARIANT, baseline_paths, []),
        (PRUNED_VARIANT, pruned_paths, modified),
    ]:
        failure = ""
        reload_success = False
        smoke: dict[str, Any] = {"reload_forward_smoke_passed": False, "failure_reason": "not_run"}
        try:
            loaded = load_model_object_artifact(paths["model_object"], device=device)
            reloaded_models[variant] = loaded
            reload_success = True
            smoke = smoke_reloaded_model(loaded, samples[0], forward_fn=adapter.forward_for_task)
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        artifact_rows.append(
            _artifact_report_row(
                variant=variant,
                model_path=paths["model_object"],
                state_path=paths["state_dict_manifest"],
                reload_success=reload_success,
                smoke=smoke,
                parameter_count=count_parameters(reloaded_models[variant]) if variant in reloaded_models else 0,
                modified_blocks=blocks,
                failure_reason=failure,
            )
        )
    write_json(reload_dir / "saved_model_artifact_report.json", artifact_rows)

    eval_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    raw_stats: dict[str, dict[str, float]] = {}
    for variant, model in reloaded_models.items():
        eval_rows.append({"variant": variant, **_eval_500(model, samples, forward_fn=adapter.forward_for_task)})
        raw_stats[variant] = _frame_latency_stats(model, samples, forward_fn=adapter.forward_for_task, device=device, warmup=args.latency_warmup)
        if selected_index is not None:
            write_json(reload_dir / f"gpu_state_after_{variant}.json", collect_gpu_state(selected_index))
    baseline_stats = raw_stats[BASELINE_VARIANT]
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
    write_csv(reload_dir / "stage512_reload_latency_500.csv", latency_rows)
    write_json(reload_dir / "stage512_reload_eval_500.json", eval_rows)
    _write_verdict(out_dir, block_rows=block_rows_after, latency_rows=latency_rows, eval_rows=eval_rows, artifact_rows=artifact_rows, failure="")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.7 Stage512 grouped per-group8 pruning")
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
    parser.add_argument("--output-dir", default="outputs/latency_lut/stage512_pergroup8_prune_v107")
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
    write_json(out_dir / "run_config.json", vars(args))
    selected_index: int | None = None
    gpu_samples: list[dict[str, Any]] = []
    failure = ""
    try:
        device, selected_index = _select_device(args, out_dir)
        if selected_index is not None:
            before = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_before.json", before)
            gpu_samples.append({"sample_reason": "before", **before})
        run_stage512_experiment(args, out_dir, device, selected_index)
        if selected_index is not None:
            after = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_after.json", after)
            gpu_samples.append({"sample_reason": "after", **after})
        write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
        write_json(out_dir / "gpu_contamination_report.json", _contamination_report(gpu_samples, selected_index))
        write_json(out_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        write_json(out_dir / "failure_report.json", {"success": False, "failure_reason": failure, "traceback": traceback.format_exc()})
        try:
            _write_verdict(out_dir, block_rows=[], latency_rows=[], eval_rows=[], artifact_rows=[], failure=failure)
        except Exception:
            pass
    print(json.dumps({"success": not failure, "failure": failure, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
    return 0 if not failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
