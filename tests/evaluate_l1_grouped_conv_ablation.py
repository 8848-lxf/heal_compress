#!/usr/bin/env python3
"""Unified evaluator for L1 grouped-conv pruning ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_UNIAD = _ROOT.parent
if str(_UNIAD) not in sys.path:
    sys.path.insert(0, str(_UNIAD))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heal_compress.utils.io_utils import ensure_dir, save_csv, save_json, save_text


MODEL_ORDER = [
    "baseline_original",
    "shared_local_mean_p25",
    "shared_local_mean_p50",
    "shared_local_mean_p75",
    "independent_group_topk_p25",
    "independent_group_topk_p50",
    "independent_group_topk_p75",
]

SUMMARY_COLUMNS = [
    "model_name",
    "checkpoint_path",
    "target_prune_ratio",
    "actual_prune_ratio",
    "importance_mode",
    "selection_mode",
    "group_conv_selection_mode",
    "group_conv_prune_mode",
    "group_conv_align",
    "align",
    "protect_residual_add",
    "allow_remove_groups",
    "gpu_id",
    "selected_gpu_id",
    "device",
    "structure_legal",
    "forward_sanity_check",
    "params",
    "params_removed",
    "params_removed_ratio",
    "coupled_channels_before",
    "coupled_channels_after",
    "coupled_channels_removed",
    "num_dependency_scopes",
    "num_coupled_channel_units",
    "num_atomic_prune_units",
    "num_concrete_pruning_groups",
    "num_grouped_conv_align_violations",
    "AP@0.3",
    "AP@0.5",
    "AP@0.7",
    "mAP",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "forward_mean_ms",
    "forward_p50_ms",
    "forward_p90_ms",
    "forward_p95_ms",
    "postprocess_mean_ms",
    "FPS",
    "speedup_vs_original",
    "AP_drop_vs_original",
    "data_loading_time_mean_ms",
    "data_loading_time_p50_ms",
    "data_loading_time_p90_ms",
    "data_loading_time_p95_ms",
    "data_loading_time_std_ms",
    "data_loading_time_num_frames",
    "data_to_gpu_time_mean_ms",
    "data_to_gpu_time_p50_ms",
    "data_to_gpu_time_p90_ms",
    "data_to_gpu_time_p95_ms",
    "data_to_gpu_time_std_ms",
    "data_to_gpu_time_num_frames",
    "total_time_std_ms",
    "forward_time_std_ms",
    "postprocess_time_std_ms",
    "num_frames",
    "speedup_basis",
    "target_unreachable",
    "unreachable_reason",
    "eval_status",
    "failure_reason",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "not_available"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float | str:
    return round(float(statistics.mean(values)), 6) if values else "not_available"


def _std(values: list[float]) -> float | str:
    return round(float(statistics.pstdev(values)), 6) if values else "not_available"


def _percentile(values: list[float], q: float) -> float | str:
    if not values:
        return "not_available"
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac), 6)


def _latency_stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": _mean(values),
        "p50_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
        "p95_ms": _percentile(values, 0.95),
        "std_ms": _std(values),
        "num_frames": len(values),
    }


def discover_model_manifests(experiment_root: str | Path) -> list[dict[str, Any]]:
    root = Path(experiment_root)
    manifests: list[dict[str, Any]] = []
    for name in MODEL_ORDER:
        path = root / name / "model_manifest.json"
        if path.is_file():
            data = _read_json(path, {})
            data.setdefault("model_name", name)
            data["_manifest_path"] = str(path)
            manifests.append(data)
    return manifests


def _eval_output_dir(output_dir: Path, model_name: str) -> Path:
    return output_dir / model_name


def build_eval_command(
    *,
    checkpoint: str,
    pruned_checkpoint: str | None,
    model_name: str,
    device: str,
    gpu_id: str,
    max_frames: int,
    warmup_frames: int,
    rounds: int,
    output_dir: Path,
) -> list[str]:
    eval_max_frames = 0 if int(max_frames) < 0 else int(max_frames)
    cmd = [
        sys.executable,
        "tests/test_prune_and_eval.py",
        "--original-checkpoint",
        checkpoint,
        "--rounds",
        str(rounds),
        "--gup-id",
        str(gpu_id),
        "--device",
        device,
        "--max-frames",
        str(eval_max_frames),
        "--warmup-frames",
        str(warmup_frames),
        "--output-dir",
        str(_eval_output_dir(output_dir, model_name)),
    ]
    if pruned_checkpoint is None:
        cmd.extend(["--eval-original", "true", "--eval-pruned", "false"])
    else:
        cmd.extend([
            "--pruned-checkpoint",
            pruned_checkpoint,
            "--eval-original",
            "false",
            "--eval-pruned",
            "true",
        ])
    return cmd


def run_eval_subprocess(cmd: list[str]) -> int:
    result = subprocess.run(cmd, cwd=_ROOT, text=True)
    return int(result.returncode)


def _load_eval_results_for_model(output_dir: Path, model_name: str, model_type: str) -> dict[str, Any]:
    model_dir = _eval_output_dir(output_dir, model_name)
    per_round = _read_csv_rows(model_dir / "per_round_summary.csv")
    rows = [row for row in per_round if row.get("model_type") == model_type]
    result: dict[str, Any] = {}
    if rows:
        for src, dst in [
            ("AP_0_30", "AP@0.3"),
            ("AP_0_50", "AP@0.5"),
            ("AP_0_70", "AP@0.7"),
            ("total_time_mean_ms", "latency_mean_ms"),
            ("total_time_p50_ms", "latency_p50_ms"),
            ("forward_time_mean_ms", "forward_mean_ms"),
            ("forward_time_p50_ms", "forward_p50_ms"),
            ("postprocess_time_mean_ms", "postprocess_mean_ms"),
            ("actual_frames", "num_frames"),
        ]:
            values = [_float(row.get(src), 0.0) for row in rows]
            result[dst] = round(float(statistics.mean(values)), 6) if values else "not_available"
        aps = [_float(result.get("AP@0.3")), _float(result.get("AP@0.5")), _float(result.get("AP@0.7"))]
        result["mAP"] = round(float(statistics.mean(aps)), 6)

    prefix = "baseline" if model_type == "baseline" else "pruned"
    frame_rows: list[dict[str, Any]] = []
    for path in sorted(model_dir.glob(f"{prefix}_per_frame_latency_round_*.csv")):
        frame_rows.extend(row for row in _read_csv_rows(path) if str(row.get("success")).lower() == "true")
    if not rows and not frame_rows:
        return {}
    for source, out_prefix in [
        ("data_loading_time_ms", "data_loading_time"),
        ("total_time_ms", "total_time"),
        ("forward_time_ms", "forward_time"),
        ("postprocess_time_ms", "postprocess_time"),
        ("data_to_gpu_time_ms", "data_to_gpu_time"),
    ]:
        values = [_float(row.get(source), 0.0) for row in frame_rows]
        stats = _latency_stats(values)
        for key, value in stats.items():
            result[f"{out_prefix}_{key}"] = value
    result["latency_p90_ms"] = result.get("total_time_p90_ms", "not_available")
    result["latency_p95_ms"] = result.get("total_time_p95_ms", "not_available")
    result["forward_p90_ms"] = result.get("forward_time_p90_ms", "not_available")
    result["forward_p95_ms"] = result.get("forward_time_p95_ms", "not_available")
    if result.get("latency_mean_ms") not in (None, "", "not_available", 0):
        result["FPS"] = round(1000.0 / _float(result["latency_mean_ms"]), 6)
    return result


def load_eval_results(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    out = Path(output_dir)
    results: dict[str, dict[str, Any]] = {}
    for name in MODEL_ORDER:
        model_type = "baseline" if name == "baseline_original" else "pruned"
        results[name] = _load_eval_results_for_model(out, name, model_type)
    return results


def _empty_row(manifest: dict[str, Any]) -> dict[str, Any]:
    row = {column: "not_available" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "model_name": manifest.get("model_name", ""),
            "checkpoint_path": manifest.get("pruned_model_path") or manifest.get("checkpoint_path") or manifest.get("source_checkpoint", ""),
            "target_prune_ratio": manifest.get("target_prune_ratio", 0.0),
            "actual_prune_ratio": manifest.get("actual_prune_ratio", 0.0),
            "importance_mode": manifest.get("importance_mode", "l1_norm"),
            "selection_mode": manifest.get("selection_mode", ""),
            "group_conv_selection_mode": manifest.get("group_conv_selection_mode", ""),
            "group_conv_prune_mode": manifest.get("group_conv_prune_mode", ""),
            "group_conv_align": manifest.get("group_conv_align", ""),
            "align": manifest.get("align", ""),
            "protect_residual_add": manifest.get("protect_residual_add", ""),
            "allow_remove_groups": manifest.get("allow_remove_groups", ""),
            "gpu_id": manifest.get("gpu_id", ""),
            "selected_gpu_id": manifest.get("selected_gpu_id", ""),
            "device": manifest.get("device", ""),
            "structure_legal": manifest.get("structure_legal", ""),
            "forward_sanity_check": manifest.get("forward_sanity_check", ""),
            "params": manifest.get("params", manifest.get("pruned_params", "")),
            "params_removed": manifest.get("params_removed", ""),
            "params_removed_ratio": manifest.get("params_removed_ratio", manifest.get("actual_prune_ratio", "")),
            "coupled_channels_before": manifest.get("coupled_channels_before", ""),
            "coupled_channels_after": manifest.get("coupled_channels_after", ""),
            "coupled_channels_removed": manifest.get("coupled_channels_removed", ""),
            "num_dependency_scopes": manifest.get("num_dependency_scopes", ""),
            "num_coupled_channel_units": manifest.get("num_coupled_channel_units", ""),
            "num_atomic_prune_units": manifest.get("num_atomic_prune_units", ""),
            "num_concrete_pruning_groups": manifest.get("num_concrete_pruning_groups", ""),
            "num_grouped_conv_align_violations": manifest.get("num_grouped_conv_align_violations", ""),
            "target_unreachable": manifest.get("target_unreachable", False),
            "unreachable_reason": manifest.get("unreachable_reason", ""),
            "eval_status": "pending",
            "failure_reason": manifest.get("failure_reason", ""),
        }
    )
    return row


def build_summary_rows_from_manifests(
    manifests: list[dict[str, Any]],
    eval_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        row = _empty_row(manifest)
        model_name = str(manifest.get("model_name", ""))
        if manifest.get("status") not in (None, "", "success"):
            row["eval_status"] = "skipped"
            row["failure_reason"] = manifest.get("failure_reason") or manifest.get("status")
            rows.append(row)
            continue
        if manifest.get("checkpoint_type") == "pruned":
            if manifest.get("structure_legal") is False:
                row["eval_status"] = "skipped"
                row["failure_reason"] = "structure_illegal"
                rows.append(row)
                continue
            if manifest.get("forward_sanity_check") is False:
                row["eval_status"] = "skipped"
                row["failure_reason"] = "forward_sanity_failed"
                rows.append(row)
                continue
        result = eval_results.get(model_name, {})
        for key, value in result.items():
            if key in row:
                row[key] = value
        if result:
            row["eval_status"] = "success"
        else:
            row["eval_status"] = "failed"
            row["failure_reason"] = row.get("failure_reason") or "evaluation_outputs_missing"
        rows.append(row)

    baseline = next((row for row in rows if row["model_name"] == "baseline_original"), None)
    baseline_forward = _float(baseline.get("forward_p50_ms") if baseline else "not_available", 0.0)
    baseline_map = _float(baseline.get("mAP") if baseline else "not_available", 0.0)
    for row in rows:
        forward = _float(row.get("forward_p50_ms"), 0.0)
        row["speedup_basis"] = "forward_p50_ms"
        row["speedup_vs_original"] = (
            round((baseline_forward - forward) / baseline_forward, 6)
            if baseline_forward > 0 and forward > 0
            else "not_available"
        )
        has_map = row.get("mAP") not in (None, "", "not_available")
        current_map = _float(row.get("mAP"), 0.0)
        row["AP_drop_vs_original"] = (
            round(baseline_map - current_map, 6)
            if baseline_map > 0 and has_map
            else "not_available"
        )
    return rows


def _subset(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "not_available") for column in columns} for row in rows]


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# L1 Grouped Conv Ablation Report",
        "",
        "## 实验配置",
        "",
        "- importance_mode: l1_norm",
        "- selection_mode: constrained_global",
        "- group_conv_prune_mode: keep_groups",
        "- remove_groups: disabled",
        "",
        "## 七个模型列表",
        "",
    ]
    for row in rows:
        lines.append(f"- {row.get('model_name')}: status={row.get('eval_status')}")
    lines.extend(["", "## 目标剪枝率与实际剪枝率", ""])
    for row in rows:
        lines.append(
            f"- {row.get('model_name')}: target={row.get('target_prune_ratio')} actual={row.get('actual_prune_ratio')}"
        )
    lines.extend(["", "## AP 与 Latency 对比", ""])
    for row in rows:
        lines.append(
            f"- {row.get('model_name')}: mAP={row.get('mAP')} AP@0.5={row.get('AP@0.5')} "
            f"forward_p50={row.get('forward_p50_ms')} p90={row.get('forward_p90_ms')} p95={row.get('forward_p95_ms')} "
            f"speedup={row.get('speedup_vs_original')}"
        )
    unreachable = [row for row in rows if str(row.get("target_unreachable")).lower() == "true"]
    lines.extend(["", "## 未达到目标剪枝率", ""])
    if unreachable:
        for row in unreachable:
            lines.append(f"- {row.get('model_name')}: {row.get('unreachable_reason')}")
    else:
        lines.append("- none")
    collapsed = [row for row in rows if _float(row.get("mAP"), 1.0) == 0.0 and row.get("eval_status") == "success"]
    lines.extend(["", "## AP 崩塌检查", ""])
    if collapsed:
        for row in collapsed:
            lines.append(f"- {row.get('model_name')}: mAP=0，可能由过高剪枝率、残差/分组约束下有效容量不足或后处理无检测框导致。")
    else:
        lines.append("- no zero-mAP model detected in summary")
    slower = [row for row in rows if isinstance(row.get("speedup_vs_original"), float) and row["speedup_vs_original"] < 0]
    lines.extend(["", "## Forward Latency 变慢检查", ""])
    if slower:
        for row in slower:
            lines.append(f"- {row.get('model_name')}: speedup={row.get('speedup_vs_original')}，需检查 kernel shape、Tensor Core 对齐、grouped conv 重排和后处理耗时。")
    else:
        lines.append("- no forward_p50 slowdown detected in summary")
    lines.extend(["", "## Grouped Conv Alignment Violations", ""])
    for row in rows:
        lines.append(f"- {row.get('model_name')}: {row.get('num_grouped_conv_align_violations')}")
    lines.extend(["", "## 结论", ""])
    lines.append("根据 ablation_eval_summary.csv 中 mAP、forward_p50/p90/p95、结构合法性和 align violation 综合判断最终策略。")
    save_text("\n".join(lines) + "\n", output_dir / "ablation_report.md")


def write_summary_outputs(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    stable_rows = [{column: row.get(column, "not_available") for column in SUMMARY_COLUMNS} for row in rows]
    save_csv(stable_rows, out / "ablation_eval_summary.csv")
    save_json(stable_rows, out / "ablation_eval_summary.json")
    save_csv(
        _subset(
            stable_rows,
            [
                "model_name",
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p90_ms",
                "latency_p95_ms",
                "forward_mean_ms",
                "forward_p50_ms",
                "forward_p90_ms",
                "forward_p95_ms",
                "postprocess_mean_ms",
                "FPS",
                "speedup_vs_original",
            ],
        ),
        out / "ablation_latency_summary.csv",
    )
    save_csv(
        _subset(
            stable_rows,
            [
                "model_name",
                "target_prune_ratio",
                "actual_prune_ratio",
                "params_removed_ratio",
                "coupled_channels_removed",
                "target_unreachable",
                "unreachable_reason",
                "structure_legal",
                "forward_sanity_check",
            ],
        ),
        out / "ablation_pruning_summary.csv",
    )
    save_csv(
        _subset(
            stable_rows,
            [
                "model_name",
                "group_conv_selection_mode",
                "group_conv_prune_mode",
                "group_conv_align",
                "num_grouped_conv_align_violations",
            ],
        ),
        out / "ablation_grouped_conv_summary.csv",
    )
    write_report(stable_rows, out)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = Path(args.experiment_root)
    output_dir = ensure_dir(args.output_dir)
    manifests = discover_model_manifests(experiment_root)
    commands: list[list[str]] = []

    if args.run_subprocess and (args.run_eval or args.run_latency):
        for manifest in manifests:
            model_name = manifest["model_name"]
            if manifest.get("status") not in (None, "", "success"):
                continue
            if manifest.get("checkpoint_type") == "pruned" and (
                manifest.get("structure_legal") is False or manifest.get("forward_sanity_check") is False
            ):
                continue
            pruned = None if model_name == "baseline_original" else manifest.get("pruned_model_path")
            cmd = build_eval_command(
                checkpoint=args.checkpoint,
                pruned_checkpoint=pruned,
                model_name=model_name,
                device=args.device,
                gpu_id=str(args.gpu_id),
                max_frames=args.max_frames,
                warmup_frames=args.warmup_frames,
                rounds=args.rounds,
                output_dir=output_dir,
            )
            commands.append(cmd)
            rc = run_eval_subprocess(cmd)
            if rc != 0 and args.fail_fast:
                raise RuntimeError(f"evaluation failed for {model_name}: returncode={rc}")

    eval_results = load_eval_results(output_dir)
    rows = build_summary_rows_from_manifests(manifests, eval_results)
    write_summary_outputs(rows, output_dir)
    save_json(
        {"commands": [" ".join(shlex.quote(part) for part in cmd) for cmd in commands]},
        output_dir / "eval_commands.json",
    )
    return {"rows": rows, "commands": commands}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate L1 grouped-conv ablation outputs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-eval", type=str2bool, default=True)
    parser.add_argument("--run-latency", type=str2bool, default=True)
    parser.add_argument("--run-subprocess", type=str2bool, default=True)
    parser.add_argument("--fail-fast", type=str2bool, default=False)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_evaluation(parse_args())
